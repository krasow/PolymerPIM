#include "allocator.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>

#include "logger.h"
#include "perfetto/trace.h"
#include "runtime.h"

namespace {
std::atomic<uint64_t> next_vector_id{1};

void assign_vector_id(detail::VectorDesc& vec) {
  vec.vector_id = next_vector_id.fetch_add(1, std::memory_order_relaxed);
}

size_t align8(size_t n) { return (n + 7) & ~size_t{7}; }

size_t total_allocated_footprint_from_layout(const detail::VectorDesc& vec) {
  size_t total = 0;
  for (const auto& segment : vec.desc) total += segment.allocated_bytes;
  return total;
}

// Every shard is allocated the same number of bytes -- see
// materialize_descriptor_layout for why.
size_t uniform_shard_bytes(size_t n, size_t reserved, size_t size_type,
                           size_t num_dpus) {
  const size_t elems_per_dpu = n / num_dpus;
  const size_t remainder = n % num_dpus;
  const size_t widest = elems_per_dpu + (remainder ? 1 : 0);
  return align8(widest * size_type + reserved);
}

size_t total_allocated_footprint(size_t n, size_t reserved, size_t size_type,
                                 size_t num_dpus) {
  if (n == 0) return 0;
  return num_dpus * uniform_shard_bytes(n, reserved, size_type, num_dpus);
}
}  // namespace

allocator::allocator(uint32_t start_addr, std::size_t dpu_mem,
                     std::size_t num_dpus)
    : start_addr_(start_addr), dpu_mem_(dpu_mem), num_dpus_(num_dpus) {
  // Ensure we don't start at address 0 to avoid NULL pointer confusion in JIT
  // kernels
  uint32_t effective_start = (start_addr_ == 0) ? 1024 : start_addr_;
  ptrs_.resize(num_dpus_, effective_start);
  sizes_.resize(num_dpus_, dpu_mem_ - (effective_start - start_addr_));
  offsets_.resize(num_dpus_, 0);
  broadcast_offset_ = 0;
  free_list_.resize(num_dpus_);
}

detail::VectorDescRef allocator::allocate_upmem_vector(std::size_t n,
                                                       std::size_t reserved,
                                                       std::size_t size_type,
                                                       bool lazy) {
  if (n == 0) {
    std::lock_guard<std::recursive_mutex> lock(this->lock);
    auto vec = std::make_shared<detail::VectorDesc>();
    assign_vector_id(*vec);
    vec->desc.resize(num_dpus_, {0, 0, 0});
    vec->ptr_allocated = true;
    vec->reserved_bytes = reserved;
    vec->element_size = size_type;
    vec->num_elements = 0;
    vec->allocated_footprint_bytes = 0;
    return vec;
  }
  bool uniform = (n % num_dpus_ == 0) && (n / num_dpus_ * size_type >= 8);
  {
    std::lock_guard<std::recursive_mutex> lock(this->lock);
    if (is_synchronized_ && uniform)
      return allocate_upmem_vector_broadcast(n, reserved, size_type, lazy);
    is_synchronized_ = false;
  }
  std::lock_guard<std::recursive_mutex> lock(this->lock);
  const size_t eff = n / num_dpus_, rem = n % num_dpus_;

  auto vec = std::make_shared<detail::VectorDesc>();
  assign_vector_id(*vec);
  vec->ptr_allocated = !lazy;
  vec->reserved_bytes = reserved;
  vec->element_size = size_type;
  vec->num_elements = n;
  vec->allocated_footprint_bytes =
      total_allocated_footprint(n, reserved, size_type, num_dpus_);
  if (lazy) {
    vec->needs_layout_materialization = true;
    return vec;
  }

  // Must match materialize_descriptor_layout exactly: the eager and lazy paths
  // produce descriptors for the same vector shape, and a transfer reads one
  // MRAM offset and one size for the whole DPU set.
  const size_t uniform_bytes =
      uniform_shard_bytes(n, reserved, size_type, num_dpus_);

  vec->desc.reserve(num_dpus_);
  for (size_t i = 0; i < num_dpus_; i++) {
    const size_t sz = (eff + (i < rem ? 1 : 0)) * size_type + reserved;
    vec->desc.push_back({raw_allocate(i, uniform_bytes), (uint32_t)sz,
                         (uint32_t)uniform_bytes});
  }
  return vec;
}

detail::VectorDescRef allocator::allocate_local_vector(std::size_t n,
                                                       std::size_t size_type) {
  std::lock_guard<std::recursive_mutex> lock(this->lock);
  auto vec = std::make_shared<detail::VectorDesc>();
  assign_vector_id(*vec);
  size_t sz = n * size_type;
  size_t aligned_sz = (sz + 7) & ~7;
  for (size_t i = 0; i < num_dpus_; i++) {
    vec->desc.push_back(
        {raw_allocate(i, aligned_sz), (uint32_t)sz, (uint32_t)aligned_sz});
  }
  vec->ptr_allocated = true;
  vec->reserved_bytes = 0;
  vec->element_size = size_type;
  vec->num_elements = n;
  vec->allocated_footprint_bytes = total_allocated_footprint_from_layout(*vec);
  return vec;
}

detail::VectorDescRef allocator::allocate_upmem_vector_broadcast(
    std::size_t n, std::size_t reserved, std::size_t size_type, bool lazy) {
  size_t sz = std::max((size_t)8, (n / num_dpus_) * size_type) + reserved;
  size_t aligned_sz = align8(sz);
  auto vec = std::make_shared<detail::VectorDesc>();
  assign_vector_id(*vec);
  vec->ptr_allocated = !lazy;
  vec->reserved_bytes = reserved;
  vec->element_size = size_type;
  vec->num_elements = n;
  vec->allocated_footprint_bytes = aligned_sz * num_dpus_;
  if (lazy) {
    vec->needs_layout_materialization = true;
    return vec;
  }

  uint32_t addr = raw_allocate(DPU_BROADCAST, aligned_sz);
  vec->desc.assign(num_dpus_, {addr, (uint32_t)sz, (uint32_t)aligned_sz});
  return vec;
}

void allocator::materialize_descriptor_layout(detail::VectorDesc* data) {
  if (!data || !data->needs_layout_materialization) return;

  size_t n = data->num_elements;
  size_t reserved = data->reserved_bytes;
  size_t size_type = data->element_size;
  if (n == 0) {
    data->desc.resize(num_dpus_, {0, 0, 0});
    data->needs_layout_materialization = false;
    return;
  }

  const size_t elems_per_dpu = n / num_dpus_;
  const size_t remainder = n % num_dpus_;

  // `size_bytes` is per-shard payload, so kernels still see the right element
  // count.  `allocated_bytes` is deliberately the SAME on every DPU: a host
  // transfer is one dpu_push_xfer, which applies a single size *and* a single
  // MRAM offset to the whole set.  Uneven allocations would both over-read the
  // short shards and let the per-DPU addresses drift out of lockstep.
  const size_t uniform = uniform_shard_bytes(n, reserved, size_type, num_dpus_);

  data->desc.reserve(num_dpus_);
  for (size_t i = 0; i < num_dpus_; i++) {
    const size_t sz =
        (elems_per_dpu + (i < remainder ? 1 : 0)) * size_type + reserved;
    data->desc.push_back({0, (uint32_t)sz, (uint32_t)uniform});
  }
  data->needs_layout_materialization = false;
}

void allocator::realize_allocation(detail::VectorDescRef data) {
  if (!data || data->ptr_allocated) return;
  std::lock_guard<std::recursive_mutex> lock(this->lock);
  if (data->ptr_allocated) return;  // double check after lock
  materialize_descriptor_layout(data.get());

  if (is_synchronized_) {
    uint32_t addr = raw_allocate(DPU_BROADCAST, data->desc[0].allocated_bytes);
    for (auto& s : data->desc) s.ptr = addr;
  } else {
    for (size_t i = 0; i < num_dpus_; i++)
      data->desc[i].ptr = raw_allocate(i, data->desc[i].allocated_bytes);
  }
  data->ptr_allocated = true;
}

uint32_t allocator::raw_allocate(int id, std::size_t n) {
  auto& fl = (id == DPU_BROADCAST) ? broadcast_free_list_ : free_list_[id];
  auto best = fl.end();
  size_t bsz = SIZE_MAX;
  for (auto it = fl.begin(); it != fl.end(); ++it)
    if (it->size >= n && it->size < bsz) {
      best = it;
      bsz = it->size;
    }

  if (best != fl.end()) {
    uint32_t addr = best->addr;
    if (best->size > n) {
      best->addr += n;
      best->size -= n;
    } else
      fl.erase(best);
    total_allocated_bytes_ += n * (id == DPU_BROADCAST ? num_dpus_ : 1);
    trace::counter("runtime", "total_bytes", total_allocated_bytes_);
    return addr;
  }

  uint32_t& off = (id == DPU_BROADCAST) ? broadcast_offset_ : offsets_[id];
  if (id == DPU_BROADCAST)
    off = *std::max_element(offsets_.begin(), offsets_.end());
  if (off + n > (id == DPU_BROADCAST ? sizes_[0] : sizes_[id])) {
    throw DpuOOMException();
  }

  uint32_t addr = (id == DPU_BROADCAST ? ptrs_[0] : ptrs_[id]) + off;
  off += n;
  if (id == DPU_BROADCAST) {
    std::fill(offsets_.begin(), offsets_.end(), off);
  }
  total_allocated_bytes_ += n * (id == DPU_BROADCAST ? num_dpus_ : 1);
  trace::counter("runtime", "total_bytes", total_allocated_bytes_);
  return addr;
}

void allocator::deallocate_upmem_vector(detail::VectorDesc* data) {
  if (!data->ptr_allocated || data->desc.empty()) return;
  data->ptr_allocated = false;
  std::lock_guard<std::recursive_mutex> lock(this->lock);
  if (is_synchronized_ && !data->desc.empty()) {
    raw_deallocate(DPU_BROADCAST, data->desc[0].ptr,
                   data->desc[0].allocated_bytes);
    return;
  }
  is_synchronized_ = false;
  for (size_t i = 0; i < num_dpus_; i++)
    raw_deallocate(i, data->desc[i].ptr, data->desc[i].allocated_bytes);
}

void allocator::deallocate_upmem_vector_broadcast(detail::VectorDesc* data) {
  deallocate_upmem_vector(data);
}

void allocator::raw_deallocate(int id, uint32_t addr, size_t sz) {
  auto& fl = (id == DPU_BROADCAST) ? broadcast_free_list_ : free_list_[id];
  auto it = std::find_if(fl.begin(), fl.end(),
                         [&](const FreeBlock& b) { return b.addr > addr; });
  auto ins = fl.insert(it, {addr, sz});
  if (ins != fl.begin()) {
    auto p = std::prev(ins);
    if (p->addr + p->size == ins->addr) {
      p->size += ins->size;
      fl.erase(ins);
      ins = p;
    }
  }
  auto nxt = std::next(ins);
  if (nxt != fl.end() && ins->addr + ins->size == nxt->addr) {
    ins->size += nxt->size;
    fl.erase(nxt);
  }

  uint32_t& off = (id == DPU_BROADCAST) ? broadcast_offset_ : offsets_[id];
  uint32_t base = (id == DPU_BROADCAST) ? ptrs_[0] : ptrs_[id];
  while (!fl.empty() && fl.back().addr + fl.back().size == base + off) {
    off -= fl.back().size;
    fl.pop_back();
  }
  if (id == DPU_BROADCAST) std::fill(offsets_.begin(), offsets_.end(), off);
  total_allocated_bytes_ -= sz * (id == DPU_BROADCAST ? num_dpus_ : 1);
  trace::counter("runtime", "total_bytes", total_allocated_bytes_);
}
