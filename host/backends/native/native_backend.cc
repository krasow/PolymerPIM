// DpuBackend over the UPMEM SDK.  One dpu_set spans every DPU, and the SDK's
// asynchronous calls on it already run in issue order, so that set is the
// stream.  Vectors live where `allocator` puts them.

#include <cstring>
#include <dpu>  // UPMEM rt syslib
#include <utility>

#include "allocator.h"
#include "backend.h"

#define CHECK_UPMEM(x) DPU_ASSERT(x)

namespace {

bool all_identical(const DPU_LAUNCH_ARGS* args, uint32_t n) {
  for (uint32_t i = 1; i < n; i++)
    if (std::memcmp(&args[0], &args[i], sizeof(args[0])) != 0) return false;
  return true;
}

dpu_error_t run_notification([[maybe_unused]] struct dpu_set_t stream,
                             [[maybe_unused]] uint32_t rank_id, void* data) {
  auto done = static_cast<std::function<void()>*>(data);
  (*done)();
  delete done;
  return DPU_OK;
}

// Reads placement straight from the allocator's per-DPU segments.  One object
// walks every DPU of a launch in turn, so prefix sums are built once.
class NativeShard final : public DpuBackend::Shard {
 public:
  uint32_t dpu() const override { return dpu_; }
  uint32_t offset(const detail::VectorDesc& vec) const override {
    return vec.desc[dpu_].ptr;
  }
  uint32_t size_bytes(const detail::VectorDesc& vec) const override {
    return vec.desc[dpu_].size_bytes;
  }
  // Accumulated, not dpu * stride: shard sizes can be ragged.
  uint32_t first_element(const detail::VectorDesc& vec) const override {
    for (const auto& [desc, prefix] : prefixes_)
      if (desc == &vec) return prefix[dpu_];
    std::vector<uint32_t> prefix(vec.desc.size());
    uint32_t sum = 0;
    for (size_t i = 0; i < vec.desc.size(); i++) {
      prefix[i] = sum;
      sum += vec.desc[i].size_bytes / vec.element_size;
    }
    prefixes_.emplace_back(&vec, std::move(prefix));
    return prefixes_.back().second[dpu_];
  }

  void at(uint32_t dpu) { dpu_ = dpu; }

 private:
  uint32_t dpu_ = 0;
  mutable std::vector<
      std::pair<const detail::VectorDesc*, std::vector<uint32_t>>>
      prefixes_;
};

class NativeBackend final : public DpuBackend {
 public:
  uint32_t open(uint32_t num_dpus, const std::string& binary) override {
    std::string profile = "backend=";
    profile += BACKEND;
    CHECK_UPMEM(dpu_alloc(num_dpus, profile.c_str(), &set_));
    CHECK_UPMEM(dpu_get_nr_dpus(set_, &num_dpus_));
    load(binary);

    constexpr size_t dpu_mem = 64 * 1024 * 1024;  // 64MB per DPU
    allocator_ = std::make_unique<allocator>(0, dpu_mem, num_dpus_);
    open_ = true;
    return num_dpus_;
  }

  void close() override {
    if (!open_) return;
    CHECK_UPMEM(dpu_free(set_));
    open_ = false;
  }

  uint32_t num_dpus() const override { return num_dpus_; }

  void load(const std::string& binary) override {
    CHECK_UPMEM(dpu_load(set_, binary.c_str(), nullptr));
  }

  detail::VectorDescRef allocate(size_t n, size_t reserved, size_t element_size,
                                 bool lazy) override {
    return allocator_->allocate_upmem_vector(n, reserved, element_size, lazy);
  }

  // One shard per DPU is exactly what sharding num_dpus * n_per_dpu
  // elements gives.
  detail::VectorDescRef allocate_per_dpu(size_t n_per_dpu, size_t reserved,
                                         size_t element_size,
                                         bool lazy) override {
    return allocate(num_dpus_ * n_per_dpu, reserved, element_size, lazy);
  }

  detail::VectorDescRef allocate_local(size_t n, size_t element_size) override {
    return allocator_->allocate_local_vector(n, element_size);
  }

  void realize(const detail::VectorDescRef& vec) override {
    allocator_->realize_allocation(vec);
  }

  void release(detail::VectorDesc& vec) override {
    allocator_->deallocate_upmem_vector(&vec);
  }

  void write(const detail::VectorDescRef& vec, const char* host) override {
    realize(vec);
    std::vector<void*> host_ptrs(num_dpus_);
    size_t element = 0;
    for (uint32_t idx_dpu = 0; idx_dpu < num_dpus_; idx_dpu++) {
      host_ptrs[idx_dpu] = const_cast<char*>(&host[element]);
      element += vec->desc[idx_dpu].size_bytes - vec->reserved_bytes;
    }
    push(DPU_XFER_TO_DPU, host_ptrs, vec->desc[0].ptr,
         vec->desc[0].allocated_bytes - vec->reserved_bytes);
  }

  detail::ShardLayout read_layout(const detail::VectorDesc& vec) override {
    return detail::shard_layout(vec);
  }

  void read(const detail::VectorDescRef& vec, char* host,
            size_t stride) override {
    realize(vec);
    std::vector<void*> host_ptrs(num_dpus_);
    for (uint32_t idx_dpu = 0; idx_dpu < num_dpus_; idx_dpu++)
      host_ptrs[idx_dpu] = &host[(size_t)idx_dpu * stride];
    push(DPU_XFER_FROM_DPU, host_ptrs, vec->desc[0].ptr, stride);
  }

  void launch(const std::vector<detail::VectorDescRef>& vectors,
              const FillArgs& fill, const InspectArgs& inspect) override {
    for (const auto& vec : vectors)
      if (vec) realize(vec);

    // Left uninitialised like the stack arrays this replaced; `fill` writes
    // what the kernel reads.
    thread_local std::unique_ptr<DPU_LAUNCH_ARGS[]> buffer;
    thread_local uint32_t capacity = 0;
    if (capacity < num_dpus_) {
      buffer.reset(new DPU_LAUNCH_ARGS[num_dpus_]);
      capacity = num_dpus_;
    }
    DPU_LAUNCH_ARGS* args = buffer.get();

    NativeShard shard;
    for (uint32_t i = 0; i < num_dpus_; i++) {
      shard.at(i);
      fill(args[i], shard);
    }
    if (inspect) inspect(args, num_dpus_);

    if (all_identical(args, num_dpus_)) {
      CHECK_UPMEM(dpu_broadcast_to(set_, "args", 0, &args[0], sizeof(args[0]),
                                   DPU_XFER_DEFAULT));
    } else {
      dpu_set_t dpu;
      uint32_t idx_dpu = 0;
      DPU_FOREACH(set_, dpu, idx_dpu) {
        CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx_dpu]));
      }
      CHECK_UPMEM(dpu_push_xfer(set_, DPU_XFER_TO_DPU, "args", 0,
                                sizeof(args[0]), DPU_XFER_DEFAULT));
    }
    CHECK_UPMEM(dpu_launch(set_, DPU_ASYNCHRONOUS));
  }

  void notify(std::function<void()> done) override {
    auto heap_done = new std::function<void()>(std::move(done));
    CHECK_UPMEM(dpu_callback(
        set_, &run_notification, heap_done,
        (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING |
                               DPU_CALLBACK_SINGLE_CALL)));
  }

  void sync() override { CHECK_UPMEM(dpu_sync(set_)); }

  void read_logs(FILE* out) override {
    dpu_set_t dpu;
    DPU_FOREACH(set_, dpu) { CHECK_UPMEM(dpu_log_read(dpu, out)); }
  }

 private:
  // Transfers host_ptrs[i] <-> DPU i's MRAM heap at `mram_offset`.
  void push(dpu_xfer_t direction, const std::vector<void*>& host_ptrs,
            uint32_t mram_offset, size_t bytes) {
    dpu_set_t dpu;
    uint32_t idx_dpu = 0;
    DPU_FOREACH(set_, dpu, idx_dpu) {
      CHECK_UPMEM(dpu_prepare_xfer(dpu, host_ptrs[idx_dpu]));
    }
    CHECK_UPMEM(dpu_push_xfer(set_, direction, DPU_MRAM_HEAP_POINTER_NAME,
                              mram_offset, bytes, DPU_XFER_ASYNC));
  }

  dpu_set_t set_{};
  uint32_t num_dpus_ = 0;
  bool open_ = false;
  std::unique_ptr<allocator> allocator_;
};

}  // namespace

std::unique_ptr<DpuBackend> make_native_backend() {
  return std::make_unique<NativeBackend>();
}
