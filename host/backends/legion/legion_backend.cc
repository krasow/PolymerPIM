// DpuBackend over upmem-legion (one processor and blocked MRAM memory per
// rank).  Vectors are regions placed by the mapper on first use; a kernel is
// an index launch with one point per rank of its placement.

#include <legion.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "backend.h"
#include "default_mapper.h"
#include "mapper.h"
#include "realm/upmem/upmem_api.h"
#include "realm/upmem/upmem_internal.h"
#include "realm/upmem/upmem_layout.h"
#include "realm/upmem/upmem_module.h"

#define CHECK_UPMEM(x) DPU_ASSERT(x)

namespace {

using namespace Legion;
using namespace polymerpim_legion;
using Realm::Upmem::UpmemModule;
using RankDPU = Realm::Upmem::DPU;

// Below LEGION_MAX_APPLICATION_TASK_ID.
enum : TaskID {
  KERNEL_TASK_ID = 0x504d1,
  WRITE_TASK_ID,
  READ_TASK_ID,
  TOP_LEVEL_TASK_ID,
};
constexpr FieldID FID_DATA = 1;

UpmemModule& upmem_module() {
  auto* mod = Realm::Runtime::get_runtime().get_module<UpmemModule>("upmem");
  if (!mod) throw std::runtime_error("Legion was built without UPMEM support");
  return *mod;
}

// Index of `rank` in the module's rank list.
uint32_t rank_index(const RankDPU* rank) {
  const auto& ranks = upmem_module().dpus;
  auto it = std::find(ranks.begin(), ranks.end(), rank);
  assert(it != ranks.end());
  return (uint32_t)(it - ranks.begin());
}

// A launch's region requirement and the vector it names.
struct Operand {
  const detail::VectorDesc* vec;
  PrivilegeMode privilege;
  size_t element_size;
  // Per-DPU vectors only: the elements each DPU holds.
  bool per_dpu;
  size_t payload_bytes;
};

// A piece's first element; the accessor must match the privilege.
template <PrivilegeMode PRIV>
const void* piece_ptr(const PhysicalRegion& region, const Rect<1>& rect,
                      size_t element_size) {
  FieldAccessor<PRIV, uint8_t, 1, coord_t,
                Realm::AffineAccessor<uint8_t, 1, coord_t>>
      acc(region, FID_DATA, element_size);
  return acc.ptr(rect.lo);
}

const void* piece_ptr(const PhysicalRegion& region, const Rect<1>& rect,
                      const Operand& op) {
  switch (op.privilege) {
    case LEGION_READ_ONLY:
      return piece_ptr<LEGION_READ_ONLY>(region, rect, op.element_size);
    case LEGION_WRITE_DISCARD:
      return piece_ptr<LEGION_WRITE_DISCARD>(region, rect, op.element_size);
    default:
      return piece_ptr<LEGION_READ_WRITE>(region, rect, op.element_size);
  }
}

struct Launch {
  DpuBackend::FillArgs fill;
  DpuBackend::InspectArgs inspect;
  std::vector<Operand> operands;  // in region requirement order
  std::string binary;
  // Per point; kept until the launch is reaped since the push is async.
  std::vector<std::vector<DPU_LAUNCH_ARGS>> args;
};

// Binary loaded on each rank; changed only by that rank's kernel task.
std::mutex resident_mutex;
std::map<const RankDPU*, std::string> resident;

// Loads `binary` on `rank` if needed; call with the rank's xfer mutex held.
void ensure_binary(RankDPU& rank, dpu_set_t* stream,
                   const std::string& binary) {
  {
    std::lock_guard<std::mutex> lock(resident_mutex);
    auto it = resident.find(&rank);
    if (it != resident.end() && it->second == binary) return;
  }
  // Earlier kernels must finish on their own binary.
  CHECK_UPMEM(dpu_sync(*stream));
  CHECK_UPMEM(dpu_load(*stream, binary.c_str(), nullptr));
  std::lock_guard<std::mutex> lock(resident_mutex);
  resident[&rank] = binary;
}

// One rank's view of a launch, from the instances mapped for its task.
class LegionShard final : public DpuBackend::Shard {
 public:
  LegionShard(const Launch& launch, const std::vector<PhysicalRegion>& regions,
              const RankDPU& rank, uint32_t first_dpu)
      : launch_(launch), first_dpu_(first_dpu) {
    for (size_t i = 0; i < regions.size(); i++) {
      const Operand& op = launch.operands[i];
      const Rect<1> rect = regions[i].get_bounds<1, coord_t>().bounds;
      Piece piece{};
      piece.first = rect.lo[0];
      if (rect.empty()) {
        pieces_.push_back(piece);
        continue;
      }
      const void* ptr = piece_ptr(regions[i], rect, op);
      if (!Realm::Upmem::get_mram_block(rank.mram->me, ptr, &piece.block,
                                        &piece.shard))
        throw std::runtime_error(
            "vector piece is not in a blocked rank memory");
      piece.base = (uintptr_t)ptr - REALM_UPMEM_RANK_PTR_BASE - piece.block;
      piece.local = piece.block / rank.nr_dpus;
      // Pieces are padded; the kernel sees only real elements.
      const coord_t end =
          std::min<coord_t>(rect.hi[0] + 1, (coord_t)op.vec->num_elements);
      piece.bytes = end > rect.lo[0] ? (end - rect.lo[0]) * op.element_size : 0;
      pieces_.push_back(piece);
    }
  }

  uint32_t dpu() const override { return first_dpu_ + dpu_; }

  uint32_t offset(const detail::VectorDesc& vec) const override {
    const Piece& p = pieces_[find(vec)];
    // Empty DPUs still get an address inside their shard.
    const uint64_t in_shard =
        lo(p) > dpu_start(p) ? std::min(lo(p) - dpu_start(p), p.shard) : 0;
    return REALM_UPMEM_MRAM_DATA_OFFSET + (uint32_t)(p.local + in_shard);
  }

  uint32_t size_bytes(const detail::VectorDesc& vec) const override {
    const size_t i = find(vec);
    const Operand& op = launch_.operands[i];
    if (op.per_dpu) return (uint32_t)(op.payload_bytes + vec.reserved_bytes);
    const Piece& p = pieces_[i];
    return (uint32_t)(hi(p) - lo(p));
  }

  uint32_t first_element(const detail::VectorDesc& vec) const override {
    const size_t i = find(vec);
    const Piece& p = pieces_[i];
    return (uint32_t)(p.first +
                      (lo(p) - p.base) / launch_.operands[i].element_size);
  }

  void at(uint32_t dpu) { dpu_ = dpu; }

 private:
  // Bytes [base, base + bytes) of a blocked allocation; DPU d holds
  // [d * shard, (d + 1) * shard).
  struct Piece {
    uint64_t block;  // rank address of the allocation
    uint64_t shard;
    uint64_t local;  // local offset of every DPU's shard
    uint64_t base;
    uint64_t bytes;
    coord_t first;  // vector index of the piece's first element
  };

  uint64_t dpu_start(const Piece& p) const { return dpu_ * p.shard; }
  // This DPU's part of the piece; empty when lo == hi.
  uint64_t lo(const Piece& p) const {
    return std::min(std::max(p.base, dpu_start(p)), p.base + p.bytes);
  }
  uint64_t hi(const Piece& p) const {
    return std::max(lo(p), std::min(p.base + p.bytes, dpu_start(p) + p.shard));
  }

  size_t find(const detail::VectorDesc& vec) const {
    for (size_t i = 0; i < launch_.operands.size(); i++)
      if (launch_.operands[i].vec == &vec) return i;
    throw std::logic_error("kernel argument names a vector it does not touch");
  }

  const Launch& launch_;
  uint32_t first_dpu_;
  uint32_t dpu_ = 0;
  std::vector<Piece> pieces_;
};

// Every sharded operand must give each DPU the same elements: nothing moves
// data between DPUs of a rank.
//
// TODO: intra-rank copies.  Legion sees one memory per rank and upmem-legion
// has no MRAM-to-MRAM channel.  A host-bounce channel would let Legion realign
// shards here, and enable ops that exchange data (shifts, gather, scans).
void check_aligned(const Launch& launch, const LegionShard& shard) {
  const Operand* ref = nullptr;
  for (const Operand& op : launch.operands) {
    if (op.per_dpu) continue;
    if (!ref) {
      ref = &op;
      continue;
    }
    const uint32_t ref_count = shard.size_bytes(*ref->vec) / ref->element_size;
    const uint32_t count = shard.size_bytes(*op.vec) / op.element_size;
    if (count == ref_count &&
        (count == 0 ||
         shard.first_element(*op.vec) == shard.first_element(*ref->vec)))
      continue;
    std::fprintf(stderr,
                 "polymerpim: DPU %u holds elements [%u, +%u) of one kernel "
                 "operand but [%u, +%u) of another; shards within a rank are "
                 "misaligned\n",
                 shard.dpu(), shard.first_element(*ref->vec), ref_count,
                 shard.first_element(*op.vec), count);
    std::abort();
  }
}

// Fills a rank's DPU arguments and starts the kernel; done when the rank
// drains.
int kernel_task(const Task* task, const std::vector<PhysicalRegion>& regions,
                Context, Runtime*) {
  Launch& launch = **static_cast<Launch* const*>(task->args);
  UpmemModule& mod = upmem_module();
  RankDPU* rank = mod.get_task_dpu();
  dpu_set_t* stream = mod.get_task_upmem_stream();
  assert(rank && stream);

  const uint32_t nr = rank->nr_dpus;
  LegionShard shard(launch, regions, *rank, rank_index(rank) * nr);
  std::vector<DPU_LAUNCH_ARGS>& args = launch.args[task->index_point[0]];
  args.resize(nr);
  for (uint32_t d = 0; d < nr; d++) {
    shard.at(d);
    check_aligned(launch, shard);
    launch.fill(args[d], shard);
  }
  if (launch.inspect) launch.inspect(args.data(), nr);

  // The UPMEM runtime isn't safe for concurrent calls on one rank.
  Realm::AutoLock<> al(rank->get_xfer_mutex());
  ensure_binary(*rank, stream, launch.binary);
  for (uint32_t d = 0; d < nr; d++)
    CHECK_UPMEM(dpu_prepare_xfer(rank->dpu_handles[d], &args[d]));
  // Async: a sync transfer behind pending async work can crash the SDK.
  CHECK_UPMEM(dpu_push_xfer(*stream, DPU_XFER_TO_DPU, "args", 0,
                            sizeof(DPU_LAUNCH_ARGS), DPU_XFER_ASYNC));
  CHECK_UPMEM(dpu_launch(*stream, DPU_ASYNCHRONOUS));
  return 0;
}

// Copies `bytes` from the start of a (possibly padded) region.
struct HostCopy {
  void* host;
  size_t bytes;
  size_t element_size;
};

void write_task(const Task* task, const std::vector<PhysicalRegion>& regions,
                Context, Runtime*) {
  const HostCopy& copy = *static_cast<const HostCopy*>(task->args);
  const Rect<1> rect = regions[0].get_bounds<1, coord_t>().bounds;
  FieldAccessor<LEGION_WRITE_DISCARD, uint8_t, 1, coord_t,
                Realm::AffineAccessor<uint8_t, 1, coord_t>>
      acc(regions[0], FID_DATA, copy.element_size);
  std::memcpy(acc.ptr(rect.lo), copy.host, copy.bytes);
}

void read_task(const Task* task, const std::vector<PhysicalRegion>& regions,
               Context, Runtime*) {
  const HostCopy& copy = *static_cast<const HostCopy*>(task->args);
  const Rect<1> rect = regions[0].get_bounds<1, coord_t>().bounds;
  FieldAccessor<LEGION_READ_ONLY, uint8_t, 1, coord_t,
                Realm::AffineAccessor<uint8_t, 1, coord_t>>
      acc(regions[0], FID_DATA, copy.element_size);
  std::memcpy(copy.host, acc.ptr(rect.lo), copy.bytes);
}

void register_tasks() {
  register_mapper();
  {
    TaskVariantRegistrar registrar(KERNEL_TASK_ID, "polymerpim_kernel");
    registrar.add_constraint(ProcessorConstraint(Processor::DPU_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int, kernel_task>(registrar,
                                                        "polymerpim_kernel");
  }
  {
    TaskVariantRegistrar registrar(WRITE_TASK_ID, "polymerpim_write");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<write_task>(registrar,
                                                  "polymerpim_write");
  }
  {
    TaskVariantRegistrar registrar(READ_TASK_ID, "polymerpim_read");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<read_task>(registrar, "polymerpim_read");
  }
}

// Legion flags, then $POLYMERPIM_LEGION_ARGS (later wins).  An 8-byte granule
// balances shards to a word; sysmem holds the host copies Legion keeps.
std::vector<std::string> legion_args(uint32_t num_dpus) {
  const uint32_t ranks =
      std::max<uint32_t>(1, (num_dpus + REALM_UPMEM_MAX_DPUS_PER_RANK - 1) /
                                REALM_UPMEM_MAX_DPUS_PER_RANK);
  const uint32_t sysmem_mb = std::min<uint32_t>(ranks * 4096, 65536);
  std::vector<std::string> args = {"polymerpim",
                                   "-ll:num_ranks",
                                   std::to_string(ranks),
                                   "-ll:dpu_layout",
                                   "blocked",
                                   "-ll:dpu_granule",
                                   "8",
                                   "-ll:csize",
                                   std::to_string(sysmem_mb)};
  if (const char* extra = std::getenv("POLYMERPIM_LEGION_ARGS")) {
    std::istringstream in(extra);
    for (std::string word; in >> word;) args.push_back(word);
  }
  return args;
}

class LegionBackend final : public DpuBackend {
 public:
  uint32_t open(uint32_t num_dpus, const std::string& binary) override {
    // Legion starts once per process and cannot be restarted.
    static bool started = false;
    if (started)
      throw std::runtime_error(
          "the Legion backend cannot be reopened after shutdown");
    started = true;

    register_tasks();
    argv_storage_ = legion_args(num_dpus);
    for (auto& arg : argv_storage_) argv_.push_back(arg.data());
    Runtime::start((int)argv_.size(), argv_.data(), true /*background*/);
    runtime_ = Runtime::get_runtime();
    ctx_ = runtime_->begin_implicit_task(TOP_LEVEL_TASK_ID, 0,
                                         Processor::LOC_PROC, "polymerpim");

    const auto& ranks = upmem_module().dpus;
    if (ranks.empty()) throw std::runtime_error("Legion found no DPU ranks");
    if (!Realm::Upmem::is_mram_blocked(ranks[0]->mram->me))
      throw std::runtime_error(
          "the Legion backend needs -ll:dpu_layout blocked");
    dpus_per_rank_ = ranks[0]->nr_dpus;
    granule_ = ranks[0]->granule;
    for (const RankDPU* rank : ranks)
      if (rank->nr_dpus != dpus_per_rank_)
        throw std::runtime_error(
            "the Legion backend needs every rank to have the same number of "
            "DPUs");
    num_ranks_ = (uint32_t)ranks.size();
    num_dpus_ = num_ranks_ * dpus_per_rank_;

    colors_ =
        runtime_->create_index_space(ctx_, Rect<1>(0, (coord_t)num_ranks_ - 1));

    load(binary);
    open_ = true;
    return num_dpus_;
  }

  void close() override {
    if (!open_) return;
    sync();
    for (const auto& [count, space] : colors_)
      runtime_->destroy_index_space(ctx_, space);
    runtime_->finish_implicit_task(ctx_);
    Runtime::wait_for_shutdown();
    open_ = false;
  }

  uint32_t num_dpus() const override { return num_dpus_; }

  // Kernel tasks load it on their own ranks (see ensure_binary).
  void load(const std::string& binary) override { binary_ = binary; }

  detail::VectorDescRef allocate(size_t n, size_t reserved, size_t element_size,
                                 bool) override {
    if (reserved != 0)
      throw std::logic_error(
          "the Legion backend has no reserved bytes for sharded vectors");
    // Padded to whole words so copies never take Realm's slow partial-word
    // path.
    const size_t unit = 8 / element_size;
    Storage s;
    s.element_size = element_size;
    s.region_elements = (std::max<size_t>(n, 1) + unit - 1) / unit * unit;
    s.region = make_region(s.region_elements, element_size);
    return add_vector(n, std::move(s));
  }

  // Each rank's piece is whole rows, one slot per DPU; sized once placed.
  detail::VectorDescRef allocate_per_dpu(size_t n_per_dpu, size_t reserved,
                                         size_t element_size, bool) override {
    const size_t slot = (n_per_dpu * element_size + reserved + 7) & ~size_t{7};
    const size_t rows = (slot + granule_ - 1) / granule_;
    Storage s;
    s.element_size = element_size;
    s.per_dpu = true;
    s.payload_bytes = n_per_dpu * element_size;
    s.piece_bytes = rows * granule_ * dpus_per_rank_;
    auto vec = add_vector(n_per_dpu * num_dpus_, std::move(s));
    vec->reserved_bytes = reserved;
    return vec;
  }

  // A per-DPU vector whose length is the local one.
  detail::VectorDescRef allocate_local(size_t n, size_t element_size) override {
    auto vec = allocate_per_dpu(n, 0, element_size, false);
    vec->num_elements = n;
    return vec;
  }

  void realize(const detail::VectorDescRef&) override {}

  void release(detail::VectorDesc& vec) override {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = storage_.find(&vec);
    if (it == storage_.end()) return;
    const Storage& s = it->second;
    if (s.region.exists()) {
      runtime_->destroy_logical_region(ctx_, s.region);
      runtime_->destroy_field_space(ctx_, s.region.get_field_space());
      runtime_->destroy_index_space(ctx_, s.region.get_index_space());
    }
    storage_.erase(it);
    vec.ptr_allocated = false;
  }

  void write(const detail::VectorDescRef& vec, const char* host) override {
    const Storage& s = storage_of(*vec);
    if (s.per_dpu)
      throw std::logic_error("per-DPU vectors are written only by kernels");
    // Waited on so the caller may free `host` once the event retires.
    HostCopy copy{const_cast<char*>(host), vec->num_elements * s.element_size,
                  s.element_size};
    TaskLauncher launcher(WRITE_TASK_ID, TaskArgument(&copy, sizeof(copy)));
    launcher.add_region_requirement(RegionRequirement(
        s.region, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, s.region));
    launcher.add_field(0, FID_DATA);
    runtime_->execute_task(ctx_, launcher).get_void_result();
  }

  // A per-DPU vector spans the ranks of the kernel that writes it.
  bool layout_ready(const detail::VectorDesc& vec) override {
    const Storage& s = storage_of(vec);
    return !s.per_dpu || s.placed;
  }

  detail::ShardLayout read_layout(const detail::VectorDesc& vec) override {
    const Storage& s = storage_of(vec);
    detail::ShardLayout layout;
    if (s.per_dpu) {
      if (!s.placed)
        throw std::logic_error(
            "per-DPU vector read before any kernel wrote it");
      layout.logical.assign(s.placement.ranks * dpus_per_rank_,
                            s.payload_bytes);
      layout.stride = s.payload_bytes;
    } else {
      const size_t bytes = vec.num_elements * s.element_size;
      layout.logical.assign(1, bytes);
      layout.stride = bytes;
    }
    for (size_t payload : layout.logical) layout.total_logical += payload;
    return layout;
  }

  void read(const detail::VectorDescRef& vec, char* host,
            size_t stride) override {
    const Storage& s = storage_of(*vec);
    if (!s.per_dpu) {
      read_region(s, host, vec->num_elements * s.element_size);
      return;
    }

    // DPU d's slot opens its shard of the rank's piece.
    const uint32_t ranks = s.placement.ranks;
    const size_t shard =
        realm_upmem_block_shard(s.piece_bytes, dpus_per_rank_, granule_);
    std::vector<char> pieces(s.piece_bytes * ranks);
    read_region(s, pieces.data(), pieces.size());
    for (uint32_t r = 0; r < ranks; r++)
      for (uint32_t d = 0; d < dpus_per_rank_; d++)
        std::memcpy(host + (size_t)(r * dpus_per_rank_ + d) * stride,
                    pieces.data() + r * s.piece_bytes + d * shard,
                    s.payload_bytes);
  }

  void launch(const std::vector<detail::VectorDescRef>& reads,
              const std::vector<detail::VectorDescRef>& writes,
              const FillArgs& fill, const InspectArgs& inspect) override {
    reap(false);
    auto pending = std::make_unique<Launch>(Launch{fill, inspect, {}, binary_});
    std::vector<Storage*> regions;
    auto add = [&](const detail::VectorDescRef& vec, PrivilegeMode privilege) {
      if (!vec) return;
      for (Operand& op : pending->operands)
        if (op.vec == vec.get()) {
          if (op.privilege != privilege) op.privilege = LEGION_READ_WRITE;
          return;
        }
      Storage& s = storage_of(*vec);
      pending->operands.push_back(
          {vec.get(), privilege, s.element_size, s.per_dpu, s.payload_bytes});
      regions.push_back(&s);
    };
    // A written vector is overwritten in full unless it is also read.
    for (const auto& vec : reads) add(vec, LEGION_READ_ONLY);
    for (const auto& vec : writes) add(vec, LEGION_WRITE_DISCARD);

    const Placement placement = place(pending->operands, regions);
    pending->args.resize(placement.ranks);

    Launch* raw = pending.get();
    IndexLauncher launcher(KERNEL_TASK_ID, colors(placement.ranks),
                           TaskArgument(&raw, sizeof(raw)), ArgumentMap());
    launcher.tag = (MappingTagID)placement.first << PLACEMENT_TAG_SHIFT;
    for (size_t i = 0; i < regions.size(); i++) {
      // Exact instances: shards depend on instance bounds (check_aligned).
      launcher.add_region_requirement(RegionRequirement(
          partition(*regions[i], placement.ranks), 0,
          pending->operands[i].privilege, LEGION_EXCLUSIVE, regions[i]->region,
          Mapping::DefaultMapper::EXACT_REGION));
      launcher.add_field((unsigned)i, FID_DATA);
    }
    Future done =
        runtime_->execute_index_space(ctx_, launcher, LEGION_REDOP_SUM_INT32);
    inflight_.push_back({done, std::move(pending)});
  }

  // Legion orders dependent work, so there is nothing to wait for.
  void notify(std::function<void()> done) override { done(); }

  // Completion is reported at issue, so draining needs no wait.
  void progress() override {}

  void sync() override {
    runtime_->issue_execution_fence(ctx_).get_void_result();
    reap(true);
  }

  void read_logs(FILE* out) override {
    (void)out;  // Realm owns the rank streams; nothing to read here yet.
  }

 private:
  struct Storage {
    // Per-DPU vectors: created when placed.
    LogicalRegion region = LogicalRegion::NO_REGION;
    size_t element_size = 0;
    size_t region_elements = 0;
    bool per_dpu = false;
    size_t payload_bytes = 0;  // per-DPU vectors: what each DPU's slot holds
    size_t piece_bytes = 0;    // per-DPU vectors: one rank's piece
    bool placed = false;
    Placement placement{};
    // Partitions by rank count.
    std::map<uint32_t, LogicalPartition> partitions;
  };

  LogicalRegion make_region(size_t elements, size_t element_size) {
    IndexSpace is = runtime_->create_index_space(
        ctx_, Rect<1>(0, (coord_t)std::max<size_t>(elements, 1) - 1));
    FieldSpace fs = runtime_->create_field_space(ctx_);
    {
      FieldAllocator alloc = runtime_->create_field_allocator(ctx_, fs);
      alloc.allocate_field(element_size, FID_DATA);
    }
    return runtime_->create_logical_region(ctx_, is, fs);
  }

  detail::VectorDescRef add_vector(size_t n, Storage s) {
    auto vec = std::make_shared<detail::VectorDesc>();
    vec->num_elements = n;
    vec->element_size = (uint32_t)s.element_size;
    vec->allocated_footprint_bytes = s.region_elements * s.element_size;
    vec->ptr_allocated = true;
    std::lock_guard<std::mutex> lock(mtx_);
    storage_[vec.get()] = std::move(s);
    return vec;
  }

  IndexSpace colors(uint32_t ranks) {
    auto it = colors_.find(ranks);
    if (it == colors_.end())
      it = colors_
               .emplace(ranks, runtime_->create_index_space(
                                   ctx_, Rect<1>(0, (coord_t)ranks - 1)))
               .first;
    return it->second;
  }

  // Runs where the first placed operand lives (others join it), else asks
  // the mapper.
  Placement place(const std::vector<Operand>& operands,
                  const std::vector<Storage*>& regions) {
    const Storage* anchor = nullptr;
    uint64_t bytes = 0;
    for (size_t i = 0; i < regions.size(); i++) {
      if (regions[i]->placed && !anchor) anchor = regions[i];
      if (!regions[i]->per_dpu)
        bytes = std::max<uint64_t>(
            bytes, operands[i].vec->num_elements * regions[i]->element_size);
    }

    Placement placement;
    if (anchor) {
      placement = anchor->placement;
    } else {
      TunableLauncher tunable(PLACEMENT_TUNABLE, 0, 0, sizeof(Placement));
      tunable.arg = UntypedBuffer(&bytes, sizeof(bytes));
      placement =
          runtime_->select_tunable_value(ctx_, tunable).get_result<Placement>();
    }

    for (Storage* s : regions) {
      if (s->placed) continue;
      s->placed = true;
      s->placement = placement;
      if (s->per_dpu) {
        s->region_elements = s->piece_bytes / s->element_size * placement.ranks;
        s->region = make_region(s->region_elements, s->element_size);
      }
    }
    return placement;
  }

  // Pieces split on words; per-DPU vectors on their whole-row pieces.
  LogicalPartition partition(Storage& s, uint32_t ranks) {
    auto it = s.partitions.find(ranks);
    if (it != s.partitions.end()) return it->second;

    const size_t unit = s.per_dpu ? s.piece_bytes / s.element_size
                                  : std::max<size_t>(8 / s.element_size, 1);
    const size_t chunk =
        (s.region_elements + ranks * unit - 1) / (ranks * unit) * unit;
    std::map<DomainPoint, Domain> pieces;
    for (uint32_t r = 0; r < ranks; r++) {
      const coord_t lo = (coord_t)std::min(r * chunk, s.region_elements);
      const coord_t hi =
          (coord_t)std::min((r + 1) * chunk, s.region_elements) - 1;
      pieces[DomainPoint(Point<1>(r))] = Domain(Rect<1>(lo, hi));
    }
    IndexPartition ip = runtime_->create_partition_by_domain(
        ctx_, s.region.get_index_space(), pieces, colors(ranks));
    LogicalPartition lp = runtime_->get_logical_partition(ctx_, s.region, ip);
    s.partitions.emplace(ranks, lp);
    return lp;
  }

  Storage& storage_of(const detail::VectorDesc& vec) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = storage_.find(&vec);
    if (it == storage_.end())
      throw std::logic_error("vector has no Legion storage");
    return it->second;
  }

  void read_region(const Storage& s, char* host, size_t bytes) {
    HostCopy copy{host, bytes, s.element_size};
    TaskLauncher launcher(READ_TASK_ID, TaskArgument(&copy, sizeof(copy)));
    launcher.add_region_requirement(RegionRequirement(
        s.region, LEGION_READ_ONLY, LEGION_EXCLUSIVE, s.region));
    launcher.add_field(0, FID_DATA);
    runtime_->execute_task(ctx_, launcher).get_void_result();
  }

  // Frees finished launches on the context's thread: their callbacks hold
  // vectors whose release calls into Legion.
  void reap(bool all) {
    while (!inflight_.empty()) {
      if (!all && !inflight_.front().first.is_ready()) break;
      inflight_.front().first.get_void_result();
      inflight_.pop_front();
    }
  }

  Runtime* runtime_ = nullptr;
  Context ctx_ = nullptr;
  std::vector<std::string> argv_storage_;
  std::string binary_;  // what launches run (see load)
  std::vector<char*> argv_;
  std::map<uint32_t, IndexSpace> colors_;  // [0, ranks) per rank count
  uint32_t num_ranks_ = 0, dpus_per_rank_ = 0, num_dpus_ = 0, granule_ = 0;
  bool open_ = false;

  std::mutex mtx_;
  std::map<const detail::VectorDesc*, Storage> storage_;
  std::deque<std::pair<Future, std::unique_ptr<Launch>>> inflight_;
};

}  // namespace

std::unique_ptr<DpuBackend> make_legion_backend() {
  return std::make_unique<LegionBackend>();
}
