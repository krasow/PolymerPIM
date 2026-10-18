// DpuBackend over Legion, using upmem-legion's Realm module: one processor and
// one interleaved MRAM memory per rank.
//
// A vector is a 1-D logical region equally partitioned over the ranks; the
// mapper decides where each piece's instance lives.  A kernel is an index
// launch with one point per rank, whose task fills that rank's per-DPU
// arguments and starts the binary asynchronously on the rank's stream -- the
// task completes once the rank drains.  Legion orders dependent launches, so
// operations report completion as soon as they are issued.
//
// Within a rank, Realm interleaves each instance over the DPUs in granules.
// Instances start on a row boundary, so a vector's shard sits at the same
// local offset on every DPU of the rank, but the elements it holds are not a
// contiguous range of the vector: kernels that need global indices are not
// supported yet.

#include <legion.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "backend.h"
#include "realm/upmem/upmem_internal.h"
#include "realm/upmem/upmem_layout.h"
#include "realm/upmem/upmem_module.h"

#define CHECK_UPMEM(x) DPU_ASSERT(x)

namespace {

using namespace Legion;
using Realm::Upmem::UpmemModule;
using RankDPU = Realm::Upmem::DPU;

// Below LEGION_MAX_APPLICATION_TASK_ID.
enum : TaskID {
  KERNEL_TASK_ID = 0x504d1,
  WRITE_TASK_ID,
  READ_TASK_ID,
  TOP_LEVEL_TASK_ID,
};
// A raw Realm task; Legion numbers its variants up from
// TASK_ID_FIRST_AVAILABLE, so this sits well past them.
constexpr Realm::Processor::TaskFuncID LOAD_TASK_ID =
    Realm::Processor::TASK_ID_FIRST_AVAILABLE + (1u << 20);
constexpr FieldID FID_DATA = 1;

UpmemModule& upmem_module() {
  auto* mod = Realm::Runtime::get_runtime().get_module<UpmemModule>("upmem");
  if (!mod) throw std::runtime_error("Legion was built without UPMEM support");
  return *mod;
}

// Index of `rank` in the module's rank list; ranks are numbered in that order.
uint32_t rank_index(const RankDPU* rank) {
  const auto& ranks = upmem_module().dpus;
  auto it = std::find(ranks.begin(), ranks.end(), rank);
  assert(it != ranks.end());
  return (uint32_t)(it - ranks.begin());
}

// How a launch's region requirement relates to the vector it names.
struct Operand {
  const detail::VectorDesc* vec;
  size_t element_size;
  // Per-DPU vectors only: the elements each DPU holds.
  bool per_dpu;
  size_t payload_bytes;
};

struct Launch {
  DpuBackend::FillArgs fill;
  DpuBackend::InspectArgs inspect;
  std::vector<Operand> operands;  // in region requirement order
};

// One rank's view of a launch: placement comes from the instances Legion
// mapped for this point task.
class LegionShard final : public DpuBackend::Shard {
 public:
  LegionShard(const Launch& launch, const std::vector<PhysicalRegion>& regions,
              const RankDPU& rank, uint32_t first_dpu)
      : launch_(launch), rank_(rank), first_dpu_(first_dpu) {
    for (size_t i = 0; i < regions.size(); i++) {
      const Operand& op = launch.operands[i];
      const Rect<1> rect = regions[i].get_bounds<1, coord_t>().bounds;
      FieldAccessor<LEGION_READ_WRITE, uint8_t, 1, coord_t,
                    Realm::AffineAccessor<uint8_t, 1, coord_t>>
          acc(regions[i], FID_DATA, op.element_size);
      const uintptr_t ptr = (uintptr_t)acc.ptr(rect.lo);
      starts_.push_back(ptr - REALM_UPMEM_RANK_PTR_BASE);
      // Pieces are padded (see allocate); the kernel sees only real elements.
      const coord_t end =
          std::min<coord_t>(rect.hi[0] + 1, (coord_t)op.vec->num_elements);
      bytes_.push_back(end > rect.lo[0] ? (end - rect.lo[0]) * op.element_size
                                        : 0);
    }
  }

  uint32_t dpu() const override { return first_dpu_ + dpu_; }

  uint32_t offset(const detail::VectorDesc& vec) const override {
    const size_t i = find(vec);
    return REALM_UPMEM_MRAM_DATA_OFFSET +
           (uint32_t)realm_upmem_local_bound(starts_[i], dpu_, rank_.nr_dpus,
                                             rank_.granule);
  }

  uint32_t size_bytes(const detail::VectorDesc& vec) const override {
    const size_t i = find(vec);
    const Operand& op = launch_.operands[i];
    if (op.per_dpu) return (uint32_t)(op.payload_bytes + vec.reserved_bytes);
    const uint64_t lo =
        realm_upmem_local_bound(starts_[i], dpu_, rank_.nr_dpus, rank_.granule);
    const uint64_t hi = realm_upmem_local_bound(starts_[i] + bytes_[i], dpu_,
                                                rank_.nr_dpus, rank_.granule);
    return (uint32_t)(hi - lo);
  }

  uint32_t first_element(const detail::VectorDesc&) const override {
    throw std::logic_error(
        "the Legion backend interleaves shards, so kernels that need global "
        "element indices are not supported yet");
  }

  void at(uint32_t dpu) { dpu_ = dpu; }

 private:
  size_t find(const detail::VectorDesc& vec) const {
    for (size_t i = 0; i < launch_.operands.size(); i++)
      if (launch_.operands[i].vec == &vec) return i;
    throw std::logic_error("kernel argument names a vector it does not touch");
  }

  const Launch& launch_;
  const RankDPU& rank_;
  uint32_t first_dpu_;
  uint32_t dpu_ = 0;
  std::vector<uint64_t> starts_;  // rank address of each region's piece
  std::vector<uint64_t> bytes_;
};

// Runs on a rank's processor: fills its DPUs' arguments and starts the
// kernel.  The task completes when the rank's stream drains.
int kernel_task(const Task* task, const std::vector<PhysicalRegion>& regions,
                Context, Runtime*) {
  const Launch& launch = **static_cast<Launch* const*>(task->args);
  UpmemModule& mod = upmem_module();
  RankDPU* rank = mod.get_task_dpu();
  dpu_set_t* stream = mod.get_task_upmem_stream();
  assert(rank && stream);

  const uint32_t nr = rank->nr_dpus;
  LegionShard shard(launch, regions, *rank, rank_index(rank) * nr);
  std::vector<DPU_LAUNCH_ARGS> args(nr);
  for (uint32_t d = 0; d < nr; d++) {
    shard.at(d);
    launch.fill(args[d], shard);
  }
  if (launch.inspect) launch.inspect(args.data(), nr);

  // The UPMEM runtime isn't safe for concurrent calls on one rank.
  Realm::AutoLock<> al(rank->get_xfer_mutex());
  for (uint32_t d = 0; d < nr; d++)
    CHECK_UPMEM(dpu_prepare_xfer(rank->dpu_handles[d], &args[d]));
  CHECK_UPMEM(dpu_push_xfer(*stream, DPU_XFER_TO_DPU, "args", 0,
                            sizeof(DPU_LAUNCH_ARGS), DPU_XFER_DEFAULT));
  CHECK_UPMEM(dpu_launch(*stream, DPU_ASYNCHRONOUS));
  return 0;
}

// Copies `bytes` from the start of a whole region; regions may be padded
// past the vector's end.
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

// Realm task on one rank's processor: replaces its program.
void load_task(const void* args, size_t, const void*, size_t,
               Realm::Processor) {
  UpmemModule& mod = upmem_module();
  RankDPU* rank = mod.get_task_dpu();
  dpu_set_t* stream = mod.get_task_upmem_stream();
  assert(rank && stream);
  Realm::AutoLock<> al(rank->get_xfer_mutex());
  CHECK_UPMEM(dpu_sync(*stream));
  CHECK_UPMEM(dpu_load(*stream, static_cast<const char*>(args), nullptr));
}

void register_tasks() {
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

// Legion's command line: enough ranks for `num_dpus`, then anything in
// $POLYMERPIM_LEGION_ARGS (later flags win).
//
// Reads and writes stage through system memory, and Legion keeps a vector's
// host copy alive after the transfer, so system memory is sized to what the
// ranks can hold rather than Realm's 512 MB default.
std::vector<std::string> legion_args(uint32_t num_dpus) {
  const uint32_t ranks =
      std::max<uint32_t>(1, (num_dpus + REALM_UPMEM_MAX_DPUS_PER_RANK - 1) /
                                REALM_UPMEM_MAX_DPUS_PER_RANK);
  const uint32_t sysmem_mb = std::min<uint32_t>(ranks * 4096, 65536);
  std::vector<std::string> args = {"polymerpim", "-ll:num_ranks",
                                   std::to_string(ranks), "-ll:csize",
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

    Realm::Processor::register_task_by_kind(
        Realm::Processor::DPU_PROC, false /*!global*/, LOAD_TASK_ID,
        Realm::CodeDescriptor(load_task), Realm::ProfilingRequestSet())
        .wait();
    load(binary);
    open_ = true;
    return num_dpus_;
  }

  void close() override {
    if (!open_) return;
    sync();
    runtime_->destroy_index_space(ctx_, colors_);
    runtime_->finish_implicit_task(ctx_);
    Runtime::wait_for_shutdown();
    open_ = false;
  }

  uint32_t num_dpus() const override { return num_dpus_; }

  void load(const std::string& binary) override {
    Machine::ProcessorQuery query(Machine::get_machine());
    query.only_kind(Processor::DPU_PROC);
    std::vector<Realm::Event> loaded;
    for (Processor proc : query)
      loaded.push_back(
          proc.spawn(LOAD_TASK_ID, binary.c_str(), binary.size() + 1));
    Realm::Event::merge_events(loaded).wait();
  }

  detail::VectorDescRef allocate(size_t n, size_t reserved, size_t element_size,
                                 bool) override {
    if (reserved != 0)
      throw std::logic_error(
          "the Legion backend has no reserved bytes for sharded vectors");
    // Every rank's piece is a whole number of 8-byte words.  Pieces start on
    // a row boundary, so copies then never split a word -- Realm handles a
    // partial word with a synchronous read-modify-write that drains the rank.
    const size_t unit = num_ranks_ * (8 / element_size);
    const size_t padded = (std::max<size_t>(n, 1) + unit - 1) / unit * unit;
    return make_vector(padded, element_size, n, false, 0);
  }

  // Every DPU needs its own slot at the same local offset, so each rank's
  // piece spans whole rows: `rows` granules on every DPU.
  detail::VectorDescRef allocate_per_dpu(size_t n_per_dpu, size_t reserved,
                                         size_t element_size, bool) override {
    const size_t slot = (n_per_dpu * element_size + reserved + 7) & ~size_t{7};
    const size_t rows = (slot + granule_ - 1) / granule_;
    const size_t piece_bytes = rows * granule_ * dpus_per_rank_;
    auto vec =
        make_vector(piece_bytes / element_size * num_ranks_, element_size,
                    n_per_dpu * num_dpus_, true, n_per_dpu * element_size);
    vec->reserved_bytes = reserved;
    return vec;
  }

  detail::VectorDescRef allocate_local(size_t, size_t) override {
    throw std::logic_error(
        "local vectors are not supported by the Legion backend yet");
  }

  void realize(const detail::VectorDescRef&) override {}

  void release(detail::VectorDesc& vec) override {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = storage_.find(&vec);
    if (it == storage_.end()) return;
    const Storage& s = it->second;
    runtime_->destroy_logical_region(ctx_, s.region);
    runtime_->destroy_field_space(ctx_, s.region.get_field_space());
    runtime_->destroy_index_space(ctx_, s.region.get_index_space());
    storage_.erase(it);
    vec.ptr_allocated = false;
  }

  void write(const detail::VectorDescRef& vec, const char* host) override {
    Storage& s = storage_of(*vec);
    if (s.per_dpu)
      throw std::logic_error("per-DPU vectors are written only by kernels");
    // Waited on so the caller's buffer can go as soon as the event retires.
    HostCopy copy{const_cast<char*>(host), vec->num_elements * s.element_size,
                  s.element_size};
    TaskLauncher launcher(WRITE_TASK_ID, TaskArgument(&copy, sizeof(copy)));
    launcher.add_region_requirement(RegionRequirement(
        s.region, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, s.region));
    launcher.add_field(0, FID_DATA);
    runtime_->execute_task(ctx_, launcher).get_void_result();
    s.written = true;
  }

  detail::ShardLayout read_layout(const detail::VectorDesc& vec) override {
    const Storage& s = storage_of(vec);
    detail::ShardLayout layout;
    if (s.per_dpu) {
      layout.logical.assign(num_dpus_, s.payload_bytes);
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

    // Each rank's piece, as rank addresses from its row-aligned start; DPU d's
    // slot is local bytes [0, payload) of it.
    const size_t piece_bytes = s.region_elements / num_ranks_ * s.element_size;
    std::vector<char> pieces(piece_bytes * num_ranks_);
    read_region(s, pieces.data(), pieces.size());
    const size_t payload = s.payload_bytes;
    for (uint32_t r = 0; r < num_ranks_; r++)
      for (uint32_t d = 0; d < dpus_per_rank_; d++) {
        char* out = host + (size_t)(r * dpus_per_rank_ + d) * stride;
        for (size_t l = 0; l < payload; l++)
          out[l] =
              pieces[r * piece_bytes +
                     realm_upmem_rank_addr(l, d, dpus_per_rank_, granule_)];
      }
  }

  void launch(const std::vector<detail::VectorDescRef>& vectors,
              const FillArgs& fill, const InspectArgs& inspect) override {
    reap(false);
    auto pending = std::make_unique<Launch>(Launch{fill, inspect, {}});
    std::vector<Storage*> regions;
    for (const auto& vec : vectors) {
      if (!vec) continue;
      const bool seen =
          std::any_of(pending->operands.begin(), pending->operands.end(),
                      [&](const Operand& op) { return op.vec == vec.get(); });
      if (seen) continue;
      Storage& s = storage_of(*vec);
      pending->operands.push_back(
          {vec.get(), s.element_size, s.per_dpu, s.payload_bytes});
      regions.push_back(&s);
    }

    Launch* raw = pending.get();
    IndexLauncher launcher(KERNEL_TASK_ID, colors_,
                           TaskArgument(&raw, sizeof(raw)), ArgumentMap());
    // The launch API does not say which operands a kernel only reads, so
    // every one is read-write and Legion orders the launch after all earlier
    // users.  A vector's first user is always the kernel producing it in full,
    // so until then its contents can be discarded.
    for (size_t i = 0; i < regions.size(); i++) {
      launcher.add_region_requirement(RegionRequirement(
          regions[i]->partition, 0,
          regions[i]->written ? LEGION_READ_WRITE : LEGION_WRITE_DISCARD,
          LEGION_EXCLUSIVE, regions[i]->region));
      launcher.add_field((unsigned)i, FID_DATA);
      regions[i]->written = true;
    }
    Future done =
        runtime_->execute_index_space(ctx_, launcher, LEGION_REDOP_SUM_INT32);
    inflight_.push_back({done, std::move(pending)});
  }

  // Legion already orders dependent work, so nothing is gained by waiting.
  void notify(std::function<void()> done) override { done(); }

  void sync() override {
    runtime_->issue_execution_fence(ctx_).get_void_result();
    reap(true);
  }

  void read_logs(FILE* out) override {
    (void)out;  // Realm owns the rank streams; nothing to read here yet.
  }

 private:
  struct Storage {
    LogicalRegion region;
    LogicalPartition partition;
    size_t element_size;
    size_t region_elements;
    bool per_dpu;
    size_t payload_bytes;  // per-DPU vectors: what each DPU's slot holds
    bool written = false;
  };

  detail::VectorDescRef make_vector(size_t region_elements, size_t element_size,
                                    size_t n, bool per_dpu,
                                    size_t payload_bytes) {
    IndexSpace is = runtime_->create_index_space(
        ctx_, Rect<1>(0, (coord_t)std::max<size_t>(region_elements, 1) - 1));
    FieldSpace fs = runtime_->create_field_space(ctx_);
    {
      FieldAllocator alloc = runtime_->create_field_allocator(ctx_, fs);
      alloc.allocate_field(element_size, FID_DATA);
    }
    LogicalRegion region = runtime_->create_logical_region(ctx_, is, fs);
    IndexPartition ip = runtime_->create_equal_partition(ctx_, is, colors_);

    auto vec = std::make_shared<detail::VectorDesc>();
    vec->num_elements = n;
    vec->element_size = (uint32_t)element_size;
    vec->allocated_footprint_bytes = region_elements * element_size;
    vec->ptr_allocated = true;
    std::lock_guard<std::mutex> lock(mtx_);
    storage_[vec.get()] = {
        region,       runtime_->get_logical_partition(ctx_, region, ip),
        element_size, region_elements,
        per_dpu,      payload_bytes};
    return vec;
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

  // Frees finished launches here, on the thread that owns the context: their
  // callbacks hold vectors whose release calls back into Legion.
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
  std::vector<char*> argv_;
  IndexSpace colors_;
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
