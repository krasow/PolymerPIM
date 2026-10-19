// Placement policy: a vector spans one rank per $POLYMERPIM_RANK_BYTES of
// data (1 MiB by default), and new placements rotate through the ranks so
// independent vectors land on different ones and their kernels can overlap.

#include "mapper.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "default_mapper.h"

namespace polymerpim_legion {
namespace {

using namespace Legion;
using namespace Legion::Mapping;

size_t rank_bytes() {
  if (const char* value = std::getenv("POLYMERPIM_RANK_BYTES")) {
    const long long parsed = std::atoll(value);
    if (parsed > 0) return (size_t)parsed;
  }
  return size_t{1} << 20;
}

class PolymerPIMMapper final : public DefaultMapper {
 public:
  PolymerPIMMapper(MapperRuntime* rt, Machine machine, Processor local)
      : DefaultMapper(rt, machine, local, "polymerpim_mapper"),
        rank_bytes_(rank_bytes()) {
    Machine::ProcessorQuery query(machine);
    query.only_kind(Processor::DPU_PROC);
    ranks_.assign(query.begin(), query.end());
    std::sort(ranks_.begin(), ranks_.end());
  }

  void slice_task(const MapperContext ctx, const Task& task,
                  const SliceTaskInput& input,
                  SliceTaskOutput& output) override {
    if (task.task_id != KERNEL_TASK_ID) {
      DefaultMapper::slice_task(ctx, task, input, output);
      return;
    }
    const Rect<1> points = input.domain;
    for (PointInRectIterator<1> p(points); p(); p++) {
      const uint64_t first = task.tag >> PLACEMENT_TAG_SHIFT;
      const Processor rank = ranks_[(first + (*p)[0]) % ranks_.size()];
      output.slices.push_back(
          TaskSlice(Domain(Rect<1>(*p, *p)), rank, false, false));
    }
  }

  void select_tunable_value(const MapperContext ctx, const Task& task,
                            const SelectTunableInput& input,
                            SelectTunableOutput& output) override {
    if (input.tunable_id != PLACEMENT_TUNABLE) {
      DefaultMapper::select_tunable_value(ctx, task, input, output);
      return;
    }
    uint64_t bytes = 0;
    if (input.size == sizeof(bytes))
      std::memcpy(&bytes, input.args, sizeof(bytes));

    const uint32_t total = (uint32_t)ranks_.size();
    Placement placement;
    placement.ranks = (uint32_t)std::clamp<uint64_t>(
        (bytes + rank_bytes_ - 1) / rank_bytes_, 1, total);
    placement.first = next_ + placement.ranks <= total ? next_ : 0;
    next_ = (placement.first + placement.ranks) % total;

    auto* value = static_cast<Placement*>(std::malloc(sizeof(Placement)));
    *value = placement;
    output.value = value;
    output.size = sizeof(Placement);
    output.take_ownership = true;
  }

 private:
  std::vector<Processor> ranks_;
  size_t rank_bytes_;
  uint32_t next_ = 0;
};

void install(Machine machine, Runtime* runtime,
             const std::set<Processor>& local_procs) {
  for (Processor proc : local_procs)
    runtime->replace_default_mapper(
        new PolymerPIMMapper(runtime->get_mapper_runtime(), machine, proc),
        proc);
}

}  // namespace

void register_mapper() { Runtime::add_registration_callback(install); }

}  // namespace polymerpim_legion
