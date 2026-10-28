// One rank per $POLYMERPIM_RANK_BYTES (32 MiB); placements rotate so
// independent vectors use different ranks.

#include "mapper.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "default_mapper.h"

namespace polymerpim_legion {

size_t rank_bytes() {
  if (const char* value = std::getenv("POLYMERPIM_RANK_BYTES")) {
    const long long parsed = std::atoll(value);
    if (parsed > 0) return (size_t)parsed;
  }
  return size_t{32} << 20;
}

namespace {

using namespace Legion;
using namespace Legion::Mapping;

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

  // Index copies write a vector's rank pieces: each point's destination goes
  // in its rank's memory, where the kernels will find it.
  void map_copy(const MapperContext ctx, const Copy& copy,
                const MapCopyInput& input, MapCopyOutput& output) override {
    if (!copy.is_index_space || copy.dst_requirements.size() != 1) {
      DefaultMapper::map_copy(ctx, copy, input, output);
      return;
    }
    output.src_instances[0] = input.src_instances[0];
    const uint64_t first = copy.tag >> PLACEMENT_TAG_SHIFT;
    const Processor rank =
        ranks_[(first + copy.index_point[0]) % ranks_.size()];
    const RegionRequirement& req = copy.dst_requirements[0];
    const Memory target = default_policy_select_target_memory(ctx, rank, req);
    bool force_new = false;
    LayoutConstraintSet constraints = runtime->find_layout_constraints(
        ctx, default_policy_select_layout_constraints(
                 ctx, target, req, COPY_MAPPING, true, force_new));
    constraints.add_constraint(
        FieldConstraint(req.privilege_fields, false, false));
    PhysicalInstance instance;
    if (!default_make_instance(ctx, target, constraints, instance, COPY_MAPPING,
                               force_new, true, req)) {
      std::fprintf(stderr, "PolymerPIM mapper: no room on rank %u for a copy\n",
                   (unsigned)(first + copy.index_point[0]));
      std::abort();
    }
    output.dst_instances[0].push_back(instance);
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
