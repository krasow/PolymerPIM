#pragma once
// PolymerPIM's Legion mapper: decides how many ranks a vector spans and
// which, and runs each kernel launch on its vectors' ranks.

#include <legion.h>

namespace polymerpim_legion {

// Below LEGION_MAX_APPLICATION_TASK_ID.
enum : Legion::TaskID {
  KERNEL_TASK_ID = 0x504d1,
  WRITE_TASK_ID,
  READ_TASK_ID,
  TOP_LEVEL_TASK_ID,
};

// A run of ranks, as indices into the DPU processors in ID order.
struct Placement {
  uint32_t first;
  uint32_t ranks;
};

// Asks the mapper where to put vectors not yet placed.  The argument is the
// bytes of the largest one (uint64_t); the result is a Placement.
constexpr Legion::TunableID PLACEMENT_TUNABLE = 0x504d;

// A kernel launch over Placement::ranks points runs point i on rank
// first + i, with first carried in its mapping tag above the bits the default
// mapper reads.
constexpr unsigned PLACEMENT_TAG_SHIFT = 8;

// Installs the mapper on every processor; call before Runtime::start.
void register_mapper();

}  // namespace polymerpim_legion
