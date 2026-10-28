#pragma once
// Legion mapper: picks the ranks each vector spans and runs launches there.

#include <legion.h>

namespace polymerpim_legion {

// Below LEGION_MAX_APPLICATION_TASK_ID.
enum : Legion::TaskID {
  KERNEL_TASK_ID = 0x504d1,
  TOP_LEVEL_TASK_ID,
};

// A run of ranks, as indices into the DPU processors in ID order.
struct Placement {
  uint32_t first;
  uint32_t ranks;
};

// Tunable: bytes of the largest unplaced operand (uint64_t) -> Placement.
constexpr Legion::TunableID PLACEMENT_TUNABLE = 0x504d;

// Launch (or index copy) point i runs on rank first + i; first rides in the
// tag, above the default mapper's bits.
constexpr unsigned PLACEMENT_TAG_SHIFT = 8;

// Bytes per rank a placement allows ($POLYMERPIM_RANK_BYTES, 32 MiB).
size_t rank_bytes();

// Installs the mapper on every processor; call before Runtime::start.
void register_mapper();

}  // namespace polymerpim_legion
