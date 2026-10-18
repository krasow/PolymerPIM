#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <utility>
#include <vector>

using std::size_t;
using std::vector;

#define DPU_BROADCAST (-1)

#include "backend.h"
#include "detail/vector_desc.h"

struct FreeBlock {
  uint32_t addr;
  size_t size;
};

class allocator {
 public:
  allocator(uint32_t start_addr, std::size_t total_size, std::size_t num_dpus);

  detail::VectorDescRef allocate_upmem_vector(std::size_t n,
                                              std::size_t reserved_mem_per_dpu,
                                              std::size_t size_type,
                                              bool lazy = false);
  detail::VectorDescRef allocate_local_vector(std::size_t n,
                                              std::size_t size_type);
  void realize_allocation(detail::VectorDescRef data);
  void deallocate_upmem_vector(detail::VectorDesc* data);

  // Broadcast allocation/deallocation (O(1))
  detail::VectorDescRef allocate_upmem_vector_broadcast(
      std::size_t n, std::size_t reserved_mem_per_dpu, std::size_t size_type,
      bool lazy = false);
  void deallocate_upmem_vector_broadcast(detail::VectorDesc* data);

  size_t get_total_allocated_bytes() const { return total_allocated_bytes_; }

 private:
  uint32_t start_addr_;  // starting base address
  std::size_t dpu_mem_;  // memory size per DPU
  std::size_t num_dpus_;

  // Per-DPU state
  vector<uint32_t> ptrs_;                        // base addresses per DPU
  vector<std::size_t> sizes_;                    // total size per DPU
  vector<uint32_t> offsets_;                     // bump pointer per DPU
  std::vector<std::list<FreeBlock>> free_list_;  // free blocks per DPU

  // Broadcast state (optimization)
  // If all DPUs are in sync, we can use these instead of per-DPU vectors
  bool is_synchronized_ = true;
  uint32_t broadcast_offset_ = 0;
  std::list<FreeBlock> broadcast_free_list_;

  // Internal helpers for raw allocation/deallocation.
  // dpu_id = DPU_BROADCAST means broadcast.
  void materialize_descriptor_layout(detail::VectorDesc* data);
  uint32_t raw_allocate(int dpu_id, std::size_t n);
  void raw_deallocate(int dpu_id, uint32_t addr, size_t size);

  std::recursive_mutex lock;
  size_t total_allocated_bytes_ = 0;
};
