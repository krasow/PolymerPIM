#include <assert.h>
#include <limits.h>
#include <mram.h>
#include <stdbool.h>
#include <string.h>

#include "stdio.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define PRODUCT(a, b) ((a) * (b))
#define SUM(a, b) ((a) + (b))

#if ENABLE_DPU_PRINTING == 1
void print_args(DPU_LAUNCH_ARGS args) {
  printf("Reduction kernel launched with arguments:\n");
  printf("  num_elements: %u\n", args.num_elements);
  printf("  rhs_offset: 0x%08X\n", args.reduction.rhs_offset);
  printf("  res_offset: 0x%08X\n", args.reduction.res_offset);
}
#else
void print_args(DPU_LAUNCH_ARGS args) {
  /* do nothing */
  (void)args; /* remove unused parameter warning */
}
#endif

#define STR(x) #x
#define XSTR(x) STR(x)

#define DEFINE_REDUCTION_KERNEL(TYPE, OP, FUNC)                                \
  int reduction_##TYPE##_##OP(void) {                                          \
    enum { stride = (MINIMUM_WRITE_SIZE / sizeof(TYPE)) };                     \
    unsigned int tasklet_id = me();                                            \
    uint32_t num_elems = args.num_elements;                                    \
    __mram_ptr TYPE *rhs_ptr = (__mram_ptr TYPE *)(args.reduction.rhs_offset); \
    __mram_ptr TYPE *res_ptr = (__mram_ptr TYPE *)(args.reduction.res_offset); \
                                                                               \
    /* WRAM working buffer (DMA aligned) */                                    \
    TYPE *rhs_block = (TYPE *)dpu_workspace[tasklet_id];                       \
    TYPE *res_block =                                                          \
        (TYPE *)&dpu_workspace[tasklet_id][BLOCK_SIZE * sizeof(TYPE)];         \
                                                                               \
    const char op_name[] = XSTR(OP);                                           \
    bool is_sum = (op_name[0] == 's' || op_name[0] == 'S');                    \
    bool is_product = (op_name[0] == 'p' || op_name[0] == 'P');                \
    bool is_min = ((op_name[0] == 'm' || op_name[0] == 'M') &&                 \
                   (op_name[1] == 'i' || op_name[1] == 'I'));                  \
    bool is_promotable = (is_sum || is_product);                               \
    bool is_sum32 =                                                            \
        (sizeof(TYPE) == 4 && is_promotable && ENABLE_PROMOTION_REDUCTIONS);   \
                                                                               \
    int64_t local_red_64 = is_sum ? 0 : 1;                                     \
    TYPE local_red;                                                            \
    if (is_sum)                                                                \
      local_red = (TYPE)0;                                                     \
    else if (is_product)                                                       \
      local_red = (TYPE)1;                                                     \
    else if (is_min)                                                           \
      local_red = (TYPE)INT32_MAX;                                             \
    else                                                                       \
      local_red = (TYPE)INT32_MIN;                                             \
    for (uint32_t block_loc = tasklet_id << BLOCK_SIZE_LOG2;                   \
         block_loc < num_elems;                                                \
         block_loc += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {                      \
      uint32_t block_elems = (block_loc + BLOCK_SIZE >= num_elems)             \
                                 ? (num_elems - block_loc)                     \
                                 : BLOCK_SIZE;                                 \
                                                                               \
      uint32_t block_bytes =                                                   \
          ((block_elems * sizeof(TYPE)) + 7) & ~(uint32_t)7;                   \
                                                                               \
      /* Copy block from MRAM to WRAM */                                       \
      mram_read((__mram_ptr void const *)(rhs_ptr + block_loc), rhs_block,     \
                block_bytes);                                                  \
                                                                               \
      /* Compute in WRAM */                                                    \
      if (is_sum32) {                                                          \
        for (uint32_t i = 0; i < block_elems; i++) {                           \
          if (is_sum)                                                          \
            local_red_64 += rhs_block[i];                                      \
          else                                                                 \
            local_red_64 *= rhs_block[i];                                      \
        }                                                                      \
      } else {                                                                 \
        for (uint32_t i = 0; i < block_elems; i++) {                           \
          local_red = FUNC(local_red, rhs_block[i]);                           \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    /* write local result into preceding reserved area (one 8-byte slot per    \
     * tasklet) */                                                             \
    uint64_t *buff_ptr =                                                       \
        (uint64_t *)&dpu_workspace[tasklet_id][BLOCK_SIZE * 2 * sizeof(TYPE)]; \
    *buff_ptr = 0;                                                             \
    if (is_sum32) {                                                            \
      *buff_ptr = (uint64_t)local_red_64;                                      \
    } else {                                                                   \
      memcpy(buff_ptr, &local_red, sizeof(TYPE));                              \
    }                                                                          \
    extern uint64_t reduction_scratchpad[];                                    \
    /* partial results go into the dedicated WRAM scratchpad */                \
    reduction_scratchpad[tasklet_id] = *buff_ptr;                              \
                                                                               \
    barrier_wait(&my_barrier);                                                 \
                                                                               \
    /* Tasklet 0 performs final reduction from partial slots */                \
    if (tasklet_id == 0) {                                                     \
      if (is_sum32) {                                                          \
        int64_t total_64 = is_sum ? 0 : 1;                                     \
        uint32_t i;                                                            \
        for (i = 0; i < NR_TASKLETS; i++) {                                    \
          if (is_sum)                                                          \
            total_64 += (int64_t)reduction_scratchpad[i];                      \
          else                                                                 \
            total_64 *= (int64_t)reduction_scratchpad[i];                      \
        }                                                                      \
        *buff_ptr = (uint64_t)total_64;                                        \
      } else {                                                                 \
        uint32_t total_slots = NR_TASKLETS * stride;                           \
        TYPE res_block_tot[NR_TASKLETS * stride] __attribute__((aligned(8)));  \
        uint32_t i;                                                            \
        /* read all slots back from WRAM scratchpad */                         \
        for (i = 0; i < NR_TASKLETS; i++) {                                    \
          res_block_tot[i * stride] = *(TYPE *)&reduction_scratchpad[i];       \
        }                                                                      \
        TYPE total = res_block_tot[0];                                         \
        for (i = 1; i < NR_TASKLETS; i++) {                                    \
          total = FUNC(total, res_block_tot[i * stride]);                      \
        }                                                                      \
        memcpy(buff_ptr, &total, sizeof(TYPE));                                \
      }                                                                        \
                                                                               \
      /* Final total goes into the data area (res_ptr) */                      \
      mram_write((void *)buff_ptr, (__mram_ptr void *)(res_ptr),               \
                 MINIMUM_WRITE_SIZE);                                          \
    }                                                                          \
    return 0;                                                                  \
  }
