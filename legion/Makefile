# Builds liblegion.so and librealm.so (UPMEM module) here for RUNTIME=legion.

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

DEBUG          ?= 0
MAX_DIM        ?= 3
OUTPUT_LEVEL   ?= LEVEL_INFO
USE_CUDA       := 0
USE_HIP        := 0
USE_GASNET     := 0
USE_HDF        := 0
USE_UPMEM      := 1
SHARED_OBJECTS := 1

OUTFILE        :=
GEN_SRC        :=
GEN_GPU_SRC    :=
GEN_UPMEM_SRC  :=
CC_FLAGS       ?= -DLEGION_MAX_NUM_PROCS=256

include $(LG_RT_DIR)/runtime.mk
