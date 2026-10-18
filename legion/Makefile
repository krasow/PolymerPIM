# Builds liblegion.so and librealm.so with the UPMEM module, for RUNTIME=legion.
# Run from the top-level Makefile, which passes LG_RT_DIR; objects land in the
# upmem-legion checkout and the libraries and *_defines.h in this directory.

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
