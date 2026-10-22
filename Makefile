# https://github.com/CMU-SAFARI/prim-benchmarks/tree/main
# leveraged the above repository to create Makefile for DPU and host code compilation
BUILDDIR ?= build
NR_TASKLETS ?= 12

BACKEND ?= hw
# Device runtime under the queue: native (UPMEM SDK) or legion (upmem-legion).
RUNTIME ?= native
DEBUG ?= 0
LOGGING ?= 3

# this option enables experimental pipeline and fusion features
PIPELINE ?= 0
# this option enables JIT compilation of pipeline kernels
JIT ?= 0
# use interpreter/eager kernels until an async JIT kernel is ready
JIT_PIPELINE_FALLBACK ?= 0
# how many unique kernels to batch before triggering a JIT compile
JIT_BATCH_SIZE ?= 16
# how many pending queue events to scan ahead when looking for fusion candidates
QUEUE_ABSORB_LIMIT ?= 32
# horizontal fusion: max independent parallel chains per kernel pass
MAX_HFUSE_CHAINS ?= 9
# vertical fusion: max RPN opcodes per chain (caps how deep chains can be fused)
MAX_VFUSE_OPS ?= 128
# vertical fusion: max distinct input vectors referenced per kernel
MAX_VFUSE_INPUTS ?= 11
# vertical fusion: max live WRAM stack vectors used by the generic pipeline interpreter
# 3 lets the interpreter run depth-3 updates, so the hybrid fallback
# engages instead of blocking on the compiler; costs one hfuse chain.
MAX_PIPELINE_STACK_DEPTH ?= 3
# DPU WRAM tile size is 2^BLOCK_SIZE_LOG2 elements per tasklet block
BLOCK_SIZE_LOG2 ?= 4

# this option enables fencing after dpu-to-host transfers automatically
# you can disable it to manually control fencing in your code with add_fence() calls
ENABLE_AUTO_FENCING ?= 1

# this option enables printing from the DPU to the host stdout
ENABLE_DPU_PRINTING ?= 0

# this option enables tracing with Perfetto
TRACE ?= 1
PERFETTO_HOME ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/opt/Perfetto)
DEPENDENCY_INSTALLER := scripts/install_dependencies.sh

# this option prevents the automatic removal of the JIT build directory at shutdown
DEBUG_KEEP_JIT_DIR ?= 0

# if you are overflowing during summation/mul reductions
ENABLE_PROMOTION_REDUCTIONS ?= 0

# the default compiler on feta supports up to C++17
# c++20 is needed for some debugging output from std::source_location
CXX_STANDARD ?= c++17

# ----------------- Edit above this line -----------------

ifneq ($(MAKECMDGOALS),dependencies)
ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif
endif

# JIT requires pipeline logic to dispatch events correctly
ifeq ($(JIT),1)
  PIPELINE := 1
endif
ifeq ($(JIT_PIPELINE_FALLBACK),1)
  JIT := 1
  PIPELINE := 1
endif

DPU_DIR := dpu
HOST_DIR := host
TEST_DIR := test

DESTDIR ?= $(CURDIR)/install

# Julia interpreter used to build the PolymerPIM.jl wrapper during `make install`.
JULIA ?= julia

CONFIG_STAMP := build.config

HOST_TARGET := ${BUILDDIR}/lib/libpolymerpim.so
DPU_TARGET := ${BUILDDIR}/bin/runtime.dpu
TEST_TARGET := ${TEST_DIR}/polymerpim_test

COMMON_DIR := common
HOST_INCLUDES := host
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.cc) $(wildcard ${HOST_DIR}/detail/*.cc) \
                $(wildcard ${HOST_DIR}/perfetto/*.cc) \
                $(wildcard ${HOST_DIR}/backends/native/*.cc)
ifeq ($(RUNTIME),legion)
HOST_SOURCES += $(wildcard ${HOST_DIR}/backends/legion/*.cc)
endif
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)
# One binary holds every suite; select subsets at run time with --filter.
TEST_SOURCES := $(wildcard ${TEST_DIR}/*.cc)
TEST_HEADERS := $(wildcard ${TEST_DIR}/*.h) $(wildcard ${TEST_DIR}/*.inl)

HOST_HEADERS := $(wildcard ${HOST_DIR}/*.inl) $(wildcard ${HOST_DIR}/*.h) \
                $(wildcard ${HOST_DIR}/detail/*.h) \
                $(wildcard ${HOST_DIR}/detail/*.inl) \
                $(wildcard ${HOST_DIR}/perfetto/*.h) \
                $(wildcard ${HOST_DIR}/perfetto/detail/*.h) \
                $(wildcard ${HOST_DIR}/backends/native/*.h)
DPU_HEADERS := $(wildcard ${DPU_DIR}/*.inl) $(wildcard ${DPU_DIR}/*.h)
COMMON_HEADERS := ${COMMON_DIR}/common.h ${COMMON_DIR}/config.h

ifeq ($(DEBUG),1)
  CXXFLAGS += -g -pg -O0 -DDEBUG -fno-omit-frame-pointer -fno-inline
  LDFLAGS  +=
  BUILD_TYPE := debug
else
  CXXFLAGS += -O3 -DNDEBUG
  CXXFLAGS += -O3 -DNDEBUG
  BUILD_TYPE := release
endif

# Debian 10 / GCC 8 requirement for filesystem
LDFLAGS += -lstdc++fs

ifeq ($(TRACE),1)
  CXXFLAGS += -pthread -I$(PERFETTO_HOME)/include
  LDFLAGS += -L$(PERFETTO_HOME)/lib -lperfetto -ldl -lpthread
endif

ifeq ($(JIT),1)
  CXXFLAGS += -pthread
  LDFLAGS += -pthread
endif

.PHONY: dependencies config_check reconfigure all clean clean-internal test build-test list-tests install install-cpp install-julia uninstall print_config make_header

GENERATED_TARGETS := dpu/kernels.h host/opinfo.h host/kernelids.h common/opcodes.h
# Same generator, but not a C header -- must stay out of the install list.
GENERATED_JULIA := julia/src/internal/opcodes.jl


# Synced at parse time so -j builds can't race it; a change cleans the build.
define CONFIG_CONTENT
NR_TASKLETS=$(NR_TASKLETS)
BACKEND=$(BACKEND)
DEVICE_RUNTIME=$(RUNTIME)
DEBUG=$(DEBUG)
ENABLE_DPU_LOGGING=$(LOGGING)
ENABLE_AUTO_FENCING=$(ENABLE_AUTO_FENCING)
ENABLE_DPU_PRINTING=$(ENABLE_DPU_PRINTING)
CXX_STANDARD=$(CXX_STANDARD)
PIPELINE=$(PIPELINE)
JIT=$(JIT)
JIT_PIPELINE_FALLBACK=$(JIT_PIPELINE_FALLBACK)
JIT_BATCH_SIZE=$(JIT_BATCH_SIZE)
TRACE=$(TRACE)
PERFETTO_HOME=$(PERFETTO_HOME)
DEBUG_KEEP_JIT_DIR=$(DEBUG_KEEP_JIT_DIR)
ENABLE_PROMOTION_REDUCTIONS=$(ENABLE_PROMOTION_REDUCTIONS)
QUEUE_ABSORB_LIMIT=$(QUEUE_ABSORB_LIMIT)
ENABLE_OOM_RECOVERY=1
MAX_HFUSE_CHAINS=$(MAX_HFUSE_CHAINS)
MAX_VFUSE_OPS=$(MAX_VFUSE_OPS)
MAX_VFUSE_INPUTS=$(MAX_VFUSE_INPUTS)
MAX_PIPELINE_STACK_DEPTH=$(MAX_PIPELINE_STACK_DEPTH)
BLOCK_SIZE_LOG2=$(BLOCK_SIZE_LOG2)
endef
$(file >$(CONFIG_STAMP).new,$(CONFIG_CONTENT))
CONFIG_STATUS := $(shell \
  if [ ! -f $(CONFIG_STAMP) ]; then mv -f $(CONFIG_STAMP).new $(CONFIG_STAMP); echo new; \
  elif cmp -s $(CONFIG_STAMP).new $(CONFIG_STAMP); then rm -f $(CONFIG_STAMP).new; echo same; \
  else mv -f $(CONFIG_STAMP).new $(CONFIG_STAMP); echo changed; fi)
ifeq ($(CONFIG_STATUS),changed)
$(info Configuration changed, cleaning build...)
__clean := $(shell $(RM) -r $(BUILDDIR) $(TEST_TARGET))
endif

__dirs := $(shell mkdir -p ${BUILDDIR} && mkdir -p ${BUILDDIR}/bin && mkdir -p ${BUILDDIR}/lib)

COMMON_FLAGS := -Wall -Wextra -I${COMMON_DIR} -I${HOST_DIR} $(EXTRA_FLAGS)
HOST_FLAGS := ${COMMON_FLAGS} ${CXXFLAGS} `dpu-pkg-config --cflags --libs dpu`

# upmem-legion lives in its own checkout; legion/ builds its shared libraries.
LG_RT_DIR ?= $(CURDIR)/opt/upmem-legion/runtime
LEGION_LIB_DIR := $(CURDIR)/legion
LEGION_LIBS := $(LEGION_LIB_DIR)/liblegion.so $(LEGION_LIB_DIR)/librealm.so
ifeq ($(RUNTIME),legion)
HOST_FLAGS += -DPOLYMERPIM_LEGION=1 -I$(LEGION_LIB_DIR) -I$(LG_RT_DIR) \
              -I$(LG_RT_DIR)/mappers
LDFLAGS += -L$(LEGION_LIB_DIR) -Wl,-rpath,$(LEGION_LIB_DIR) -llegion -lrealm
endif
# DPU-specific flags
DPU_FLAGS := ${COMMON_FLAGS} -Os -DNR_TASKLETS=${NR_TASKLETS}

all: dependencies $(GENERATED_TARGETS) $(GENERATED_JULIA) config_check print_config ${HOST_TARGET} ${DPU_TARGET}
	@echo "Build complete: $(BUILD_TYPE) \n"

dependencies: $(DEPENDENCY_INSTALLER) scripts/install_perfetto.sh
	@$(DEPENDENCY_INSTALLER)


$(GENERATED_TARGETS) $(GENERATED_JULIA): tools/generate.py
	@echo "Generating kernel headers..."
	python3 tools/generate.py


# Explicit rule for config.h
common/config.h: tools/generate_config.py $(CONFIG_STAMP)
	@echo "Generating config header..."
	python3 tools/generate_config.py

make_header: common/config.h

# Kept for callers; the configuration is synced when the Makefile is read.
reconfigure config_check:

ifeq ($(RUNTIME),legion)
${HOST_TARGET}: $(LEGION_LIBS)
endif
${HOST_TARGET}: ${HOST_SOURCES} ${HOST_HEADERS} ${COMMON_HEADERS} $(GENERATED_TARGETS) | dependencies
	$(CXX) -std=${CXX_STANDARD} -shared -fPIC -o $@ ${HOST_SOURCES} ${HOST_FLAGS} $(LDFLAGS)

${DPU_TARGET}: ${DPU_SOURCES} ${DPU_HEADERS} ${COMMON_HEADERS} $(GENERATED_TARGETS)
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

$(TEST_TARGET): ${TEST_SOURCES} ${TEST_HEADERS} ${HOST_TARGET} ${DPU_TARGET}
	@echo "Building test target: $@"
	$(CXX) -std=${CXX_STANDARD} $(CXXFLAGS) $(COMMON_FLAGS) -pthread -o $@ $(TEST_SOURCES) -I$(HOST_INCLUDES)  \
		-L$(BUILDDIR)/lib -Wl,-rpath,$(BUILDDIR)/lib -lpolymerpim

# One sub-make builds both libraries.
$(LEGION_LIB_DIR)/liblegion.so: scripts/install_legion.sh legion/Makefile
	scripts/install_legion.sh
	$(MAKE) -C legion LG_RT_DIR=$(LG_RT_DIR)
$(LEGION_LIB_DIR)/librealm.so: $(LEGION_LIB_DIR)/liblegion.so

clean-internal:
	$(RM) -r $(BUILDDIR) $(TEST_TARGET)

clean: clean-internal
	$(RM) -r $(CONFIG_STAMP) $(GENERATED_TARGETS) $(GENERATED_JULIA) common/config.h

# ANSI color codes
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
CYAN   := \033[0;36m
NC     := \033[0m  # No color

print_config: reconfigure
	@echo "\n$(CYAN)Current build configuration:$(NC)"
	@cat $(CONFIG_STAMP) | while read line; do \
	    key=$${line%%=*}; \
	    value=$${line#*=}; \
	    echo "  $(YELLOW)$${key}=$(GREEN)$${value}$(NC)"; \
	done
	@echo "\n"

# TEST_ARGS is forwarded to the runner, e.g.
#   make test TEST_ARGS="--filter=hfuse --stats"
#   make test TEST_ARGS="--dpus=8 --isolate=0"
#
# --isolate runs every test in its own process.  It is the default because a
# handful of known runtime bugs (see test/README.md) crash or deadlock the
# process, and because state left behind by one failing test otherwise changes
# the outcome of later ones.
TEST_ARGS ?= --isolate

test: all $(TEST_TARGET)
	@printf "\n$(CYAN)Running tests...$(NC)\n\n"
	UPMEM_NO_OS_WARNING=1 ./$(TEST_TARGET) $(TEST_ARGS)

# Build the test binary without running it.
build-test: all $(TEST_TARGET)

list-tests: all $(TEST_TARGET)
	@./$(TEST_TARGET) --list
bindir := $(DESTDIR)/bin
libdir := $(DESTDIR)/lib
includedir := $(DESTDIR)/include/polymerpim
# Consumers link against this prefix, not the source tree, so the prefix has to
# be able to say what it holds without anyone guessing from mtimes.
sharedir := $(DESTDIR)/share/polymerpim
jitincludedir := $(sharedir)/jit

install: install-cpp
	@$(MAKE) install-julia

install-cpp: all
	@echo "Installing to $(DESTDIR)..."
	install -d $(bindir) $(libdir) $(includedir) $(jitincludedir)
	install -m 644 $(DPU_TARGET) $(bindir)
	install -m 644 $(HOST_TARGET) $(libdir)
	install -m 644 $(HOST_DIR)/polymerpim.h $(COMMON_DIR)/config.h $(includedir)
	install -m 644 $(COMMON_HEADERS) common/opcodes.h $(jitincludedir)
	# build.config verbatim, so it matches BUILD_CONFIG_STRING in the shipped library.
	install -d $(sharedir)
	install -m 644 $(CONFIG_STAMP) $(sharedir)/build.config
	@{ \
	  echo "INSTALL_DATE=$$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
	  echo "INSTALL_PREFIX=$$(cd $(DESTDIR) && pwd)"; \
	  echo "SOURCE_DIR=$$(pwd)"; \
	  echo "GIT_REV=$$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	  echo "GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"; \
	  echo "GIT_DIRTY=$$(git status --porcelain 2>/dev/null | head -1 | wc -l)"; \
	  echo "BUILD_TYPE=$(BUILD_TYPE)"; \
	  echo "CXX=$$($(CXX) --version 2>/dev/null | head -1)"; \
	  echo "HOST=$$(uname -n)"; \
	} > $(sharedir)/install.config
	@echo "Installed configuration recorded in $(sharedir)/"

# The Julia package binds ops that only exist under PIPELINE=1 JIT=1, so other
# configurations install the C++ library alone.
install-julia:
	@if [ "$(JIT)" != "1" ] || [ "$(PIPELINE)" != "1" ]; then \
	    echo "Skipping PolymerPIM.jl: needs PIPELINE=1 JIT=1 (have PIPELINE=$(PIPELINE) JIT=$(JIT))"; \
	elif ! command -v $(JULIA) >/dev/null 2>&1; then \
	    echo "Skipping PolymerPIM.jl: $(JULIA) not on PATH"; \
	else \
	    $(MAKE) -C julia wrapper POLYMERPIM_ROOT=$(abspath $(DESTDIR)); \
	fi

uninstall:
	@echo "Removing from $(prefix)..."
	rm -f $(bindir)/$(notdir $(DPU_TARGET))
	rm -f $(libdir)/$(notdir $(HOST_TARGET))
	rm -rf $(includedir)
	rm -rf $(sharedir)
