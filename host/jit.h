#pragma once

#include <cstdint>
#include <string>
#include <utility>  // For std::pair
#include <vector>

#include "config.h"

// Number of statically compiled kernels in the default DPU binary.
// JIT-compiled kernels are assigned IDs starting at this offset.
static constexpr uint32_t JIT_STATIC_KERNEL_COUNT = 17;

using Signature = std::pair<std::vector<uint8_t>, std::string>;

#if JIT

// Canonicalizes host/C++ type names before they become part of a JIT cache key.
std::string jit_canonical_type_name(const char* raw_type_name);

// Stable identifiers derived from the exact JIT cache keys.
std::string jit_signature_hash(const Signature& sig);
std::string jit_batch_hash(const std::vector<Signature>& kernels);

// Compiles a batch of unique RPN sequences into a single DPU binary
std::string jit_compile(const std::vector<Signature>& kernels);

// The generated C for one kernel, rendered without compiling or writing files.
// Same text jit_compile() puts in <jit_build_dir()>/k_<jit_signature_hash()>.c.
std::string jit_kernel_source(const Signature& sig);

// The scaffolding every kernel in a batch shares: launch args, tasklet barrier,
// WRAM workspace.  Emitted once per batch as main_<n>.c.
std::string jit_main_source();

// Where jit_compile() writes its sources, objects and linked binaries, and what
// jit_cleanup() removes: $POLYMERPIM_JIT_DIR/polymerpim-jit-<pid> when the
// variable is set, otherwise build/jit under the working directory.  Absolute.
std::string jit_build_dir();

// Returns true if the kernel is found in the specified binary
bool jit_find_kernel_in_binary(const Signature& sig,
                               const std::string& bin_path, int& out_idx);

// Cleanup JIT files at shutdown
void jit_cleanup();

#endif
