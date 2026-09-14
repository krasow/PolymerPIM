// JIT orchestration: hashing a kernel signature, batching kernels so they share
// a binary, caching the results, and cleaning up.
//
// Source generation lives in host/detail/jit_codegen.cc and toolchain
// invocation in host/detail/jit_toolchain.cc.

#include <jit.h>

#if JIT
#include <common.h>
#include <detail/fusion.h>
#include <detail/jit_codegen.h>
#include <detail/jit_toolchain.h>
#include <detail/vector.h>
#include <dlfcn.h>
#include <logger.h>
#include <opcodes.h>
#include <perfetto/trace.h>
#include <queue.h>
#include <runtime.h>
#include <stats.h>

#include <algorithm>
#if JIT_PIPELINE_FALLBACK
#include <atomic>
#include <condition_variable>
#endif
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

using detail::compile_dpu_source;
using detail::get_include_flags;
using detail::link_dpu_objects;
using detail::write_dpu_main_header;
using detail::write_kernel_function;

namespace {
using CacheKey = std::vector<Signature>;
std::map<CacheKey, std::string> g_jit_cache;
std::map<Signature, std::string> g_kernel_obj_cache;
std::map<Signature, std::shared_future<std::string>> g_kernel_obj_inflight;
std::recursive_mutex g_jit_cache_mutex;
std::mutex g_jit_link_mutex;
#if JIT_PIPELINE_FALLBACK
std::atomic<int> g_binary_counter{0};
std::mutex g_jit_compile_mutex;
std::condition_variable g_jit_compile_ready;
size_t g_jit_compiles = 0;
constexpr size_t kMaxJitCompiles = 2;

class JitCompileSlot {
 public:
  JitCompileSlot() {
    std::unique_lock<std::mutex> lock(g_jit_compile_mutex);
    g_jit_compile_ready.wait(lock,
                             [] { return g_jit_compiles < kMaxJitCompiles; });
    ++g_jit_compiles;
  }

  ~JitCompileSlot() {
    {
      std::lock_guard<std::mutex> lock(g_jit_compile_mutex);
      --g_jit_compiles;
    }
    g_jit_compile_ready.notify_one();
  }
};
#else
int g_binary_counter = 0;
#endif

// The DPU IRAM cannot hold an unbounded batch, so cap how many kernels share a
// binary regardless of JIT_BATCH_SIZE.
size_t jit_link_batch_limit() {
  constexpr size_t kIramSafeLinkBatch = 6;
  return JIT_BATCH_SIZE < kIramSafeLinkBatch ? JIT_BATCH_SIZE
                                             : kIramSafeLinkBatch;
}
}  // namespace

std::string jit_canonical_type_name(const char* raw_type_name) {
  if (!raw_type_name) return "int32_t";
  std::string tn = raw_type_name;
  if (tn == "i" || tn == "int" || tn == "int32_t") return "int32_t";
  if (tn == "j" || tn == "uint32_t") return "uint32_t";
  if (tn == "f" || tn == "float") return "float";
  if (tn == "d" || tn == "double") return "double";
  return raw_type_name;
}

std::string jit_signature_hash(const Signature& sig) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint8_t byte) {
    h ^= byte;
    h *= 1099511628211ull;
  };
  for (char c : sig.second) mix(static_cast<uint8_t>(c));
  mix(0xff);
  for (uint8_t b : sig.first) mix(b);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return std::string(buf);
}

std::string jit_batch_hash(const std::vector<Signature>& kernels) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint8_t byte) {
    h ^= byte;
    h *= 1099511628211ull;
  };
  for (const auto& sig : kernels) {
    std::string hash = jit_signature_hash(sig);
    for (char c : hash) mix(static_cast<uint8_t>(c));
    mix(0xfe);
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return std::string(buf);
}

static std::string compile_kernel_object_uncached(
    const Signature& sig, const std::string& build_dir,
    const std::string& include_flags, const std::string& kernel_hash) {
  VECTORDPU_NOTE(jit_kernel_compiles);
#if ENABLE_DPU_LOGGING >= 1
  {
    auto log = DpuRuntime::get().get_logger().lock(logcat::JIT_COMPILER);
    log.first() << "compile kernel object";
    log.second() << "kernel_hash=" << kernel_hash << " type=" << sig.second
                 << " "
                 << detail::fusion_rpn_fields(
                        detail::summarize_fusion_rpn(sig.first))
                 << std::endl;
  }
#endif

  const std::string c_path = build_dir + "/k_" + kernel_hash + ".c";
  const std::string obj_path = build_dir + "/k_" + kernel_hash + ".o";
  {
    std::ofstream out(c_path);
    write_kernel_function(out, "k_" + kernel_hash, sig.first, sig.second);
  }
#if ENABLE_DPU_LOGGING >= 1
  {
    auto log = DpuRuntime::get().get_logger().lock(logcat::JIT_DEBUG, 2);
    log.first() << "wrote kernel source";
    log.second() << "kernel_hash=" << kernel_hash << " path=" << c_path
                 << std::endl;
  }
#endif

#if JIT_PIPELINE_FALLBACK
  JitCompileSlot compile_slot;
#endif
  if (!compile_dpu_source(c_path, obj_path, true, include_flags)) {
    trace::jit_compile_end();
    throw std::runtime_error("JIT Compilation failed for " + c_path);
  }
#if ENABLE_DPU_LOGGING >= 1
  DpuRuntime::get().get_logger().lock(logcat::JIT_DEBUG, 2)
      << "compiled " << obj_path << std::endl;
#endif
  return obj_path;
}

static std::string compile_kernel_object(const Signature& sig,
                                         const std::string& build_dir,
                                         const std::string& include_flags) {
  const std::string kernel_hash = jit_signature_hash(sig);
  std::shared_future<std::string> object_future;
  std::shared_ptr<std::promise<std::string>> object_promise;
  {
    std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
    auto cached = g_kernel_obj_cache.find(sig);
    if (cached != g_kernel_obj_cache.end()) {
      VECTORDPU_NOTE(jit_kernel_cache_hits);
#if ENABLE_DPU_LOGGING >= 2
      auto log = DpuRuntime::get().get_logger().lock(logcat::JIT_COMPILER, 2);
      log.first() << "cache hit kernel object";
      log.second() << "kernel_hash=" << kernel_hash << " type=" << sig.second
                   << " path=" << cached->second << std::endl;
#endif
      return cached->second;
    }

    auto inflight = g_kernel_obj_inflight.find(sig);
    if (inflight != g_kernel_obj_inflight.end()) {
      VECTORDPU_NOTE(jit_kernel_cache_hits);
      object_future = inflight->second;
    } else {
      object_promise = std::make_shared<std::promise<std::string>>();
      object_future = object_promise->get_future().share();
      g_kernel_obj_inflight[sig] = object_future;
    }
  }

  if (!object_promise) return object_future.get();

  try {
    std::string obj_path = compile_kernel_object_uncached(
        sig, build_dir, include_flags, kernel_hash);

    {
      std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
      g_kernel_obj_cache[sig] = obj_path;
      g_kernel_obj_inflight.erase(sig);
    }
    object_promise->set_value(obj_path);
    return obj_path;
  } catch (...) {
    {
      std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
      g_kernel_obj_inflight.erase(sig);
    }
    object_promise->set_exception(std::current_exception());
    throw;
  }
}

// Writes the batch's main(), which dispatches on args.kernel to the right
// sub-kernel.  Kernel ids start at JIT_STATIC_KERNEL_COUNT, after the kernels
// baked into the default binary.
static void write_dpu_main(const std::string& path,
                           const std::vector<Signature>& kernels) {
  std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
  std::ofstream out(path);
  write_dpu_main_header(out);

  for (const Signature& sig : kernels)
    out << "extern int k_" << jit_signature_hash(sig) << "(void);\n";

  out << "\nint main() {\n  switch (args.kernel) {\n";
  for (size_t k = 0; k < kernels.size(); ++k)
    out << "    case " << (JIT_STATIC_KERNEL_COUNT + k) << ": return k_"
        << jit_signature_hash(kernels[k]) << "();\n";
  out << "    default: return -1;\n  }\n}\n";
}

std::string jit_compile(
    const std::vector<std::pair<std::vector<uint8_t>, std::string>>& kernels) {
  const std::string batch_hash = jit_batch_hash(kernels);
  {
    std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
    auto it = g_jit_cache.find(kernels);
    if (it != g_jit_cache.end()) {
      VECTORDPU_NOTE(jit_batch_cache_hits);
#if ENABLE_DPU_LOGGING >= 1
      auto log = DpuRuntime::get().get_logger().lock(logcat::JIT_COMPILER);
      log.first() << "cache hit linked binary";
      log.second() << "batch_hash=" << batch_hash
                   << " kernels=" << kernels.size() << " path=" << it->second
                   << std::endl;
#endif
      return it->second;
    }
  }

  trace::jit_compile_begin(kernels);

  const std::string include_flags = get_include_flags();
  const std::string build_dir = jit_build_dir();
  fs::create_directories(build_dir);

  std::vector<std::string> object_files;
  for (const auto& sig : kernels)
    object_files.push_back(
        compile_kernel_object(sig, build_dir, include_flags));

    // Generate a main() that dispatches on args.kernel to the right sub-kernel.
#if JIT_PIPELINE_FALLBACK
  int binary_id = g_binary_counter.fetch_add(1, std::memory_order_relaxed);
#else
  int binary_id = g_binary_counter++;
#endif
  const std::string main_c_path =
      build_dir + "/main_" + std::to_string(binary_id) + ".c";
  const std::string binpath = main_c_path + ".dpu";

  write_dpu_main(main_c_path, kernels);

  bool linked;
  {
    std::lock_guard<std::mutex> lock(g_jit_link_mutex);
    linked = link_dpu_objects(main_c_path, object_files, binpath, include_flags,
                              batch_hash);
  }
  if (!linked) {
    trace::jit_compile_end();
    throw std::runtime_error("JIT Linking failed for " + binpath);
  }
#if ENABLE_DPU_LOGGING >= 1
  DpuRuntime::get().get_logger().lock(logcat::JIT_DEBUG, 2)
      << "linked " << binpath << std::endl;
#endif

  {
    std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
    VECTORDPU_NOTE(jit_batch_links);
    g_jit_cache[kernels] = binpath;
  }
  trace::jit_compile_end();
  return binpath;
}

void EventQueue::flush_jit_batch() {
  if (pending_unique_kernels_.empty()) return;

  std::vector<std::pair<std::vector<uint8_t>, std::string>> batch =
      pending_unique_kernels_;

  // Re-link the previous kernels too, so a mix of old and new shapes does not
  // ping-pong between binaries.  New kernels stay in front to keep slots.
  for (const auto& prior : last_linked_kernels_) {
    if (batch.size() >= jit_link_batch_limit()) break;
    bool present = false;
    for (const auto& have : batch) present = present || (have == prior);
    if (!present) batch.push_back(prior);
  }

#if ENABLE_DPU_LOGGING >= 1
  auto log = DpuRuntime::get().get_logger().lock(logcat::QUEUE_JIT);
  log.first() << "flush JIT batch";
  log.second() << "batch_hash=" << jit_batch_hash(batch)
               << " kernels=" << batch.size() << std::endl;
#endif

  std::shared_future<std::string> future = std::async(
#if JIT_PIPELINE_FALLBACK
      std::launch::async,
#else
      std::launch::deferred,
#endif
      [batch]() { return jit_compile(batch); });
  for (auto& ev : pending_jit_events_) ev->jit_future = future;
  // Every kernel in the superset now resolves to this binary.
  for (size_t i = 0; i < batch.size(); ++i)
    inflight_jit_kernels_[batch[i]] = {future, (int)i};
  last_linked_kernels_ = batch;

  pending_jit_events_.clear();
  pending_unique_kernels_.clear();
}

void EventQueue::lock_for_jit(std::shared_ptr<Event> e) {
  if (e->op != Event::OperationType::COMPUTE || e->is_locked_for_jit ||
      !e->jit_binary_path.empty())
    return;
#if JIT_PIPELINE_FALLBACK
  if (unused_compiles_ >= kStopCompilingAfter) return;
#endif
  if (e->inputs.empty()) return;  // no input: no RPN form (e.g. a fill)
  e->is_locked_for_jit = true;

  if (e->rpn_ops.empty()) {
    e->rpn_ops.push_back(OP_PUSH_INPUT);
    if (e->is_scalar) {
      e->rpn_ops.push_back(detail::map_to_var_op(e->opcode));
      e->rpn_ops.push_back(0);
      e->scalars.push_back(e->scalar_value);
    } else {
      if (e->inputs.size() > 1) e->rpn_ops.push_back(OP_PUSH_OPERAND_0);
      e->rpn_ops.push_back(e->opcode);
    }
  }

  const char* raw_type_name =
      (e->output && e->output->type_name) ? e->output->type_name : "int32_t";
  std::string canonical_type = jit_canonical_type_name(raw_type_name);
  Signature sig = {e->rpn_ops, canonical_type};
  e->jit_kernel_hash = jit_signature_hash(sig);

  // Check if this signature already has a slot in the current batch.
  for (size_t i = 0; i < pending_unique_kernels_.size(); ++i) {
    if (pending_unique_kernels_[i] == sig) {
      e->jit_sub_kernel_idx = i;
      pending_jit_events_.push_back(e);
#if ENABLE_DPU_LOGGING >= 2
      auto log = DpuRuntime::get().get_logger().lock(logcat::QUEUE_JIT, 2);
      log.first() << "cache hit pending JIT batch";
      log.second() << "event_id=" << e->id
                   << " kernel_hash=" << e->jit_kernel_hash
                   << " sub_kernel=" << i
                   << " pending_events=" << pending_jit_events_.size()
                   << std::endl;
#endif
      if (pending_jit_events_.size() >= jit_link_batch_limit())
        flush_jit_batch();
      return;
    }
  }

  // Reuse the binary a kernel already sits in, to avoid binary ping-pong.
  auto inflight = inflight_jit_kernels_.find(sig);
  if (inflight != inflight_jit_kernels_.end() &&
      pending_unique_kernels_.empty()) {
    e->jit_future = inflight->second.binary;
    e->jit_sub_kernel_idx = inflight->second.slot;
    return;
  }

  if (pending_unique_kernels_.size() >= jit_link_batch_limit())
    flush_jit_batch();

  e->jit_sub_kernel_idx = pending_unique_kernels_.size();
  pending_unique_kernels_.push_back(sig);
  pending_jit_events_.push_back(e);
  if (pending_jit_events_.size() >= jit_link_batch_limit()) flush_jit_batch();
}

#if JIT_PIPELINE_FALLBACK
void EventQueue::await_jit_compilations() {
  for (const auto& [signature, kernel] : inflight_jit_kernels_) {
    (void)signature;
    kernel.binary.get();
  }
}
#endif

bool jit_find_kernel_in_binary(const Signature& sig,
                               const std::string& bin_path, int& out_idx) {
  std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
  for (const auto& [kernels, path] : g_jit_cache) {
    if (path == bin_path) {
      for (size_t i = 0; i < kernels.size(); ++i) {
        if (kernels[i] == sig) {
          out_idx = (int)i;
          return true;
        }
      }
    }
  }
  return false;
}

std::string jit_build_dir() { return "build/jit"; }

// Rendering the source without compiling it: what @code_jitted shows in Julia.
std::string jit_kernel_source(const Signature& sig) {
  std::ostringstream out;
  write_kernel_function(out, "k_" + jit_signature_hash(sig), sig.first,
                        sig.second);
  return out.str();
}

std::string jit_main_source() {
  std::ostringstream out;
  write_dpu_main_header(out);
  return out.str();
}

void jit_cleanup() {
  std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
#if DEBUG_KEEP_JIT_DIR
  return;
#endif
  const std::string build_dir = jit_build_dir();
  if (fs::exists(build_dir)) {
    try {
      fs::remove_all(build_dir);
    } catch (...) {
    }
  }
}

#endif  // JIT
