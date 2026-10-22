#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

// dl function info
#include <dlfcn.h>
#include <libgen.h>
#include <limits.h>

#include "jit.h"
#include "perfetto/trace.h"
#include "runtime.h"

// (Moved to trace.cc)

std::string get_runtime_dpu_binary();

EventQueue& DpuRuntime::get_event_queue() { return *event_queue_; }
Logger& DpuRuntime::get_logger() { return *logger_; }
DpuBackend& DpuRuntime::backend() { return *backend_; }
uint32_t DpuRuntime::num_dpus() const { return num_dpus_; }

uint32_t DpuRuntime::configured_num_dpus() {
  const char* env_val = std::getenv("NR_DPUS");
  if (env_val == nullptr) {
    return 8;
  }
  int parsed = std::atoi(env_val);
  return parsed > 0 ? (uint32_t)parsed : 8;
}
uint32_t DpuRuntime::num_tasklets() const { return NR_TASKLETS; }

std::string DpuRuntime::get_default_binary_path() const {
  return get_runtime_dpu_binary();
}

extern "C" void polymerpim_dladdr_anchor() {}

std::string get_runtime_dpu_binary() {
  Dl_info info;
  void* fptr = (void*)&polymerpim_dladdr_anchor;
  if (dladdr(fptr, &info) == 0) {
    throw std::runtime_error("Failed to get library path");
  }

  // Use std::filesystem to resolve path
  fs::path lib_path = fs::absolute(info.dli_fname);

  // Try relative to library: lib/libpolymerpim.so -> ../bin/runtime.dpu
  fs::path bin_path =
      lib_path.parent_path().parent_path() / "bin" / "runtime.dpu";

  if (!fs::exists(bin_path)) {
    // Fallback: try relative from CWD/build
    // If running from source root: build/bin/runtime.dpu
    bin_path = fs::absolute("build/bin/runtime.dpu");
    if (!fs::exists(bin_path)) {
      std::stringstream ss;
      ss << "Failed to resolve runtime.dpu path.\n"
         << "lib_path: " << lib_path << "\n"
         << "CWD: " << fs::current_path() << "\n"
         << "Checked: " << bin_path;
      throw std::runtime_error(ss.str());
    }
  }

  return fs::absolute(bin_path).string();
}

void DpuRuntime::init(uint32_t num_dpus) {
  if (initialized_) return;  // idempotent
  TRACE_INIT();

  trace::scoped_event trace_scoped("runtime", "DpuRuntime::init");

  num_dpus_ = num_dpus;
  logger_ = std::make_unique<Logger>();

#if ENABLE_DPU_LOGGING >= 1
  logger_->lock(logcat::RUNTIME) << "Initializing DPU runtime with "
                                 << num_dpus_ << " DPUs..." << std::endl;
#endif

  // Allocate the DPUs and load the default binary
  std::string dpu_file = get_runtime_dpu_binary();
  backend_ = make_native_backend();
  num_dpus_ = backend_->open(num_dpus_, dpu_file);

#if ENABLE_DPU_LOGGING >= 1
  logger_->lock(logcat::RUNTIME)
      << "Allocated " << num_dpus_ << " DPUs..." << std::endl;
  logger_->lock(logcat::RUNTIME)
      << "DPU runtime initialized with backend=" << BACKEND << std::endl;
#endif

  event_queue_ = std::make_unique<EventQueue>();

  initialized_ = true;
}

void DpuRuntime::shutdown() {
  if (!initialized_) return;

  {
    trace::scoped_event trace_scoped("runtime", "DpuRuntime::shutdown");

#if ENABLE_DPU_LOGGING >= 1
    if (logger_)
      logger_->lock(logcat::RUNTIME)
          << "Shutting down DPU runtime..." << std::endl;
#endif

    if (event_queue_ && event_queue_->has_pending()) {
      if (logger_)
        logger_->lock(logcat::RUNTIME)
            << "Flushing pending events..." << std::endl;
      event_queue_->process_events(UINT64_MAX);
    }

    if (logger_)
      logger_->lock(logcat::RUNTIME)
          << "Waiting for active events and callbacks..." << std::endl;
    while (event_queue_) {
      std::list<std::shared_ptr<Event>> active;
      {
        std::lock_guard<std::recursive_mutex> lock(event_queue_->get_mutex());
        active = event_queue_->get_active_events();
        if (active.empty() && event_queue_->outstanding_callbacks_.load() == 0)
          break;
      }
      for (auto& e : active) {
        if (e) e->wait();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (logger_)
      logger_->lock(logcat::RUNTIME) << "Freeing DPU set..." << std::endl;
    if (backend_) {
      backend_->close();
      backend_.reset();
    }
    initialized_ = false;
  }  // trace_scoped ends here, before TRACE_SHUTDOWN

  if (logger_)
    logger_->lock(logcat::RUNTIME) << "Tracing shutdown..." << std::endl;
  TRACE_SHUTDOWN();

#if JIT_PIPELINE_FALLBACK
  if (event_queue_) event_queue_->await_jit_compilations();
#endif
#if JIT
  if (logger_)
    logger_->lock(logcat::RUNTIME) << "Cleaning up JIT files..." << std::endl;
  jit_cleanup();
#endif

  if (logger_)
    logger_->lock(logcat::RUNTIME) << "Shutdown complete." << std::endl;

  // Reset core systems explicitly so they don't hang in static destructor
  event_queue_.reset();
  logger_.reset();
}

void DpuRuntime::debug_read_dpu_log() { backend().read_logs(stdout); }
