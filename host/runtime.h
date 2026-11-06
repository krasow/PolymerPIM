#pragma once
#include <memory>

#include "backend.h"
#include "logger.h"
#include "queue.h"

class DpuRuntime {
 private:
  DpuRuntime() : initialized_(false) {}
  ~DpuRuntime() = default;

  bool initialized_;
  std::unique_ptr<DpuBackend> backend_;
  uint32_t num_dpus_;
  std::unique_ptr<EventQueue> event_queue_;
  std::unique_ptr<Logger> logger_;

 public:
  // Delete copy/move
  DpuRuntime(const DpuRuntime&) = delete;
  DpuRuntime& operator=(const DpuRuntime&) = delete;

  // Get singleton instance
  static DpuRuntime& get() {
    static DpuRuntime instance;
    return instance;
  }

  void init(uint32_t num_dpus);
  bool is_initialized() const { return initialized_; }

  // The count init() will be given on first use: $NR_DPUS, or 8.  Readable
  // before the runtime exists, unlike num_dpus().
  static uint32_t configured_num_dpus();

  void debug_read_dpu_log();

  EventQueue& get_event_queue();
  Logger& get_logger();
  DpuBackend& backend();
  uint32_t num_dpus() const;
  uint32_t num_tasklets() const;

  std::string get_default_binary_path() const;

  void shutdown();
};
