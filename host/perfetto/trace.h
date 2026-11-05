#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../opinfo.h"
#if __has_include("../config.h")
#include "../config.h"
#else
#include "config.h"
#endif

// Descriptive helpers available even if TRACE=0
#include "../queue.h"
std::string operationtype_to_string(Event::OperationType op);

namespace trace {

// Library-internal opaque tracing hooks to avoid pulling Perfetto into user
// headers. These are implemented in trace.cc using the full Perfetto SDK.
#if TRACE == 1
bool enabled();
void internal_reduction_begin(uint64_t flow_id);
void internal_reduction_end();
void internal_to_cpu_begin(uint64_t flow_id);
void internal_to_cpu_end();
void internal_from_cpu_begin();
void internal_from_cpu_end();

// Opaque hooks for general tracing
void counter(const char* cat, const char* name, int64_t value);
void event_begin(const char* cat, const char* name);
void event_end(const char* cat);
void jit_compile_begin(const std::vector<uint8_t>& rpn_ops,
                       const char* type_name);
void jit_compile_begin(
    const std::vector<std::pair<std::vector<uint8_t>, std::string>>& kernels);
void jit_compile_end();
void jit_binary_switch(const std::string& previous, const std::string& current);
#else
inline bool enabled() { return false; }
inline void internal_reduction_begin(uint64_t flow_id) { (void)flow_id; }
inline void internal_reduction_end() {}
inline void internal_to_cpu_begin(uint64_t flow_id) { (void)flow_id; }
inline void internal_to_cpu_end() {}
inline void internal_from_cpu_begin() {}
inline void internal_from_cpu_end() {}

inline void counter(const char* cat, const char* name, int64_t value) {
  (void)cat;
  (void)name;
  (void)value;
}
inline void event_begin(const char* cat, const char* name) {
  (void)cat;
  (void)name;
}
inline void event_end(const char* cat) { (void)cat; }
inline void jit_compile_begin(const std::vector<uint8_t>& rpn_ops,
                              const char* type_name) {
  (void)rpn_ops;
  (void)type_name;
}
inline void jit_compile_begin(
    const std::vector<std::pair<std::vector<uint8_t>, std::string>>& kernels) {
  (void)kernels;
}
inline void jit_compile_end() {}
inline void jit_binary_switch(const std::string& previous,
                              const std::string& current) {
  (void)previous;
  (void)current;
}
#endif

struct scoped_event {
  const char* category;
  scoped_event(const char* cat, const char* name) : category(cat) {
    event_begin(cat, name);
  }
  ~scoped_event() { event_end(category); }
};

// RAII helpers for vector template operations.
// These call the opaque functions above so the user program doesn't need
// <perfetto.h>.
struct reduction_cpu {
  reduction_cpu(uint64_t flow_id) { internal_reduction_begin(flow_id); }
  ~reduction_cpu() { internal_reduction_end(); }
};

struct to_cpu {
  to_cpu(uint64_t flow_id) { internal_to_cpu_begin(flow_id); }
  ~to_cpu() { internal_to_cpu_end(); }
};

struct from_cpu {
  from_cpu() { internal_from_cpu_begin(); }
  ~from_cpu() { internal_from_cpu_end(); }
};

}  // namespace trace

// Standard Trace Init/Shutdown
#if TRACE == 1
namespace trace {
void initialize();
void shutdown();
// Records a trace into `path` until stop(); throws if one is recording.
void start(const std::string& path);
// Writes the trace start() began and returns its absolute path.
std::string stop();
}  // namespace trace
#define TRACE_INIT() trace::initialize()
#define TRACE_SHUTDOWN() trace::shutdown()
#else
namespace trace {
[[noreturn]] inline void start(const std::string& path) {
  (void)path;
  throw std::runtime_error("libpolymerpim was built with TRACE=0");
}
[[noreturn]] inline std::string stop() {
  throw std::runtime_error("libpolymerpim was built with TRACE=0");
}
}  // namespace trace
#define TRACE_INIT()
#define TRACE_SHUTDOWN()
#endif
