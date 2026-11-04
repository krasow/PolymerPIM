#pragma once
extern "C" {
#include <dpu.h>
}

#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <string>

#include "../../queue.h"
#include "../trace.h"
#include "config.h"

std::string operationtype_to_string(Event::OperationType op);

#define DPU_TRACK_ID uint64_t(0x1000)
#define EVENT_TRACK_BASE uint64_t(0x2000)
#define RANK_TRACK_BASE uint64_t(0x3000)

namespace trace {
// The queue event a backend launch runs for.
struct LaunchInfo {
  uint64_t id = 0;
  std::string name = "kernel";
};
using LaunchTag = std::shared_ptr<const LaunchInfo>;

#if TRACE == 1
void event_enqueued(std::shared_ptr<Event> e,
                    const std::deque<std::shared_ptr<Event>>& ops,
                    const std::list<std::shared_ptr<Event>>& running);
void event_fused(std::shared_ptr<Event> e, std::shared_ptr<Event> into,
                 const std::string& fused_ops);
void inqueue_end(std::shared_ptr<Event> e);
void execution_begin(std::shared_ptr<Event> e);
void execution_end();
void active_ops_counter(size_t count);
void ensure_callback_thread_named();

// Ranks run kernels concurrently, so the queue's order is only issue order.
void set_concurrent_ranks();
// Brackets dispatching `e`; its launches are named after it.
void dispatch_begin(const std::shared_ptr<Event>& e);
void dispatch_end();
// What a launch issued now runs for; null when not tracing.
LaunchTag current_launch();
// Queue on `set` (ranks first_rank.. of the system) the start and end of the
// kernel launched between them, on each rank's track; the event runs until
// every end fires.  `flow` links it to its queue event.  Call with the ranks'
// xfer lock held, passing begin's result to end.
void* rank_launch_begin(dpu_set_t set, uint32_t first_rank, uint32_t nr_ranks,
                        LaunchTag tag, bool flow);
void rank_launch_end(dpu_set_t set, void* launch);
// A host-side step on one rank's track, e.g. loading a binary.
void rank_span_begin(uint32_t rank, const char* name);
void rank_span_end(uint32_t rank);
#else
inline void event_enqueued(std::shared_ptr<Event> e,
                           const std::deque<std::shared_ptr<Event>>& ops,
                           const std::list<std::shared_ptr<Event>>& running) {
  (void)e;
  (void)ops;
  (void)running;
}
inline void event_fused(std::shared_ptr<Event> e, std::shared_ptr<Event> into,
                        const std::string& fused_ops) {
  (void)e;
  (void)into;
  (void)fused_ops;
}
inline void inqueue_end(std::shared_ptr<Event> e) { (void)e; }
inline void execution_begin(std::shared_ptr<Event> e) { (void)e; }
inline void execution_end() {}
inline void active_ops_counter(size_t count) { (void)count; }
inline void ensure_callback_thread_named() {}
inline void set_concurrent_ranks() {}
inline void dispatch_begin(const std::shared_ptr<Event>& e) { (void)e; }
inline void dispatch_end() {}
inline LaunchTag current_launch() { return nullptr; }
inline void* rank_launch_begin(dpu_set_t set, uint32_t first_rank,
                               uint32_t nr_ranks, LaunchTag tag, bool flow) {
  (void)set;
  (void)first_rank;
  (void)nr_ranks;
  (void)tag;
  (void)flow;
  return nullptr;
}
inline void rank_launch_end(dpu_set_t set, void* launch) {
  (void)set;
  (void)launch;
}
inline void rank_span_begin(uint32_t rank, const char* name) {
  (void)rank;
  (void)name;
}
inline void rank_span_end(uint32_t rank) { (void)rank; }
#endif

struct dispatch_scope {
  explicit dispatch_scope(const std::shared_ptr<Event>& e) {
    dispatch_begin(e);
  }
  ~dispatch_scope() { dispatch_end(); }
};
}  // namespace trace
