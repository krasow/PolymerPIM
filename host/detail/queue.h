#pragma once
// Low-level queue mechanics.  host/queue.cc holds the policy -- what the queue
// decides to do -- and calls in here for how each step is carried out.

#include <queue.h>

#include <memory>

namespace detail {

// Sets the Perfetto slice name for `e` from its kernel and JIT binary.
void name_event(const std::shared_ptr<Event>& e);

// The event this thread is dispatching, or null.
const Event* dispatching_event();

// Compiles a lone fused kernel that never joined a JIT batch.
void compile_kernel_if_unbatched(const std::shared_ptr<Event>& e);

// Logging wrappers; each compiles away when its level is not enabled.
void log_next_operation(const std::shared_ptr<Event>& e);
void log_launch(const std::shared_ptr<Event>& e);
void log_oom_caught(const std::shared_ptr<Event>& e);

}  // namespace detail
