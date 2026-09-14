// The event queue's decision making: what to fuse, what to enqueue, and the
// order in which an event's stages run.
//
// The mechanics of each stage live in host/detail/queue.cc.

#include <detail/fusion.h>
#include <detail/queue.h>
#include <detail/vector.h>
#include <jit.h>
#include <opinfo.h>
#include <perfetto/detail/trace.h>
#include <perfetto/trace.h>
#include <queue.h>
#include <runtime.h>
#include <stats.h>

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <sstream>
#include <thread>

#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

using detail::compile_kernel_if_unbatched;
using detail::log_launch;
using detail::log_next_operation;
using detail::log_oom_caught;
using detail::name_event;

void EventQueue::sync() {
  size_t last_id;
  {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
#if PIPELINE
    // No later submission is guaranteed to follow the final full expression.
    // Reap its dead temporaries here so they cannot split the last fusion
    // group while the queue drains.
    retire_inlined_producers(nullptr);
    compact_fusable_operations();
#endif
    if (operations_.empty() && running_events_.empty()) return;
    last_id = counter_ - 1;
  }
  process_events(last_id);

  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();
  CHECK_UPMEM(dpu_sync(dpu_set));
  finalize_finished_events();
}

constexpr bool NO_PROGRESS = false;
constexpr bool YES_PROGRESS = true;

bool EventQueue::process_next() {
  std::shared_ptr<Event> e = take_next_operation();
  if (!e) return NO_PROGRESS;

  log_next_operation(e);
  trace::inqueue_end(e);

  // Its consumer recomputes this inline and nothing else can read it, so the
  // event is redundant.  Retire it without launching, keeping its id range so
  // anything waiting on it is released.
  if (!output_still_needed(e)) {
    VECTORDPU_NOTE(absorbed_producers);
    begin_running(e);
    e->mark_finished();
    return YES_PROGRESS;
  }

  await_dependencies(e);
  grow_fusion_batch(e);
  await_jit_binary(e);

  name_event(e);
  log_launch(e);

  begin_running(e);
#if JIT_PIPELINE_FALLBACK
  // Gate both compile paths together: skipping only the batch path diverts
  // here, which compiles synchronously and blocks.
  if (unused_compiles_ < kStopCompilingAfter) compile_kernel_if_unbatched(e);
  if (e->is_locked_for_jit && !e->jit_pipeline_fallback) unused_compiles_ = 0;
#else
  compile_kernel_if_unbatched(e);
#endif
  switch_dpu_binary(e);

  try {
    dispatch(e);
  } catch (const DpuOOMException&) {
    log_oom_caught(e);
#if !ENABLE_OOM_RECOVERY
    throw DpuOOMException("DPU OOM: event id=" + std::to_string(e->id));
#else
    VECTORDPU_NOTE(oom_retries);
    if (++e->oom_retries > OOM_RECOVERY_RETRIES)
      throw DpuOOMException("DPU OOM: event id=" + std::to_string(e->id) +
                            " failed after " +
                            std::to_string(OOM_RECOVERY_RETRIES) + " retries");
    requeue_after_oom(e);
    return NO_PROGRESS;
#endif
  }

  debug_active_events();
  debug_print_queue();
  return YES_PROGRESS;
}

void EventQueue::process_events(size_t wait_for_id) {
#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock(logcat::QUEUE_WAIT, 2)
      << "begin wait_for_id=" << wait_for_id << std::endl;
#endif
  while (true) {
    bool progress = this->process_next();

    if (this->get_last_finished_id() >= wait_for_id) break;

    {
      std::lock_guard<std::recursive_mutex> lock(mtx_);
      if (operations_.empty() && running_events_.empty()) break;
    }

    auto& runtime = DpuRuntime::get();
    dpu_set_t& dpu_set = runtime.dpu_set();
#if ENABLE_DPU_LOGGING >= 2
    logger.lock(logcat::QUEUE_WAIT, 2)
        << "dpu_sync wait_for_id=" << wait_for_id
        << " last_finished=" << this->get_last_finished_id() << std::endl;
#endif
    CHECK_UPMEM(dpu_sync(dpu_set));
    finalize_finished_events();
#if ENABLE_DPU_LOGGING >= 2
    logger.lock(logcat::QUEUE_WAIT, 2)
        << "dpu_sync done wait_for_id=" << wait_for_id
        << " last_finished=" << this->get_last_finished_id() << std::endl;
#endif

    if (!progress) std::this_thread::sleep_for(std::chrono::milliseconds(1));
#if ENABLE_DPU_LOGGING >= 2
    static size_t loop_count = 0;
    if (++loop_count % 1000 == 0) {
      std::lock_guard<std::recursive_mutex> lock(mtx_);
      logger.lock(logcat::QUEUE_HEARTBEAT, 2)
          << "process_events waiting for " << wait_for_id
          << " (last_finished=" << this->get_last_finished_id()
          << " ops=" << operations_.size()
          << " running=" << running_events_.size() << ")" << std::endl;
    }
#endif
  }
#if ENABLE_DPU_LOGGING >= 2
  logger.lock(logcat::QUEUE_WAIT, 2)
      << "end wait_for_id=" << wait_for_id
      << " last_finished=" << this->get_last_finished_id() << std::endl;
#endif
}

// Thin dispatcher: classifies the fusion as vertical or horizontal and
// delegates to the appropriate implementation in vfuse.cc / hfuse.cc.
bool EventQueue::try_fuse(std::shared_ptr<Event> last,
                          std::shared_ptr<Event> e) {
#if PIPELINE
  if (last->op != Event::OperationType::COMPUTE ||
      e->op != Event::OperationType::COMPUTE || last->output == nullptr)
    return false;
  if (last->is_locked_for_jit || e->is_locked_for_jit) return false;
  // A fill has no vector input, so it has no RPN form: every program starts
  // with OP_PUSH_INPUT.
  if (last->inputs.empty() || e->inputs.empty()) return false;

  bool dependent = false;
  for (const auto& in : e->inputs) {
    if (in == last->output) {
      dependent = true;
      break;
    }
    for (const auto& out : last->extra_outputs)
      if (in == out) {
        dependent = true;
        break;
      }
  }

  if (dependent) {
    bool fused = try_vfuse(last, e);
    if (fused) retarget_inlined_producers(e->id, last->id);
    return fused;
  }

  // If e already inlined last's output via absorbed_rpn, horizontally fusing
  // last as a separate chain would duplicate that work and shift result slots.
  if (e->absorbed_producer_ids.find(last->id) != e->absorbed_producer_ids.end())
    return false;

  // Horizontal: independent chains, same element count, operand budget fits.
  if (last->inputs.empty() || e->inputs.empty()) return false;
  if (last->inputs[0]->num_elements != e->inputs[0]->num_elements) return false;

  std::vector<detail::VectorDescRef> unique = last->inputs;
  for (const auto& in : e->inputs) {
    bool found = false;
    for (const auto& u : unique)
      if (in == u) {
        found = true;
        break;
      }
    if (!found) unique.push_back(in);
  }
  if (unique.size() > MAX_COMBINED_INPUTS) return false;

  // Hold back events a vfuse candidate is likely to want: hfusing them now
  // spends an extra_output slot and blocks the deeper vertical chain.  A fresh
  // scalar op is usually about to be consumed by an accumulator (linreg's
  // `error += dx[j]*dw[j]`), and an output marked for absorption will be
  // inlined into its consumer anyway.
  if (e->is_scalar && e->rpn_ops.empty()) return false;
  if (e->output && !e->output->absorbed_rpn.empty()) return false;

  bool fused = try_hfuse(last, e);
  if (fused) retarget_inlined_producers(e->id, last->id);
  return fused;
#else
  return false;
#endif
}

// Merges `e` into the tail of the queue, then walks backwards collapsing any
// chain the merge just made fusable (K1->K2->K3 becomes one kernel).
bool EventQueue::fuse_into_queue_tail(const std::shared_ptr<Event>& e) {
#if PIPELINE
  if (e->op != Event::OperationType::COMPUTE || operations_.empty())
    return false;
  if (!try_fuse(operations_.back(), e)) return false;

  while (operations_.size() >= 2) {
    auto& prev = operations_[operations_.size() - 2];
    auto& tail = operations_.back();
    if (!try_fuse(prev, tail)) break;
    operations_.pop_back();
  }
  return true;
#else
  (void)e;
  return false;
#endif
}

void EventQueue::enqueue(const std::shared_ptr<Event>& e) {
#if JIT
  // Only a non-COMPUTE event ends the run of fusable kernels, so only that
  // is a safe point to flush.  Reductions stay batchable so process_next's
  // look-ahead can absorb them instead of forcing a mid-stream binary swap.
  if (e->op != Event::OperationType::COMPUTE) {
    bool any_locked = false;
    for (auto& op : operations_) {
      if (op->op == Event::OperationType::COMPUTE && !op->is_locked_for_jit) {
        lock_for_jit(op);
        any_locked = true;
      }
    }
    if (any_locked) flush_jit_batch();
  }
  if (e->op == Event::OperationType::COMPUTE && e->rpn_ops.empty() &&
      JIT_BATCH_SIZE > 0)
    e->kid = e->pipeline_kid;
#endif

  for (const auto& in : e->inputs)
    if (in && in->last_producer_id != 0)
      e->dependencies.insert(in->last_producer_id);
  if (e->output) e->output->last_producer_id = e->id;
  for (const auto& out : e->extra_outputs)
    if (out) out->last_producer_id = e->id;

  operations_.push_back(e);

#if PIPELINE
  // Record how to recompute this output so a later consumer can inline it
  // instead of reading MRAM (see EventQueue::expand_absorbed_inputs).
  //
  // Never for an in-place op.  There the output *is* an input, so the recorded
  // expression is self-referential and only valid until this event runs -- a
  // consumer that inlines it afterwards re-reads a buffer this event already
  // overwrote, applying it twice (`a += b; a -= b` yielded a + b).
  const bool writes_in_place =
      e->output && std::find(e->inputs.begin(), e->inputs.end(), e->output) !=
                       e->inputs.end();
  if (e->op == Event::OperationType::COMPUTE && !IS_OP_REDUCTION(e->opcode) &&
      e->output && !e->inputs.empty() && e->extra_outputs.empty() &&
      !writes_in_place) {
    std::vector<uint8_t> rpn;
    std::vector<uint32_t> scalars;
    detail::build_default_rpn(e, rpn, scalars);
    e->output->absorbed_rpn = rpn;
    e->output->absorbed_scalars = scalars;
    e->output->absorbed_inputs = e->inputs;
    e->output->is_shared_intermediate = true;
  }
#endif
}

void EventQueue::submit(std::shared_ptr<Event> e) {
  std::lock_guard<std::recursive_mutex> lock(mtx_);

#if PIPELINE
  retire_inlined_producers(e);
  compact_fusable_operations();
#endif

  await_queue_space();

  e->id = counter_++;
  e->max_id = e->id;
  VECTORDPU_NOTE(events_submitted);

#if PIPELINE
  expand_absorbed_inputs(e);
#endif

  trace::event_enqueued(e, operations_, running_events_);
  trace::active_ops_counter(operations_.size());

  if (!fuse_into_queue_tail(e)) enqueue(e);
}

void EventQueue::retire_inlined_producers(
    const std::shared_ptr<Event>& pending_consumer) {
#if PIPELINE
  bool changed = false;
  for (auto it = operations_.begin(); it != operations_.end();) {
    const auto producer = *it;
    if (!producer->extra_outputs.empty() ||
        output_still_needed(producer, pending_consumer)) {
      ++it;
      continue;
    }

    auto consumer = std::find_if(
        operations_.begin(), operations_.end(), [&](const auto& candidate) {
          return candidate != producer &&
                 candidate->id == producer->inlined_into;
        });
    if (producer->inlined_into == 0 || consumer == operations_.end()) {
      ++it;
      continue;
    }

    if (producer->output && producer->output->last_producer_id == producer->id)
      producer->output->last_producer_id = (*consumer)->id;
    retarget_inlined_producers(producer->id, (*consumer)->id);
    trace::event_fused(producer, *consumer, "deferred inline");
    it = operations_.erase(it);
    VECTORDPU_NOTE(absorbed_producers);
    changed = true;
  }
  if (changed) trace::active_ops_counter(operations_.size());
#else
  (void)pending_consumer;
#endif
}

void EventQueue::retarget_inlined_producers(size_t old_consumer_id,
                                            size_t new_consumer_id) {
#if PIPELINE
  for (const auto& producer : operations_)
    if (producer->inlined_into == old_consumer_id)
      producer->inlined_into = new_consumer_id;
#else
  (void)old_consumer_id;
  (void)new_consumer_id;
#endif
}

void EventQueue::compact_fusable_operations() {
#if PIPELINE
  // Retirement can expose fusable neighbours anywhere in the bounded queue,
  // not just at its tail.  Reconsider the preceding pair after every merge
  // because the larger left-hand program may unlock a vertical fusion.
  size_t i = 1;
  while (i < operations_.size()) {
    if (try_fuse(operations_[i - 1], operations_[i])) {
      operations_.erase(operations_.begin() + i);
      if (i > 1) --i;
    } else {
      ++i;
    }
  }
#endif
}
