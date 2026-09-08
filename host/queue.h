#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <typeindex>
#include <variant>

#include "common.h"
#include "config.h"
#include "detail/vector_desc.h"
#include "jit.h"

class Event : public std::enable_shared_from_this<Event> {
 public:
  enum class OperationType { COMPUTE, DPU_TRANSFER, HOST_TRANSFER, FENCE };

  OperationType op;
  std::function<void()> cb;

  // Metadata for fusion
  std::vector<detail::VectorDescRef> inputs;
  detail::VectorDescRef output;
  std::vector<detail::VectorDescRef> extra_outputs;
  std::vector<uint8_t> rpn_ops;
  std::vector<uint32_t> scalars;
  std::vector<uint32_t> extra_scalars;
  std::set<size_t> absorbed_producer_ids;
  KernelID kid = 0;
  KernelID pipeline_kid = 0;  // For lazy promotion
  uint8_t opcode = 0;         // For lazy promotion
  bool is_scalar = false;     // Prevent fusion of scalar ops for now
  uint32_t scalar_value = 0;  // Scalar operand for fusion
  void* host_ptr = nullptr;   // For transfers
  size_t transfer_size = 0;   // For transfers

  // JIT
  std::string jit_binary_path;
  std::string jit_kernel_hash;
  bool is_locked_for_jit = false;
#if JIT_PIPELINE_FALLBACK
  bool jit_pipeline_fallback = false;
  bool jit_eager_fallback = false;
#endif
  // This event's output was inlined into a later consumer, so it may not need
  // to run.  Whether it does is decided later, once the vector's last handle
  // has gone: see EventQueue::output_still_needed.  `inlined_into` names the
  // consumer that absorbed it and proves the computation is still represented
  // if this producer is retired.
  bool output_was_inlined = false;
  size_t inlined_into = 0;
  int jit_sub_kernel_idx = -1;
  std::shared_future<std::string> jit_future;
  std::shared_future<void> dpu_future;

  Event(OperationType t) : op(t) {}

  template <typename Callable>
  Event(OperationType t, Callable&& c) : op(t), cb(std::forward<Callable>(c)) {}

  void wait() {
    if (jit_future.valid()) jit_future.wait();
    if (dpu_future.valid()) dpu_future.wait();
  }

  size_t id = 0;
  std::string slice_name;
  std::set<size_t> dependencies;
  std::atomic<bool> finished{false};
  bool started = false;
  size_t max_id = 0;  // The highest ID represented by this event (if fused)
  int oom_retries = 0;

  void add_completion_callback(std::shared_ptr<Event> self);
  void mark_finished() { this->finished.store(true); }
  bool operator==(const Event& other) const { return this->id == other.id; }
};

class EventQueue {
 public:
  static constexpr size_t DEFAULT_MAX_QUEUE_DEPTH = 64;

  EventQueue() = default;
  ~EventQueue() = default;

  bool try_fuse(std::shared_ptr<Event> last, std::shared_ptr<Event> e);
  void lock_for_jit(std::shared_ptr<Event> e);
  void flush_jit_batch();
#if JIT_PIPELINE_FALLBACK
  void await_jit_compilations();
#endif

  void submit(std::shared_ptr<Event> e);
  void set_max_queue_depth(size_t depth) { max_queue_depth_ = depth; }
  size_t max_queue_depth() const { return max_queue_depth_; }

  void add_fence(std::shared_ptr<Event> e);

  void sync();
  bool process_next();
  void process_events(size_t wait_for_id);
  void finalize_finished_events();
  void debug_print_queue();
  void debug_active_events();
  size_t count_internal_references(detail::VectorDescRef vec);

  bool has_pending() const { return !operations_.empty(); }
  std::size_t pending_count() const { return operations_.size(); }

  std::shared_ptr<Event> get_curr_event() const { return current_event_; }
  size_t get_curr_event_id() const {
    if (current_event_ != nullptr) {
      return current_event_->id;
    } else {
      return SIZE_MAX;
    }
  }

  std::deque<std::shared_ptr<Event>>::iterator begin() {
    return operations_.begin();
  }
  std::deque<std::shared_ptr<Event>>::iterator end() {
    return operations_.end();
  }
  size_t size() const { return operations_.size(); }

  size_t get_last_finished_id() const { return last_finished_id_.load(); }

  std::list<std::shared_ptr<Event>>& get_active_events() {
    return running_events_;
  }

  std::recursive_mutex& get_mutex() { return mtx_; }

  std::atomic<size_t> last_finished_id_{0};
  std::atomic<int> outstanding_callbacks_{0};

  bool try_vfuse(std::shared_ptr<Event> last, std::shared_ptr<Event> e);
  bool try_hfuse(std::shared_ptr<Event> last, std::shared_ptr<Event> e);
  void expand_absorbed_inputs(std::shared_ptr<Event> e);

 private:
  // process_next() stages, in the order it runs them.
  std::shared_ptr<Event> take_next_operation();
  void await_dependencies(const std::shared_ptr<Event>& e);
  void grow_fusion_batch(const std::shared_ptr<Event>& e);
  void await_jit_binary(const std::shared_ptr<Event>& e);
  void begin_running(const std::shared_ptr<Event>& e);
  void switch_dpu_binary(const std::shared_ptr<Event>& e);
  void dispatch(const std::shared_ptr<Event>& e);
  void requeue_after_oom(const std::shared_ptr<Event>& e);
  bool output_still_needed(
      const std::shared_ptr<Event>& e,
      const std::shared_ptr<Event>& pending_consumer = nullptr);

  // submit() stages.
  void retarget_inlined_producers(size_t old_consumer_id,
                                  size_t new_consumer_id);
  void retire_inlined_producers(const std::shared_ptr<Event>& pending_consumer);
  void compact_fusable_operations();
  void await_queue_space();
  bool fuse_into_queue_tail(const std::shared_ptr<Event>& e);
  void enqueue(const std::shared_ptr<Event>& e);

  std::recursive_mutex mtx_;
  size_t counter_ = 1;
  size_t max_queue_depth_ = DEFAULT_MAX_QUEUE_DEPTH;
  std::shared_ptr<Event> current_event_ = nullptr;
  std::deque<std::shared_ptr<Event>> operations_;
  std::list<std::shared_ptr<Event>> running_events_;

  // JIT Batching State
  std::vector<std::pair<std::vector<uint8_t>, std::string>>
      pending_unique_kernels_;
  std::vector<std::shared_ptr<Event>> pending_jit_events_;
  struct InflightJitKernel {
    std::shared_future<std::string> binary;
    int slot;
  };
  std::map<Signature, InflightJitKernel> inflight_jit_kernels_;
  // Last binary's kernels; the next link re-includes them so it is a superset.
  std::vector<std::pair<std::vector<uint8_t>, std::string>>
      last_linked_kernels_;
#if JIT_PIPELINE_FALLBACK
  // Interpreted launches in a row whose compile never landed; past the
  // threshold compiling is pure overhead.
  static constexpr uint32_t kStopCompilingAfter = 8;
  uint32_t unused_compiles_ = 0;
#endif

  std::string current_binary_path_;
};
