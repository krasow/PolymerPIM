#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "perfetto/trace.h"

namespace detail {
// Handle bookkeeping for VectorDesc::handle_count.  Fusion consults it to tell
// "nobody can build another op on this vector" from "the queue still has a
// reference".
inline void retain_handle(const VectorDescRef& desc) {
  if (desc) desc->handle_count++;
}
inline void release_handle(const VectorDescRef& desc) {
  if (desc && desc->handle_count > 0) desc->handle_count--;
}
}  // namespace detail

template <typename T>
dpu_vector<T>::dpu_vector() noexcept : size_(0), reserved_(0) {}

template <typename T>
dpu_vector<T>::dpu_vector(size_t n, uint32_t reserved, bool lazy,
                          std::string_view name, std::source_location loc)
    : size_(n),
      reserved_(reserved),
      debug_name(name.data()),
      debug_file(loc.file_name()),
      debug_line(loc.line()) {
  auto& runtime = DpuRuntime::get();
  if (runtime.is_initialized() == false) {
    runtime.init(DpuRuntime::configured_num_dpus());
  }
  bind_storage(runtime.backend().allocate(n, reserved, sizeof(T), lazy));
}

template <typename T>
dpu_vector<T>::dpu_vector(per_dpu_tag, size_t n_per_dpu, uint32_t reserved,
                          bool lazy, std::string_view name,
                          std::source_location loc)
    : reserved_(reserved),
      debug_name(name.data()),
      debug_file(loc.file_name()),
      debug_line(loc.line()) {
  auto& runtime = DpuRuntime::get();
  if (runtime.is_initialized() == false) {
    runtime.init(DpuRuntime::configured_num_dpus());
  }
  size_ = n_per_dpu * runtime.num_dpus();
  bind_storage(
      runtime.backend().allocate_per_dpu(n_per_dpu, reserved, sizeof(T), lazy));
}

template <typename T>
dpu_vector<T> dpu_vector<T>::per_dpu(size_t n_per_dpu, uint32_t reserved,
                                     bool lazy, std::string_view name,
                                     std::source_location loc) {
  return dpu_vector<T>(per_dpu_tag{}, n_per_dpu, reserved, lazy, name, loc);
}

template <typename T>
void dpu_vector<T>::bind_storage(detail::VectorDescRef data) {
  data_ = std::move(data);
  data_->type_name = typeid(T).name();
  data_->debug_name = debug_name;
  data_->debug_file = debug_file;
  data_->debug_line = debug_line;
  detail::retain_handle(data_);
#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  log_allocation(logger, typeid(T), size_, data_->vector_id,
                 data_->allocated_footprint_bytes,
                 !data_->needs_layout_materialization, debug_name, debug_file,
                 debug_line);
#endif
}

template <typename T>
dpu_vector<T>::dpu_vector(const dpu_vector& other)
    : data_(other.data_),
      size_(other.size_),
      reserved_(other.reserved_),
      debug_name(other.debug_name),
      debug_file(other.debug_file),
      debug_line(other.debug_line),
      copied(false) {
  other.copied = true;
  detail::retain_handle(data_);
}

template <typename T>
dpu_vector<T>::dpu_vector(dpu_vector&& other) noexcept
    : data_(std::move(other.data_)),
      size_(other.size_),
      reserved_(other.reserved_),
      debug_name(other.debug_name),
      debug_file(other.debug_file),
      debug_line(other.debug_line),
      copied(false) {
  // The moved-from handle gives up its reference, so the count is unchanged.
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator=(const dpu_vector& other) {
  if (this != &other) {
    detail::release_handle(data_);
    detail::retain_handle(other.data_);
    data_ = other.data_;
    size_ = other.size_;
    reserved_ = other.reserved_;
    debug_name = other.debug_name;
    debug_file = other.debug_file;
    debug_line = other.debug_line;
  }
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator=(dpu_vector&& other) noexcept {
  if (this != &other) {
    detail::release_handle(data_);
    data_ = std::move(other.data_);
    size_ = other.size_;
    reserved_ = other.reserved_;
    debug_name = other.debug_name;
    debug_file = other.debug_file;
    debug_line = other.debug_line;
  }
  return *this;
}

template <typename T>
dpu_vector<T>::~dpu_vector() {
  detail::release_handle(data_);
}

template <typename T>
void dpu_vector<T>::add_fence() {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::FENCE);

  event_queue.submit(e);
  event_queue.process_events(e->id);
}

inline void dpu_fence() {
  // Nothing has been submitted before the first vector is constructed, and the
  // event queue's mutex does not exist yet -- locking it segfaults.  Fencing an
  // uninitialised runtime is vacuously satisfied.
  auto& runtime = DpuRuntime::get();
  if (!runtime.is_initialized()) return;
  runtime.get_event_queue().sync();
}

template <typename T>
dpu_vector<T> dpu_vector<T>::from_cpu(T* cpu_data, size_t n,
                                      std::string_view name,
                                      std::source_location loc) {
  dpu_vector<T> vec(n, 0, false, name, loc);
  auto desc = vec.data_desc_ref();

  char* cpu_buffer = reinterpret_cast<char*>(cpu_data);
  auto bound_cb = std::bind(detail::vec_xfer_to_dpu, cpu_buffer, desc);

  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::DPU_TRANSFER, bound_cb);
  e->output = desc;
  e->host_ptr = cpu_buffer;
  e->transfer_size = n * sizeof(T);

  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock(logcat::TRANSFER, 2)
      << "type=DPU_TRANSFER action=submit id=" << e->id << " size=" << n
      << " bytes=" << e->transfer_size << std::endl;
  logger.lock(logcat::QUEUE_APPEND, 2)
      << "type=DPU_TRANSFER size=" << n << std::endl;
#endif
  return vec;
}

template <typename T>
dpu_vector<T> dpu_vector<T>::from_cpu(std::vector<T>& cpu_vec,
                                      std::string_view name,
                                      std::source_location loc) {
  return from_cpu(cpu_vec.data(), cpu_vec.size(), name, loc);
}

template <typename T>
vector<T> dpu_vector<T>::to_cpu() {
  auto desc = this->data_desc_ref();
  DpuRuntime::get().backend().realize(desc);
  const size_t result_elems =
      DpuRuntime::get().backend().read_layout(*desc).total_logical / sizeof(T);

  vector<T> cpu_vec(result_elems);
  to_cpu_into(cpu_vec.data(), result_elems);
  return cpu_vec;
}

template <typename T>
size_t dpu_vector<T>::to_cpu_into(T* out, size_t capacity) {
  auto desc = this->data_desc_ref();
  auto& runtime = DpuRuntime::get();
  runtime.backend().realize(desc);

  // One transfer size applies to every DPU, so ragged shards have to land in a
  // padded staging buffer and be compacted afterwards.  When every shard's
  // payload already fills the stride we read straight into the result.
  const detail::ShardLayout layout =
      DpuRuntime::get().backend().read_layout(*desc);
  const size_t result_elems = layout.total_logical / sizeof(T);
  const size_t n = std::min(result_elems, capacity);

  // The transfer lands all result_elems, so it can only target `out` when
  // `out` has room; a short buffer gets its own and keeps the first n.
  const bool fits = capacity >= result_elems;
  vector<T> spill;
  if (!fits) spill.resize(result_elems);
  T* dest = fits ? out : spill.data();

  vector<char> staging;
  char* cpu_buffer;
  if (layout.needs_padding) {
    staging.resize(layout.padded_bytes());
    cpu_buffer = staging.data();
  } else {
    cpu_buffer = reinterpret_cast<char*>(dest);
  }

  const size_t stride = layout.stride;
  auto bound_cb = [cpu_buffer, desc, stride]() {
    detail::vec_xfer_from_dpu_strided(cpu_buffer, desc, stride);
  };
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::HOST_TRANSFER, bound_cb);
  e->inputs = {desc};
  e->host_ptr = cpu_buffer;
  e->transfer_size =
      layout.needs_padding ? layout.padded_bytes() : result_elems * sizeof(T);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
#endif
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  logger.lock(logcat::TRANSFER, 2)
      << "type=HOST_TRANSFER action=submit id=" << e->id
      << " size=" << result_elems << " bytes=" << e->transfer_size << std::endl;
  logger.lock(logcat::QUEUE_APPEND, 2)
      << "type=HOST_TRANSFER size=" << result_elems << std::endl;
#endif

  // A staged read has to complete before it can be compacted, so it fences
  // regardless of ENABLE_AUTO_FENCING.
  const bool must_wait = layout.needs_padding || ENABLE_AUTO_FENCING == 1;
  if (must_wait) {
#if ENABLE_DPU_LOGGING >= 2
    logger.lock(logcat::TRANSFER, 2)
        << "type=HOST_TRANSFER action=wait id=" << e->id << std::endl;
#endif
    event_queue.process_events(e->id);
#if ENABLE_DPU_LOGGING >= 2
    logger.lock(logcat::TRANSFER, 2)
        << "type=HOST_TRANSFER action=done id=" << e->id << std::endl;
#endif
#if ENABLE_DPU_PRINTING == 1
    // Needs the event finished before the DPU log can be read.
    runtime.debug_read_dpu_log();
#endif
  }

  if (layout.needs_padding) {
    char* compacted = reinterpret_cast<char*>(dest);
    for (size_t dpu = 0; dpu < layout.logical.size(); ++dpu) {
      std::memcpy(compacted, staging.data() + dpu * stride,
                  layout.logical[dpu]);
      compacted += layout.logical[dpu];
    }
  }

  if (!fits) std::memcpy(out, spill.data(), n * sizeof(T));
  return n;
}

template <typename T>
typename dpu_vector<T>::reduction_result_t reduction_cpu(dpu_vector<T>& da,
                                                         KernelID kernel_id) {
  // block and send to cpu
  auto a = da.to_cpu();

  uint64_t flow_id =
      (da.data_desc_ref() ? da.data_desc_ref()->last_producer_id : 0);
  trace::reduction_cpu _trace(flow_id);

  auto& runtime = DpuRuntime::get();
  assert(a.size() % runtime.num_dpus() == 0);
  size_t stride = a.size() / runtime.num_dpus();
  // initialize accumulator with the first partial result
  typename dpu_vector<T>::reduction_result_t acc = a[0];

  // reduce over the remaining DPUs
  auto op = kernel_infos[kernel_id].op;
  for (size_t i = stride; i < a.size(); i += stride) {
    typename dpu_vector<T>::reduction_result_t x = a[i];
    switch (op) {
      case KERNEL_OP_SUM:
        acc += x;
        break;
      case KERNEL_OP_PRODUCT:
        acc *= x;
        break;
      case KERNEL_OP_MAX:
        acc = (x > acc) ? x : acc;
        break;
      case KERNEL_OP_MIN:
        acc = (x < acc) ? x : acc;
        break;
      default:
        assert(false && "Unknown reduction operation");
    }
  }
  return acc;
}

// Binary operators
template <typename T>
dpu_vector<T> operator+(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                        rhs.data_desc_ref(), OpInfo<T>::add, OpInfo<T>::add_op,
                        OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                        rhs.data_desc_ref(), OpInfo<T>::sub, OpInfo<T>::sub_op,
                        OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator*(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                        rhs.data_desc_ref(), OpInfo<T>::mul, OpInfo<T>::mul_op,
                        OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator/(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                        rhs.data_desc_ref(), OpInfo<T>::div, OpInfo<T>::div_op,
                        OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator+=(const dpu_vector<T>& other) {
  detail::launch_binary(this->data_desc_ref(), this->data_desc_ref(),
                        other.data_desc_ref(), OpInfo<T>::add,
                        OpInfo<T>::add_op, OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator-=(const dpu_vector<T>& other) {
  detail::launch_binary(this->data_desc_ref(), this->data_desc_ref(),
                        other.data_desc_ref(), OpInfo<T>::sub,
                        OpInfo<T>::sub_op, OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator*=(const dpu_vector<T>& other) {
  detail::launch_binary(this->data_desc_ref(), this->data_desc_ref(),
                        other.data_desc_ref(), OpInfo<T>::mul,
                        OpInfo<T>::mul_op, OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator/=(const dpu_vector<T>& other) {
  detail::launch_binary(this->data_desc_ref(), this->data_desc_ref(),
                        other.data_desc_ref(), OpInfo<T>::div,
                        OpInfo<T>::div_op, OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator+=(T scalar) {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::add_scalar,
                               OpInfo<T>::add_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator-=(T scalar) {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::sub_scalar,
                               OpInfo<T>::sub_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator*=(T scalar) {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::mul_scalar,
                               OpInfo<T>::mul_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator/=(T scalar) {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::div_scalar,
                               OpInfo<T>::div_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T>& dpu_vector<T>::operator>>=(T scalar) {
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(this->data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::asr_scalar,
                               OpInfo<T>::asr_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return *this;
}

template <typename T>
dpu_vector<T> dpu_vector<T>::operator-() const {
  dpu_vector<T> res(this->size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_unary(res.data_desc_ref(), this->data_desc_ref(),
                       OpInfo<T>::negate, OpInfo<T>::negate_op,
                       OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator>>(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(),
                               static_cast<uint32_t>(rhs),
                               OpInfo<T>::asr_scalar, OpInfo<T>::asr_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> dpu_vector<T>::operator==(T scalar) const {
  dpu_vector<T> res(this->size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  uint32_t scalar_bits = 0;
  std::memcpy(&scalar_bits, &scalar, sizeof(T) < 4 ? sizeof(T) : 4);
  detail::launch_binary_scalar(res.data_desc_ref(), this->data_desc_ref(),
                               scalar_bits, OpInfo<T>::eq_scalar,
                               OpInfo<T>::eq_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator+(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(),
                               static_cast<uint32_t>(rhs),
                               OpInfo<T>::add_scalar, OpInfo<T>::add_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator+(T lhs, const dpu_vector<T>& rhs) {
  return rhs + lhs;
}

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(),
                               static_cast<uint32_t>(rhs),
                               OpInfo<T>::sub_scalar, OpInfo<T>::sub_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator*(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(),
                               static_cast<uint32_t>(rhs),
                               OpInfo<T>::mul_scalar, OpInfo<T>::mul_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> operator*(T lhs, const dpu_vector<T>& rhs) {
  return rhs * lhs;
}

template <typename T>
dpu_vector<T> operator/(const dpu_vector<T>& lhs, T rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(),
                               static_cast<uint32_t>(rhs),
                               OpInfo<T>::div_scalar, OpInfo<T>::div_scalar_op,
                               OpInfo<T>::universal_pipeline);
  return res;
}

#if PIPELINE
template <typename T>
pipeline_result<T> dpu_vector<T>::pipeline(const std::vector<uint8_t>& ops) {
  return pipeline(ops, {});
}
#endif

#if PIPELINE
template <typename T>
std::vector<uint8_t> dpu_vector<T>::prepare_rpn(
    const std::vector<uint8_t>& ops) {
  std::vector<uint8_t> rpn_ops;
  bool is_rpn = !ops.empty() && (ops[0] >= OP_PUSH_INPUT);
  if (is_rpn) {
    rpn_ops = ops;
  } else {
    // Check if ops are already RPN (contain PUSH instructions)
    bool is_raw_rpn = false;
    for (uint8_t op : ops) {
      if (op == OP_PUSH_INPUT ||
          (op >= OP_PUSH_OPERAND_0 && op <= OP_PUSH_OPERAND_7)) {
        is_raw_rpn = true;
        break;
      }
    }

    if (is_raw_rpn) {
      rpn_ops = ops;
    } else {
      // Translate Linear -> RPN
      if (!ops.empty()) {
        rpn_ops.push_back(OP_PUSH_INPUT);
        size_t next_operand = 0;
        for (uint8_t op : ops) {
          bool is_binary = (op >= OP_ADD && op <= OP_DIV);
          if (is_binary) {
            if (next_operand < MAX_VFUSE_INPUTS) {
              rpn_ops.push_back(OP_PUSH_OPERAND_0 + next_operand);
              next_operand++;
            }
          }
          rpn_ops.push_back(op);
        }
      }
    }
  }
  return rpn_ops;
}
#endif

#if PIPELINE
template <typename T>
pipeline_result<T> dpu_vector<T>::pipeline(
    const std::vector<uint8_t>& ops, const std::vector<dpu_vector<T>>& operands,
    const std::vector<uint32_t>& scalars) {
  if (operands.size() > MAX_VFUSE_INPUTS) {
    throw std::logic_error("pipeline operand count exceeds MAX_VFUSE_INPUTS");
  }
  dpu_vector<T> res(this->size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "pipeline_intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  std::vector<uint8_t> rpn_ops = prepare_rpn(ops);
  std::vector<detail::VectorDescRef> operand_refs;
  std::vector<detail::VectorDescRef> absorbed_inputs;
  absorbed_inputs.push_back(this->data_desc_ref());
  for (const auto& op : operands) {
    operand_refs.push_back(op.data_desc_ref());
    absorbed_inputs.push_back(op.data_desc_ref());
  }

  // Mark the JIT-produced vector as an absorbable intermediate so a later
  // consumer can inline it instead of forcing an MRAM round-trip.
  res.data_desc_ref()->absorbed_rpn = rpn_ops;
  res.data_desc_ref()->absorbed_scalars = scalars;
  res.data_desc_ref()->absorbed_inputs = absorbed_inputs;
  res.data_desc_ref()->is_shared_intermediate = true;

  detail::launch_universal_pipeline(res.data_desc_ref(), this->data_desc_ref(),
                                    rpn_ops, operand_refs,
                                    OpInfo<T>::universal_pipeline, scalars);
  return pipeline_result<T>(std::move(res));
}
#endif

#if JIT
#include "jit.h"

template <typename T>
pipeline_result<T> dpu_vector<T>::jit(const std::vector<uint8_t>& ops) {
  return jit(ops, {});
}

template <typename T>
pipeline_result<T> dpu_vector<T>::jit(
    const std::vector<uint8_t>& ops, const std::vector<dpu_vector<T>>& operands,
    const std::vector<uint32_t>& scalars) {
  if (operands.size() > MAX_VFUSE_INPUTS) {
    throw std::logic_error("JIT operand count exceeds MAX_VFUSE_INPUTS");
  }
  dpu_vector<T> res(this->size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "jit_result";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;

  std::vector<uint8_t> rpn_ops = prepare_rpn(ops);
  std::vector<detail::VectorDescRef> operand_refs;
  std::vector<detail::VectorDescRef> absorbed_inputs;
  absorbed_inputs.push_back(this->data_desc_ref());
  for (const auto& op : operands) {
    operand_refs.push_back(op.data_desc_ref());
    absorbed_inputs.push_back(op.data_desc_ref());
  }

  // Allow later consumers to inline this JIT result rather than materializing
  // it in MRAM if the queue can absorb it.
  res.data_desc_ref()->absorbed_rpn = rpn_ops;
  res.data_desc_ref()->absorbed_scalars = scalars;
  res.data_desc_ref()->absorbed_inputs = absorbed_inputs;
  res.data_desc_ref()->is_shared_intermediate = true;

  // Compiler invocation
  const char* tname;
  if (std::is_same<T, int>::value) {
    tname = "int32_t";
  } else if (std::is_same<T, uint32_t>::value) {
    tname = "uint32_t";
  } else {
    tname = typeid(T).name();
  }
  std::vector<std::pair<std::vector<uint8_t>, std::string>> kernels = {
      {rpn_ops, tname}};
  std::string binary_path = jit_compile(kernels);

  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE);

  e->jit_binary_path = binary_path;
  e->slice_name = "JIT Kernel";

  // Reuse the pipeline arguments structure
  e->output = res.data_desc_ref();
  e->inputs.push_back(this->data_desc_ref());
  e->inputs.insert(e->inputs.end(), operand_refs.begin(), operand_refs.end());
  e->rpn_ops = rpn_ops;
  e->scalars = scalars;
  e->kid = 0;  // JIT kernel doesn't use standard IDs
  e->pipeline_kid = 0;

  event_queue.submit(e);

  return pipeline_result<T>(std::move(res));
}

#endif

template <typename T>
typename dpu_vector<T>::reduction_result_t lazy_reduction_result<T>::get() {
  if (!vec.data_desc_ref()) return {};
  return reduction_cpu(vec, rid);
}

template <typename T>
arg_reduction_result<T> lazy_arg_reduction_result<T>::get() {
  if (ready) return cached;
  auto partials = vec.to_cpu();
  auto& runtime = DpuRuntime::get();
  if (partials.empty() || runtime.num_dpus() == 0) return {T{}, 0};
  size_t stride = partials.size() / runtime.num_dpus();
  assert(stride >= 2);
  arg_reduction_result<T> best = {partials[0],
                                  static_cast<uint32_t>(partials[1])};
  for (size_t i = stride; i < partials.size(); i += stride) {
    T value = partials[i];
    uint32_t index = static_cast<uint32_t>(partials[i + 1]);
    if ((want_max ? value > best.value : value < best.value) ||
        (value == best.value && index < best.index)) {
      best = {value, index};
    }
  }
  cached = best;
  ready = true;
  return cached;
}

#if PIPELINE
template <typename T>
lazy_reduction_result<T> dpu_vector<T>::pipeline_reduce(
    const std::vector<uint8_t>& ops, const std::vector<dpu_vector<T>>& operands,
    const std::vector<uint32_t>& scalars) {
  if (operands.size() > MAX_VFUSE_INPUTS) {
    throw std::logic_error(
        "pipeline reduction operand count exceeds MAX_VFUSE_INPUTS");
  }
  auto& runtime = DpuRuntime::get();

  // Identify rid from last op
  assert(!ops.empty());
  uint8_t last_op = ops.back();
  KernelID rid = OpInfo<T>::sum;  // default
  switch (last_op) {
    case OP_MIN:
      rid = OpInfo<T>::min;
      break;
    case OP_MAX:
      rid = OpInfo<T>::max;
      break;
    case OP_SUM:
      rid = OpInfo<T>::sum;
      break;
    case OP_PRODUCT:
      rid = OpInfo<T>::product;
      break;
  }

  // We always allocate 8 bytes per DPU for reduction results to satisfy
  // mram_write minimum alignment and size requirements.
  size_t elems_per_dpu = (8 + sizeof(T) - 1) / sizeof(T);

  auto res =
      dpu_vector<T>::per_dpu(elems_per_dpu, runtime.num_tasklets() * 8, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->is_reduction_result = true;
  res.data_desc_ref()->reduction_rid = rid;

  std::vector<detail::VectorDescRef> operand_descs;
  for (const auto& op : operands) operand_descs.push_back(op.data_desc_ref());

  detail::launch_universal_pipeline(res.data_desc_ref(), this->data_desc_ref(),
                                    ops, operand_descs,
                                    OpInfo<T>::universal_pipeline, scalars);

  return lazy_reduction_result<T>(std::move(res), rid);
}

template <typename T>
lazy_arg_reduction_result<T> dpu_vector<T>::pipeline_argreduce(
    const std::vector<uint8_t>& ops, const std::vector<dpu_vector<T>>& operands,
    const std::vector<uint32_t>& scalars) {
  if (operands.size() > MAX_VFUSE_INPUTS) {
    throw std::logic_error(
        "pipeline arg-reduction operand count exceeds MAX_VFUSE_INPUTS");
  }
  assert(!ops.empty() &&
         (ops.back() == OP_ARGMIN_REDUCE || ops.back() == OP_ARGMAX_REDUCE));
  auto& runtime = DpuRuntime::get();

  // One 8-byte {value,index} result per DPU.
  auto res = dpu_vector<T>::per_dpu(2, runtime.num_tasklets() * 8, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->is_reduction_result = true;
  res.data_desc_ref()->reduction_rid =
      ops.back() == OP_ARGMIN_REDUCE ? OpInfo<T>::min : OpInfo<T>::max;

  std::vector<detail::VectorDescRef> operand_descs;
  for (const auto& operand : operands)
    operand_descs.push_back(operand.data_desc_ref());
  detail::launch_universal_pipeline(res.data_desc_ref(), this->data_desc_ref(),
                                    ops, operand_descs,
                                    OpInfo<T>::universal_pipeline, scalars);
  return lazy_arg_reduction_result<T>(std::move(res),
                                      ops.back() == OP_ARGMAX_REDUCE);
}

template <typename T>
lazy_reduction_result<T> dpu_vector<T>::pipeline_reduce(
    const dpu_pipeline_expr<T>& expr,
    const std::vector<dpu_vector<T>>& operands,
    const std::vector<uint32_t>& scalars) {
  return pipeline_reduce(expr.ops(), operands, scalars);
}

#endif

// Unary operators
template <typename T>
dpu_vector<T> abs(const dpu_vector<T>& a) {
  dpu_vector<T> res(a.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "intermediate";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_unary(res.data_desc_ref(), a.data_desc_ref(), OpInfo<T>::abs,
                       OpInfo<T>::abs_op, OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
lazy_reduction_result<T> sum(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  auto buf = dpu_vector<T>::per_dpu(1, runtime.num_tasklets() * sizeof(size_t));
  buf.data_desc_ref()->type_name = typeid(T).name();
  buf.data_desc_ref()->debug_name = "reduction_buffer";
  buf.data_desc_ref()->debug_file = __FILE__;
  buf.data_desc_ref()->debug_line = __LINE__;
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::sum, OpInfo<T>::sum_op,
                           OpInfo<T>::universal_pipeline);
  return lazy_reduction_result<T>(std::move(buf), OpInfo<T>::sum);
}

template <typename T>
lazy_reduction_result<T> product(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  auto buf = dpu_vector<T>::per_dpu(1, runtime.num_tasklets() * sizeof(size_t));
  buf.data_desc_ref()->type_name = typeid(T).name();
  buf.data_desc_ref()->debug_name = "reduction_buffer";
  buf.data_desc_ref()->debug_file = __FILE__;
  buf.data_desc_ref()->debug_line = __LINE__;
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::product, OpInfo<T>::product_op,
                           OpInfo<T>::universal_pipeline);
  return lazy_reduction_result<T>(std::move(buf), OpInfo<T>::product);
}

template <typename T>
lazy_reduction_result<T> min(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  auto buf =
      dpu_vector<T>::per_dpu(1, runtime.num_tasklets() * sizeof(size_t), true);
  buf.data_desc_ref()->type_name = typeid(T).name();
  buf.data_desc_ref()->debug_name = "reduction_buffer";
  buf.data_desc_ref()->debug_file = __FILE__;
  buf.data_desc_ref()->debug_line = __LINE__;
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::min, OpInfo<T>::min_op,
                           OpInfo<T>::universal_pipeline);
  return lazy_reduction_result<T>(std::move(buf), OpInfo<T>::min);
}

template <typename T>
lazy_reduction_result<T> max(const dpu_vector<T>& a) {
  auto& runtime = DpuRuntime::get();
  auto buf = dpu_vector<T>::per_dpu(1, runtime.num_tasklets() * sizeof(size_t));
  buf.data_desc_ref()->type_name = typeid(T).name();
  buf.data_desc_ref()->debug_name = "reduction_buffer";
  buf.data_desc_ref()->debug_file = __FILE__;
  buf.data_desc_ref()->debug_line = __LINE__;
  detail::launch_reduction(buf.data_desc_ref(), a.data_desc_ref(),
                           OpInfo<T>::max, OpInfo<T>::max_op,
                           OpInfo<T>::universal_pipeline);
  return lazy_reduction_result<T>(std::move(buf), OpInfo<T>::max);
}

template <typename T>
dpu_vector<T> operator<(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  dpu_vector<T> res(lhs.size(), 0, true);
  res.data_desc_ref()->type_name = typeid(T).name();
  res.data_desc_ref()->debug_name = "lt_result";
  res.data_desc_ref()->debug_file = __FILE__;
  res.data_desc_ref()->debug_line = __LINE__;
  detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                        rhs.data_desc_ref(), OpInfo<T>::lt, OpInfo<T>::lt_op,
                        OpInfo<T>::universal_pipeline);
  return res;
}

template <typename T>
dpu_vector<T> select(const dpu_vector<T>& cond, const dpu_vector<T>& then_vec,
                     const dpu_vector<T>& else_vec) {
  const std::vector<uint8_t> ops = {
      (uint8_t)OP_PUSH_INPUT,
      (uint8_t)OP_PUSH_OPERAND_0,
      (uint8_t)OP_PUSH_OPERAND_1,
      (uint8_t)OP_SELECT,
  };
#if JIT
  return const_cast<dpu_vector<T>&>(cond).jit(ops, {then_vec, else_vec}).vec;
#elif PIPELINE
  return const_cast<dpu_vector<T>&>(cond)
      .pipeline(ops, {then_vec, else_vec})
      .vec;
#else
  static_assert(sizeof(T) == 0, "select requires JIT or PIPELINE support");
#endif
}

namespace detail {
inline uint8_t local_reduce_opcode(dpu_local_reduce_op op) {
  switch (op) {
    case dpu_local_reduce_op::sum:
      return OP_SUM;
    case dpu_local_reduce_op::product:
      return OP_PRODUCT;
    case dpu_local_reduce_op::min:
      return OP_MIN;
    case dpu_local_reduce_op::max:
      return OP_MAX;
  }
  return OP_SUM;
}

template <typename T>
T local_reduce_identity(dpu_local_reduce_op op) {
  switch (op) {
    case dpu_local_reduce_op::sum:
      return static_cast<T>(0);
    case dpu_local_reduce_op::product:
      return static_cast<T>(1);
    case dpu_local_reduce_op::min:
      return std::numeric_limits<T>::max();
    case dpu_local_reduce_op::max:
      return std::numeric_limits<T>::lowest();
  }
  return static_cast<T>(0);
}

template <typename T>
void local_reduce_apply(T& dst, const T& value, dpu_local_reduce_op op) {
  switch (op) {
    case dpu_local_reduce_op::sum:
      dst += value;
      return;
    case dpu_local_reduce_op::product:
      dst *= value;
      return;
    case dpu_local_reduce_op::min:
      if (value < dst) dst = value;
      return;
    case dpu_local_reduce_op::max:
      if (value > dst) dst = value;
      return;
  }
}
}  // namespace detail

#if PIPELINE
template <typename T>
uint8_t dpu_pipeline_context<T>::local_id(dpu_local_vector<T>& local) {
  detail::VectorDescRef desc = local.data_desc_ref();
  for (size_t i = 0; i < locals_.size(); ++i) {
    if (locals_[i] == desc) return (uint8_t)i;
  }
  // WRAM only has room for MAX_LOCAL_SCRATCH_VECTORS of these (see
  // TASKLET_WORKSPACE_SIZE); the codegen happily emits slots up to
  // MAX_HFUSE_CHAINS, so without this bound a second local vector writes past
  // its region and silently reads back as zeros.
  if (locals_.size() >= MAX_LOCAL_SCRATCH_VECTORS) {
    throw std::logic_error(
        "dpu_pipeline_context allows at most " +
        std::to_string(MAX_LOCAL_SCRATCH_VECTORS) +
        " local vector(s) per program (MAX_LOCAL_SCRATCH_VECTORS)");
  }
  locals_.push_back(desc);
  return (uint8_t)(locals_.size() - 1);
}

template <typename T>
void dpu_pipeline_context<T>::local_reduce(dpu_local_vector<T>& local,
                                           const dpu_pipeline_expr<T>& index,
                                           const dpu_pipeline_expr<T>& value) {
  uint8_t id = local_id(local);
  local_reductions_.push_back({id,
                               detail::local_reduce_opcode(local.reduce_op()),
                               index.ops(), value.ops()});
}

template <typename T>
std::vector<uint8_t> dpu_pipeline_context<T>::materialize_ops() const {
  std::vector<uint8_t> out = ops_;
  if (local_reductions_.empty()) return out;

  size_t raw_common = local_reductions_[0].index_ops.size();
  for (size_t r = 1; r < local_reductions_.size(); ++r) {
    const auto& rhs = local_reductions_[r].index_ops;
    raw_common = std::min(raw_common, rhs.size());
    size_t i = 0;
    while (i < raw_common && local_reductions_[0].index_ops[i] == rhs[i]) {
      ++i;
    }
    raw_common = i;
  }

  size_t common = 0;
  for (size_t i = 0; i < local_reductions_[0].index_ops.size();) {
    size_t next = i + 1 + OP_INLINE_BYTES(local_reductions_[0].index_ops[i]);
    if (next > raw_common) break;
    common = next;
    i = next;
  }

  if (common > 0) {
    out.insert(out.end(), local_reductions_[0].index_ops.begin(),
               local_reductions_[0].index_ops.begin() + common);
  }

  for (const auto& reduction : local_reductions_) {
    if (common > 0) {
      out.push_back(OP_DUP);
      out.insert(out.end(), reduction.index_ops.begin() + common,
                 reduction.index_ops.end());
    } else {
      out.insert(out.end(), reduction.index_ops.begin(),
                 reduction.index_ops.end());
    }
    out.insert(out.end(), reduction.value_ops.begin(),
               reduction.value_ops.end());
    out.push_back(OP_APPLY_INDIRECT);
    out.push_back(reduction.local_id);
    out.push_back(reduction.reduce_op);
  }

  return out;
}
#endif

template <typename T>
dpu_local_vector<T>::dpu_local_vector(uint32_t n, std::string_view name,
                                      std::source_location loc)
    : dpu_local_vector(n, dpu_local_reduce_op::sum, name, loc) {}

template <typename T>
dpu_local_vector<T>::dpu_local_vector(uint32_t n, dpu_local_reduce_op reduce_op,
                                      std::string_view name,
                                      std::source_location loc)
    : size_(n), reduce_op_(reduce_op) {
  auto& runtime = DpuRuntime::get();
  data_ = runtime.backend().allocate_local(n, sizeof(T));
  data_->is_local_vector = true;
  data_->local_reduce_opcode = detail::local_reduce_opcode(reduce_op_);
  data_->type_name = typeid(T).name();
  data_->debug_name = name.data();
  data_->debug_file = loc.file_name();
  data_->debug_line = loc.line();
#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = runtime.get_logger();
  log_allocation(logger, typeid(T), n, data_->vector_id,
                 data_->allocated_footprint_bytes,
                 !data_->needs_layout_materialization, data_->debug_name,
                 data_->debug_file, data_->debug_line);
#endif
}

template <typename T>
vector<T> dpu_local_vector<T>::to_cpu() {
  auto& runtime = DpuRuntime::get();
  uint32_t nr_dpus = runtime.num_dpus();
#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = runtime.get_logger();
  logger.lock(logcat::TRANSFER, 2)
      << "type=COMPUTE action=wait_dependency last_producer_id="
      << data_->last_producer_id << " size=" << size_ << " dpus=" << nr_dpus
      << std::endl;
#endif
  runtime.get_event_queue().process_events(data_->last_producer_id);
#if ENABLE_DPU_LOGGING >= 2
  logger.lock(logcat::TRANSFER, 2)
      << "type=COMPUTE action=dependency_done last_producer_id="
      << data_->last_producer_id << std::endl;
#endif

  if (data_->last_producer_id == 0) {
    return std::vector<T>(size_, detail::local_reduce_identity<T>(reduce_op_));
  }

  // DPU transfers use the aligned shard stride, not the logical payload size.
  // Stage the padding and merge only logical elements.
  const detail::ShardLayout layout = runtime.backend().read_layout(*data_);
  const size_t stride = layout.stride;
  std::vector<char> all_data(layout.padded_bytes());
  char* cpu_buffer = all_data.data();
  auto bound_cb = [cpu_buffer, data = data_, stride]() {
    detail::vec_xfer_from_dpu_strided(cpu_buffer, data, stride);
  };
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::HOST_TRANSFER, bound_cb);
  e->inputs = {data_};
  e->host_ptr = cpu_buffer;
  e->transfer_size = all_data.size();

  event_queue.submit(e);
#if ENABLE_DPU_LOGGING >= 2
  logger.lock(logcat::TRANSFER, 2)
      << "type=HOST_TRANSFER action=submit id=" << e->id
      << " bytes=" << e->transfer_size << std::endl;
  logger.lock(logcat::TRANSFER, 2)
      << "type=HOST_TRANSFER action=wait id=" << e->id << std::endl;
#endif
  event_queue.process_events(e->id);
#if ENABLE_DPU_LOGGING >= 2
  logger.lock(logcat::TRANSFER, 2)
      << "type=HOST_TRANSFER action=done id=" << e->id << std::endl;
#endif

  std::vector<T> merged(size_, detail::local_reduce_identity<T>(reduce_op_));
  for (uint32_t d = 0; d < nr_dpus; d++) {
    for (uint32_t i = 0; i < size_; i++) {
      T value;
      std::memcpy(&value,
                  all_data.data() + (size_t)d * stride + (size_t)i * sizeof(T),
                  sizeof(T));
      detail::local_reduce_apply(merged[i], value, reduce_op_);
    }
  }
  return merged;
}

#if JIT
template <typename T, typename F>
void dpu_jit_foreach(dpu_vector<T>& primary,
                     const std::vector<dpu_vector<T>>& operands,
                     const std::vector<uint32_t>& scalars, F f) {
  if (operands.size() > MAX_VFUSE_INPUTS) {
    throw std::logic_error(
        "dpu_jit_foreach operand count exceeds MAX_VFUSE_INPUTS");
  }

  std::vector<dpu_expr<T>> vars;
  vars.reserve(operands.size() + 1);
  vars.push_back(dpu_expr<T>::input());
  for (size_t i = 0; i < operands.size(); ++i) {
    vars.push_back(dpu_expr<T>::operand((uint8_t)i));
  }

  dpu_pipeline_context<T> ctx;
  f(vars, ctx);
  std::vector<uint8_t> ops = ctx.materialize_ops();
  if (ops.empty()) return;

  std::vector<detail::VectorDescRef> operand_refs;
  operand_refs.reserve(operands.size());
  for (const auto& operand : operands) {
    operand_refs.push_back(operand.data_desc_ref());
  }

  detail::launch_universal_pipeline(
      detail::VectorDescRef{}, primary.data_desc_ref(), ops, operand_refs,
      OpInfo<T>::universal_pipeline, scalars, {}, ctx.locals());
}
#endif
