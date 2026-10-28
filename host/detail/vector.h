#pragma once

#include <cstring>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "common.h"
#include "config.h"
#include "jit.h"
#include "kernelids.h"
#include "opinfo.h"
#include "runtime.h"
#include "timer.h"
#include "vector_desc.h"

#if __cplusplus < 202002L
// Fake source_location for pre-C++20
namespace std {
struct source_location {
  constexpr source_location(const char* file = "unknown", int line = 0,
                            int column = 0, const char* function = "unknown")
      : file_(file), line_(line), column_(column), function_(function) {}
  static source_location current() { return {}; }
  constexpr const char* file_name() const { return file_; }
  constexpr int line() const { return line_; }
  constexpr int column() const { return column_; }
  constexpr const char* function_name() const { return function_; }

 private:
  const char* file_;
  int line_;
  int column_;
  const char* function_;
};
};  // namespace std
#define VECTORDPU_SOURCE_LOCATION \
  std::source_location(__FILE__, __LINE__, 0, __func__)
#else
#include <source_location>
#define VECTORDPU_SOURCE_LOCATION std::source_location::current()
#endif

using std::vector;

#define LOGGER_ARGS_WITH_DEFAULTS \
  std::string_view name = "",     \
                   std::source_location loc = std::source_location::current()

// Forward declarations
void dpu_fence();

template <typename T>
class dpu_vector;

template <typename T>
struct lazy_reduction_result;

template <typename T>
struct arg_reduction_result {
  T value;
  uint32_t index;
};

template <typename T>
struct lazy_arg_reduction_result;

template <typename T>
class dpu_local_vector;

template <typename T>
class dpu_pipeline_context;

#if PIPELINE
template <typename T>
class dpu_pipeline_expr;

template <typename T>
using dpu_expr = dpu_pipeline_expr<T>;
#endif

#if PIPELINE
template <typename T>
struct pipeline_result;
#endif

// ============================
// DPU Vector
// ============================

template <typename T>
struct reduction_result {
  using type = T;
};

#if ENABLE_PROMOTION_REDUCTIONS
template <>
struct reduction_result<int32_t> {
  using type = int64_t;
};
#endif

// A vector living in DPU memory.
//
// This is a *handle*: copying one shares the underlying MRAM rather than
// duplicating it, so a write through any copy is visible through all of them.
// Duplicating would mean a silent device-side transfer on every assignment;
// take a snapshot with to_cpu() instead.
template <typename T>
class dpu_vector {
 public:
  dpu_vector() noexcept;
  dpu_vector(size_t n, uint32_t reserved = 0, bool lazy = false,
             LOGGER_ARGS_WITH_DEFAULTS);

  ~dpu_vector();

  dpu_vector(const dpu_vector& other);                 // copy constructor
  dpu_vector(dpu_vector&& other) noexcept;             // move constructor
  dpu_vector& operator=(const dpu_vector& other);      // copy assignment
  dpu_vector& operator=(dpu_vector&& other) noexcept;  // move assignment

  vector<T> to_cpu();
  // Reads into caller-owned memory, returning how many elements landed.  A
  // short `capacity`, or shards needing compaction, still stages internally.
  size_t to_cpu_into(T* out, size_t capacity);

  static dpu_vector<T> from_cpu(std::vector<T>& cpu_vec,
                                LOGGER_ARGS_WITH_DEFAULTS);
  // Borrows caller-owned memory instead of copying into a std::vector.  The
  // transfer is queued, so the buffer must outlive it -- add_fence() first.
  static dpu_vector<T> from_cpu(T* cpu_data, size_t n,
                                LOGGER_ARGS_WITH_DEFAULTS);
  // `n_per_dpu` elements per DPU, read back in DPU order (e.g. partials).
  static dpu_vector<T> per_dpu(size_t n_per_dpu, uint32_t reserved = 0,
                               bool lazy = false, LOGGER_ARGS_WITH_DEFAULTS);
  void add_fence();
  dpu_vector<T>& operator+=(const dpu_vector<T>& other);
  dpu_vector<T>& operator-=(const dpu_vector<T>& other);
  dpu_vector<T>& operator*=(const dpu_vector<T>& other);
  dpu_vector<T>& operator/=(const dpu_vector<T>& other);

  dpu_vector<T>& operator+=(T scalar);
  dpu_vector<T>& operator-=(T scalar);
  dpu_vector<T>& operator*=(T scalar);
  dpu_vector<T>& operator/=(T scalar);
  dpu_vector<T>& operator>>=(T scalar);

  dpu_vector<T> operator-() const;
  dpu_vector<T> operator==(T scalar) const;

  const detail::VectorDesc& data_desc() const { return *data_; }
  detail::VectorDescRef data_desc_ref() const { return data_; }

  size_t size() const { return size_; }
  uint32_t reserved() const { return reserved_; }

 private:
  detail::VectorDescRef data_;
  size_t size_;
  uint32_t reserved_ = 0;
  const char* debug_name = nullptr;

  const char* debug_file = nullptr;
  int debug_line = -1;
  mutable bool copied = false;

  struct per_dpu_tag {};
  dpu_vector(per_dpu_tag, size_t n_per_dpu, uint32_t reserved, bool lazy,
             std::string_view name, std::source_location loc);
  void bind_storage(detail::VectorDescRef data);

  static std::vector<uint8_t> prepare_rpn(const std::vector<uint8_t>& ops);

 public:
  using reduction_result_t = typename reduction_result<T>::type;

#if PIPELINE
  pipeline_result<T> pipeline(const std::vector<uint8_t>& ops);
  pipeline_result<T> pipeline(const std::vector<uint8_t>& ops,
                              const std::vector<dpu_vector<T>>& operands,
                              const std::vector<uint32_t>& scalars = {});
  lazy_reduction_result<T> pipeline_reduce(
      const std::vector<uint8_t>& ops,
      const std::vector<dpu_vector<T>>& operands = {},
      const std::vector<uint32_t>& scalars = {});
  lazy_reduction_result<T> pipeline_reduce(
      const dpu_pipeline_expr<T>& expr,
      const std::vector<dpu_vector<T>>& operands = {},
      const std::vector<uint32_t>& scalars = {});
  lazy_arg_reduction_result<T> pipeline_argreduce(
      const std::vector<uint8_t>& ops,
      const std::vector<dpu_vector<T>>& operands = {},
      const std::vector<uint32_t>& scalars = {});
#endif
#if JIT
  pipeline_result<T> jit(const std::vector<uint8_t>& ops);
  pipeline_result<T> jit(const std::vector<uint8_t>& ops,
                         const std::vector<dpu_vector<T>>& operands,
                         const std::vector<uint32_t>& scalars = {});

#endif
};

template <typename T>
struct lazy_reduction_result {
  dpu_vector<T> vec;
  KernelID rid = 0;
  lazy_reduction_result() noexcept = default;
  lazy_reduction_result(dpu_vector<T> v, KernelID r)
      : vec(std::move(v)), rid(r) {}
  typename dpu_vector<T>::reduction_result_t get();
  operator typename dpu_vector<T>::reduction_result_t() { return get(); }
#if ENABLE_PROMOTION_REDUCTIONS
  operator T() { return (T)get(); }
#endif
};

template <typename T>
struct lazy_arg_reduction_result {
  dpu_vector<T> vec;
  bool want_max = true;
  bool ready = false;
  arg_reduction_result<T> cached{};
  lazy_arg_reduction_result() noexcept = default;
  lazy_arg_reduction_result(dpu_vector<T> v, bool max)
      : vec(std::move(v)), want_max(max) {}
  arg_reduction_result<T> get();
};

template <typename T>
using dpu_future = lazy_reduction_result<T>;

template <typename T>
using dpu_arg_future = lazy_arg_reduction_result<T>;

enum class dpu_local_reduce_op : uint8_t {
  sum,
  product,
  min,
  max,
};

#if PIPELINE
template <typename T>
class dpu_pipeline_expr {
 public:
  dpu_pipeline_expr() = default;
  explicit dpu_pipeline_expr(std::vector<uint8_t> ops) : ops_(std::move(ops)) {}

  static dpu_pipeline_expr input() {
    return dpu_pipeline_expr({(uint8_t)OP_PUSH_INPUT});
  }

  static dpu_pipeline_expr operand(uint8_t idx) {
    return dpu_pipeline_expr({(uint8_t)(OP_PUSH_OPERAND_0 + idx)});
  }

  static dpu_pipeline_expr scalar(T value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T) < 4 ? sizeof(T) : 4);
    return dpu_pipeline_expr({
        (uint8_t)OP_PUSH_SCALAR,
        (uint8_t)(bits & 0xFF),
        (uint8_t)((bits >> 8) & 0xFF),
        (uint8_t)((bits >> 16) & 0xFF),
        (uint8_t)((bits >> 24) & 0xFF),
    });
  }

  static dpu_pipeline_expr scalar_var(uint8_t idx) {
    return dpu_pipeline_expr({
        (uint8_t)OP_PUSH_SCALAR_VAR,
        idx,
    });
  }

  dpu_pipeline_expr dup() const { return append(OP_DUP); }
  dpu_pipeline_expr sqr() const { return dup().append(OP_MUL); }
  dpu_pipeline_expr min() const { return append(OP_MIN); }
  dpu_pipeline_expr max() const { return append(OP_MAX); }
  dpu_pipeline_expr sum() const { return append(OP_SUM); }
  dpu_pipeline_expr product() const { return append(OP_PRODUCT); }

  dpu_pipeline_expr operator+(T rhs) const {
    return append_scalar_op(OP_ADD_SCALAR, rhs);
  }
  dpu_pipeline_expr operator-(T rhs) const {
    return append_scalar_op(OP_SUB_SCALAR, rhs);
  }
  dpu_pipeline_expr operator*(T rhs) const {
    return append_scalar_op(OP_MUL_SCALAR, rhs);
  }
  dpu_pipeline_expr operator/(T rhs) const {
    return append_scalar_op(OP_DIV_SCALAR, rhs);
  }
  dpu_pipeline_expr operator==(T rhs) const {
    return append_scalar_op(OP_EQ_SCALAR, rhs);
  }

  const std::vector<uint8_t>& ops() const { return ops_; }

  dpu_pipeline_expr operator+(const dpu_pipeline_expr& rhs) const {
    return combine(rhs, OP_ADD);
  }
  dpu_pipeline_expr operator-(const dpu_pipeline_expr& rhs) const {
    return combine(rhs, OP_SUB);
  }
  dpu_pipeline_expr operator<(const dpu_pipeline_expr& rhs) const {
    return combine(rhs, OP_LT);
  }
  dpu_pipeline_expr operator*(const dpu_pipeline_expr& rhs) const {
    return combine(rhs, OP_MUL);
  }
  dpu_pipeline_expr operator/(const dpu_pipeline_expr& rhs) const {
    return combine(rhs, OP_DIV);
  }

  dpu_pipeline_expr select(const dpu_pipeline_expr& then_expr,
                           const dpu_pipeline_expr& else_expr) const {
    dpu_pipeline_expr out;
    out.ops_.reserve(ops_.size() + then_expr.ops_.size() +
                     else_expr.ops_.size() + 1);
    out.ops_.insert(out.ops_.end(), ops_.begin(), ops_.end());
    out.ops_.insert(out.ops_.end(), then_expr.ops_.begin(),
                    then_expr.ops_.end());
    out.ops_.insert(out.ops_.end(), else_expr.ops_.begin(),
                    else_expr.ops_.end());
    out.ops_.push_back(OP_SELECT);
    return out;
  }

 private:
  std::vector<uint8_t> ops_;

  dpu_pipeline_expr append(uint8_t op) const {
    auto out = *this;
    out.ops_.push_back(op);
    return out;
  }

  dpu_pipeline_expr append_scalar_op(uint8_t op, T value) const {
    auto out = *this;
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T) < 4 ? sizeof(T) : 4);
    out.ops_.push_back(op);
    out.ops_.push_back((uint8_t)(bits & 0xFF));
    out.ops_.push_back((uint8_t)((bits >> 8) & 0xFF));
    out.ops_.push_back((uint8_t)((bits >> 16) & 0xFF));
    out.ops_.push_back((uint8_t)((bits >> 24) & 0xFF));
    return out;
  }

  dpu_pipeline_expr combine(const dpu_pipeline_expr& rhs, uint8_t op) const {
    // Keep scalar variables as scalar operations instead of materializing
    // them as a third stack value. Besides shortening the RPN, this lets the
    // generic interpreter evaluate accumulator + column * scalar_var with its
    // two-vector stack.
    if (rhs.ops_.size() == 2 && rhs.ops_[0] == OP_PUSH_SCALAR_VAR &&
        op >= OP_ADD && op <= OP_LE) {
      auto out = *this;
      out.ops_.push_back((uint8_t)(op + (OP_ADD_SCALAR_VAR - OP_ADD)));
      out.ops_.push_back(rhs.ops_[1]);
      return out;
    }

    dpu_pipeline_expr out;
    out.ops_.reserve(ops_.size() + rhs.ops_.size() + 1);
    out.ops_.insert(out.ops_.end(), ops_.begin(), ops_.end());
    out.ops_.insert(out.ops_.end(), rhs.ops_.begin(), rhs.ops_.end());
    out.ops_.push_back(op);
    return out;
  }
};

// Horizontal argmin/argmax over K lanes, evaluated per element. `.label` is
// the winning lane index (lowers to the variadic OP_ARGMIN_K/OP_ARGMAX_K op);
// `.value` is the winning value (an elementwise min/max select-chain, since no
// dedicated min-of-vectors op exists yet). Both use a strict comparison so
// ties keep the lowest lane index. kmeans reads `.label` to drive the
// centroid scatter-add; inertia/convergence reads `.value`.
template <typename T>
struct dpu_arg_expr {
  dpu_pipeline_expr<T> label;
  dpu_pipeline_expr<T> value;
};

template <typename T>
dpu_arg_expr<T> detail_arg_k_lanes(
    const std::vector<dpu_pipeline_expr<T>>& lanes, uint8_t arg_op) {
  std::vector<uint8_t> label_ops;
  for (const auto& lane : lanes) {
    const auto& o = lane.ops();
    label_ops.insert(label_ops.end(), o.begin(), o.end());
  }
  label_ops.push_back(arg_op);
  label_ops.push_back((uint8_t)lanes.size());

  const bool is_min = (arg_op == (uint8_t)OP_ARGMIN_K);
  dpu_pipeline_expr<T> best = lanes[0];
  for (size_t j = 1; j < lanes.size(); ++j) {
    dpu_pipeline_expr<T> take = is_min ? (lanes[j] < best) : (best < lanes[j]);
    best = take.select(lanes[j], best);
  }
  return {dpu_pipeline_expr<T>(std::move(label_ops)), best};
}

// Over K candidate expressions, per element.
template <typename T>
dpu_arg_expr<T> argmin(const std::vector<dpu_pipeline_expr<T>>& lanes) {
  return detail_arg_k_lanes(lanes, (uint8_t)OP_ARGMIN_K);
}
template <typename T>
dpu_arg_expr<T> argmax(const std::vector<dpu_pipeline_expr<T>>& lanes) {
  return detail_arg_k_lanes(lanes, (uint8_t)OP_ARGMAX_K);
}

// Over K whole vectors, returning the per-element winning lane.
#if JIT
template <typename T>
dpu_vector<T> argmin(std::vector<dpu_vector<T>>& lanes) {
  std::vector<dpu_vector<T>> operands(lanes.begin() + 1, lanes.end());
  std::vector<dpu_pipeline_expr<T>> expressions;
  expressions.reserve(lanes.size());
  expressions.push_back(dpu_pipeline_expr<T>::input());
  for (size_t i = 0; i < operands.size(); ++i) {
    expressions.push_back(dpu_pipeline_expr<T>::operand((uint8_t)i));
  }
  return lanes[0].jit(argmin(expressions).label.ops(), operands).vec;
}
template <typename T>
dpu_vector<T> argmax(std::vector<dpu_vector<T>>& lanes) {
  std::vector<dpu_vector<T>> operands(lanes.begin() + 1, lanes.end());
  std::vector<dpu_pipeline_expr<T>> expressions;
  expressions.reserve(lanes.size());
  expressions.push_back(dpu_pipeline_expr<T>::input());
  for (size_t i = 0; i < operands.size(); ++i) {
    expressions.push_back(dpu_pipeline_expr<T>::operand((uint8_t)i));
  }
  return lanes[0].jit(argmax(expressions).label.ops(), operands).vec;
}
#endif  // JIT
#endif

#if PIPELINE
template <typename T>
struct pipeline_result {
  dpu_vector<T> vec;
  explicit pipeline_result(dpu_vector<T> value) : vec(std::move(value)) {}
};
#endif

template <typename T>
lazy_reduction_result<T> sum(const dpu_vector<T>& a);
template <typename T>
lazy_reduction_result<T> product(const dpu_vector<T>& a);
template <typename T>
lazy_reduction_result<T> min(const dpu_vector<T>& a);
template <typename T>
lazy_reduction_result<T> max(const dpu_vector<T>& a);
template <typename T>
dpu_vector<T> operator<(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs);
template <typename T>
dpu_vector<T> select(const dpu_vector<T>& cond, const dpu_vector<T>& then_vec,
                     const dpu_vector<T>& else_vec);

template <typename T>
class dpu_local_vector {
 public:
  dpu_local_vector(uint32_t n, LOGGER_ARGS_WITH_DEFAULTS);
  dpu_local_vector(uint32_t n, dpu_local_reduce_op reduce_op,
                   LOGGER_ARGS_WITH_DEFAULTS);
  ~dpu_local_vector() = default;

  vector<T> to_cpu();
  dpu_local_reduce_op reduce_op() const { return reduce_op_; }

  const detail::VectorDesc& data_desc() const { return *data_; }
  detail::VectorDescRef data_desc_ref() const { return data_; }
  uint32_t size() const { return size_; }

 private:
  detail::VectorDescRef data_;
  uint32_t size_;
  dpu_local_reduce_op reduce_op_ = dpu_local_reduce_op::sum;
};

#if PIPELINE
template <typename T>
class dpu_pipeline_context {
 public:
  void local_reduce(dpu_local_vector<T>& local,
                    const dpu_pipeline_expr<T>& index,
                    const dpu_pipeline_expr<T>& value);
  void local_reduce(dpu_local_vector<T>& local,
                    const dpu_pipeline_expr<T>& index, T value) {
    local_reduce(local, index, dpu_pipeline_expr<T>::scalar(value));
  }
  void local_sum(dpu_local_vector<T>& local, const dpu_pipeline_expr<T>& index,
                 const dpu_pipeline_expr<T>& value) {
    local_reduce(local, index, value);
  }
  void local_sum(dpu_local_vector<T>& local, const dpu_pipeline_expr<T>& index,
                 T value) {
    local_reduce(local, index, value);
  }

  std::vector<uint8_t> materialize_ops() const;
  const std::vector<detail::VectorDescRef>& locals() const { return locals_; }

 private:
  struct LocalReduce {
    uint8_t local_id;
    uint8_t reduce_op;
    std::vector<uint8_t> index_ops;
    std::vector<uint8_t> value_ops;
  };

  std::vector<uint8_t> ops_;
  std::vector<detail::VectorDescRef> locals_;
  std::vector<LocalReduce> local_reductions_;

  uint8_t local_id(dpu_local_vector<T>& local);
};
#endif

#if JIT
template <typename T, typename F>
void dpu_jit_foreach(dpu_vector<T>& primary,
                     const std::vector<dpu_vector<T>>& operands,
                     const std::vector<uint32_t>& scalars, F f);
#endif

namespace detail {
void launch_binary(VectorDescRef res, VectorDescRef lhs, VectorDescRef rhs,
                   KernelID kernel_id, uint8_t opcode, KernelID pipeline_kid);
void launch_binary_scalar(VectorDescRef res, VectorDescRef lhs, uint32_t scalar,
                          KernelID kernel_id, uint8_t opcode,
                          KernelID pipeline_kid);
void launch_fill(VectorDescRef res, uint32_t value, KernelID kernel_id);
void launch_unary(VectorDescRef res, VectorDescRef rhs, KernelID kernel_id,
                  uint8_t opcode, KernelID pipeline_kid);
void launch_reduction(VectorDescRef buf, VectorDescRef rhs, KernelID kernel_id,
                      uint8_t opcode, KernelID pipeline_kid);

void internal_launch_unary(VectorDescRef res, VectorDescRef lhs,
                           KernelID kernel_id);

#if PIPELINE
void launch_universal_pipeline(
    VectorDescRef res, VectorDescRef init, const std::vector<uint8_t>& ops,
    const std::vector<VectorDescRef>& operands, KernelID kernel_id,
    const std::vector<uint32_t>& scalars = {},
    const std::vector<uint32_t>& extra_scalars = {},
    const std::vector<VectorDescRef>& extra_outputs = {});

void internal_launch_universal_pipeline(
    VectorDescRef res, VectorDescRef init, const std::vector<uint8_t>& ops,
    const std::vector<VectorDescRef>& operands, KernelID kernel_id,
    const std::vector<uint32_t>& scalars,
    const std::vector<uint32_t>& extra_scalars = {},
    const std::vector<VectorDescRef>& extra_outputs = {},
    std::string_view kernel_hash = {});

void internal_launch_jit(const std::string& binary_path, VectorDescRef output,
                         const std::vector<VectorDescRef>& inputs,
                         const std::vector<uint8_t>& rpn_ops,
                         const std::vector<uint32_t>& extra_scalars = {},
                         const std::vector<VectorDescRef>& extra_outputs = {});
#endif
}  // namespace detail

#include "vector.inl"
