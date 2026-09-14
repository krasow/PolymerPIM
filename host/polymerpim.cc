#include "polymerpim.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "detail/vector.h"
#include "stats.h"

namespace polymerpim {
namespace {

using T = int32_t;
using BackendVector = ::dpu_vector<T>;

enum class ExprOp {
  input,
  scalar,
  add,
  sub,
  mul,
  div,
  lt,
  eq,
  shift_right,
  negate,
  absolute,
  square,
  select,
  argmin,
  argmax,
};

struct Node {
  ExprOp op;
  BackendVector input;
  T scalar = 0;
  // True only for a fill's own constant.  A runtime scalar must not be folded
  // on its value: that would make the program shape depend on the data and
  // force a JIT recompile whenever the value changes.
  bool structural = false;
  // Cached subtree op count; the deferral cap reads it once per op.
  size_t ops = 0;
  // Cached RPN stack depth this subtree needs to evaluate.
  size_t stack_depth = 0;
  // Upper bound on distinct inputs; shared vectors are counted once per use.
  size_t input_count = 0;
  std::vector<std::shared_ptr<Node>> children;
};

using NodeRef = std::shared_ptr<Node>;

NodeRef node(ExprOp op, std::vector<NodeRef> children = {}) {
  auto result = std::make_shared<Node>();
  result->op = op;
  result->ops = (op == ExprOp::input || op == ExprOp::scalar) ? 0 : 1;
  for (const auto& child : children) {
    if (child) result->ops += child->ops;
  }
  result->input_count = (op == ExprOp::input) ? 1 : 0;
  for (const auto& child : children) {
    if (child) result->input_count += child->input_count;
  }
  // Child i sits above i already-pushed slots.
  result->stack_depth = 1;
  for (size_t i = 0; i < children.size(); ++i) {
    if (!children[i]) continue;
    const size_t needed = i + children[i]->stack_depth;
    if (needed > result->stack_depth) result->stack_depth = needed;
  }
  result->children = std::move(children);
  return result;
}

NodeRef scalar_node(T value, bool structural = false) {
  auto result = node(ExprOp::scalar);
  result->scalar = value;
  result->structural = structural;
  return result;
}

// An identity operand drops, so a sized vector's zero fill costs nothing on its
// first use.
NodeRef fold_binary(ExprOp op, const NodeRef& lhs, const NodeRef& rhs) {
  auto is_value = [](const NodeRef& n, T value) {
    return n && n->op == ExprOp::scalar && n->structural && n->scalar == value;
  };
  switch (op) {
    case ExprOp::add:
      if (is_value(lhs, 0)) return rhs;
      if (is_value(rhs, 0)) return lhs;
      break;
    case ExprOp::sub:
      if (is_value(rhs, 0)) return lhs;
      break;
    case ExprOp::mul:
      if (is_value(lhs, 1)) return rhs;
      if (is_value(rhs, 1)) return lhs;
      break;
    default:
      break;
  }
  return node(op, {lhs, rhs});
}

uint8_t binary_opcode(ExprOp op) {
  switch (op) {
    case ExprOp::add:
      return OP_ADD;
    case ExprOp::sub:
      return OP_SUB;
    case ExprOp::mul:
      return OP_MUL;
    case ExprOp::div:
      return OP_DIV;
    case ExprOp::lt:
      return OP_LT;
    case ExprOp::eq:
      return OP_EQ;
    default:
      throw std::logic_error("invalid binary expression");
  }
}

// Ops a pending expression contributes to a fused program.
size_t expression_ops(const NodeRef& value) { return value ? value->ops : 0; }

// RPN stack slots needed; past the interpreter's stack it cannot run at all.
size_t expression_depth(const NodeRef& value) {
  return value ? value->stack_depth : 0;
}

// Deferring past the input budget makes lowering throw; materialise instead.
size_t expression_inputs(const NodeRef& value) {
  return value ? value->input_count : 0;
}

uint8_t scalar_opcode(ExprOp op) {
  switch (op) {
    case ExprOp::add:
      return OP_ADD_SCALAR_VAR;
    case ExprOp::sub:
      return OP_SUB_SCALAR_VAR;
    case ExprOp::mul:
      return OP_MUL_SCALAR_VAR;
    case ExprOp::div:
      return OP_DIV_SCALAR_VAR;
    case ExprOp::lt:
      return OP_LT_SCALAR_VAR;
    case ExprOp::eq:
      return OP_EQ_SCALAR_VAR;
    case ExprOp::shift_right:
      return OP_ASR_SCALAR_VAR;
    default:
      throw std::logic_error("invalid scalar expression");
  }
}

class Compiler {
 public:
  std::vector<uint8_t> expression(const NodeRef& root) {
    std::vector<uint8_t> ops;
    emit(root, ops);
    return ops;
  }

  const std::vector<BackendVector>& inputs() const { return inputs_; }
  const std::vector<uint32_t>& scalars() const { return scalars_; }

 private:
  std::vector<BackendVector> inputs_;
  std::vector<uint32_t> scalars_;
  std::unordered_map<const Node*, uint8_t> scalar_slots_;

  uint8_t input_slot(const BackendVector& input) {
    auto desc = input.data_desc_ref();
    for (size_t i = 0; i < inputs_.size(); ++i) {
      if (inputs_[i].data_desc_ref() == desc) return static_cast<uint8_t>(i);
    }
    if (inputs_.size() >= MAX_COMBINED_INPUTS) {
      throw std::logic_error("expression has too many inputs");
    }
    inputs_.push_back(input);
    return static_cast<uint8_t>(inputs_.size() - 1);
  }

  uint8_t scalar_slot(const NodeRef& value) {
    auto found = scalar_slots_.find(value.get());
    if (found != scalar_slots_.end()) return found->second;
    if (scalars_.size() >= MAX_PIPELINE_SCALARS) {
      throw std::logic_error("expression has too many scalars");
    }
    uint32_t bits;
    std::memcpy(&bits, &value->scalar, sizeof(bits));
    scalars_.push_back(bits);
    uint8_t slot = static_cast<uint8_t>(scalars_.size() - 1);
    scalar_slots_.emplace(value.get(), slot);
    return slot;
  }

  void emit(const NodeRef& value, std::vector<uint8_t>& ops) {
    if (!value) throw std::logic_error("empty expression");
    switch (value->op) {
      case ExprOp::input: {
        uint8_t slot = input_slot(value->input);
        ops.push_back(slot == 0
                          ? static_cast<uint8_t>(OP_PUSH_INPUT)
                          : static_cast<uint8_t>(OP_PUSH_OPERAND_0 + slot - 1));
        return;
      }
      case ExprOp::scalar:
        ops.push_back(OP_PUSH_SCALAR_VAR);
        ops.push_back(scalar_slot(value));
        return;
      case ExprOp::negate:
      case ExprOp::absolute:
      case ExprOp::square:
        emit(value->children[0], ops);
        if (value->op == ExprOp::square) {
          ops.push_back(OP_DUP);
          ops.push_back(OP_MUL);
        } else {
          ops.push_back(value->op == ExprOp::negate ? OP_NEGATE : OP_ABS);
        }
        return;
      case ExprOp::select:
        emit(value->children[0], ops);
        emit(value->children[1], ops);
        emit(value->children[2], ops);
        ops.push_back(OP_SELECT);
        return;
      case ExprOp::argmin:
      case ExprOp::argmax:
        for (const auto& child : value->children) emit(child, ops);
        ops.push_back(value->op == ExprOp::argmin ? OP_ARGMIN_K : OP_ARGMAX_K);
        ops.push_back(static_cast<uint8_t>(value->children.size()));
        return;
      case ExprOp::shift_right:
        emit(value->children[0], ops);
        ops.push_back(scalar_opcode(value->op));
        ops.push_back(scalar_slot(value->children[1]));
        return;
      default:
        emit(value->children[0], ops);
        if (value->children[1]->op == ExprOp::scalar) {
          ops.push_back(scalar_opcode(value->op));
          ops.push_back(scalar_slot(value->children[1]));
        } else {
          emit(value->children[1], ops);
          ops.push_back(binary_opcode(value->op));
        }
    }
  }
};

#if PIPELINE && !JIT
bool can_interpret(const std::vector<uint8_t>& ops) {
  if (ops.empty() || ops.size() > MAX_VFUSE_OPS) return false;
  size_t depth = 0;
  bool reduced = false;
  for (size_t i = 0; i < ops.size(); ++i) {
    if (reduced) return false;
    uint8_t op = ops[i];
    size_t inline_bytes = OP_INLINE_BYTES(op);
    if (inline_bytes > ops.size() - i - 1) return false;
    if (IS_OP_STACK(op) || op == OP_PUSH_SCALAR || op == OP_PUSH_SCALAR_VAR) {
      ++depth;
    } else if (op == OP_DUP) {
      if (depth == 0) return false;
      ++depth;
    } else if (IS_OP_UNARY(op) || IS_OP_SCALAR(op) || IS_OP_SCALAR_VAR(op)) {
      if (depth == 0) return false;
    } else if (IS_OP_BINARY(op)) {
      if (depth < 2) return false;
      --depth;
    } else if (IS_OP_TERNARY(op)) {
      if (depth < 3) return false;
      depth -= 2;
    } else if (IS_OP_ARG_K(op)) {
      uint8_t lanes = ops[i + 1];
      if (lanes == 0 || depth < lanes) return false;
      depth -= lanes - 1;
    } else if (IS_OP_REDUCTION(op)) {
      if (depth != 1) return false;
      depth = 0;
      reduced = true;
    } else {
      return false;
    }
    if (depth > MAX_PIPELINE_STACK_DEPTH) return false;
    i += inline_bytes;
  }
  return reduced ? depth == 0 : depth == 1;
}
#endif

// Filled on the DPUs.  Staging it from the host instead would cost a
// full-length buffer and one host-to-DPU transfer of the whole vector.
BackendVector fill_vector(T value, size_t length, std::string_view name = "") {
  BackendVector vec(length);
  if (!name.empty()) vec.data_desc_ref()->debug_name = name.data();
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  ::detail::launch_fill(vec.data_desc_ref(), bits, OpInfo<T>::fill);
  return vec;
}

BackendVector eager(const NodeRef& value) {
  switch (value->op) {
    case ExprOp::input:
      return value->input;
    case ExprOp::negate:
      return -eager(value->children[0]);
    case ExprOp::absolute:
      return ::abs(eager(value->children[0]));
    case ExprOp::square: {
      auto input = eager(value->children[0]);
      return input * input;
    }
    case ExprOp::select: {
      auto condition = eager(value->children[0]).to_cpu();
      auto then_value = eager(value->children[1]).to_cpu();
      auto else_value = eager(value->children[2]).to_cpu();
      std::vector<T> output(condition.size());
      for (size_t i = 0; i < output.size(); ++i) {
        output[i] = condition[i] ? then_value[i] : else_value[i];
      }
      auto result = BackendVector::from_cpu(output);
      result.add_fence();
      return result;
    }
    case ExprOp::shift_right:
      return eager(value->children[0]) >> value->children[1]->scalar;
    case ExprOp::argmin:
    case ExprOp::argmax:
    case ExprOp::scalar:
      throw std::logic_error("expression requires fused execution");
    default:
      break;
  }

  auto lhs = eager(value->children[0]);
  if (value->children[1]->op == ExprOp::scalar) {
    T rhs = value->children[1]->scalar;
    switch (value->op) {
      case ExprOp::add:
        return lhs + rhs;
      case ExprOp::sub:
        return lhs - rhs;
      case ExprOp::mul:
        return lhs * rhs;
      case ExprOp::div:
        return lhs / rhs;
      case ExprOp::lt: {
        uint32_t bits;
        std::memcpy(&bits, &rhs, sizeof(bits));
        BackendVector result(lhs.size(), 0, true);
        ::detail::launch_binary_scalar(
            result.data_desc_ref(), lhs.data_desc_ref(), bits,
            OpInfo<T>::lt_scalar, OpInfo<T>::lt_scalar_op,
            OpInfo<T>::universal_pipeline);
        return result;
      }
      case ExprOp::eq:
        return lhs == rhs;
      default:
        throw std::logic_error("unsupported scalar expression");
    }
  }

  auto rhs = eager(value->children[1]);
  switch (value->op) {
    case ExprOp::add:
      return lhs + rhs;
    case ExprOp::sub:
      return lhs - rhs;
    case ExprOp::mul:
      return lhs * rhs;
    case ExprOp::div:
      return lhs / rhs;
    case ExprOp::lt:
      return lhs < rhs;
    case ExprOp::eq:
      return (lhs - rhs) == static_cast<T>(0);
    default:
      throw std::logic_error("unsupported binary expression");
  }
}

BackendVector materialize(const NodeRef& root) {
  if (!root) throw std::logic_error("empty expression");
  if (root->op == ExprOp::input) return root->input;
#if PIPELINE
  Compiler compiler;
  auto ops = compiler.expression(root);
  const auto& inputs = compiler.inputs();
  if (inputs.empty()) throw std::logic_error("expression has no DPU input");
  std::vector<BackendVector> operands(inputs.begin() + 1, inputs.end());
#if !JIT
  if (!can_interpret(ops)) return eager(root);
#endif
  return const_cast<BackendVector&>(inputs[0])
      .pipeline(ops, operands, compiler.scalars())
      .vec;
#else
  return eager(root);
#endif
}

template <typename F>
auto translate_oom(F&& operation) -> decltype(operation()) {
  try {
    return operation();
  } catch (const DpuOOMException& error) {
    throw OutOfMemory(error.what());
  }
}

std::mutex local_mutex;
std::vector<DPULocalVector<T>*> pending_locals;

RuntimeStatistics public_stats(const StatsSnapshot& value) {
  RuntimeStatistics out;
#define COPY_STAT(name) out.name = value.name
  COPY_STAT(events_submitted);
  COPY_STAT(compute_launches);
  COPY_STAT(dpu_transfers);
  COPY_STAT(host_transfers);
  COPY_STAT(fences);
  COPY_STAT(vertical_fusions);
  COPY_STAT(horizontal_fusions);
  COPY_STAT(absorbed_producers);
  COPY_STAT(binary_switches);
  COPY_STAT(oom_retries);
  COPY_STAT(jit_kernel_compiles);
  COPY_STAT(jit_kernel_cache_hits);
  COPY_STAT(jit_batch_links);
  COPY_STAT(jit_batch_cache_hits);
#if JIT_PIPELINE_FALLBACK
  COPY_STAT(jit_pipeline_fallbacks);
  COPY_STAT(jit_eager_fallbacks);
#endif
#undef COPY_STAT
  return out;
}

}  // namespace

// Contents may still be an unevaluated expression: assignment records it and
// storage waits for a host read or a fence, so a chain of assignments reaches a
// reduction as one kernel.  A sized vector starts as a bare scalar -- a fill --
// which is what `length` sizes.
struct DPUVector<T>::Impl {
  explicit Impl(BackendVector storage) : value(std::move(storage)) {}

  Impl(NodeRef expression, size_t length, std::string debug_name = {})
      : pending(std::move(expression)),
        length(length),
        name(std::move(debug_name)) {}

  bool is_fill() const { return pending && pending->op == ExprOp::scalar; }

  BackendVector& storage() const {
    if (!pending) return value;
    const bool fill = is_fill();
    value = fill ? fill_vector(pending->scalar, length, name)
                 : materialize(pending);
    // A consumer that already fused this chain shares the node: point it at
    // the buffer so it is not evaluated twice.  A fill stays scalar.
    if (consumed && !fill) {
      pending->op = ExprOp::input;
      pending->input = value;
      pending->children.clear();
      pending->ops = 0;
      pending->stack_depth = 1;
      pending->input_count = 1;
    }
    pending = nullptr;
    return value;
  }

  size_t size() const {
    if (!pending) return value.size();
    if (is_fill()) return length;
    Compiler compiler;
    compiler.expression(pending);
    return compiler.inputs().empty() ? 0 : compiler.inputs()[0].size();
  }

  mutable BackendVector value;
  mutable NodeRef pending;
  size_t length = 0;
  std::string name;
  // The first consumer fuses the expression; later ones read storage, since
  // re-fusing into each recomputes it per element (1.7x on nine reductions).
  mutable bool consumed = false;
};

struct DpuLazy<T>::Impl {
  explicit Impl(NodeRef root) : root(std::move(root)) {}
  NodeRef root;
};

struct DpuFuture<T>::Impl {
  explicit Impl(::dpu_future<T> value) : value(std::move(value)) {}
  ::dpu_future<T> value;
};

struct DpuFuture<ArgResult>::Impl {
  explicit Impl(::dpu_arg_future<T> value) : value(std::move(value)) {}
  explicit Impl(ArgResult result) : cached(result), ready(true) {}
  ::dpu_arg_future<T> value;
  ArgResult cached{};
  bool ready = false;
};

struct DPULocalVector<T>::Impl {
  struct Update {
    NodeRef index;
    NodeRef value;
  };

  Impl(uint32_t size, std::string_view name) : value(size, name) {}
  ::dpu_local_vector<T> value;
  std::vector<Update> updates;
};

DpuLazy<T>::DpuLazy(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

size_t DpuLazy<T>::size() const {
  if (!impl_ || !impl_->root) return 0;
  Compiler compiler;
  compiler.expression(impl_->root);
  return compiler.inputs().empty() ? 0 : compiler.inputs()[0].size();
}

std::vector<T> DpuLazy<T>::to_cpu() const {
  return translate_oom([&] { return materialize(impl_->root).to_cpu(); });
}

DPUVector<T>::DPUVector(size_t count, std::string_view name)
    : DPUVector(count, (T)0, name) {}

DPUVector<T>::DPUVector(size_t count, T value, std::string_view name)
#if PIPELINE
    : impl_(std::make_shared<Impl>(scalar_node(value, /*structural=*/true),
                                   count, std::string(name))) {
}
#else
    // without fusion we can't do this scalar impl
    : impl_(std::make_shared<Impl>(fill_vector(value, count, name))) {
}
#endif

DPUVector<T>::DPUVector(std::vector<T>& values, std::string_view name)
    : impl_(translate_oom([&] {
        return std::make_shared<Impl>(BackendVector::from_cpu(values, name));
      })) {}

DPUVector<T>::DPUVector(T* values, size_t count, std::string_view name)
    : impl_(translate_oom([&] {
        return std::make_shared<Impl>(
            BackendVector::from_cpu(values, count, name));
      })) {}

// Naming a value means wanting it: construction runs the expression, so a
// vector several consumers read is computed once.  Assignment is what defers,
// because `acc = acc + x` replaces the value it just consumed.
DPUVector<T>::DPUVector(const DpuLazy<T>& expression) {
  if (!expression.impl_) return;
  impl_ = translate_oom([&] {
    return std::make_shared<Impl>(materialize(expression.impl_->root));
  });
}

DPUVector<T>::DPUVector(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

DPUVector<T>& DPUVector<T>::operator=(const DpuLazy<T>& expression) {
  impl_ = expression.impl_
              ? std::make_shared<Impl>(expression.impl_->root, size())
              : nullptr;
  return *this;
}

DPUVector<T>& DPUVector<T>::operator+=(const DpuLazy<T>& rhs) {
  return *this = static_cast<DpuLazy<T>>(*this) + rhs;
}

DPUVector<T>& DPUVector<T>::operator-=(const DpuLazy<T>& rhs) {
  return *this = static_cast<DpuLazy<T>>(*this) - rhs;
}

DPUVector<T>& DPUVector<T>::operator*=(const DpuLazy<T>& rhs) {
  return *this = static_cast<DpuLazy<T>>(*this) * rhs;
}

DPUVector<T>& DPUVector<T>::operator+=(T rhs) {
  return *this = static_cast<DpuLazy<T>>(*this) + rhs;
}

DPUVector<T>& DPUVector<T>::operator-=(T rhs) {
  return *this = static_cast<DpuLazy<T>>(*this) - rhs;
}

DPUVector<T>& DPUVector<T>::operator*=(T rhs) {
  return *this = static_cast<DpuLazy<T>>(*this) * rhs;
}

std::vector<T> DPUVector<T>::to_cpu() {
  if (!impl_) return {};
  return translate_oom([&] { return impl_->storage().to_cpu(); });
}

size_t DPUVector<T>::to_cpu_into(T* output, size_t capacity) {
  if (!impl_) return 0;
  return translate_oom(
      [&] { return impl_->storage().to_cpu_into(output, capacity); });
}

size_t DPUVector<T>::size() const { return impl_ ? impl_->size() : 0; }

DPUVector<T>::operator DpuLazy<T>() const {
  if (!impl_) return {};
  // Past the cap the chain cannot fuse into one kernel, so extending it only
  // mints another program shape for the JIT to compile.
  if (impl_->pending && !impl_->consumed &&
      expression_ops(impl_->pending) < MAX_VFUSE_OPS &&
      expression_depth(impl_->pending) <= MAX_PIPELINE_STACK_DEPTH &&
      expression_inputs(impl_->pending) < MAX_COMBINED_INPUTS) {
    impl_->consumed = true;
    return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(impl_->pending));
  }
  auto root = node(ExprOp::input);
  root->input = translate_oom([&] { return impl_->storage(); });
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(std::move(root)));
}

DpuLazy<T> operator+(const DpuLazy<T>& lhs, const DpuLazy<T>& rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      fold_binary(ExprOp::add, lhs.impl_->root, rhs.impl_->root)));
}
DpuLazy<T> operator-(const DpuLazy<T>& lhs, const DpuLazy<T>& rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      fold_binary(ExprOp::sub, lhs.impl_->root, rhs.impl_->root)));
}
DpuLazy<T> operator*(const DpuLazy<T>& lhs, const DpuLazy<T>& rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      fold_binary(ExprOp::mul, lhs.impl_->root, rhs.impl_->root)));
}
DpuLazy<T> operator/(const DpuLazy<T>& lhs, const DpuLazy<T>& rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      node(ExprOp::div, {lhs.impl_->root, rhs.impl_->root})));
}
DpuLazy<T> operator<(const DpuLazy<T>& lhs, const DpuLazy<T>& rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      node(ExprOp::lt, {lhs.impl_->root, rhs.impl_->root})));
}
DpuLazy<T> operator==(const DpuLazy<T>& lhs, const DpuLazy<T>& rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      node(ExprOp::eq, {lhs.impl_->root, rhs.impl_->root})));
}
DpuLazy<T> operator+(const DpuLazy<T>& lhs, T rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      fold_binary(ExprOp::add, lhs.impl_->root, scalar_node(rhs))));
}
DpuLazy<T> operator-(const DpuLazy<T>& lhs, T rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      fold_binary(ExprOp::sub, lhs.impl_->root, scalar_node(rhs))));
}
DpuLazy<T> operator*(const DpuLazy<T>& lhs, T rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      fold_binary(ExprOp::mul, lhs.impl_->root, scalar_node(rhs))));
}
DpuLazy<T> operator/(const DpuLazy<T>& lhs, T rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      node(ExprOp::div, {lhs.impl_->root, scalar_node(rhs)})));
}
DpuLazy<T> operator<(const DpuLazy<T>& lhs, T rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      node(ExprOp::lt, {lhs.impl_->root, scalar_node(rhs)})));
}
DpuLazy<T> operator==(const DpuLazy<T>& lhs, T rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      node(ExprOp::eq, {lhs.impl_->root, scalar_node(rhs)})));
}
DpuLazy<T> operator>>(const DpuLazy<T>& lhs, T rhs) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      node(ExprOp::shift_right, {lhs.impl_->root, scalar_node(rhs)})));
}
DpuLazy<T> operator-(const DpuLazy<T>& value) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      node(ExprOp::negate, {value.impl_->root})));
}
DpuLazy<T> abs(const DpuLazy<T>& value) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      node(ExprOp::absolute, {value.impl_->root})));
}
DpuLazy<T> sqr(const DpuLazy<T>& value) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      node(ExprOp::square, {value.impl_->root})));
}

DpuLazy<T> select(const DpuLazy<T>& condition, const DpuLazy<T>& then_value,
                  const DpuLazy<T>& else_value) {
  return DpuLazy<T>(std::make_shared<DpuLazy<T>::Impl>(
      node(ExprOp::select, {condition.impl_->root, then_value.impl_->root,
                            else_value.impl_->root})));
}

namespace {
NodeRef arg_k(const std::vector<NodeRef>& lanes, ExprOp op) {
  if (lanes.empty()) throw std::logic_error("argmin/argmax requires a lane");
  if (lanes.size() > UINT8_MAX)
    throw std::logic_error("argmin/argmax has too many lanes");
  return node(op, lanes);
}
}  // namespace

DpuLazy<T> argmin(const std::vector<DpuLazy<T>>& lanes) {
  std::vector<NodeRef> roots;
  for (const auto& lane : lanes) roots.push_back(lane.impl_->root);
  return DpuLazy<T>(
      std::make_shared<DpuLazy<T>::Impl>(arg_k(roots, ExprOp::argmin)));
}
DpuLazy<T> argmax(const std::vector<DpuLazy<T>>& lanes) {
  std::vector<NodeRef> roots;
  for (const auto& lane : lanes) roots.push_back(lane.impl_->root);
  return DpuLazy<T>(
      std::make_shared<DpuLazy<T>::Impl>(arg_k(roots, ExprOp::argmax)));
}

DpuFuture<T>::DpuFuture(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

DpuFuture<T>::result_type DpuFuture<T>::get() {
  if (!impl_) throw std::logic_error("empty future");
  return translate_oom([&] { return impl_->value.get(); });
}

DpuFuture<ArgResult>::DpuFuture(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

DpuFuture<ArgResult>::result_type DpuFuture<ArgResult>::get() {
  if (!impl_) throw std::logic_error("empty arg-reduction future");
  return translate_oom([&] {
    if (!impl_->ready) {
      const auto result = impl_->value.get();
      impl_->cached = {result.value, result.index};
      impl_->ready = true;
    }
    return impl_->cached;
  });
}

namespace {
std::shared_ptr<DpuFuture<T>::Impl> reduce(const NodeRef& root, uint8_t op) {
  return translate_oom([&] {
#if PIPELINE
    Compiler compiler;
    auto ops = compiler.expression(root);
    ops.push_back(op);
    const auto& inputs = compiler.inputs();
    if (inputs.empty()) throw std::logic_error("reduction has no DPU input");
#if !JIT
    if (!can_interpret(ops)) {
      auto value = eager(root);
      if (op == OP_SUM)
        return std::make_shared<DpuFuture<T>::Impl>(::sum(value));
      if (op == OP_PRODUCT)
        return std::make_shared<DpuFuture<T>::Impl>(::product(value));
      if (op == OP_MIN)
        return std::make_shared<DpuFuture<T>::Impl>(::min(value));
      return std::make_shared<DpuFuture<T>::Impl>(::max(value));
    }
#endif
    std::vector<BackendVector> operands(inputs.begin() + 1, inputs.end());
    auto future = const_cast<BackendVector&>(inputs[0]).pipeline_reduce(
        ops, operands, compiler.scalars());
    return std::make_shared<DpuFuture<T>::Impl>(std::move(future));
#else
    auto value = eager(root);
    if (op == OP_SUM) return std::make_shared<DpuFuture<T>::Impl>(::sum(value));
    if (op == OP_PRODUCT)
      return std::make_shared<DpuFuture<T>::Impl>(::product(value));
    if (op == OP_MIN) return std::make_shared<DpuFuture<T>::Impl>(::min(value));
    return std::make_shared<DpuFuture<T>::Impl>(::max(value));
#endif
  });
}

std::shared_ptr<DpuFuture<ArgResult>::Impl> arg_reduce(const NodeRef& root,
                                                       uint8_t op) {
  return translate_oom([&] {
#if PIPELINE
    Compiler compiler;
    auto ops = compiler.expression(root);
    ops.push_back(op);
    const auto& inputs = compiler.inputs();
    if (inputs.empty())
      throw std::logic_error("arg-reduction has no DPU input");
    std::vector<BackendVector> operands(inputs.begin() + 1, inputs.end());
    auto future = const_cast<BackendVector&>(inputs[0]).pipeline_argreduce(
        ops, operands, compiler.scalars());
    return std::make_shared<DpuFuture<ArgResult>::Impl>(std::move(future));
#else
    auto values = eager(root).to_cpu();
    if (values.empty()) throw std::logic_error("arg-reduction of empty vector");
    ArgResult best{values[0], 0};
    for (uint32_t i = 1; i < values.size(); ++i) {
      if ((op == OP_ARGMAX_REDUCE ? values[i] > best.value
                                  : values[i] < best.value))
        best = {values[i], i};
    }
    return std::make_shared<DpuFuture<ArgResult>::Impl>(best);
#endif
  });
}
}  // namespace

DpuFuture<T> sum(const DpuLazy<T>& expression) {
  return DpuFuture<T>(reduce(expression.impl_->root, OP_SUM));
}
DpuFuture<T> product(const DpuLazy<T>& expression) {
  return DpuFuture<T>(reduce(expression.impl_->root, OP_PRODUCT));
}
DpuFuture<T> minimum(const DpuLazy<T>& expression) {
  return DpuFuture<T>(reduce(expression.impl_->root, OP_MIN));
}
DpuFuture<T> maximum(const DpuLazy<T>& expression) {
  return DpuFuture<T>(reduce(expression.impl_->root, OP_MAX));
}

DpuFuture<ArgResult> argmin(const DpuLazy<T>& expression) {
  if (!expression.impl_) throw std::logic_error("empty expression");
  if (expression.size() == 0)
    throw std::logic_error("argmin of empty expression");
  return DpuFuture<ArgResult>(
      arg_reduce(expression.impl_->root, OP_ARGMIN_REDUCE));
}

DpuFuture<ArgResult> argmax(const DpuLazy<T>& expression) {
  if (!expression.impl_) throw std::logic_error("empty expression");
  if (expression.size() == 0)
    throw std::logic_error("argmax of empty expression");
  return DpuFuture<ArgResult>(
      arg_reduce(expression.impl_->root, OP_ARGMAX_REDUCE));
}

DPULocalVector<T>::Reference::Reference(DPULocalVector& owner, DpuLazy<T> index)
    : owner_(&owner), index_(std::move(index)) {}

void DPULocalVector<T>::Reference::operator+=(T value) {
  owner_->add(index_, value);
}

void DPULocalVector<T>::Reference::operator+=(const DpuLazy<T>& value) {
  owner_->add(index_, value);
}

DPULocalVector<T>::DPULocalVector(uint32_t size, std::string_view name)
    : impl_(translate_oom([&] { return std::make_unique<Impl>(size, name); })) {
  std::lock_guard<std::mutex> lock(local_mutex);
  pending_locals.push_back(this);
}

DPULocalVector<T>::~DPULocalVector() {
  std::lock_guard<std::mutex> lock(local_mutex);
  pending_locals.erase(
      std::remove(pending_locals.begin(), pending_locals.end(), this),
      pending_locals.end());
}

DPULocalVector<T>::Reference DPULocalVector<T>::operator[](
    const DpuLazy<T>& index) {
  return Reference(*this, index);
}

void DPULocalVector<T>::add(const DpuLazy<T>& index, T value) {
  impl_->updates.push_back({index.impl_->root, scalar_node(value)});
}

void DPULocalVector<T>::add(const DpuLazy<T>& index, const DpuLazy<T>& value) {
  impl_->updates.push_back({index.impl_->root, value.impl_->root});
}

void DPULocalVector<T>::flush() {
  if (impl_->updates.empty()) return;
#if !JIT
  throw std::logic_error("local vector updates require JIT=1");
#else
  translate_oom([&] {
    Compiler compiler;
    std::vector<::dpu_expr<T>> indices;
    std::vector<::dpu_expr<T>> values;
    for (const auto& update : impl_->updates) {
      indices.emplace_back(compiler.expression(update.index));
      values.emplace_back(compiler.expression(update.value));
    }
    const auto& inputs = compiler.inputs();
    if (inputs.empty()) throw std::logic_error("local update has no DPU input");
    std::vector<BackendVector> operands(inputs.begin() + 1, inputs.end());
    ::dpu_jit_foreach<T>(
        const_cast<BackendVector&>(inputs[0]), operands, compiler.scalars(),
        [&](const std::vector<::dpu_expr<T>>&,
            ::dpu_pipeline_context<T>& context) {
          for (size_t i = 0; i < indices.size(); ++i) {
            context.local_sum(impl_->value, indices[i], values[i]);
          }
        });
    impl_->updates.clear();
  });
#endif
}

std::vector<T> DPULocalVector<T>::to_cpu() {
  flush();
  return translate_oom([&] { return impl_->value.to_cpu(); });
}

size_t RuntimeStatistics::total_launches() const {
  return compute_launches + dpu_transfers + host_transfers + fences;
}

size_t RuntimeStatistics::fused_away() const {
  return vertical_fusions + horizontal_fusions + absorbed_producers;
}

std::string RuntimeStatistics::to_string() const {
  return "compute=" + std::to_string(compute_launches) +
         " transfers=" + std::to_string(dpu_transfers + host_transfers) +
         " fusions=" + std::to_string(fused_away());
}

RuntimeStatistics operator-(const RuntimeStatistics& lhs,
                            const RuntimeStatistics& rhs) {
  RuntimeStatistics out;
#define SUB_STAT(name) out.name = lhs.name - rhs.name
  SUB_STAT(events_submitted);
  SUB_STAT(compute_launches);
  SUB_STAT(dpu_transfers);
  SUB_STAT(host_transfers);
  SUB_STAT(fences);
  SUB_STAT(vertical_fusions);
  SUB_STAT(horizontal_fusions);
  SUB_STAT(absorbed_producers);
  SUB_STAT(binary_switches);
  SUB_STAT(oom_retries);
  SUB_STAT(jit_kernel_compiles);
  SUB_STAT(jit_kernel_cache_hits);
  SUB_STAT(jit_batch_links);
  SUB_STAT(jit_batch_cache_hits);
  SUB_STAT(jit_pipeline_fallbacks);
  SUB_STAT(jit_eager_fallbacks);
#undef SUB_STAT
  return out;
}

RuntimeStatistics statistics() {
  return public_stats(RuntimeStats::get().snapshot());
}

void init(uint32_t dpus) {
  translate_oom([&] { DpuRuntime::get().init(dpus); });
}

uint32_t ndpus() {
  auto& runtime = DpuRuntime::get();
  return runtime.is_initialized() ? runtime.num_dpus()
                                  : DpuRuntime::configured_num_dpus();
}

uint32_t ntasklets() { return DpuRuntime::get().num_tasklets(); }

void sync() {
  std::vector<DPULocalVector<T>*> locals;
  {
    std::lock_guard<std::mutex> lock(local_mutex);
    locals = pending_locals;
  }
  for (auto* local : locals) local->flush();
  translate_oom([] { ::dpu_fence(); });
}

void fence(DPUVector<T>& vector) {
  if (vector.impl_) vector.impl_->storage().add_fence();
}

void shutdown() {
  if (!DpuRuntime::get().is_initialized()) return;
  sync();
  DpuRuntime::get().shutdown();
}

}  // namespace polymerpim
