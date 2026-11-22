// The DPU source generator.  See detail/jit_codegen.h.

#include <jit.h>

#if JIT
#include <common.h>
#include <detail/fusion.h>
#include <detail/jit_codegen.h>
#include <dlfcn.h>
#include <logger.h>
#include <opcodes.h>
#include <perfetto/trace.h>
#include <queue.h>
#include <runtime.h>
#include <stats.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

#include "vector.h"

namespace fs = std::filesystem;

namespace {
// One primary result slot + one per extra horizontal chain.
static constexpr int MAX_RESULT_SLOTS = MAX_HFUSE_CHAINS + 1;

// Substitutes $name placeholders in a DPU-C template.  `$` is used rather than
// braces because the emitted text is C, which is full of braces.
using Subst = std::initializer_list<std::pair<std::string_view, std::string>>;

std::string fill(std::string_view text, Subst vars) {
  std::string out;
  out.reserve(text.size() + 128);
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '$') {
      out += text[i];
      continue;
    }
    size_t start = i + 1;
    size_t end = start;
    while (end < text.size() &&
           (std::isalnum((unsigned char)text[end]) || text[end] == '_'))
      ++end;
    std::string_view name = text.substr(start, end - start);
    bool found = false;
    for (const auto& [key, value] : vars) {
      if (key != name) continue;
      out += value;
      found = true;
      break;
    }
    if (!found) {
      fprintf(stderr, "[JIT] unbound placeholder $%.*s in kernel template\n",
              (int)name.size(), name.data());
      abort();
    }
    i = end - 1;
  }
  return out;
}
// The C operator for a vector-vector arithmetic or comparison opcode.  The
// OP_*_SCALAR and OP_*_SCALAR_VAR forms share their counterpart's symbol.
const char* binary_op_symbol(uint8_t op) {
  switch (op) {
    case OP_ADD:
      return "+";
    case OP_SUB:
      return "-";
    case OP_MUL:
      return "*";
    case OP_DIV:
      return "/";
    case OP_ASR:
      return ">>";
    case OP_EQ:
      return "==";
    case OP_LT:
      return "<";
    case OP_GT:
      return ">";
    case OP_GE:
      return ">=";
    case OP_LE:
      return "<=";
    default:
      return nullptr;
  }
}

// `target <reduce> value` as one C statement.  Used for the per-element
// accumulator, the local-vector slots, and the cross-tasklet merge, which all
// fold the same four ways.
std::string fold_statement(uint8_t reduce_op, const std::string& target,
                           const std::string& value) {
  switch (reduce_op) {
    case OP_PRODUCT:
      return target + " *= " + value + ";";
    case OP_MIN:
      return "if (" + value + " < " + target + ") " + target + " = " + value +
             ";";
    case OP_MAX:
      return "if (" + value + " > " + target + ") " + target + " = " + value +
             ";";
    default:  // OP_SUM
      return target + " += " + value + ";";
  }
}
}  // namespace

namespace {

// One horizontal chain of the RPN program: a half-open opcode range that either
// writes a result vector or folds into a scalar accumulator.
struct Chain {
  size_t start_op = 0;
  size_t end_op = 0;
  bool is_reduction = false;
  uint8_t reduction_op = 0;
};

// What the emitter needs to know about an RPN program before writing any C:
// where the chains are, and which inputs, local vectors and scalar slots they
// touch (so only those are declared and only those are read from MRAM).
struct KernelPlan {
  std::vector<Chain> chains;
  bool uses_input = false;
  bool uses_op[MAX_VFUSE_INPUTS] = {false};
  bool uses_local[MAX_HFUSE_CHAINS] = {false};
  bool uses_scalar[MAX_PIPELINE_SCALARS] = {false};

  bool any_local() const {
    return std::any_of(std::begin(uses_local), std::end(uses_local),
                       [](bool x) { return x; });
  }
  bool any_reduction() const {
    return std::any_of(chains.begin(), chains.end(),
                       [](const Chain& c) { return c.is_reduction; });
  }
  bool any_promotable_reduction() const {
    return std::any_of(chains.begin(), chains.end(), [](const Chain& c) {
      return c.is_reduction &&
             (c.reduction_op == OP_SUM || c.reduction_op == OP_PRODUCT);
    });
  }
};

KernelPlan analyze_rpn(const std::vector<uint8_t>& rpn_ops) {
  KernelPlan plan;

  // Records one chain and the slots its opcodes reference.
  auto scan_chain = [&](size_t start, size_t end) {
    Chain chain;
    chain.start_op = start;
    chain.end_op = end;
    for (size_t i = start; i < end; ++i) {
      uint8_t op = rpn_ops[i];
      if (op == OP_PUSH_INPUT) {
        plan.uses_input = true;
      } else if (op >= OP_PUSH_OPERAND_0 &&
                 op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
        plan.uses_op[op - OP_PUSH_OPERAND_0] = true;
      } else if (IS_OP_INDIRECT_UPDATE(op)) {
        if (i + 1 < end) plan.uses_local[rpn_ops[i + 1]] = true;
        i += OP_INLINE_BYTES(op);
      } else if (op == OP_PUSH_SCALAR_VAR || IS_OP_SCALAR_VAR(op)) {
        if (i + 1 < end) plan.uses_scalar[rpn_ops[i + 1]] = true;
        i += OP_INLINE_BYTES(op);
      } else if (OP_INLINE_BYTES(op) > 0) {
        i += OP_INLINE_BYTES(op);
      } else if (IS_OP_REDUCTION(op)) {
        chain.is_reduction = true;
        chain.reduction_op = op;
      }
    }
    plan.chains.push_back(chain);
  };

  size_t chain_start = 0;
  for (size_t i = 0; i < rpn_ops.size(); ++i) {
    uint8_t op = rpn_ops[i];
    if (op == OP_NEXT_CHAIN) {
      scan_chain(chain_start, i);
      chain_start = i + 1;
    } else if (OP_INLINE_BYTES(op) > 0) {
      i += OP_INLINE_BYTES(op);
    }
  }
  scan_chain(chain_start, rpn_ops.size());
  return plan;
}

// Compiles one chain's RPN into C statements inside the per-element loop.
//
// Values live on a compile-time stack as C expressions.  Every emitted
// sub-expression is memoised under a signature built from its operands'
// identities, so a sub-expression that appears twice in a chain is computed
// once into a temporary and reused.
class ChainCompiler {
 public:
  ChainCompiler(std::ostream& out, size_t chain_index,
                const std::string& stack_type, const std::string& type_name)
      : out_(out),
        chain_index_(chain_index),
        stack_type_(stack_type),
        type_name_(type_name) {}

  // Emits the chain's statements.  Returns the expression holding the
  // per-element result, or an empty string for a reduction chain, which folds
  // into an accumulator instead of producing a value.
  std::string compile(const std::vector<uint8_t>& rpn, const Chain& chain) {
    for (size_t i = chain.start_op; i < chain.end_op; ++i) {
      uint8_t op = rpn[i];

      if (op == OP_PUSH_INPUT) {
        push(leaf("((" + stack_type_ + ")input_blk[i])", "input"));
      } else if (op >= OP_PUSH_OPERAND_0 &&
                 op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
        push_operand(op - OP_PUSH_OPERAND_0);
      } else if (IS_OP_SCALAR(op) || IS_OP_SCALAR_VAR(op)) {
        apply_scalar_op(rpn, i, op);
      } else if (op == OP_DUP) {
        push(stack_.back());
      } else if (IS_OP_UNARY(op)) {
        apply_unary(op);
      } else if (IS_OP_BINARY(op)) {
        apply_binary(op);
      } else if (IS_OP_TERNARY(op)) {
        apply_select(op);
      } else if (IS_OP_ARG_K(op)) {
        apply_arg_k(rpn, i, op);
      } else if (IS_OP_REDUCTION(op)) {
        apply_reduction(op);
      } else if (op == OP_PUSH_INDEX) {
        push(leaf("(blk + i)", "idx"));
      } else if (op == OP_PUSH_GLOBAL_INDEX) {
        push(leaf("(args.pipeline.index_base + blk + i)", "gidx"));
      } else if (op == OP_LOAD_INDIRECT) {
        apply_load_indirect(rpn, i);
      } else if (op == OP_ADD_INDIRECT || op == OP_APPLY_INDIRECT) {
        apply_indirect_update(rpn, i, op);
      } else if (op == OP_PUSH_SCALAR || op == OP_PUSH_SCALAR_VAR) {
        push_literal(rpn, i, op);
      }
    }

    if (chain.is_reduction || stack_.empty()) return {};
    return stack_.back().expr;
  }

 private:
  struct Value {
    std::string expr;  // the C expression
    std::string id;    // identity used to key the CSE table
  };

  static Value leaf(const std::string& expr, const std::string& id) {
    return Value{expr, id};
  }

  void push(const Value& v) { stack_.push_back(v); }

  Value pop() {
    Value v = stack_.back();
    stack_.pop_back();
    return v;
  }

  void require(size_t n, const char* what, uint8_t op) const {
    if (stack_.size() >= n) return;
    fprintf(stderr, "[JIT-DBG] STACK UNDERFLOW at %s op %u, stack size=%zu\n",
            what, (unsigned)op, stack_.size());
    abort();
  }

  // Emits `T tmp = expr;` unless an identical expression was already emitted
  // in this chain, in which case its temporary is reused.
  Value emit(const std::string& signature, const std::string& expr) {
    auto it = cse_.find(signature);
    if (it != cse_.end()) return it->second;

    std::string tmp =
        "t_" + std::to_string(chain_index_) + "_" + std::to_string(next_tmp_++);
    Value value{tmp, "e" + std::to_string(next_id_++)};
    out_ << "            " << stack_type_ << " " << tmp << " = " << expr
         << ";\n";
    cse_.emplace(signature, value);
    return value;
  }

  // Reads a 4-byte little-endian immediate and advances past it.
  int32_t read_immediate(const std::vector<uint8_t>& rpn, size_t& i) const {
    int32_t value =
        (int32_t)((uint32_t)rpn[i + 1] | ((uint32_t)rpn[i + 2] << 8) |
                  ((uint32_t)rpn[i + 3] << 16) | ((uint32_t)rpn[i + 4] << 24));
    i += SCALAR_INLINE_BYTES;
    return value;
  }

  void push_operand(uint8_t slot) {
    push(leaf("((" + stack_type_ + ")op_blks[" + std::to_string(slot) + "][i])",
              "op" + std::to_string(slot)));
  }

  // vector OP scalar.  The scalar is cast to the stack type, except for a
  // shift count, which stays an integer.
  std::string scalar_expr(uint8_t base, const std::string& lhs,
                          const std::string& rhs) const {
    if (base < OP_ADD_SCALAR || base > OP_LE_SCALAR) return "0";
    if (base == OP_ASR_SCALAR) return lhs + " >> " + rhs;
    const char* symbol = binary_op_symbol(base - OP_ADD_SCALAR + OP_ADD);
    return lhs + " " + symbol + " (" + stack_type_ + ")" + rhs;
  }

  void apply_scalar_op(const std::vector<uint8_t>& rpn, size_t& i, uint8_t op) {
    std::string rhs, rhs_id;
    if (IS_OP_SCALAR(op)) {
      rhs = std::to_string(read_immediate(rpn, i));
      rhs_id = "scalar:" + rhs;
    } else {
      uint8_t slot = rpn[i + 1];
      i += SCALAR_VAR_INDEX_BYTES;
      rhs = "scalar_vars[" + std::to_string(slot) + "]";
      rhs_id = "scalar_var:" + std::to_string(slot);
    }

    // Both scalar forms share the operator symbol of the OP_*_SCALAR opcode,
    // so normalise before looking it up.
    uint8_t base =
        IS_OP_SCALAR_VAR(op) ? (op - (OP_ADD_SCALAR_VAR - OP_ADD_SCALAR)) : op;
    Value lhs = pop();
    push(emit("scalar_op:" + std::to_string(base) + ":" + lhs.id + ":" + rhs_id,
              scalar_expr(base, lhs.expr, rhs)));
  }

  void apply_unary(uint8_t op) {
    Value v = pop();
    std::string expr =
        op == OP_NEGATE ? "-" + v.expr
                        : "(" + v.expr + " < 0) ? -" + v.expr + " : " + v.expr;
    push(emit("unary:" + std::to_string(op) + ":" + v.id, expr));
  }

  void apply_binary(uint8_t op) {
    require(2, "binary", op);
    Value rhs = pop();
    Value lhs = pop();
    const char* symbol = binary_op_symbol(op);
    std::string expr =
        symbol ? lhs.expr + " " + symbol + " " + rhs.expr : std::string("0");
    push(emit("binary:" + std::to_string(op) + ":" + lhs.id + ":" + rhs.id,
              expr));
  }

  void apply_select(uint8_t op) {
    Value else_val = pop();
    Value then_val = pop();
    Value cond = pop();
    if (op != OP_SELECT) return;
    push(emit(
        "select:" + cond.id + ":" + then_val.id + ":" + else_val.id,
        "(" + cond.expr + " != 0) ? " + then_val.expr + " : " + else_val.expr));
  }

  // Variadic horizontal argmin/argmax over k stacked lanes (lane 0 pushed
  // first).  Folds a running (best value, best index) with a strict
  // comparison, so ties keep the lowest lane index, and pushes the index.
  void apply_arg_k(const std::vector<uint8_t>& rpn, size_t& i, uint8_t op) {
    uint8_t k = rpn[++i];
    if (k == 0 || stack_.size() < k) {
      fprintf(stderr,
              "[JIT-DBG] STACK UNDERFLOW at arg op %u, k=%u, stack=%zu\n",
              (unsigned)op, (unsigned)k, stack_.size());
      abort();
    }

    std::vector<Value> lanes(stack_.end() - k, stack_.end());
    stack_.erase(stack_.end() - k, stack_.end());

    const char* cmp = (op == OP_ARGMIN_K) ? " < " : " > ";
    const std::string tag = std::to_string(op);
    Value best_value = lanes[0];
    Value best_index = leaf("0", "argidx0");
    for (uint8_t j = 1; j < k; ++j) {
      const std::string lane = std::to_string((int)j);
      Value wins =
          emit("argcmp:" + tag + ":" + lanes[j].id + ":" + best_value.id,
               "(" + lanes[j].expr + cmp + best_value.expr + ")");
      best_index = emit(
          "argidx:" + tag + ":" + lane + ":" + wins.id + ":" + best_index.id,
          wins.expr + " ? " + lane + " : " + best_index.expr);
      best_value =
          emit("argval:" + tag + ":" + lanes[j].id + ":" + wins.id + ":" +
                   best_value.id,
               wins.expr + " ? " + lanes[j].expr + " : " + best_value.expr);
    }
    push(best_index);
  }

  void apply_reduction(uint8_t op) {
    Value v = pop();
    if (op == OP_ARGMIN_REDUCE || op == OP_ARGMAX_REDUCE) {
      const std::string c = std::to_string(chain_index_);
      const std::string cmp = op == OP_ARGMAX_REDUCE ? ">" : "<";
      out_ << "            int32_t arg_v_" << c << " = (int32_t)(" << v.expr
           << ");\n"
           << "            uint32_t arg_i_" << c
           << " = args.pipeline.index_base + blk + i;\n"
           << "            if (arg_v_" << c << " " << cmp << " acc_" << c
           << ".value || (arg_v_" << c << " == acc_" << c << ".value && arg_i_"
           << c << " < acc_" << c << ".index)) "
           << "acc_" << c << " = (arg_result_t){arg_v_" << c << ", arg_i_" << c
           << "};\n";
      return;
    }
    out_ << "            "
         << fold_statement(op, "acc_" + std::to_string(chain_index_), v.expr)
         << "\n";
  }

  void apply_load_indirect(const std::vector<uint8_t>& rpn, size_t& i) {
    uint8_t slot = rpn[++i];
    Value index = pop();
    push(emit("load_indirect:" + std::to_string(slot) + ":" + index.id,
              "((__mram_ptr " + type_name_ +
                  " *)args.pipeline.binary_operands[" +
                  std::to_string((int)slot) + "])[" + index.expr + "]"));
  }

  // Scatter-update of a local vector slot; the reduction kind is implicit for
  // OP_ADD_INDIRECT and an inline byte for OP_APPLY_INDIRECT.
  void apply_indirect_update(const std::vector<uint8_t>& rpn, size_t& i,
                             uint8_t op) {
    uint8_t local_id = rpn[++i];
    uint8_t reduce_op = (op == OP_ADD_INDIRECT) ? OP_SUM : rpn[++i];
    Value value = pop();
    Value index = pop();
    std::string slot =
        "local_accum_" + std::to_string((int)local_id) + "[" + index.expr + "]";
    out_ << "            " << fold_statement(reduce_op, slot, value.expr)
         << "\n";
  }

  void push_literal(const std::vector<uint8_t>& rpn, size_t& i, uint8_t op) {
    if (op == OP_PUSH_SCALAR_VAR) {
      uint8_t slot = rpn[i + 1];
      i += SCALAR_VAR_INDEX_BYTES;
      push(leaf("scalar_vars[" + std::to_string((uint32_t)slot) + "]",
                "scalar_var:" + std::to_string((uint32_t)slot)));
    } else {
      int32_t value = read_immediate(rpn, i);
      push(leaf(std::to_string(value), "scalar:" + std::to_string(value)));
    }
  }

  std::ostream& out_;
  size_t chain_index_;
  std::string stack_type_;
  std::string type_name_;
  std::vector<Value> stack_;
  std::map<std::string, Value> cse_;
  int next_tmp_ = 0;
  int next_id_ = 0;
};
}  // namespace

// Widen the accumulator type when a reduction over int32 would otherwise
// overflow.  Only the on-stack type changes; MRAM still holds `type_name`.
// The single main() translation unit shared by every kernel in a JIT batch.
void detail::write_dpu_main_header(std::ostream& out) {
  out << R"(#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

__host DPU_LAUNCH_ARGS args;
BARRIER_INIT(my_barrier, NR_TASKLETS);
// Scratchpad for cross-tasklet reduction: one slot per tasklet per chain.
// Oversized for alignment safety; actual usage is MAX_RESULT_SLOTS * NR_TASKLETS.
uint64_t reduction_scratchpad[NR_TASKLETS * 16] __attribute__((aligned(8)));
__dma_aligned uint8_t dpu_workspace[NR_TASKLETS][TASKLET_WORKSPACE_SIZE];

)";
}

static std::string select_stack_type(const KernelPlan& plan,
                                     const std::string& type_name) {
#if ENABLE_PROMOTION_REDUCTIONS == 1
  if (plan.any_promotable_reduction() && type_name == "int32_t")
    return "int64_t";
#else
  (void)plan;
#endif
  return type_name;
}

// The identity value a reduction accumulator starts from.
static std::string reduction_identity(uint8_t reduce_op,
                                      const std::string& stack_type) {
  const bool is_float = stack_type == "float";
  switch (reduce_op) {
    case OP_PRODUCT:
      return "1";
    case OP_MIN:
      return is_float ? "3.402823466e+38f" : "INT32_MAX";
    case OP_MAX:
      return is_float ? "-3.402823466e+38f" : "INT32_MIN";
    default:  // OP_SUM
      return "0";
  }
}

// Includes, function header, per-chain result pointers, and the WRAM workspace.
static void write_kernel_prologue(std::ostream& out,
                                  const std::string& func_name,
                                  const std::string& type_name,
                                  const KernelPlan& plan) {
  // necessary to include these headers for the generated kernel code
  // each fused kernel is a separate compilation unit
  out << R"(#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "common.h"
extern barrier_t my_barrier;
extern uint64_t reduction_scratchpad[NR_TASKLETS * 16];

typedef struct {
    int32_t value;
    uint32_t index;
} arg_result_t;

)";

  // WRAM workspace layout, one slot of BLOCK_SIZE * MINIMUM_WRITE_SIZE bytes
  // each: slot 0 is the input block, the next MAX_VFUSE_INPUTS are the operand
  // blocks, and the MAX_RESULT_SLOTS after those are the per-chain result
  // blocks.  The layout is fixed, but the pointers this kernel sets up are not:
  // every index the body emits is a constant this plan knows, so declaring and
  // loading the slots it never touches would be dead stores in every kernel.
  // (The interpreter in dpu/pipeline.inl indexes these dynamically and does
  // need the full arrays.)
  static constexpr char kPrologueTemplate[] = R"C(int $fn(void) {
    unsigned int id = me();
    uint32_t n = args.num_elements;
    __mram_ptr $T *in_ptr = (__mram_ptr $T *)(args.pipeline.init_offset);

    __mram_ptr $T *res_ptrs[$slots];
    res_ptrs[0] = (__mram_ptr $T *)(args.pipeline.res_offset);
)C";

  // One result slot per chain, and operand slots up to the highest one the RPN
  // references (Julia and the fusion pass both allocate them densely, so this
  // is normally exactly the count in use).
  const int nresults = (int)plan.chains.size();
  int nop_slots = 0;
  for (int k = 0; k < MAX_VFUSE_INPUTS; ++k) {
    if (plan.uses_op[k]) nop_slots = k + 1;
  }
  assert(nresults > 0 && nresults <= MAX_RESULT_SLOTS);

  const std::string slots = std::to_string(nresults);
  out << fill(kPrologueTemplate,
              {{"fn", func_name}, {"T", type_name}, {"slots", slots}});
  for (int i = 0; i + 1 < nresults; ++i) {
    out << fill(
        "    res_ptrs[$n] = (__mram_ptr $T "
        "*)(args.pipeline.extra_res_offsets[$i]);\n",
        {{"n", std::to_string(i + 1)},
         {"T", type_name},
         {"i", std::to_string(i)}});
  }

  out << fill("\n    $T *input_blk = ($T *)dpu_workspace[id];\n",
              {{"T", type_name}});
  if (nop_slots > 0) {
    out << fill("    $T *op_blks[$n];\n",
                {{"T", type_name}, {"n", std::to_string(nop_slots)}});
    for (int k = 0; k < nop_slots; ++k) {
      if (!plan.uses_op[k]) continue;
      out << fill(
          "    op_blks[$k] = ($T *)&dpu_workspace[id][$slot * BLOCK_SIZE * "
          "MINIMUM_WRITE_SIZE];\n",
          {{"T", type_name},
           {"k", std::to_string(k)},
           {"slot", std::to_string(k + 1)}});
    }
  }
  out << fill("    $T *res_blks[$n];\n", {{"T", type_name}, {"n", slots}});
  for (int c = 0; c < nresults; ++c) {
    out << fill(
        "    res_blks[$c] = ($T *)&dpu_workspace[id][(MAX_VFUSE_INPUTS + $slot)"
        " * BLOCK_SIZE * MINIMUM_WRITE_SIZE];\n",
        {{"T", type_name},
         {"c", std::to_string(c)},
         {"slot", std::to_string(c + 1)}});
  }
  out << "\n";
}

// Everything the per-element loop needs in scope: local-vector pointers,
// reduction accumulators, the scalar table, and local accumulator seeding.
static void write_kernel_declarations(std::ostream& out, const KernelPlan& plan,
                                      const std::string& type_name,
                                      const std::string& stack_type) {
  const std::vector<Chain>& chains = plan.chains;
  for (int k = 0; k < MAX_HFUSE_CHAINS; ++k) {
    if (!plan.uses_local[k]) continue;
    out << "    uint32_t local_size_" << k << " = args.pipeline.local_sizes["
        << k << "];\n"
        << "    " << type_name << " *local_accum_" << k << " = (" << type_name
        << " *)&dpu_workspace[id][BASE_TASKLET_WORKSPACE_SIZE + " << k
        << " * LOCAL_VECTOR_WORKSPACE_BYTES];\n";
  }
  if (plan.any_local()) out << "\n";

  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    if (!chains[c_idx].is_reduction) continue;
    if (chains[c_idx].reduction_op == OP_ARGMIN_REDUCE ||
        chains[c_idx].reduction_op == OP_ARGMAX_REDUCE) {
      const char* identity = chains[c_idx].reduction_op == OP_ARGMAX_REDUCE
                                 ? "INT32_MIN"
                                 : "INT32_MAX";
      out << "    arg_result_t acc_" << c_idx << " = {" << identity
          << ", UINT32_MAX};\n";
    } else {
      out << "    " << stack_type << " acc_" << c_idx << " = "
          << reduction_identity(chains[c_idx].reduction_op, stack_type)
          << ";\n";
    }
  }

  int nscalars = 0;
  for (int k = 0; k < MAX_PIPELINE_SCALARS; ++k) {
    if (plan.uses_scalar[k]) nscalars = k + 1;
  }
  if (nscalars > 0) {
    out << "    " << stack_type << " scalar_vars[" << nscalars << "] = {0};\n";
    for (int k = 0; k < nscalars; ++k) {
      if (!plan.uses_scalar[k]) continue;
      // Scalars travel as uint32: convert through the element type, so a
      // negative scalar stays negative on a wider (promoted) stack.
      out << "    scalar_vars[" << k << "] = (" << stack_type << ")("
          << type_name << ")args.pipeline.scalars[" << k << "];\n";
    }
    out << "\n";
  }
  // Seed each local accumulator with the identity for its reduction.
  static constexpr char kLocalInitTemplate[] =
      R"C(    if (local_size_$c > MAX_LOCAL_VECTOR_SIZE)
        return -2;
    {
        uint32_t local_init_$c = local_size_$c;
        switch (args.pipeline.local_reduce_ops[$c]) {
            case OP_SUM:
                for (uint32_t j = 0; j < local_init_$c; ++j) local_accum_$c[j] = 0;
                break;
            case OP_PRODUCT:
                for (uint32_t j = 0; j < local_init_$c; ++j) local_accum_$c[j] = 1;
                break;
            case OP_MIN:
                for (uint32_t j = 0; j < local_init_$c; ++j) local_accum_$c[j] = INT32_MAX;
                break;
            case OP_MAX:
                for (uint32_t j = 0; j < local_init_$c; ++j) local_accum_$c[j] = INT32_MIN;
                break;
        }
    }
)C";

  for (int k = 0; k < MAX_HFUSE_CHAINS; ++k) {
    if (!plan.uses_local[k]) continue;
    out << fill(kLocalInitTemplate, {{"c", std::to_string(k)}});
  }
  if (plan.any_local()) out << "\n";
}

// The tiled main loop: read a block per input, compile every chain over its
// elements, then write each chain's block back.
static void write_kernel_block_loop(std::ostream& out, const KernelPlan& plan,
                                    const std::vector<uint8_t>& rpn_ops,
                                    const std::string& type_name,
                                    const std::string& stack_type) {
  const std::vector<Chain>& chains = plan.chains;
  // Main loop: each tasklet strides over the vector one BLOCK_SIZE tile at a
  // time, and the final tile of the last block may be short.
  static constexpr char kBlockLoopTemplate[] =
      R"C(    uint32_t blk, i, b_e, b_b, b_b_aligned;
    for (blk = id << BLOCK_SIZE_LOG2; blk < n; blk += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {
        b_e = (blk + BLOCK_SIZE >= n) ? (n - blk) : BLOCK_SIZE;
        b_b = b_e * sizeof($T);
        b_b_aligned = (b_b + 7) & ~7;

)C";
  out << fill(kBlockLoopTemplate, {{"T", type_name}});

  if (plan.uses_input)
    out << "        mram_read((__mram_ptr void const *)(in_ptr + blk), "
           "input_blk, b_b_aligned);\n";

  // An unbound operand slot is a null pointer, so the read is guarded.
  static constexpr char kOperandReadTemplate[] = R"C(        {
            __mram_ptr $T *p = (__mram_ptr $T *)(args.pipeline.binary_operands[$k]);
            if (p) mram_read((__mram_ptr void const *)(p + blk), op_blks[$k], b_b_aligned);
        }
)C";
  for (int k = 0; k < MAX_VFUSE_INPUTS; k++) {
    if (!plan.uses_op[k]) continue;
    out << fill(kOperandReadTemplate,
                {{"T", type_name}, {"k", std::to_string(k)}});
  }

  out << "        for (i = 0; i < b_e; i++) {\n";

  std::vector<bool> chain_has_output(chains.size(), false);
  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    const auto& chain = chains[c_idx];
    out << "            // Chain " << c_idx << "\n";

    ChainCompiler compiler(out, c_idx, stack_type, type_name);
    std::string result = compiler.compile(rpn_ops, chain);
    if (!result.empty()) {
      out << "            res_blks[" << c_idx << "][i] = " << result << ";\n";
      chain_has_output[c_idx] = true;
    }
  }  // c_idx

  out << "        }\n";  // end inner element loop

  // Write computed blocks back to MRAM for non-reduction chains.
  static constexpr char kResultWriteTemplate[] = R"C(        if (res_ptrs[$c])
            mram_write(res_blks[$c], (__mram_ptr void *)(res_ptrs[$c] + blk), b_b_aligned);
)C";

  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    if (chains[c_idx].is_reduction) continue;
    if (!chain_has_output[c_idx]) continue;
    out << fill(kResultWriteTemplate, {{"c", std::to_string(c_idx)}});
  }
  out << "    }\n";  // end block loop
}

// Cross-tasklet merges: scalar accumulators via the scratchpad, local vectors
// in WRAM, both folded by tasklet 0 and written to MRAM once.
static void write_kernel_epilogue(std::ostream& out, const KernelPlan& plan,
                                  const std::string& type_name,
                                  const std::string& stack_type) {
  const std::vector<Chain>& chains = plan.chains;
  // Cross-tasklet reduction for scalar reduction chains: each tasklet parks its
  // accumulator in the scratchpad, tasklet 0 folds the per-tasklet partials,
  // and one per-DPU result goes back to MRAM.  The memcpys move the value
  // through a uint64_t because the scratchpad is a uint64_t array.
  static constexpr char kParkAccumulatorTemplate[] = R"C(    {
        uint64_t bf_scratch_$c = 0;
        memcpy(&bf_scratch_$c, &acc_$c, sizeof($T));
        reduction_scratchpad[id * 16 + $c] = bf_scratch_$c;
    }
)C";

  static constexpr char kMergeAccumulatorTemplate[] =
      R"C(        if (res_ptrs[$c]) {
            $T tot_$c;
            memcpy(&tot_$c, &reduction_scratchpad[$c], sizeof($T));
            for (uint32_t t = 1; t < NR_TASKLETS; ++t) {
                $T v_$c;
                memcpy(&v_$c, &reduction_scratchpad[t * 16 + $c], sizeof($T));
                $fold
            }
            uint64_t bf_final_$c = 0;
            memcpy(&bf_final_$c, &tot_$c, sizeof($T));
            mram_write(&bf_final_$c, (__mram_ptr void *)res_ptrs[$c], MINIMUM_WRITE_SIZE);
        }
)C";

  bool has_reduction_chain = false;
  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    if (!chains[c_idx].is_reduction) continue;
    has_reduction_chain = true;
    const bool is_arg = chains[c_idx].reduction_op == OP_ARGMIN_REDUCE ||
                        chains[c_idx].reduction_op == OP_ARGMAX_REDUCE;
    out << fill(kParkAccumulatorTemplate,
                {{"T", is_arg ? "arg_result_t" : stack_type},
                 {"c", std::to_string(c_idx)}});
  }

  if (has_reduction_chain) {
    out << "    barrier_wait(&my_barrier);\n"
        << "    if (id == 0) {\n";
    for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
      if (!chains[c_idx].is_reduction) continue;
      const std::string c = std::to_string(c_idx);
      const uint8_t op = chains[c_idx].reduction_op;
      if (op == OP_ARGMIN_REDUCE || op == OP_ARGMAX_REDUCE) {
        const std::string cmp = op == OP_ARGMAX_REDUCE ? ">" : "<";
        out << fill(R"C(        if (res_ptrs[$c]) {
            arg_result_t tot_$c;
            memcpy(&tot_$c, &reduction_scratchpad[$c], sizeof(tot_$c));
            for (uint32_t t = 1; t < NR_TASKLETS; ++t) {
                arg_result_t v_$c;
                memcpy(&v_$c, &reduction_scratchpad[t * 16 + $c], sizeof(v_$c));
                if (v_$c.value $cmp tot_$c.value ||
                    (v_$c.value == tot_$c.value && v_$c.index < tot_$c.index))
                    tot_$c = v_$c;
            }
            mram_write(&tot_$c, (__mram_ptr void *)res_ptrs[$c], MINIMUM_WRITE_SIZE);
        }
)C",
                    {{"c", c}, {"cmp", cmp}});
      } else {
        out << fill(kMergeAccumulatorTemplate,
                    {{"T", stack_type},
                     {"c", c},
                     {"fold", fold_statement(op, "tot_" + c, "v_" + c)}});
      }
    }
    out << "    }\n"
        << "    barrier_wait(&my_barrier);\n";
  }

  // Cross-tasklet reduction for local vectors: tasklet 0 merges every other
  // tasklet's WRAM shard into its own, then writes the combined vector to MRAM
  // once.
  static constexpr char kLocalMergeTemplate[] = R"C(    {
        barrier_wait(&my_barrier);
        if (id == 0) {
            __mram_ptr $T *local_ptr = (__mram_ptr $T *)(args.pipeline.extra_res_offsets[$c]);
            if (local_ptr) {
                for (uint32_t t = 1; t < NR_TASKLETS; ++t) {
                    $T *src = ($T *)&dpu_workspace[t][BASE_TASKLET_WORKSPACE_SIZE + $c * LOCAL_VECTOR_WORKSPACE_BYTES];
                    for (uint32_t j = 0; j < local_size_$c; ++j) {
                        switch (args.pipeline.local_reduce_ops[$c]) {
                            case OP_SUM:
                                local_accum_$c[j] += src[j];
                                break;
                            case OP_PRODUCT:
                                local_accum_$c[j] *= src[j];
                                break;
                            case OP_MIN:
                                if (src[j] < local_accum_$c[j]) local_accum_$c[j] = src[j];
                                break;
                            case OP_MAX:
                                if (src[j] > local_accum_$c[j]) local_accum_$c[j] = src[j];
                                break;
                        }
                    }
                }
                uint32_t local_bytes = local_size_$c * sizeof($T);
                uint32_t local_bytes_aligned = (local_bytes + 7) & ~7;
                mram_write(local_accum_$c, (__mram_ptr void *)local_ptr, local_bytes_aligned);
            }
        }
        barrier_wait(&my_barrier);
    }
)C";

  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    if (!plan.uses_local[c_idx]) continue;
    out << fill(kLocalMergeTemplate,
                {{"T", type_name}, {"c", std::to_string(c_idx)}});
  }

  out << "    return 0;\n}\n\n";
}

void detail::write_kernel_function(std::ostream& out,
                                   const std::string& func_name,
                                   const std::vector<uint8_t>& rpn_ops,
                                   const std::string& type_name) {
  const KernelPlan plan = analyze_rpn(rpn_ops);
  const std::string stack_type = select_stack_type(plan, type_name);

  write_kernel_prologue(out, func_name, type_name, plan);
  write_kernel_declarations(out, plan, type_name, stack_type);
  write_kernel_block_loop(out, plan, rpn_ops, type_name, stack_type);
  write_kernel_epilogue(out, plan, type_name, stack_type);
}

#endif  // JIT
