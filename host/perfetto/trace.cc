#include "perfetto/trace.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#include "perfetto/detail/trace.h"
#include "runtime.h"

std::string operationtype_to_string(Event::OperationType op) {
  switch (op) {
    case Event::OperationType::COMPUTE:
      return "COMPUTE";
    case Event::OperationType::DPU_TRANSFER:
      return "DPU_TRANSFER";
    case Event::OperationType::HOST_TRANSFER:
      return "HOST_TRANSFER";
    case Event::OperationType::FENCE:
      return "FENCE";
    default:
      return "UNKNOWN";
  }
}

std::string opcode_to_string(uint8_t op) {
  switch (op) {
    case OP_IDENTITY:
      return "IDENTITY";
    case OP_NEGATE:
      return "NEGATE";
    case OP_ABS:
      return "ABS";
    case OP_ADD:
      return "ADD";
    case OP_SUB:
      return "SUB";
    case OP_MUL:
      return "MUL";
    case OP_DIV:
      return "DIV";
    case OP_ASR:
      return "ASR";
    case OP_EQ:
      return "EQ";
    case OP_LT:
      return "LT";
    case OP_GT:
      return "GT";
    case OP_GE:
      return "GE";
    case OP_LE:
      return "LE";
    case OP_ADD_SCALAR:
      return "ADD_SCALAR";
    case OP_SUB_SCALAR:
      return "SUB_SCALAR";
    case OP_MUL_SCALAR:
      return "MUL_SCALAR";
    case OP_DIV_SCALAR:
      return "DIV_SCALAR";
    case OP_ASR_SCALAR:
      return "ASR_SCALAR";
    case OP_EQ_SCALAR:
      return "EQ_SCALAR";
    case OP_LT_SCALAR:
      return "LT_SCALAR";
    case OP_GT_SCALAR:
      return "GT_SCALAR";
    case OP_GE_SCALAR:
      return "GE_SCALAR";
    case OP_LE_SCALAR:
      return "LE_SCALAR";
    case OP_ADD_SCALAR_VAR:
      return "ADD_SCALAR_VAR";
    case OP_SUB_SCALAR_VAR:
      return "SUB_SCALAR_VAR";
    case OP_MUL_SCALAR_VAR:
      return "MUL_SCALAR_VAR";
    case OP_DIV_SCALAR_VAR:
      return "DIV_SCALAR_VAR";
    case OP_ASR_SCALAR_VAR:
      return "ASR_SCALAR_VAR";
    case OP_EQ_SCALAR_VAR:
      return "EQ_SCALAR_VAR";
    case OP_LT_SCALAR_VAR:
      return "LT_SCALAR_VAR";
    case OP_GT_SCALAR_VAR:
      return "GT_SCALAR_VAR";
    case OP_GE_SCALAR_VAR:
      return "GE_SCALAR_VAR";
    case OP_LE_SCALAR_VAR:
      return "LE_SCALAR_VAR";
    case OP_MIN:
      return "MIN";
    case OP_MAX:
      return "MAX";
    case OP_SUM:
      return "SUM";
    case OP_PRODUCT:
      return "PRODUCT";
    case OP_ARGMIN_REDUCE:
      return "ARGMIN_REDUCE";
    case OP_ARGMAX_REDUCE:
      return "ARGMAX_REDUCE";
    case OP_SELECT:
      return "SELECT";
    case OP_DUP:
      return "DUP";
    case OP_PUSH_INDEX:
      return "PUSH_INDEX";
    case OP_PUSH_GLOBAL_INDEX:
      return "PUSH_GLOBAL_INDEX";
    case OP_LOAD_INDIRECT:
      return "LOAD_INDIRECT";
    case OP_ADD_INDIRECT:
      return "ADD_INDIRECT";
    case OP_APPLY_INDIRECT:
      return "APPLY_INDIRECT";
    case OP_PUSH_SCALAR:
      return "PUSH_SCALAR";
    case OP_PUSH_SCALAR_VAR:
      return "PUSH_SCALAR_VAR";
    case OP_NEXT_CHAIN:
      return "NEXT_CHAIN";
    case OP_PUSH_INPUT:
    case OP_PUSH_OPERAND_0:
    case OP_PUSH_OPERAND_1:
    case OP_PUSH_OPERAND_2:
    case OP_PUSH_OPERAND_3:
    case OP_PUSH_OPERAND_4:
    case OP_PUSH_OPERAND_5:
    case OP_PUSH_OPERAND_6:
    case OP_PUSH_OPERAND_7:
    case OP_PUSH_OPERAND_8:
    case OP_PUSH_OPERAND_9:
    case OP_PUSH_OPERAND_10:
    case OP_PUSH_OPERAND_11:
    case OP_PUSH_OPERAND_12:
    case OP_PUSH_OPERAND_13:
    case OP_PUSH_OPERAND_14:
    case OP_PUSH_OPERAND_15:
      return "";
    default:
      return "UNK(" + std::to_string(op) + ")";
  }
}

#if TRACE == 1 && __has_include(<perfetto.h>)
#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("runtime").SetDescription(
        "Events related to runtime init and shutdown"),
    perfetto::Category("queue").SetDescription(
        "Events related to the event queue"),
    perfetto::Category("transfer")
        .SetDescription("Events related to MRAM transfers"),
    perfetto::Category("events").SetDescription(
        "Actual operation execution events"));

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

static std::unique_ptr<perfetto::TracingSession> tracing_session_;
static bool tracing_enabled_ = false;
static bool concurrent_ranks_ = false;

static bool truthy_env(const char* value) {
  if (value == nullptr || value[0] == '\0') return false;
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "FALSE") != 0 && std::strcmp(value, "off") != 0 &&
         std::strcmp(value, "OFF") != 0;
}

bool trace::enabled() { return tracing_enabled_; }

static bool requested() {
  return truthy_env(std::getenv("POLYMERPIM_TRACE")) ||
         truthy_env(std::getenv("TRACE_OUTPUT"));
}

static std::string output_path() {
  const char* polymerpim_trace = std::getenv("POLYMERPIM_TRACE");
  if (truthy_env(polymerpim_trace) && std::strcmp(polymerpim_trace, "1") != 0 &&
      std::strcmp(polymerpim_trace, "true") != 0 &&
      std::strcmp(polymerpim_trace, "TRUE") != 0 &&
      std::strcmp(polymerpim_trace, "on") != 0 &&
      std::strcmp(polymerpim_trace, "ON") != 0) {
    return polymerpim_trace;
  }
  const char* trace_output = std::getenv("TRACE_OUTPUT");
  return truthy_env(trace_output) ? trace_output : "trace.perfetto-trace";
}

static std::string trace_string(std::string value, size_t max_bytes = 8192) {
  if (value.size() <= max_bytes) return value;
  value.resize(max_bytes);
  value += "\n... truncated ...";
  return value;
}

static size_t count_rpn_chains(const std::vector<uint8_t>& ops) {
  if (ops.empty()) return 0;
  size_t chains = 1;
  for (size_t i = 0; i < ops.size(); ++i) {
    uint8_t op = ops[i];
    if (op == OP_NEXT_CHAIN) chains++;
    if (OP_INLINE_BYTES(op) > 0) i += OP_INLINE_BYTES(op);
  }
  return chains;
}

static bool is_stack_push_op(uint8_t op) {
  return op >= OP_PUSH_INPUT && op <= OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS - 1;
}

static std::vector<std::string> rpn_tokens(const std::vector<uint8_t>& ops,
                                           bool include_stack_pushes) {
  std::vector<std::string> tokens;
  for (size_t i = 0; i < ops.size(); ++i) {
    uint8_t op = ops[i];
    if (op == OP_NEXT_CHAIN) {
      if (include_stack_pushes) tokens.push_back("NEXT_CHAIN");
    } else if (!include_stack_pushes &&
               (is_stack_push_op(op) || op == OP_PUSH_SCALAR ||
                op == OP_PUSH_SCALAR_VAR)) {
      // Operand pushes are VM plumbing, not useful when reading a trace.
    } else {
      std::string name = opcode_to_string(op);
      if (!name.empty()) tokens.push_back(name);
    }
    if (OP_INLINE_BYTES(op) > 0) i += OP_INLINE_BYTES(op);
  }
  return tokens;
}

static size_t count_decoded_rpn_ops(const std::vector<uint8_t>& ops) {
  size_t count = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    count++;
    if (OP_INLINE_BYTES(ops[i]) > 0) i += OP_INLINE_BYTES(ops[i]);
  }
  return count;
}

static std::string summarize_top_counts(const std::vector<std::string>& tokens,
                                        size_t limit) {
  std::map<std::string, size_t> counts;
  for (const auto& token : tokens) counts[token]++;

  std::vector<std::pair<std::string, size_t>> sorted(counts.begin(),
                                                     counts.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second > b.second;
    return a.first < b.first;
  });

  std::ostringstream out;
  for (size_t i = 0; i < sorted.size() && i < limit; ++i) {
    if (i) out << ", ";
    out << sorted[i].first << "=" << sorted[i].second;
  }
  return out.str();
}

static std::string summarize_repeated_windows(
    const std::vector<std::string>& tokens, size_t limit) {
  std::map<std::string, size_t> windows;
  for (size_t width : {size_t{6}, size_t{5}, size_t{4}, size_t{3}}) {
    if (tokens.size() < width) continue;
    for (size_t i = 0; i + width <= tokens.size(); ++i) {
      std::string key;
      for (size_t j = 0; j < width; ++j) {
        if (j) key += ", ";
        key += tokens[i + j];
      }
      windows[key]++;
    }
  }

  std::vector<std::pair<std::string, size_t>> sorted;
  for (const auto& entry : windows)
    if (entry.second > 1) sorted.push_back(entry);
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second > b.second;
    if (a.first.size() != b.first.size())
      return a.first.size() > b.first.size();
    return a.first < b.first;
  });

  std::ostringstream out;
  for (size_t i = 0; i < sorted.size() && i < limit; ++i) {
    out << "  [" << sorted[i].first << "] x" << sorted[i].second << "\n";
  }
  return out.str();
}

static size_t estimate_jit_unique_exprs(const std::vector<uint8_t>& ops) {
  std::vector<std::string> stack;
  std::map<std::string, std::string> ids;
  size_t next_id = 0;

  auto leaf = [](const std::string& id) { return id; };
  auto make_expr = [&](const std::string& sig) {
    auto it = ids.find(sig);
    if (it != ids.end()) return it->second;
    std::string id = "e" + std::to_string(next_id++);
    ids.emplace(sig, id);
    return id;
  };

  for (size_t i = 0; i < ops.size(); ++i) {
    uint8_t op = ops[i];
    if (op == OP_NEXT_CHAIN) {
      stack.clear();
    } else if (op == OP_PUSH_INPUT) {
      stack.push_back(leaf("input"));
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op <= OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS - 1) {
      stack.push_back(leaf("op" + std::to_string(op - OP_PUSH_OPERAND_0)));
    } else if (op == OP_DUP) {
      if (!stack.empty()) stack.push_back(stack.back());
    } else if (IS_OP_SCALAR(op) || IS_OP_SCALAR_VAR(op)) {
      std::string rhs_id;
      uint8_t base = op;
      if (IS_OP_SCALAR(op)) {
        uint8_t b0 = ops[i + 1], b1 = ops[i + 2], b2 = ops[i + 3],
                b3 = ops[i + 4];
        int32_t val = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        rhs_id = "scalar:" + std::to_string(val);
        i += SCALAR_INLINE_BYTES;
      } else {
        uint8_t idx = ops[i + 1];
        rhs_id = "scalar_var:" + std::to_string(idx);
        base = op - (OP_ADD_SCALAR_VAR - OP_ADD_SCALAR);
        i += SCALAR_VAR_INDEX_BYTES;
      }
      if (stack.empty()) continue;
      std::string lhs = stack.back();
      stack.pop_back();
      stack.push_back(make_expr("scalar_op:" + std::to_string(base) + ":" +
                                lhs + ":" + rhs_id));
    } else if (IS_OP_UNARY(op)) {
      if (stack.empty()) continue;
      std::string v = stack.back();
      stack.pop_back();
      stack.push_back(make_expr("unary:" + std::to_string(op) + ":" + v));
    } else if (IS_OP_BINARY(op)) {
      if (stack.size() < 2) continue;
      std::string rhs = stack.back();
      stack.pop_back();
      std::string lhs = stack.back();
      stack.pop_back();
      stack.push_back(
          make_expr("binary:" + std::to_string(op) + ":" + lhs + ":" + rhs));
    } else if (IS_OP_TERNARY(op)) {
      if (stack.size() < 3) continue;
      std::string false_val = stack.back();
      stack.pop_back();
      std::string true_val = stack.back();
      stack.pop_back();
      std::string cond = stack.back();
      stack.pop_back();
      stack.push_back(
          make_expr("select:" + cond + ":" + true_val + ":" + false_val));
    } else if (IS_OP_REDUCTION(op)) {
      if (!stack.empty()) stack.pop_back();
    } else if (op == OP_PUSH_INDEX) {
      stack.push_back(leaf("idx"));
    } else if (op == OP_PUSH_GLOBAL_INDEX) {
      stack.push_back(leaf("gidx"));
    } else if (op == OP_LOAD_INDIRECT) {
      if (stack.empty()) {
        i += 1;
        continue;
      }
      uint8_t operand = ops[++i];
      std::string idx = stack.back();
      stack.pop_back();
      stack.push_back(
          make_expr("load_indirect:" + std::to_string(operand) + ":" + idx));
    } else if (op == OP_ADD_INDIRECT || op == OP_APPLY_INDIRECT) {
      size_t extra = (op == OP_APPLY_INDIRECT) ? 2 : 1;
      if (i + extra < ops.size()) i += extra;
      if (!stack.empty()) stack.pop_back();
      if (!stack.empty()) stack.pop_back();
    } else if (op == OP_PUSH_SCALAR || op == OP_PUSH_SCALAR_VAR) {
      if (op == OP_PUSH_SCALAR_VAR) {
        uint8_t idx = ops[i + 1];
        stack.push_back(leaf("scalar_var:" + std::to_string(idx)));
        i += SCALAR_VAR_INDEX_BYTES;
      } else {
        uint8_t b0 = ops[i + 1], b1 = ops[i + 2], b2 = ops[i + 3],
                b3 = ops[i + 4];
        int32_t val = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        stack.push_back(leaf("scalar:" + std::to_string(val)));
        i += SCALAR_INLINE_BYTES;
      }
    }
  }
  return ids.size();
}

static std::string summarize_rpn_ops(const std::vector<uint8_t>& ops) {
  std::vector<std::string> tokens = rpn_tokens(ops, false);
  size_t decoded_ops = count_decoded_rpn_ops(ops);
  size_t jit_unique_exprs = estimate_jit_unique_exprs(ops);
  size_t chains = count_rpn_chains(ops);
  size_t reductions = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    uint8_t op = ops[i];
    if (IS_OP_REDUCTION(op)) reductions++;
    if (OP_INLINE_BYTES(op) > 0) i += OP_INLINE_BYTES(op);
  }

  std::ostringstream out;
  out << "decoded_ops=" << decoded_ops << ", rpn_bytes=" << ops.size()
      << ", jit_unique_exprs=" << jit_unique_exprs << ", chains=" << chains
      << ", reductions=" << reductions << "\n";

  std::string histogram = summarize_top_counts(tokens, 12);
  if (!histogram.empty()) out << "top_ops: " << histogram << "\n";

  if (chains > 1) {
    size_t chain_idx = 0;
    size_t chain_ops = 0;
    size_t chain_reductions = 0;
    out << "chains:\n";
    for (size_t i = 0; i < ops.size(); ++i) {
      uint8_t op = ops[i];
      if (op == OP_NEXT_CHAIN) {
        out << "  #" << chain_idx++ << ": ops=" << chain_ops
            << ", reductions=" << chain_reductions << "\n";
        chain_ops = 0;
        chain_reductions = 0;
      } else {
        chain_ops++;
        if (IS_OP_REDUCTION(op)) chain_reductions++;
      }
      if (OP_INLINE_BYTES(op) > 0) i += OP_INLINE_BYTES(op);
    }
    out << "  #" << chain_idx << ": ops=" << chain_ops
        << ", reductions=" << chain_reductions << "\n";
  }

  std::string repeated = summarize_repeated_windows(tokens, 6);
  if (!repeated.empty()) out << "repeated_patterns:\n" << repeated;
  return trace_string(out.str(), 8192);
}

static std::string vector_to_string(detail::VectorDescRef vec) {
  if (!vec) return "NULL";
  char buf[128];
  uint32_t addr = (vec->desc.empty() ? 0 : vec->desc[0].ptr);
  snprintf(buf, sizeof(buf), "[ptr=0x%x, size=%zu, elems=%zu]", addr,
           (size_t)vec->num_elements * vec->element_size, vec->num_elements);
  return std::string(buf);
}

static std::string get_pipeline_breakdown(const Event& e) {
  if (e.rpn_ops.empty()) return "";
  if (count_decoded_rpn_ops(e.rpn_ops) > 256)
    return summarize_rpn_ops(e.rpn_ops);

  std::string breakdown;
  std::vector<std::string> stack;
  int op_idx = 1;
  const uint8_t* ops = e.rpn_ops.data();
  size_t size = e.rpn_ops.size();

  for (size_t i = 0; i < size; ++i) {
    uint8_t op = ops[i];
    if (op == OP_PUSH_INPUT) {
      stack.push_back("In[0]");
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op <= OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS - 1) {
      stack.push_back("In[" + std::to_string(op - OP_PUSH_OPERAND_0 + 1) + "]");
    } else if (op == OP_DUP) {
      if (stack.empty()) {
        breakdown += "!!DUP_ERR!!\n";
        break;
      }
      stack.push_back(stack.back());
    } else if (IS_OP_UNARY(op)) {
      if (stack.size() < 1) {
        breakdown += "!!STK_ERR!!\n";
        break;
      }
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = "st[" + std::to_string(stack.size()) + "]";
      breakdown += std::to_string(op_idx++) + ". " + res + " = " +
                   opcode_to_string(op) + "(" + s1 + ")\n";
      stack.push_back(res);
    } else if (IS_OP_BINARY(op)) {
      if (stack.size() < 2) {
        breakdown += "!!STK_ERR!!\n";
        break;
      }
      std::string s2 = stack.back();
      stack.pop_back();
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = "st[" + std::to_string(stack.size()) + "]";
      breakdown += std::to_string(op_idx++) + ". " + res + " = " +
                   opcode_to_string(op) + "(" + s1 + ", " + s2 + ")\n";
      stack.push_back(res);
    } else if (IS_OP_TERNARY(op)) {
      if (stack.size() < 3) {
        breakdown += "!!STK_ERR!!\n";
        break;
      }
      std::string false_val = stack.back();
      stack.pop_back();
      std::string true_val = stack.back();
      stack.pop_back();
      std::string cond = stack.back();
      stack.pop_back();
      std::string res = "st[" + std::to_string(stack.size()) + "]";
      breakdown += std::to_string(op_idx++) + ". " + res + " = SELECT(" + cond +
                   ", " + true_val + ", " + false_val + ")\n";
      stack.push_back(res);
    } else if (IS_OP_SCALAR(op)) {
      if (stack.size() < 1) {
        breakdown += "!!STK_ERR!!\n";
        break;
      }
      if (i + sizeof(uint32_t) >= size) {
        breakdown += "!!SCALAR_ERR!!\n";
        break;
      }
      uint32_t scalar;
      memcpy(&scalar, &ops[i + 1], sizeof(uint32_t));
      i += sizeof(uint32_t);
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = "st[" + std::to_string(stack.size()) + "]";
      breakdown += std::to_string(op_idx++) + ". " + res + " = " +
                   opcode_to_string(op) + "(" + s1 + ", " +
                   std::to_string(scalar) + ")\n";
      stack.push_back(res);
    } else if (IS_OP_SCALAR_VAR(op)) {
      if (stack.size() < 1) {
        breakdown += "!!STK_ERR!!\n";
        break;
      }
      if (i + 1 >= size) {
        breakdown += "!!SCALAR_ERR!!\n";
        break;
      }
      uint8_t scalar_idx = ops[i + 1];
      i += 1;
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = "st[" + std::to_string(stack.size()) + "]";
      breakdown += std::to_string(op_idx++) + ". " + res + " = " +
                   opcode_to_string(op) + "(" + s1 + ", VAR[" +
                   std::to_string(scalar_idx) + "])\n";
      stack.push_back(res);
    } else if (IS_OP_REDUCTION(op)) {
      if (stack.size() < 1) {
        breakdown += "!!STK_ERR!!\n";
        break;
      }
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = "RED_RES";
      breakdown += std::to_string(op_idx++) + ". " + res + " = " +
                   opcode_to_string(op) + "(" + s1 + ")\n";
      stack.push_back(res);
    } else if (op == OP_PUSH_INDEX) {
      stack.push_back("IDX");
    } else if (op == OP_PUSH_GLOBAL_INDEX) {
      stack.push_back("GIDX");
    } else if (op == OP_PUSH_SCALAR || op == OP_PUSH_SCALAR_VAR) {
      if (op == OP_PUSH_SCALAR_VAR) {
        if (i + 1 >= size) {
          breakdown += "!!SCALAR_VAR_ERR!!\n";
          break;
        }
        uint8_t idx = ops[++i];
        stack.push_back("SCALAR[" + std::to_string(idx) + "]");
      } else {
        if (i + sizeof(uint32_t) >= size) {
          breakdown += "!!SCALAR_ERR!!\n";
          break;
        }
        uint32_t scalar;
        memcpy(&scalar, &ops[i + 1], sizeof(uint32_t));
        i += sizeof(uint32_t);
        stack.push_back(std::to_string(scalar));
      }
    } else if (op == OP_LOAD_INDIRECT) {
      if (stack.size() < 1 || i + 1 >= size) {
        breakdown += "!!INDIRECT_ERR!!\n";
        break;
      }
      uint8_t operand_idx = ops[++i];
      std::string idx = stack.back();
      stack.pop_back();
      std::string res = "st[" + std::to_string(stack.size()) + "]";
      breakdown += std::to_string(op_idx++) + ". " + res + " = " +
                   opcode_to_string(op) + "(In[" +
                   std::to_string(operand_idx + 1) + "], " + idx + ")\n";
      stack.push_back(res);
    } else if (op == OP_ADD_INDIRECT || op == OP_APPLY_INDIRECT) {
      size_t extra = (op == OP_APPLY_INDIRECT) ? 2 : 1;
      if (stack.size() < 2 || i + extra >= size) {
        breakdown += "!!INDIRECT_ERR!!\n";
        break;
      }
      uint8_t local_idx = ops[++i];
      uint8_t reduce_op = (op == OP_APPLY_INDIRECT) ? ops[++i] : OP_SUM;
      std::string val = stack.back();
      stack.pop_back();
      std::string idx = stack.back();
      stack.pop_back();
      breakdown += std::to_string(op_idx++) + ". LOCAL[" +
                   std::to_string(local_idx) + "][" + idx +
                   "] = " + opcode_to_string(reduce_op) + "(LOCAL[" +
                   std::to_string(local_idx) + "][" + idx + "], " + val + ")\n";
    }
  }
  if (!stack.empty()) breakdown += "Final Output: " + stack.back();
  return trace_string(std::move(breakdown), 16384);
}

static void add_event_metadata(perfetto::EventContext& ctx,
                               std::shared_ptr<Event> e) {
  if (e->op == Event::OperationType::COMPUTE) {
    for (size_t i = 0; i < e->inputs.size(); ++i) {
      ctx.AddDebugAnnotation(
          perfetto::DynamicString("in[" + std::to_string(i) + "]"),
          vector_to_string(e->inputs[i]));
    }
    if (e->output) ctx.AddDebugAnnotation("out", vector_to_string(e->output));
    if (!e->rpn_ops.empty()) {
      ctx.AddDebugAnnotation("rpn_ops_count", (uint64_t)e->rpn_ops.size());
      ctx.AddDebugAnnotation("rpn_chain_count",
                             (uint64_t)count_rpn_chains(e->rpn_ops));
    }
    std::string breakdown = get_pipeline_breakdown(*e);
    if (!breakdown.empty())
      ctx.AddDebugAnnotation("pipeline_breakdown", breakdown);
  } else if (e->op == Event::OperationType::DPU_TRANSFER ||
             e->op == Event::OperationType::HOST_TRANSFER) {
    if (e->host_ptr) {
      char buf[64];
      snprintf(buf, sizeof(buf), "0x%p", e->host_ptr);
      ctx.AddDebugAnnotation("host_buffer", std::string(buf));
    }
    if (e->transfer_size > 0)
      ctx.AddDebugAnnotation("size_bytes", (uint64_t)e->transfer_size);
    if (e->op == Event::OperationType::DPU_TRANSFER) {
      ctx.AddDebugAnnotation("direction", "Host -> DPU");
      if (e->output)
        ctx.AddDebugAnnotation("dpu_dest", vector_to_string(e->output));
    } else {
      ctx.AddDebugAnnotation("direction", "DPU -> Host");
      if (!e->inputs.empty() && e->inputs[0])
        ctx.AddDebugAnnotation("dpu_src", vector_to_string(e->inputs[0]));
    }
  }
}

namespace trace {

void initialize() {
  tracing_enabled_ = requested();
  if (!tracing_enabled_) return;
  if (tracing_session_) return;

  perfetto::TracingInitArgs args;
  args.backends |= perfetto::kInProcessBackend;
  perfetto::Tracing::Initialize(args);
  perfetto::TrackEvent::Register();

  perfetto::TraceConfig cfg;
  cfg.add_buffers()->set_size_kb(64 * 1024);
  auto* ds_cfg = cfg.add_data_sources()->mutable_config();
  ds_cfg->set_name("track_event");

  tracing_session_ = perfetto::Tracing::NewTrace(perfetto::kInProcessBackend);
  tracing_session_->Setup(cfg);
  tracing_session_->StartBlocking();

  auto jit_track_desc = perfetto::Track(8080).Serialize();
  jit_track_desc.set_name("JIT Compiler");
  jit_track_desc.set_parent_uuid(perfetto::ProcessTrack::Current().uuid);
  perfetto::TrackEvent::SetTrackDescriptor(perfetto::Track(8080),
                                           jit_track_desc);
}

void shutdown() {
  if (!tracing_enabled_) return;
  if (tracing_session_) {
    Logger& logger = DpuRuntime::get().get_logger();
    logger.lock(logcat::TRACE_IO)
        << "Flushing TrackEvent buffers..." << std::endl;
    perfetto::TrackEvent::Flush();
    logger.lock(logcat::TRACE_IO) << "Flushing tracing session..." << std::endl;
    bool flushed = tracing_session_->FlushBlocking(5000);
    logger.lock(logcat::TRACE_IO)
        << "FlushBlocking result=" << (flushed ? "ok" : "timeout") << std::endl;
    logger.lock(logcat::TRACE_IO) << "Stopping tracing session..." << std::endl;
    tracing_session_->StopBlocking();
    logger.lock(logcat::TRACE_IO) << "Reading trace data..." << std::endl;
    std::vector<char> trace_data = tracing_session_->ReadTraceBlocking();

    std::string filename = output_path();

    std::error_code ec;
    std::filesystem::path trace_path(filename);
    if (trace_path.has_parent_path()) {
      std::filesystem::create_directories(trace_path.parent_path(), ec);
      if (ec) {
        logger.lock(logcat::TRACE_IO) << "Failed to create trace directory "
                                      << trace_path.parent_path().string()
                                      << ": " << ec.message() << std::endl;
      }
    }

    std::ofstream out(filename, std::ios::binary);
    if (!out) {
      logger.lock(logcat::TRACE_IO)
          << "Failed to open trace output " << filename << std::endl;
      return;
    }
    out.write(trace_data.data(), trace_data.size());
    out.close();
    if (!out) {
      logger.lock(logcat::TRACE_IO)
          << "Failed while writing trace output " << filename << std::endl;
      return;
    }
    logger.lock(logcat::TRACE_IO)
        << "Trace written to " << filename << " (" << trace_data.size()
        << " bytes)" << std::endl;
    tracing_session_.reset();
    perfetto::Tracing::Shutdown();
    logger.lock(logcat::TRACE_IO) << "Perfetto shutdown complete." << std::endl;
  }
  tracing_enabled_ = false;
}

void internal_reduction_begin(uint64_t flow_id) {
  if (!enabled()) return;
  TRACE_EVENT_BEGIN("events", "reduction_cpu",
                    [flow_id](perfetto::EventContext& ctx) {
                      if (flow_id) perfetto::Flow::ProcessScoped(flow_id)(ctx);
                    });
}
void internal_reduction_end() {
  if (!enabled()) return;
  TRACE_EVENT_END("events");
}
void internal_to_cpu_begin(uint64_t flow_id) {
  if (!enabled()) return;
  TRACE_EVENT_BEGIN("transfer", "dpu_vector::to_cpu",
                    [flow_id](perfetto::EventContext& ctx) {
                      if (flow_id) perfetto::Flow::ProcessScoped(flow_id)(ctx);
                    });
}
void internal_to_cpu_end() {
  if (!enabled()) return;
  TRACE_EVENT_END("transfer");
}

void internal_from_cpu_begin() {
  if (!enabled()) return;
  TRACE_EVENT_BEGIN("transfer", "dpu_vector::from_cpu");
}
void internal_from_cpu_end() {
  if (!enabled()) return;
  TRACE_EVENT_END("transfer");
}

void counter(const char* cat, const char* name, int64_t value) {
  if (!enabled()) return;
  if (std::string(cat) == "runtime")
    TRACE_COUNTER("runtime", perfetto::DynamicString(name), value);
  else if (std::string(cat) == "queue")
    TRACE_COUNTER("queue", perfetto::DynamicString(name), value);
}
void event_begin(const char* cat, const char* name) {
  if (!enabled()) return;
  if (std::string(cat) == "runtime")
    TRACE_EVENT_BEGIN("runtime", perfetto::DynamicString(name));
  else if (std::string(cat) == "queue")
    TRACE_EVENT_BEGIN("queue", perfetto::DynamicString(name));
}
void event_end(const char* cat) {
  if (!enabled()) return;
  if (std::string(cat) == "runtime")
    TRACE_EVENT_END("runtime");
  else if (std::string(cat) == "queue")
    TRACE_EVENT_END("queue");
}

void event_enqueued(std::shared_ptr<Event> e,
                    const std::deque<std::shared_ptr<Event>>& ops,
                    const std::list<std::shared_ptr<Event>>& running) {
  if (!enabled()) return;
  TRACE_EVENT_INSTANT(
      "queue", "EventEnqueued", perfetto::Track(EVENT_TRACK_BASE + e->id),
      [e](perfetto::EventContext& ctx) {
        perfetto::Flow::ProcessScoped(e->id)(ctx);
      },
      "type", operationtype_to_string(e->op), "id", e->id);

  std::string waiting_on;
  for (const auto& active : running) {
    if (!waiting_on.empty()) waiting_on += ", ";
    waiting_on += "Run[" + std::to_string(active->id) +
                  "]:" + operationtype_to_string(active->op);
  }
  for (const auto& queued : ops) {
    if (!waiting_on.empty()) waiting_on += ", ";
    waiting_on += "Wait[" + std::to_string(queued->id) +
                  "]:" + operationtype_to_string(queued->op);
  }

  std::string in_queue_name = "InQueue: " + operationtype_to_string(e->op);
  if (e->op == Event::OperationType::COMPUTE) {
    if (!e->rpn_ops.empty())
      in_queue_name = "InQueue: Fused Pipeline";
    else if (e->opcode != 0)
      in_queue_name = "InQueue: " + opcode_to_string(e->opcode);
    else
      in_queue_name = "InQueue: " + std::string(kernel_id_to_string(e->kid));
  }

  TRACE_EVENT_BEGIN(
      "queue", perfetto::DynamicString(in_queue_name),
      perfetto::Track(EVENT_TRACK_BASE + e->id), "id", e->id,
      [e](perfetto::EventContext& ctx) {
        perfetto::Flow::ProcessScoped(e->id)(ctx);
      },
      "waiting_on_details", trace_string(std::move(waiting_on), 4096),
      "queue_depth", (int)ops.size(), "running_count", (int)running.size());
}

void event_fused(std::shared_ptr<Event> e, std::shared_ptr<Event> into,
                 const std::string& fused_ops) {
  if (!enabled()) return;
  TRACE_EVENT_INSTANT(
      "queue",
      perfetto::DynamicString("Fused [" + fused_ops + "] into #" +
                              std::to_string(into->id)),
      perfetto::Track(EVENT_TRACK_BASE + e->id), "into_id", into->id,
      "new_ops_count", (int)into->rpn_ops.size());
  TRACE_EVENT_END("queue", perfetto::Track(EVENT_TRACK_BASE + e->id));
}

void inqueue_end(std::shared_ptr<Event> e) {
  if (!enabled()) return;
  TRACE_EVENT_END("queue", perfetto::Track(EVENT_TRACK_BASE + e->id));
}

// Native retires events in order on "DPU Hardware"; Legion's hand-offs go on
// "Dispatch".  Each is named on first use so a trace only has its own.
static perfetto::Track dpu_track() {
  static std::once_flag named;
  const uint64_t id = concurrent_ranks_ ? DISPATCH_TRACK_ID : DPU_TRACK_ID;
  std::call_once(named, [id] {
    auto desc = perfetto::Track(id).Serialize();
    desc.set_name(id == DPU_TRACK_ID ? "DPU Hardware" : "Dispatch");
    desc.set_parent_uuid(perfetto::ProcessTrack::Current().uuid);
    perfetto::TrackEvent::SetTrackDescriptor(perfetto::Track(id), desc);
  });
  return perfetto::Track(id);
}

// A slice for `e` on the DPU track, flowing from its queue events.
static void begin_on_dpu_track(const std::shared_ptr<Event>& e) {
  auto base_lambda = [e](perfetto::EventContext& ctx) {
    perfetto::Flow::ProcessScoped(e->id)(ctx);
    for (size_t dep_id : e->dependencies)
      perfetto::Flow::ProcessScoped(dep_id)(ctx);
  };

  if (!e->rpn_ops.empty()) {
    std::string slice_name = trace_string(e->slice_name, 4096);
    std::string ops_summary = summarize_rpn_ops(e->rpn_ops);
    TRACE_EVENT_BEGIN("events", perfetto::DynamicString(slice_name),
                      dpu_track(), "id", e->id, "ops_summary",
                      perfetto::DynamicString(ops_summary),
                      [e, base_lambda](perfetto::EventContext& ctx) {
                        base_lambda(ctx);
                        add_event_metadata(ctx, e);
                      });
  } else {
    TRACE_EVENT_BEGIN("events", perfetto::DynamicString(e->slice_name),
                      dpu_track(), "id", e->id,
                      [e, base_lambda](perfetto::EventContext& ctx) {
                        base_lambda(ctx);
                        add_event_metadata(ctx, e);
                      });
  }
}

void execution_begin(std::shared_ptr<Event> e) {
  if (!enabled() || concurrent_ranks_) return;
  begin_on_dpu_track(e);
}

void execution_end() {
  if (!enabled() || concurrent_ranks_) return;
  TRACE_EVENT_END("events", dpu_track());
}
void active_ops_counter(size_t count) {
  if (!enabled()) return;
  TRACE_COUNTER("queue", "Active DPU Ops", (int)count);
}

void ensure_callback_thread_named() {
  if (!enabled()) return;
  static thread_local bool thread_named = false;
  if (!thread_named) {
    auto track = perfetto::ThreadTrack::Current();
    auto desc = track.Serialize();
    desc.mutable_thread()->set_thread_name("UPMEM Callback");
    perfetto::TrackEvent::SetTrackDescriptor(track, desc);
    thread_named = true;
  }
}

static std::string rpn_ops_to_string(const std::vector<uint8_t>& rpn_ops) {
  return summarize_rpn_ops(rpn_ops);
}

void jit_compile_begin(const std::vector<uint8_t>& rpn_ops,
                       const char* type_name) {
  if (!enabled()) return;
  std::string ops_str = trace_string(rpn_ops_to_string(rpn_ops), 8192);
  TRACE_EVENT_BEGIN("runtime", "jit_compile", perfetto::Track(8080), "type",
                    perfetto::DynamicString(type_name), "ops",
                    perfetto::DynamicString(ops_str));
}

void jit_compile_begin(
    const std::vector<std::pair<std::vector<uint8_t>, std::string>>& kernels) {
  if (!enabled()) return;
  std::string summary =
      "Batched " + std::to_string(kernels.size()) + " kernels\n";
  for (size_t i = 0; i < kernels.size(); ++i) {
    summary += "K" + std::to_string(i) + " [" + kernels[i].second +
               "]: " + rpn_ops_to_string(kernels[i].first) + "\n";
  }
  summary = trace_string(std::move(summary), 16384);
  TRACE_EVENT_BEGIN("runtime", "jit_compile_batch", perfetto::Track(8080),
                    "kernels", (int)kernels.size(), "details",
                    perfetto::DynamicString(summary));
}

void jit_compile_end() {
  if (!enabled()) return;
  TRACE_EVENT_END("runtime", perfetto::Track(8080));
}

void jit_binary_switch(const std::string& previous,
                       const std::string& current) {
  if (!enabled()) return;
  TRACE_EVENT_INSTANT("runtime", "binary_switch", perfetto::Track(8080), "from",
                      perfetto::DynamicString(previous), "to",
                      perfetto::DynamicString(current));
}

static thread_local LaunchTag dispatching_;

void set_concurrent_ranks() {
  concurrent_ranks_ = true;
}

void dispatch_begin(const std::shared_ptr<Event>& e) {
  if (!enabled()) return;
  // The event runs until its last launch finishes on its last rank, when the
  // final reference to its tag drops.
  auto info = std::shared_ptr<LaunchInfo>(new LaunchInfo, [](LaunchInfo* p) {
    TRACE_EVENT_END("events", perfetto::Track(EVENT_TRACK_BASE + p->id));
    delete p;
  });
  info->id = e->id;
  if (!e->slice_name.empty()) info->name = trace_string(e->slice_name, 256);
  TRACE_EVENT_BEGIN("events", perfetto::DynamicString("Running: " + info->name),
                    perfetto::Track(EVENT_TRACK_BASE + e->id), "id", e->id);
  dispatching_ = std::move(info);
  if (concurrent_ranks_) begin_on_dpu_track(e);
}

void dispatch_end() {
  if (!enabled()) return;
  dispatching_.reset();
  if (concurrent_ranks_)
    TRACE_EVENT_END("events", dpu_track());
}

LaunchTag current_launch() {
  if (!enabled()) return nullptr;
  static const LaunchTag undispatched = std::make_shared<LaunchInfo>();
  return dispatching_ ? dispatching_ : undispatched;
}

static perfetto::Track rank_track(uint32_t rank) {
  static std::mutex mutex;
  static std::vector<bool> named;
  const perfetto::Track parent(RANK_TRACK_BASE);
  const perfetto::Track track(RANK_TRACK_BASE + 1 + rank, parent);
  std::lock_guard<std::mutex> lock(mutex);
  if (named.empty()) {
    auto desc = parent.Serialize();
    desc.set_name("DPU Ranks");
    desc.set_parent_uuid(perfetto::ProcessTrack::Current().uuid);
    desc.set_child_ordering(perfetto::protos::gen::TrackDescriptor::EXPLICIT);
    perfetto::TrackEvent::SetTrackDescriptor(parent, desc);
  }
  if (named.size() <= rank) named.resize(rank + 1);
  if (!named[rank]) {
    auto desc = track.Serialize();
    desc.set_name("Rank " + std::to_string(rank));
    desc.set_sibling_order_rank((int32_t)rank);
    perfetto::TrackEvent::SetTrackDescriptor(track, desc);
    named[rank] = true;
  }
  return track;
}

static std::atomic<int64_t> busy_ranks_{0};

namespace {
// Shared by the ranks of one launch; the last callback frees it.
struct RankLaunch {
  LaunchTag tag;
  uint32_t first_rank;
  bool flow;
  std::atomic<uint32_t> left;
};
}  // namespace

static dpu_error_t on_rank_begin(dpu_set_t, uint32_t rank_id, void* arg) {
  auto* launch = static_cast<RankLaunch*>(arg);
  const bool flow = launch->flow && rank_id == 0;
  const uint64_t id = launch->tag->id;
  TRACE_EVENT_BEGIN("events", perfetto::DynamicString(launch->tag->name),
                    rank_track(launch->first_rank + rank_id), "id", id,
                    [flow, id](perfetto::EventContext& ctx) {
                      if (flow) perfetto::Flow::ProcessScoped(id)(ctx);
                    });
  TRACE_COUNTER("events", "Busy ranks", ++busy_ranks_);
  if (--launch->left == 0) delete launch;
  return DPU_OK;
}

static dpu_error_t on_rank_end(dpu_set_t, uint32_t rank_id, void* arg) {
  auto* launch = static_cast<RankLaunch*>(arg);
  TRACE_EVENT_END("events", rank_track(launch->first_rank + rank_id));
  TRACE_COUNTER("events", "Busy ranks", --busy_ranks_);
  if (--launch->left == 0) delete launch;
  return DPU_OK;
}

// Blocking, so each rank's callbacks run in stream order.
static void queue_rank_callback(dpu_set_t set,
                                dpu_error_t (*fn)(dpu_set_t, uint32_t, void*),
                                void* arg) {
  DPU_ASSERT(dpu_callback(set, fn, arg, DPU_CALLBACK_ASYNC));
}

void* rank_launch_begin(dpu_set_t set, uint32_t first_rank, uint32_t nr_ranks,
                        LaunchTag tag, bool flow) {
  if (!enabled() || !tag) return nullptr;
  auto* launch =
      new RankLaunch{std::move(tag), first_rank, flow, {2 * nr_ranks}};
  queue_rank_callback(set, on_rank_begin, launch);
  return launch;
}

void rank_launch_end(dpu_set_t set, void* launch) {
  if (launch) queue_rank_callback(set, on_rank_end, launch);
}

void rank_span_begin(uint32_t rank, const char* name) {
  if (!enabled()) return;
  TRACE_EVENT_BEGIN("events", perfetto::DynamicString(name), rank_track(rank));
}

void rank_span_end(uint32_t rank) {
  if (!enabled()) return;
  TRACE_EVENT_END("events", rank_track(rank));
}

}  // namespace trace
#endif
