#include "logger.h"

#include <filesystem>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "kernelids.h"
#include "queue.h"

// External declaration for KERNEL_COUNT from generated headers if possible,
// or use a safe check.
extern KernelInfo kernel_infos[];
static const size_t KERNEL_COUNT_VAL =
    sizeof(kernel_infos) / sizeof(KernelInfo);

namespace {
std::mutex memory_tracker_mutex;
std::unordered_map<uint64_t, size_t> live_vector_bytes;
size_t active_vector_bytes = 0;

std::string format_bytes(size_t bytes) {
  static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB"};
  double value = static_cast<double>(bytes);
  size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < sizeof(units) / sizeof(units[0])) {
    value /= 1024.0;
    unit++;
  }

  std::ostringstream out;
  if (unit == 0) {
    out << bytes << " " << units[unit];
  } else {
    out << std::fixed << std::setprecision(2) << value << " " << units[unit];
  }
  return out.str();
}

std::string absolute_source_path(const char* path) {
  if (path == nullptr || std::string_view(path) == "unknown") return "unknown";
  std::filesystem::path source_path(path);
  if (source_path.is_relative())
    source_path = std::filesystem::absolute(source_path);
  return source_path.lexically_normal().string();
}

struct MemoryTrackerUpdate {
  size_t event_bytes;
  size_t active_bytes;
};

MemoryTrackerUpdate update_memory_tracker(uint64_t vector_id,
                                          size_t memory_bytes,
                                          bool is_allocation) {
  std::lock_guard<std::mutex> lock(memory_tracker_mutex);
  if (vector_id == 0) return {memory_bytes, active_vector_bytes};

  if (is_allocation) {
    auto [it, inserted] = live_vector_bytes.emplace(vector_id, memory_bytes);
    if (!inserted) {
      active_vector_bytes -= it->second;
      it->second = memory_bytes;
    }
    active_vector_bytes += memory_bytes;
    return {memory_bytes, active_vector_bytes};
  }

  auto it = live_vector_bytes.find(vector_id);
  if (it == live_vector_bytes.end()) {
    return {memory_bytes, active_vector_bytes};
  }

  size_t released_bytes = it->second;
  active_vector_bytes -= released_bytes;
  live_vector_bytes.erase(it);
  return {released_bytes, active_vector_bytes};
}
}  // namespace

const char* kernel_id_to_string(KernelID id) {
  if (id >= KERNEL_COUNT_VAL) return "JIT/BATCH";
  // JIT kernels often use small IDs that collide with standard kernels if
  // treated as global IDs
  return kernel_infos[id].name;
}

const char* ktype_to_string(KernelCategory ktype) {
  switch (ktype) {
    case KERNEL_BINARY:
      return "BINARY";
    case KERNEL_UNARY:
      return "UNARY";
    case KERNEL_REDUCTION:
      return "REDUCTION";
    case KERNEL_BINARY_SCALAR:
      return "BINARY_SCALAR";
    case KERNEL_PIPELINE:
      return "PIPELINE";
    default:
      return "UNKNOWN_KTYPE";
  }
}

static bool is_fused_pipeline_launch(const DPU_LAUNCH_ARGS& args) {
  // The universal fused/JIT path stores its bytecode in the pipeline union even
  // though the legacy launch category is currently KERNEL_UNARY.
  return args.ktype == KERNEL_UNARY && args.pipeline.num_ops > 0 &&
         args.pipeline.num_ops <= MAX_VFUSE_OPS;
}

void print_vector_desc(Logger& logger, detail::VectorDescRef desc,
                       uint32_t reserved) {
  auto out = logger.lock(logcat::VECTOR_DESC);
  out << "  " << std::left << std::setw(6) << "DPU" << std::setw(14) << "PTR"
      << std::setw(14) << "ALLOC(bytes)" << std::setw(14) << "VEC_SIZE(bytes)\n"
      << std::string(51, '-') << "\n";

  for (size_t i = 0; i < desc->desc.size(); i++) {
    std::ostringstream ptr_hex;
    ptr_hex << "0x" << std::hex << std::setw(8) << std::setfill('0')
            << desc->desc[i].ptr;

    out << "  " << std::left << std::setw(6) << i << std::setw(14)
        << ptr_hex.str() << std::setw(14) << std::dec
        << desc->desc[i].size_bytes << std::dec
        << (desc->desc[i].size_bytes - reserved) << "\n";
  }
}

void log_allocation(Logger& logger, const std::type_info& type, size_t n,
                    uint64_t vector_id, size_t memory_bytes,
                    bool has_materialized_layout, std::string_view debug_name,
                    const char* debug_file, int debug_line,
                    bool is_allocation) {
  log_allocation(logger, type.name(), n, vector_id, memory_bytes,
                 has_materialized_layout, debug_name, debug_file, debug_line,
                 is_allocation);
}

void log_allocation(Logger& logger, const char* type_name, size_t n,
                    uint64_t vector_id, size_t memory_bytes,
                    bool has_materialized_layout, std::string_view debug_name,
                    const char* debug_file, int debug_line,
                    bool is_allocation) {
  if (type_name == nullptr) type_name = "unknown";
  MemoryTrackerUpdate memory =
      update_memory_tracker(vector_id, memory_bytes, is_allocation);
  auto log = logger.lock(logcat::MEMORY, has_materialized_layout ? 1 : 2);
  log.first() << (is_allocation ? "alloc" : "free ") << " vec#" << vector_id
              << " dpu_vector<" << type_name << ">"
              << " size=" << n << " bytes=" << format_bytes(memory.event_bytes)
              << " active=" << format_bytes(memory.active_bytes);
  if (!debug_name.empty()) log << " name=\"" << debug_name << "\"";
  if (!has_materialized_layout) log << " lazy";

  bool has_source = debug_file != nullptr &&
                    std::string_view(debug_file) != "unknown" && debug_line > 0;
  if (has_source) {
    log.second();
    log << "at " << absolute_source_path(debug_file) << ":" << debug_line;
  }
  log << std::endl;
}

#if ENABLE_DPU_LOGGING >= 1
void log_dpu_launch_args(Logger& logger, const DPU_LAUNCH_ARGS* args,
                         uint32_t nr_of_dpus, size_t fused_event_ops,
                         size_t fused_event_chains,
                         std::string_view kernel_hash) {
  const auto& first = args[0];
  const bool fused_pipeline = is_fused_pipeline_launch(first);
  auto log = logger.lock(logcat::TASK);
  if (fused_pipeline) {
    log.first() << "launch=fused_pipeline";
    log.second() << "dpus=" << nr_of_dpus << "  base_task="
                 << kernel_id_to_string(static_cast<KernelID>(first.kernel))
                 << "  base_type="
                 << ktype_to_string(static_cast<KernelCategory>(first.ktype))
                 << "  fused_event_ops=" << fused_event_ops
                 << "  fused_event_chains=" << fused_event_chains;
    if (!kernel_hash.empty()) log << "  kernel_hash=" << kernel_hash;
    log << std::endl;
  } else {
    log.first() << "launch=kernel";
    log.second() << "dpus=" << nr_of_dpus << "  kernel="
                 << kernel_id_to_string(static_cast<KernelID>(first.kernel))
                 << "  type="
                 << ktype_to_string(static_cast<KernelCategory>(first.ktype))
                 << std::endl;
  }

// the following code is gross, but it's just for logging purposes
// it creates a table of the launch arguments for each DPU
#if ENABLE_DPU_LOGGING >= 3
  if (!logger.enabled(3)) return;
  // Determine which columns to show
  bool show_rhs = false;
  bool show_lhs = false;
  bool show_src = false;
  bool show_res = false;
  bool show_pipeline = false;

  const auto& a = args[0];
  if (a.ktype == KERNEL_BINARY) {
    show_rhs = true;
    show_lhs = true;
    show_res = true;
  } else if (a.ktype == KERNEL_UNARY || a.ktype == KERNEL_REDUCTION) {
    // Note: KERNEL_UNARY is sometimes used for the universal pipeline
    if (a.kernel >= KERNEL_COUNT_VAL || is_fused_pipeline_launch(a)) {
      show_pipeline = true;
    } else {
      show_rhs = true;
      show_res = true;
    }
  } else if (a.ktype == KERNEL_BINARY_SCALAR) {
    show_lhs = true;
    show_res = true;
  }

  log << "  " << std::left << std::setw(6) << "DPU" << std::setw(12) << "KTYPE"
      << std::setw(12) << "NUM_ELEMS" << std::setw(9) << "SIZE_T";

  if (show_rhs) log << std::setw(13) << "RHS_OFFSET";
  if (show_lhs) log << std::setw(13) << "LHS_OFFSET";
  if (show_src) log << std::setw(13) << "SRC_OFFSET";
  if (show_res) log << std::setw(13) << "RES_OFFSET";
  if (show_pipeline)
    log << std::setw(13) << "INIT_OFF" << std::setw(13) << "RES_OFF";

  log << "\n" << std::string(38, '-');
  if (a.ktype == KERNEL_BINARY) {
    log << std::string(39, '-');
  } else {
    log << std::string(26, '-');
  }
  log << "\n";

  auto fmt_hex = [](uint32_t v) {
    std::ostringstream ss;
    ss << "0x" << std::hex << std::setw(8) << std::setfill('0') << v;
    return ss.str();
  };

  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    const auto& a = args[i];

    std::ostringstream rhs, lhs, src, res, p_init, p_res;
    rhs << "";
    lhs << "";
    src << "";
    res << "";
    p_init << "";
    p_res << "";

    if (a.ktype == KERNEL_BINARY) {
      rhs << fmt_hex(a.binary.rhs_offset);
      lhs << fmt_hex(a.binary.lhs_offset);
      res << fmt_hex(a.binary.res_offset);
    } else if (a.ktype == KERNEL_UNARY) {
      if (a.kernel >= KERNEL_COUNT_VAL || a.pipeline.num_ops > 0) {
        p_init << fmt_hex(a.pipeline.init_offset);
        p_res << fmt_hex(a.pipeline.res_offset);
      } else {
        rhs << fmt_hex(a.unary.rhs_offset);
        res << fmt_hex(a.unary.res_offset);
      }
    } else if (a.ktype == KERNEL_REDUCTION) {
      rhs << fmt_hex(a.reduction.rhs_offset);
      res << fmt_hex(a.reduction.res_offset);
    } else if (a.ktype == KERNEL_BINARY_SCALAR) {
      lhs << fmt_hex(a.binary_scalar.lhs_offset);
      res << fmt_hex(a.binary_scalar.res_offset);
    }

    std::string ktype_str =
        ktype_to_string(static_cast<KernelCategory>(a.ktype));
    if (is_fused_pipeline_launch(a)) {
      ktype_str = "FUSED_PIPE";
    }

    log << "  " << std::left << std::setw(6) << i << std::setw(12) << ktype_str
        << std::setw(12) << a.num_elements << std::setw(9) << a.size_type;

    if (show_rhs) log << std::setw(13) << rhs.str();
    if (show_lhs) log << std::setw(13) << lhs.str();
    if (show_src) log << std::setw(13) << src.str();
    if (show_res) log << std::setw(13) << res.str();
    if (show_pipeline)
      log << std::setw(13) << p_init.str() << std::setw(13) << p_res.str();

    log << "\n";
  }
#endif
}
#endif
