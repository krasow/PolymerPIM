#include "vector.h"

#include <detail/fusion.h>

#include <stdexcept>

#include "logger.h"
#include "perfetto/trace.h"
#include "vector_desc.h"

namespace detail {
VectorDesc::~VectorDesc() {
  auto& runtime = DpuRuntime::get();
  // The runtime owns the logger and the allocator, and shutdown() destroys
  // both.  A descriptor outliving it has nothing left to report to and no MRAM
  // to give back -- the DPU set is already freed -- so there is nothing to do.
  // This is the normal case for a garbage-collected binding, where finalizers
  // run after atexit.
  if (!runtime.is_initialized()) return;

  if (vector_id != 0) {
#if ENABLE_DPU_LOGGING >= 1
    Logger& logger = runtime.get_logger();
    log_deallocation(logger, type_name, num_elements, vector_id,
                     allocated_footprint_bytes, !needs_layout_materialization,
                     (debug_name ? debug_name : ""), debug_file, debug_line);
#endif
  }
  if (ptr_allocated) {
    runtime.backend().release(*this);
  }
}

void vec_xfer_to_dpu(char* cpu, VectorDescRef desc) {
  trace::scoped_event trace_scoped("transfer", "vec_xfer_to_dpu");
  DpuRuntime::get().backend().write(desc, cpu);
}

// Reads every shard into `cpu` at a uniform `stride`.  The caller must have
// sized `cpu` for stride * num_dpus, since one transfer size applies to the
// whole DPU set; see ShardLayout.
void vec_xfer_from_dpu_strided(char* cpu, VectorDescRef desc, size_t stride) {
  trace::scoped_event trace_scoped("transfer", "vec_xfer_from_dpu");
  DpuRuntime::get().backend().read(desc, cpu, stride);
}

namespace {
DpuBackend::InspectArgs log_launch_args() {
#if ENABLE_DPU_LOGGING >= 1
  return [](const DPU_LAUNCH_ARGS* args, uint32_t nr_of_dpus) {
    log_dpu_launch_args(DpuRuntime::get().get_logger(), args, nr_of_dpus);
  };
#else
  return {};
#endif
}
}  // namespace

void internal_launch_binary_scalar(VectorDescRef res, VectorDescRef lhs,
                                   uint32_t scalar, KernelID kernel_id) {
  auto fill = [&](DPU_LAUNCH_ARGS& args, const DpuBackend::Shard& s) {
    args = {};
    args.kernel = static_cast<uint32_t>(kernel_id);
    args.ktype = static_cast<uint8_t>(KERNEL_BINARY_SCALAR);
    args.num_elements = s.size_bytes(*lhs) / lhs->element_size;
    args.size_type = lhs->element_size;
    args.binary_scalar.lhs_offset = s.offset(*lhs);
    args.binary_scalar.rhs_scalar = scalar;
    args.binary_scalar.res_offset = s.offset(*res);
  };
  DpuRuntime::get().backend().launch({res, lhs}, fill, log_launch_args());
}

void internal_launch_fill(VectorDescRef res, uint32_t value,
                          KernelID kernel_id) {
  auto fill = [&](DPU_LAUNCH_ARGS& args, const DpuBackend::Shard& s) {
    args = {};
    args.kernel = static_cast<uint32_t>(kernel_id);
    args.ktype = static_cast<uint8_t>(KERNEL_FILL);
    args.num_elements = s.size_bytes(*res) / res->element_size;
    args.size_type = res->element_size;
    args.binary_scalar.lhs_offset = 0;
    args.binary_scalar.rhs_scalar = value;
    args.binary_scalar.res_offset = s.offset(*res);
  };
  DpuRuntime::get().backend().launch({res}, fill, log_launch_args());
}

void internal_launch_binary(VectorDescRef res, VectorDescRef lhs,
                            VectorDescRef rhs, KernelID kernel_id) {
  auto fill = [&](DPU_LAUNCH_ARGS& args, const DpuBackend::Shard& s) {
    args.kernel = static_cast<uint32_t>(kernel_id);
    args.ktype = static_cast<uint8_t>(KERNEL_BINARY);
    args.num_elements = s.size_bytes(*rhs) / rhs->element_size;
    args.size_type = rhs->element_size;
    args.binary.lhs_offset = s.offset(*lhs);
    args.binary.rhs_offset = s.offset(*rhs);
    args.binary.res_offset = s.offset(*res);
  };
  DpuRuntime::get().backend().launch({res, lhs, rhs}, fill, log_launch_args());
}

void internal_launch_unary(VectorDescRef res, VectorDescRef rhs,
                           KernelID kernel_id) {
  auto fill = [&](DPU_LAUNCH_ARGS& args, const DpuBackend::Shard& s) {
    args.kernel = static_cast<uint32_t>(kernel_id);
    args.ktype = static_cast<uint8_t>(KERNEL_PIPELINE);
    args.num_elements = s.size_bytes(*rhs) / rhs->element_size;
    args.size_type = rhs->element_size;
    args.unary.rhs_offset = s.offset(*rhs);
    args.unary.res_offset = s.offset(*res);
  };
  DpuRuntime::get().backend().launch({res, rhs}, fill, log_launch_args());
}

void internal_launch_reduction(VectorDescRef res, VectorDescRef rhs,
                               KernelID kernel_id) {
  auto fill = [&](DPU_LAUNCH_ARGS& args, const DpuBackend::Shard& s) {
    args.kernel = static_cast<uint32_t>(kernel_id);
    args.ktype = static_cast<uint8_t>(KERNEL_REDUCTION);
    args.num_elements = s.size_bytes(*rhs) / rhs->element_size;
    args.size_type = rhs->element_size;
    args.reduction.rhs_offset = s.offset(*rhs);
    args.reduction.res_offset = s.offset(*res);
  };
  DpuRuntime::get().backend().launch({res, rhs}, fill, log_launch_args());
}

#if PIPELINE
void internal_launch_universal_pipeline(
    VectorDescRef res, VectorDescRef init, const std::vector<uint8_t>& ops,
    const std::vector<VectorDescRef>& operands, KernelID kernel_id,
    const std::vector<uint32_t>& scalars,
    const std::vector<uint32_t>& extra_scalars,
    const std::vector<VectorDescRef>& extra_outputs,
    std::string_view kernel_hash) {
#if !JIT
  // The interpreter reads args.pipeline.ops, which is MAX_VFUSE_OPS long.
  // Truncating here would silently drop the tail of the program, so fail
  // loudly instead -- every path that builds an RPN is supposed to bound it.
  if (ops.size() > MAX_VFUSE_OPS)
    throw std::runtime_error("RPN program of " + std::to_string(ops.size()) +
                             " opcodes exceeds MAX_VFUSE_OPS (" +
                             std::to_string(MAX_VFUSE_OPS) +
                             "); rebuild with JIT=1 or raise MAX_VFUSE_OPS");
#endif

  std::vector<VectorDescRef> vectors = {res, init};
  vectors.insert(vectors.end(), operands.begin(), operands.end());
  vectors.insert(vectors.end(), extra_outputs.begin(), extra_outputs.end());

  auto fill = [&](DPU_LAUNCH_ARGS& args, const DpuBackend::Shard& s) {
    args.kernel = static_cast<uint32_t>(kernel_id);
    args.ktype = static_cast<uint8_t>(KERNEL_UNARY);

    args.num_elements = init ? (s.size_bytes(*init) / init->element_size) : 0;
    args.size_type = init ? init->element_size : (res ? res->element_size : 4);

    args.pipeline.init_offset = init ? s.offset(*init) : 0;
    args.pipeline.res_offset = res ? s.offset(*res) : 0;
    args.pipeline.index_base = init ? s.first_element(*init) : 0;

    for (size_t j = 0; j < MAX_HFUSE_CHAINS; ++j) {
      if (j < extra_outputs.size()) {
        args.pipeline.extra_res_offsets[j] = s.offset(*extra_outputs[j]);
        args.pipeline.local_sizes[j] = extra_outputs[j]->num_elements;
        args.pipeline.local_reduce_ops[j] =
            extra_outputs[j]->local_reduce_opcode;
      } else {
        args.pipeline.extra_res_offsets[j] = 0;
        args.pipeline.local_sizes[j] = 0;
        args.pipeline.local_reduce_ops[j] = OP_SUM;
      }
    }

    args.pipeline.num_ops = std::min((size_t)ops.size(), (size_t)MAX_VFUSE_OPS);

    for (size_t j = 0; j < args.pipeline.num_ops; ++j) {
      args.pipeline.ops[j] = ops[j];
    }

    // Map operands by index (0..MAX_VFUSE_INPUTS-1)
    for (size_t j = 0; j < MAX_VFUSE_INPUTS; ++j) {
      if (j < operands.size()) {
        args.pipeline.binary_operands[j] = s.offset(*operands[j]);
      } else {
        args.pipeline.binary_operands[j] = 0;
      }
    }

    // Map scalar arguments
    for (size_t j = 0; j < MAX_PIPELINE_SCALARS; ++j) {
      if (j < scalars.size()) {
        args.pipeline.scalars[j] = scalars[j];
      } else {
        args.pipeline.scalars[j] = 0;
      }
    }

    // Map extra JIT scalars
    for (size_t j = 0; j < 8; ++j) {
      if (j < extra_scalars.size()) {
        args.pipeline.extra_scalars[j] = extra_scalars[j];
      } else {
        args.pipeline.extra_scalars[j] = 0;
      }
    }
  };

  DpuBackend::InspectArgs inspect;
#if ENABLE_DPU_LOGGING >= 1
  inspect = [&](const DPU_LAUNCH_ARGS* args, uint32_t nr_of_dpus) {
    detail::FusionRpnSummary rpn_summary = detail::summarize_fusion_rpn(ops);
    log_dpu_launch_args(DpuRuntime::get().get_logger(), args, nr_of_dpus,
                        rpn_summary.decoded_ops, rpn_summary.chains,
                        kernel_hash);
  };
#else
  (void)kernel_hash;
#endif

  DpuRuntime::get().backend().launch(vectors, fill, inspect);
}
#endif

void launch_unary(VectorDescRef res, VectorDescRef rhs, KernelID kernel_id,
                  uint8_t opcode, KernelID pipeline_kid) {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

#if PIPELINE
  std::shared_ptr<Event> e = std::make_shared<Event>(
      Event::OperationType::COMPUTE,
      std::bind(internal_launch_unary, res, rhs, kernel_id));
  e->pipeline_kid = pipeline_kid;
#else
  (void)pipeline_kid;
  auto bound_cb = std::bind(internal_launch_unary, res, rhs, kernel_id);
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);
#endif
  e->inputs = {rhs};
  e->output = res;
  e->kid = kernel_id;
  e->opcode = opcode;
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock(logcat::QUEUE_APPEND, 2)
      << "type=COMPUTE (unary) kernel=" << kernel_id_to_string(kernel_id)
      << std::endl;
#endif
}

#if PIPELINE
void launch_universal_pipeline(
    VectorDescRef res, VectorDescRef init, const std::vector<uint8_t>& ops,
    const std::vector<VectorDescRef>& operands, KernelID kernel_id,
    const std::vector<uint32_t>& scalars,
    const std::vector<uint32_t>& extra_scalars,
    const std::vector<VectorDescRef>& extra_outputs) {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE);

  e->inputs = {init};
  for (auto& op : operands) {
    if (op) e->inputs.push_back(op);
  }
  e->output = res;
  e->rpn_ops = ops;
  e->kid = kernel_id;
  // The interpreter path dispatches pipeline_kid, and the universal pipeline is
  // the only kernel that can run an arbitrary RPN program -- so it is both.
  // Leaving it 0 silently ran kernel 0 (binary add) for every explicit
  // pipeline() call under JIT=0.
  e->pipeline_kid = kernel_id;
  e->scalars = scalars;
  e->extra_scalars = extra_scalars;
  e->extra_outputs = extra_outputs;

  // Detect reduction and flag result descriptor synchronously
  for (size_t i = 0; i < ops.size(); ++i) {
    uint8_t op = ops[i];
    if (res && IS_OP_REDUCTION(op)) {
      res->is_reduction_result = true;
      res->reduction_rid = static_cast<KernelID>(op);
    }
    if (OP_INLINE_BYTES(op) > 0) i += OP_INLINE_BYTES(op);
  }

  event_queue.submit(e);
}
#endif

void launch_binary(VectorDescRef res, VectorDescRef lhs, VectorDescRef rhs,
                   KernelID kernel_id, uint8_t opcode, KernelID pipeline_kid) {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

#if PIPELINE
  assert(lhs->num_elements == rhs->num_elements);

  std::shared_ptr<Event> e = std::make_shared<Event>(
      Event::OperationType::COMPUTE,
      std::bind(internal_launch_binary, res, lhs, rhs, kernel_id));

  e->pipeline_kid = pipeline_kid;
#else
  (void)pipeline_kid;
  auto bound_cb = std::bind(internal_launch_binary, res, lhs, rhs, kernel_id);
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);
#endif
  e->inputs = {lhs, rhs};
  e->output = res;
  e->kid = kernel_id;
  e->opcode = opcode;
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock(logcat::QUEUE_APPEND, 2)
      << "type=COMPUTE (binary) kernel=" << kernel_id_to_string(kernel_id)
      << std::endl;
#endif
}

void launch_fill(VectorDescRef res, uint32_t value, KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  auto bound_cb = std::bind(internal_launch_fill, res, value, kernel_id);
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);
  // Left with no inputs on purpose: that is what keeps the fill out of
  // fusion and JIT promotion, neither of which can express it.
  e->scalar_value = value;
  e->output = res;
  event_queue.submit(e);
}

void launch_binary_scalar(VectorDescRef res, VectorDescRef lhs, uint32_t scalar,
                          KernelID kernel_id, uint8_t opcode,
                          KernelID pipeline_kid) {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

#if PIPELINE
  std::shared_ptr<Event> e = std::make_shared<Event>(
      Event::OperationType::COMPUTE,
      std::bind(internal_launch_binary_scalar, res, lhs, scalar, kernel_id));
  e->pipeline_kid = pipeline_kid;
#else
  (void)pipeline_kid;
  auto bound_cb =
      std::bind(internal_launch_binary_scalar, res, lhs, scalar, kernel_id);
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);
#endif
  e->is_scalar = true;
  e->scalar_value = scalar;
  e->inputs = {lhs};
  e->output = res;
  e->kid = kernel_id;
  e->opcode = opcode;
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock(logcat::QUEUE_APPEND, 2)
      << "type=COMPUTE (binary_scalar) kernel="
      << kernel_id_to_string(kernel_id) << std::endl;
#endif
}

void launch_reduction(VectorDescRef res, VectorDescRef rhs, KernelID kernel_id,
                      uint8_t opcode, KernelID pipeline_kid) {
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

#if PIPELINE
  std::shared_ptr<Event> e = std::make_shared<Event>(
      Event::OperationType::COMPUTE,
      std::bind(internal_launch_reduction, res, rhs, kernel_id));

  e->pipeline_kid = pipeline_kid;
  // Mark result description as reduction synchronously
  res->is_reduction_result = true;
  res->reduction_rid = static_cast<KernelID>(opcode);
#else
  (void)pipeline_kid;
  auto bound_cb = std::bind(internal_launch_reduction, res, rhs, kernel_id);
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);
#endif
  e->inputs = {rhs};
  e->output = res;
  e->kid = kernel_id;
  e->opcode = opcode;
  event_queue.submit(e);

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock(logcat::QUEUE_APPEND, 2)
      << "type=COMPUTE (reduction) kernel=" << kernel_id_to_string(kernel_id)
      << std::endl;
#endif
}

}  // namespace detail
