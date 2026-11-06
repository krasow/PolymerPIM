"""
    reduce_lazy(v, opcode) -> DpuFuture

Queue the reduction identified by `opcode` over `v`. Returns an unread future
so adjacent independent reductions may fuse before a result is requested.
"""
function reduce_lazy(v::Parent.DPUArray, opcode::UInt8)
    handle = Parent.retry_on_oom(
        () -> Parent.launch_reduction_lazy(v.handle, opcode))
    return Parent.DpuFuture(handle)
end

function _check_program(::DpuExpr, operands)
    n = length(operands)
    n <= Parent.MAX_VFUSE_INPUTS || throw(ArgumentError(
        "$n operands exceeds MAX_VFUSE_INPUTS ($(Parent.MAX_VFUSE_INPUTS))"))
end

"""
    dpu_pipeline(v, program; operands=DPUArray[], scalars=Int32[]) -> DPUArray

Run an elementwise RPN `program`. `input()` reads `v`, `operand(i)` reads
`operands[i]`, and `scalar_var(i)` reads `scalars[i]`. The result remains on the
DPUs, shaped as `v`.
"""
function dpu_pipeline(v, e::DpuExpr;
                      operands::AbstractVector{<:Parent.DPUArray} = Parent.DPUArray[],
                      scalars::AbstractVector{<:Integer} = Int32[])
    _check_program(e, operands)
    sc = Int32.(collect(scalars))
    handle = Parent.retry_on_oom(() -> Parent.launch_pipeline(
        Parent._force(v).handle, e.ops, Parent._veclist(operands), sc))
    return Parent._like(v, handle)
end

"""
    dpu_pipeline_reduce(v, program; operands=DPUArray[], scalars=Int32[]) -> DpuFuture

Run an RPN `program` ending in `sum`, `prod`, `minimum`, or `maximum`. Operand
and scalar slots match [`dpu_pipeline`](@ref). The unread future allows adjacent
reductions to fuse.
"""
function dpu_pipeline_reduce(v, e::DpuExpr;
                             operands::AbstractVector{<:Parent.DPUArray} = Parent.DPUArray[],
                             scalars::AbstractVector{<:Integer} = Int32[])
    _check_program(e, operands)
    isempty(e.ops) && throw(ArgumentError("empty program"))
    Opcodes.is_reduction(e.ops[end]) || throw(ArgumentError(
        "program must end in a reduction terminal"))
    sc = Int32.(collect(scalars))
    handle = Parent.retry_on_oom(() -> Parent.launch_pipeline_reduce(
        Parent._force(v).handle, e.ops, Parent._veclist(operands), sc))
    return Parent.DpuFuture(handle)
end

# Internal bridge for the native 8-byte {Int32 value, UInt32 index} terminal.
# The wrapper transports those bytes as a UInt64 only after the DPU reduction;
# this is not scalar packing in the kernel.
function _dpu_argreduce(v, e::DpuExpr;
                        operands::AbstractVector{<:Parent.DPUArray} = Parent.DPUArray[],
                        scalars::AbstractVector{<:Integer} = Int32[])
    _check_program(e, operands)
    isempty(e.ops) && throw(ArgumentError("empty program"))
    e.ops[end] in (Opcodes.OP_ARGMIN_REDUCE, Opcodes.OP_ARGMAX_REDUCE) ||
        throw(ArgumentError("program must end in an arg-reduction terminal"))
    sc = Int32.(collect(scalars))
    bits = Parent.retry_on_oom(() -> Parent._argreduce(
        Parent._force(v).handle, e.ops, Parent._veclist(operands), sc))
    value = reinterpret(Int32, UInt32(bits & typemax(UInt32)))
    index = UInt32(bits >> 32)
    return Int64(value), Int(index) + 1
end

"""
    transform(f, v, operands...; scalars=Int32[]) -> DPUArray

Build and run one elementwise RPN expression. `f` receives the input and
operand leaves; `scalars` supplies `scalar_var(i)` values.
"""
function transform(f, v, operands...;
                   scalars::AbstractVector{<:Integer} = Int32[])
    exprs = DpuExpr[input()]
    append!(exprs, (operand(i) for i in 1:length(operands)))
    return dpu_pipeline(Parent._force(v), f(exprs);
                        operands = Parent.DPUArray[map(Parent._force, operands)...],
                        scalars = scalars)
end

"""
    reduce_expr(f, v, operands...; scalars=Int32[]) -> DpuFuture

Build and run one RPN expression ending in a reduction terminal.
"""
function reduce_expr(f, v, operands...;
                     scalars::AbstractVector{<:Integer} = Int32[])
    exprs = DpuExpr[input()]
    append!(exprs, (operand(i) for i in 1:length(operands)))
    return dpu_pipeline_reduce(Parent._force(v), f(exprs);
                               operands = Parent.DPUArray[map(Parent._force, operands)...],
                               scalars = scalars)
end

"""
    dpu_pipeline_multi(v, programs; operands=DPUArray[], scalars=Int32[])
        -> Vector{DPUArray}

Run independent RPN `programs` in one kernel. All programs share the same input,
operand, and scalar slots. Returns one device-resident vector per program.
"""
function dpu_pipeline_multi(
    v::Parent.DPUArray, chains::AbstractVector{DpuExpr};
    operands::AbstractVector{<:Parent.DPUArray} = Parent.DPUArray[],
    scalars::AbstractVector{<:Integer} = Int32[])
    isempty(chains) && throw(ArgumentError("need at least one chain"))
    length(chains) <= Parent.MAX_CHAINS || throw(ArgumentError(
        "$(length(chains)) chains exceeds MAX_HFUSE_CHAINS ($(Parent.MAX_CHAINS))"))
    program = chain(chains...)
    _check_program(program, operands)
    dests = [similar(v) for _ in chains]
    sc = Int32.(collect(scalars))
    Parent.retry_on_oom(() -> Parent.launch_pipeline_multi(
        Parent._veclist(dests), v.handle, program.ops,
        Parent._veclist(operands), sc))
    return dests
end
