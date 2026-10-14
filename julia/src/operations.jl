# Operation dispatch for DPUVector.
#
# Ops are named by their generated internal opcode definitions, which the generator emits
# alongside common/opcodes.h; the C++ wrapper switches on the same value.

# ---- generic dispatch functions ----

function binary_op(a::DPUVector, b::DPUVector, op::UInt8)
    handle = retry_on_oom(() -> PolymerPIM.launch_binary(a.handle, b.handle, op))
    return DPUVector(handle)
end

function scalar_op(a::DPUVector, s::Integer, op::UInt8)
    handle = retry_on_oom(() -> PolymerPIM.launch_binary_scalar(a.handle, Int32(s), op))
    return DPUVector(handle)
end

function unary_op(a::DPUVector, op::UInt8)
    handle = retry_on_oom(() -> PolymerPIM.launch_unary(a.handle, op))
    return DPUVector(handle)
end

function select_op(cond::DPUVector, a::DPUVector, b::DPUVector)
    handle = retry_on_oom(() -> PolymerPIM.launch_select(cond.handle, a.handle, b.handle))
    return DPUVector(handle)
end

# ---- Base overloads: binary vector ⊕ vector ----

Base.:+(a::DPUVector, b::DPUVector) = binary_op(a, b, Internal.Opcodes.OP_ADD)
Base.:-(a::DPUVector, b::DPUVector) = binary_op(a, b, Internal.Opcodes.OP_SUB)
Base.:*(a::DPUVector, b::DPUVector) = binary_op(a, b, Internal.Opcodes.OP_MUL)
Base.div(a::DPUVector, b::DPUVector) = binary_op(a, b, Internal.Opcodes.OP_DIV)
Base.:<(a::DPUVector, b::DPUVector)  = binary_op(a, b, Internal.Opcodes.OP_LT)

# ---- Base overloads: vector ⊕ scalar / scalar ⊕ vector ----

Base.:+(a::DPUVector, s::Integer) = scalar_op(a, s, Internal.Opcodes.OP_ADD_SCALAR)
Base.:+(s::Integer, a::DPUVector) = scalar_op(a, s, Internal.Opcodes.OP_ADD_SCALAR)
Base.:-(a::DPUVector, s::Integer) = scalar_op(a, s, Internal.Opcodes.OP_SUB_SCALAR)
Base.:*(a::DPUVector, s::Integer) = scalar_op(a, s, Internal.Opcodes.OP_MUL_SCALAR)
Base.:*(s::Integer, a::DPUVector) = scalar_op(a, s, Internal.Opcodes.OP_MUL_SCALAR)
Base.div(a::DPUVector, s::Integer) = scalar_op(a, s, Internal.Opcodes.OP_DIV_SCALAR)
Base.:>>(a::DPUVector, s::Integer) = scalar_op(a, s, Internal.Opcodes.OP_ASR_SCALAR)
Base.:(==)(a::DPUVector, s::Integer) = scalar_op(a, s, Internal.Opcodes.OP_EQ_SCALAR)

# ---- Base overloads: unary ----

Base.:-(a::DPUVector)  = unary_op(a, Internal.Opcodes.OP_NEGATE)
Base.abs(a::DPUVector) = unary_op(a, Internal.Opcodes.OP_ABS)

# ---- Base overloads: reductions ----

# Futures, not numbers: left unread, independent reductions share a kernel.
Base.sum(v::DPUVector)     = Internal.reduce_lazy(v, Internal.Opcodes.OP_SUM)
Base.prod(v::DPUVector)    = Internal.reduce_lazy(v, Internal.Opcodes.OP_PRODUCT)
Base.minimum(v::DPUVector) = Internal.reduce_lazy(v, Internal.Opcodes.OP_MIN)
Base.maximum(v::DPUVector) = Internal.reduce_lazy(v, Internal.Opcodes.OP_MAX)

# `sum(f, v)`: f is traced once over a DpuExpr and the terminal appended, so
# unlike `sum(f.(v))` there is no intermediate.
for (f, terminal) in ((:sum, :sum), (:prod, :prod),
                      (:minimum, :minimum), (:maximum, :maximum))
    @eval Base.$f(f, v::DPUVector) =
        Internal.dpu_pipeline_reduce(v, $terminal(_trace(f)))
end

const MAPREDUCE_TERMINALS = Dict{Any,Function}(
    (+) => sum, (*) => prod, min => minimum, max => maximum)

# ---- Base overloads: which element won ----
# One pair-valued terminal carries the Int32 value and UInt32 global index.
function _arg_reduce(v::DPUVector, want_max::Bool)
    length(v) > 0 || throw(ArgumentError("collection must be non-empty"))
    terminal = want_max ? Internal._argmax_terminal : Internal._argmin_terminal
    return Internal._dpu_argreduce(v, terminal(Internal.input()))
end

"""
    findmax(v::DPUVector) -> (value, index)
    findmin(v::DPUVector) -> (value, index)

The extreme value and the first index holding it, as Base's do. The value is an
`Int64`, the type a DPU reduction returns, not the vector's `Int32`.

One DPU pass; only the value/index pair comes back and the vector stays put.
"""
Base.findmax(v::DPUVector) = _arg_reduce(v, true)
Base.findmin(v::DPUVector) = _arg_reduce(v, false)

"""
    argmax(v::DPUVector) -> Int
    argmin(v::DPUVector) -> Int

The first index holding the extreme value; see [`findmax`](@ref).
"""
Base.argmax(v::DPUVector) = _arg_reduce(v, true)[2]
Base.argmin(v::DPUVector) = _arg_reduce(v, false)[2]

"""
    mapreduce(f, op, v::DPUVector)

One kernel pass: `f` is traced into the program and `op` becomes its reduction
terminal. `op` must be `+`, `*`, `min` or `max`.
"""
function Base.mapreduce(f, op, v::DPUVector)
    terminal = get(MAPREDUCE_TERMINALS, op, nothing)
    terminal === nothing && throw(ArgumentError(
        "mapreduce over a DPUVector needs op in (+, *, min, max), got $op"))
    return get(Internal.dpu_pipeline_reduce(v, terminal(_trace(f))))
end

# Trace a host function over the builders; an unsupported op raises rather than
# falling back to the host.
function _trace(f)
    e = f(input())
    e isa DpuExpr || throw(ArgumentError(
        "$f did not build a DPUVector expression (got $(typeof(e)))"))
    return e
end

# ---- in-place operations ----
#
# Write through the existing buffer, so chaining them allocates no
# intermediate.

"""
    add!(a, b) / sub!(a, b) / mul!(a, b) / div!(a, b)

Apply an operation to `a` in place. `b` may be a `DPUVector` or an integer.
Returns `a`.
"""
function apply!(a::DPUVector, b::DPUVector, op::UInt8)
    retry_on_oom(() -> PolymerPIM.var"apply_binary!"(a.handle, b.handle, op))
    return a
end

function apply!(a::DPUVector, s::Integer, op::UInt8)
    retry_on_oom(() -> PolymerPIM.var"apply_scalar!"(a.handle, Int32(s), op))
    return a
end

add!(a::DPUVector, b::DPUVector) = apply!(a, b, Internal.Opcodes.OP_ADD)
sub!(a::DPUVector, b::DPUVector) = apply!(a, b, Internal.Opcodes.OP_SUB)
mul!(a::DPUVector, b::DPUVector) = apply!(a, b, Internal.Opcodes.OP_MUL)
div!(a::DPUVector, b::DPUVector) = apply!(a, b, Internal.Opcodes.OP_DIV)

add!(a::DPUVector, s::Integer) = apply!(a, s, Internal.Opcodes.OP_ADD_SCALAR)
sub!(a::DPUVector, s::Integer) = apply!(a, s, Internal.Opcodes.OP_SUB_SCALAR)
mul!(a::DPUVector, s::Integer) = apply!(a, s, Internal.Opcodes.OP_MUL_SCALAR)
div!(a::DPUVector, s::Integer) = apply!(a, s, Internal.Opcodes.OP_DIV_SCALAR)
shr!(a::DPUVector, s::Integer) = apply!(a, s, Internal.Opcodes.OP_ASR_SCALAR)

export apply!, add!, sub!, mul!, div!, shr!

# ---- broadcasting ----
#
# The tree lowers to one RPN program, so `a .+ b .* c` is one pass by
# construction rather than three ops the runtime has to fuse back together.

struct DpuStyle <: Base.Broadcast.BroadcastStyle end

Base.broadcastable(v::DPUVector) = v
Base.BroadcastStyle(::Type{DPUVector}) = DpuStyle()
Base.BroadcastStyle(::DpuStyle, ::Base.Broadcast.BroadcastStyle) = DpuStyle()

# Operators reachable inside a broadcast, mapped to the expression builder.
const _BCAST_BINARY = Dict{Any,Function}(
    (+) => (+), (-) => (-), (*) => (*), div => div, (>>) => (>>),
    (==) => (==), (<) => (<), (>) => (>), (<=) => (<=), (>=) => (>=),
)
const _BCAST_UNARY = Dict{Any,Function}(
    (-) => (-), abs => abs, identity => identity,
    # abs2 is x*x, and `sqr` loads x once (OP_DUP) where `x .* x` would twice.
    abs2 => sqr,
)
const _BCAST_SCALAR_VAR = Dict{Any,Function}(
    (+) => add_var, (-) => sub_var, (*) => mul_var, div => divide_var,
    (>>) => shr_var, (==) => eq_var, (<) => lt_var, (>) => gt_var,
    (<=) => le_var, (>=) => ge_var,
)

# Julia has already evaluated a scalar argument by the time it builds a
# Broadcasted tree.  Replace each integer occurrence with an identity-carrying
# leaf at that boundary: its value is captured now, while its opcode slot is
# assigned later when the lazy expression is submitted.
_capture_scalars(x::Integer) = _DpuScalar(Int32(x))
_capture_scalars(x::Base.RefValue{<:Integer}) = _capture_scalars(x[])
_capture_scalars(x) = x
function _capture_scalars(bc::Base.Broadcast.Broadcasted{Style}) where {Style}
    args = map(_capture_scalars, bc.args)
    return Base.Broadcast.Broadcasted{Style}(bc.f, args, bc.axes)
end

# Which vector became input(), and the operand slots so far.  Matched by object
# identity, so a vector used twice is loaded once.
mutable struct _Lowering
    primary::Union{Nothing,DPUVector}
    operands::Vector{DPUVector}
    inlined::Vector{Any}    # the lazy values folded into this program
    scalar_slots::IdDict{_DpuScalar,Int}
    scalars::Vector{Int32}  # launch-time values, in slot order
end
_Lowering() = _Lowering(nothing, DPUVector[], Any[],
                        IdDict{_DpuScalar,Int}(), Int32[])

function _leaf(v::DPUVector, st::_Lowering)
    if st.primary === nothing
        st.primary = v
        return input()
    end
    st.primary === v && return input()
    for (i, o) in enumerate(st.operands)
        o === v && return operand(i)
    end
    length(st.operands) + 1 <= MAX_VFUSE_INPUTS || throw(ArgumentError(
        "expression needs more than $MAX_VFUSE_INPUTS operand slots; running a " *
        "broadcast splits it, inspecting one does not"))
    push!(st.operands, v)
    return operand(length(st.operands))
end

_lower(v::DPUVector, st::_Lowering) = _leaf(v, st)

# Reusing the same leaf reuses its slot.  This matters for scatter, which lowers
# a shared index once per update before `_scatter_program` folds the common
# prefix back to one copy.  Independently constructed leaves remain distinct
# even when their values are equal, so the program depends on expression shape
# rather than on whether two runtime values happen to coincide.
function _scalar_slot(p::_DpuScalar, st::_Lowering)
    slot = get(st.scalar_slots, p, 0)
    slot != 0 && return slot
    length(st.scalars) < MAX_PIPELINE_SCALARS || throw(ArgumentError(
        "more than $MAX_PIPELINE_SCALARS runtime scalars in one program"))
    push!(st.scalars, p.value)
    slot = length(st.scalars)
    st.scalar_slots[p] = slot
    return slot
end
_lower(p::_DpuScalar, st::_Lowering) = scalar_var(_scalar_slot(p, st))
_lower(x::Integer, st::_Lowering) = constant(x)
_lower(e::DpuExpr, ::_Lowering) = e
_lower(x::Base.RefValue, st::_Lowering) = _lower(x[], st)

function _lower(bc::Base.Broadcast.Broadcasted, st::_Lowering)
    f = bc.f
    args = bc.args
    if f isa LaneArg   # any number of lanes, each possibly an expression
        lanes = DpuExpr[_lower(a, st) for a in args]
        return f isa LaneArg{true} ? argmax(lanes) : argmin(lanes)
    end
    if length(args) == 1
        haskey(_BCAST_UNARY, f) || throw(ArgumentError(
            "$f is not supported inside a DPUVector broadcast"))
        return _BCAST_UNARY[f](_lower(args[1], st))
    elseif length(args) == 2
        # ifelse is the broadcast spelling of a per-lane select
        f === ifelse && throw(ArgumentError("ifelse needs three arguments"))
        haskey(_BCAST_BINARY, f) || throw(ArgumentError(
            "$f is not supported inside a DPUVector broadcast"))
        op = _BCAST_BINARY[f]
        a, b = args
        # A host scalar in a lazy broadcast is a launch parameter by default.
        # Use the compact in-place scalar-var opcode for the common rhs form.
        if b isa _DpuScalar && !(a isa _DpuScalar)
            lhs = _lower(a, st)
            return _BCAST_SCALAR_VAR[f](lhs, _scalar_slot(b, st))
        end
        # An integer operand becomes an immediate rather than a pushed value.
        if b isa Integer && !(a isa Integer)
            return op(_lower(a, st), b)
        elseif a isa Integer && !(b isa Integer)
            # only the commutative/reversible ones have a scalar-first form
            f === (+) && return _lower(b, st) + a
            f === (*) && return _lower(b, st) * a
            return op(constant(a), _lower(b, st))
        end
        return op(_lower(a, st), _lower(b, st))
    elseif length(args) == 3 && f === ifelse
        return select(_lower(args[1], st), _lower(args[2], st),
                      _lower(args[3], st))
    end
    throw(ArgumentError("$f with $(length(args)) arguments is not supported " *
                        "inside a DPUVector broadcast"))
end

# To (program, primary, operands).  Not via Broadcast.flatten: that rewrites the
# tree into a closure over Pick{} leaves, erasing the operator identities this
# dispatches on.
#
# Inlining lets SROA drop `_leaf`'s store to st.primary when a caller discards it
# and a good broadcast then reports "contains no DPUVector".
@noinline function _lower_tree(bc::Base.Broadcast.Broadcasted;
                               consume::Bool = true)
    bc = _capture_scalars(bc)
    consume && (bc = _split_to_fit(bc))
    st = _Lowering()
    e = _lower(bc, st)
    st.primary === nothing &&
        throw(ArgumentError("broadcast contains no DPUVector"))
    # Folded in, so it needs no run of its own -- unless nothing was submitted,
    # i.e. the program is only being inspected.
    consume && for x in st.inlined
        x.consumed = true
        x.uses += 1
    end
    return e, st.primary, st.operands, st.scalars
end

# ---- lazy results ----
#
# `a .+ b` is the program, not the vector: inlined where it is used, so a chain
# of statements is one pass and a reduction over one has no intermediate.

"""
    DpuLazy

An expression built but not run. `Array`, `DPUVector`, indexing, `fence` or a
reduction runs it; using it inside another expression inlines it instead.
"""
mutable struct DpuLazy
    bc::Base.Broadcast.Broadcasted
    len::Int
    forced::Any     # the DPUVector once it has been run, so it runs once
    consumed::Bool  # inlined into a submitted program, so it needs no run of its own
    uses::Int       # how many submitted programs folded it in
end

# Registered weakly so `sync()` can run what nothing else will: a kept value,
# as opposed to a step towards one.
const _LAZY_REGISTRY = WeakRef[]

function DpuLazy(bc::Base.Broadcast.Broadcasted, len::Int)
    x = DpuLazy(_capture_scalars(bc), len, nothing, false, 0)
    push!(_LAZY_REGISTRY, WeakRef(x))
    return x
end

Base.copy(bc::Base.Broadcast.Broadcasted{DpuStyle}) = DpuLazy(bc, _bclength(bc))

# The element count is the primary vector's, found without lowering anything.
_bclength(x::DPUVector) = length(x)
_bclength(x::DpuLazy) = x.len
_bclength(::Any) = nothing
function _bclength(bc::Base.Broadcast.Broadcasted)
    for a in bc.args
        n = _bclength(a)
        n === nothing || return n
    end
    return nothing
end

Base.length(x::DpuLazy) = x.len
Base.size(x::DpuLazy) = (x.len,)
Base.eltype(::DpuLazy) = Int32
Base.axes(x::DpuLazy) = (Base.OneTo(x.len),)
Base.broadcastable(x::DpuLazy) = x
Base.BroadcastStyle(::Type{DpuLazy}) = DpuStyle()

# Inlined where it is used, unless already run -- then that result is cheaper.
#
# A second consumer re-derives the whole expression in its own program.
# Materialising here instead is worse: the first consumer is already queued, so
# the buffer lands mid-stream and the rest stop fusing (11 launches against 2,
# no faster).  So warn, until submission defers far enough to know every
# consumer before queueing any.
function _lower(x::DpuLazy, st::_Lowering)
    x.forced === nothing || return _leaf(x.forced, st)
    if x.uses > 0
        @warn """an unrun expression is being folded into a second program, so it \
                 is computed once per consumer.  Hoist it with `DPUVector(x)` to \
                 compute it once and share the result.""" maxlog = 3
    end
    push!(st.inlined, x)
    return _lower(x.bc, st)
end

"""
    DPUVector(x::DpuLazy)

Run `x`, keeping the result on the DPUs. [`fence`](@ref) says the same thing
without reading as a conversion.
"""
function DPUVector(x::DpuLazy)
    x.forced === nothing || return x.forced
    # An expression that cannot be lowered is not retried: otherwise it stays
    # in the registry and the next `sync()` raises it again, far from whoever
    # wrote it.
    e, primary, operands, scalars = try
        _lower_tree(x.bc)
    catch
        x.consumed = true
        rethrow()
    end
    x.forced = Internal.dpu_pipeline(
        primary, e; operands = operands, scalars = scalars)
    return x.forced
end

Base.Array(x::DpuLazy) = Array(DPUVector(x))
Base.Vector(x::DpuLazy) = Array(x)
Base.collect(x::DpuLazy) = Array(x)
Base.getindex(x::DpuLazy, i::Integer) = Array(x)[i]

"""
    fence(x::DpuLazy)

Run `x`, block until it is done, return it. `fence(v::DPUVector)` waits for a
vector's queued work; this also submits an expression that has not run.

Only the named value runs. Its intermediates stay unrun -- forcing those would
cost a kernel and an MRAM buffer each.

    res = op(da, db)     # nothing has run
    fence(res)           # runs here, so this is where the time is spent
    Array(res)           # already computed
"""
fence(x::DpuLazy) = (fence(DPUVector(x)); x)

const _Lane = Union{DPUVector,DpuLazy}

# An operand slot needs a real vector; anything else passes through.
_force(x) = x
_force(x::DpuLazy) = DPUVector(x)

# ---- splitting an expression too wide for the operand table ----
#
# The runtime only declines to merge, never cuts: a chain of statements is
# already legal unfused.  A tree arrives as one program, so cutting one that
# needs more vectors than the table holds has to happen here.

const _MAX_PROGRAM_VECTORS = MAX_VFUSE_INPUTS + 1   # the primary holds slot 0

# A subtree and the vectors it reads, by identity as `_leaf` matches them.
struct _Fitted
    node::Any
    vecs::Base.IdSet{Any}
end

_vecs!(seen, ::Any) = nothing
_vecs!(seen, v::DPUVector) = (push!(seen, v); nothing)
_vecs!(seen, x::DpuLazy) =
    x.forced === nothing ? _vecs!(seen, x.bc) : _vecs!(seen, x.forced)
_vecs!(seen, bc::Base.Broadcast.Broadcasted) =
    (foreach(a -> _vecs!(seen, a), bc.args); nothing)

_vecs(x) = (seen = Base.IdSet{Any}(); _vecs!(seen, x); seen)

_fits(vecs) = length(vecs) <= _MAX_PROGRAM_VECTORS

function _all_vecs(parts)
    vecs = Base.IdSet{Any}()
    for p in parts
        union!(vecs, p.vecs)
    end
    return vecs
end

# A cut result reaches no caller, so the GC never sees the MRAM it holds and a
# query loop exhausts the DPUs.  Released once the next expression is lowered:
# by then the program reading it is submitted, which is where C++ drops its own
# handle.  `release!` only decrements the handle count, so a queued reader is
# unaffected.
const _CUT_TEMPS = DPUVector[]

function _release_cut_temps!()
    for v in _CUT_TEMPS
        release!(v)
    end
    empty!(_CUT_TEMPS)
    return nothing
end

# Run a subtree now, so what encloses it reads one vector instead of many.
function _cut(bc::Base.Broadcast.Broadcasted)
    e, primary, operands, scalars = _lower_tree(bc)
    v = Internal.dpu_pipeline(primary, e; operands = operands,
                              scalars = scalars)
    push!(_CUT_TEMPS, v)
    return v
end

# Cut the tree, not the wrapper: forcing it would hand the caller a vector we
# are about to release.  It is folded into the cut program, so mark it consumed
# or `sync()` runs it a second time.
function _cut(x::DpuLazy)
    v = _cut(x.bc)
    x.consumed = true
    x.uses += 1
    return v
end

_fit(x) = _Fitted(x, _vecs(x))

# A statement chain wraps each step in a lazy value and the lowering inlines
# those, so this must too -- otherwise every wrapper reads as its own cut point.
function _fit(x::DpuLazy)
    x.forced === nothing || return _fit(x.forced)
    fitted = _fit(x.bc)
    x.bc = fitted.node
    return _Fitted(x, fitted.vecs)
end

function _fit(bc::Base.Broadcast.Broadcasted{Style}) where {Style}
    parts = _Fitted[_fit(a) for a in bc.args]
    vecs = _all_vecs(parts)
    # Widest first, so each program spent buys back the most slots.
    for i in sortperm(parts; by = p -> length(p.vecs), rev = true)
        _fits(vecs) && break
        length(parts[i].vecs) > 1 || break   # sorted, so the rest are leaves
        cut = _cut(parts[i].node)
        parts[i] = _Fitted(cut, _vecs(cut))
        vecs = _all_vecs(parts)
    end
    args = ntuple(i -> parts[i].node, length(parts))
    return _Fitted(Base.Broadcast.Broadcasted{Style}(bc.f, args, bc.axes), vecs)
end

const _SPLIT_DEPTH = Ref(0)

# One walk when it already fits, which is the usual case.
function _split_to_fit(bc::Base.Broadcast.Broadcasted)
    # Nested calls come from `_cut`; releasing there would drop an intermediate
    # before the program that reads it has been submitted.
    _SPLIT_DEPTH[] == 0 && _release_cut_temps!()
    _SPLIT_DEPTH[] += 1
    try
        return _fits(_vecs(bc)) ? bc : _fit(bc).node
    finally
        _SPLIT_DEPTH[] -= 1
    end
end

# Operators on an unrun expression keep it unrun.  On a DPUVector they stay
# eager: `a + b` is a statically compiled kernel.
_lazy(bc::Base.Broadcast.Broadcasted) = DpuLazy(bc, _bclength(bc))

Base.:-(x::DpuLazy) = _lazy(Base.broadcasted(-, x))
Base.abs(x::DpuLazy) = _lazy(Base.broadcasted(abs, x))

for f in (:+, :-, :*, :div, :(>>), :(==), :<, :>, :<=, :>=)
    @eval Base.$f(x::DpuLazy, y::Union{DpuLazy,DPUVector,Integer}) =
        _lazy(Base.broadcasted($f, x, y))
    @eval Base.$f(x::Union{DPUVector,Integer}, y::DpuLazy) =
        _lazy(Base.broadcasted($f, x, y))
end

# One pass: the terminal joins the program rather than reducing a materialised
# intermediate.
for (f, terminal) in ((:sum, :sum), (:prod, :prod),
                      (:minimum, :minimum), (:maximum, :maximum))
    @eval function Base.$f(x::DpuLazy)
        # Already run: reduce that result rather than re-deriving the program.
        x.forced === nothing || return Base.$f(x.forced)
        e, primary, operands, scalars = _lower_tree(x.bc)
        x.consumed = true       # reduced here, so `sync()` must not run it too
        x.uses += 1
        return Internal.dpu_pipeline_reduce(
            primary, $terminal(e); operands = operands, scalars = scalars)
    end
end

Base.sum(f, x::DpuLazy) = sum(Base.broadcasted(f, x))
Base.prod(f, x::DpuLazy) = prod(Base.broadcasted(f, x))
Base.minimum(f, x::DpuLazy) = minimum(Base.broadcasted(f, x))
Base.maximum(f, x::DpuLazy) = maximum(Base.broadcasted(f, x))

# Fuse a lazy score expression directly into the native pair terminal.
function _arg_reduce(x::DpuLazy, want_max::Bool)
    x.forced === nothing || return _arg_reduce(x.forced, want_max)
    length(x) > 0 || throw(ArgumentError("collection must be non-empty"))
    e, primary, operands, scalars = _lower_tree(x.bc)
    x.consumed = true
    x.uses += 1
    terminal = want_max ? Internal._argmax_terminal : Internal._argmin_terminal
    return Internal._dpu_argreduce(
        primary, terminal(e); operands = operands, scalars = scalars)
end

Base.findmin(x::DpuLazy) = _arg_reduce(x, false)
Base.findmax(x::DpuLazy) = _arg_reduce(x, true)
Base.argmin(x::DpuLazy) = findmin(x)[2]
Base.argmax(x::DpuLazy) = findmax(x)[2]

"""
    dest .= expr

Writes through `dest`'s existing buffer, so other handles to it observe the
result. One kernel pass.
"""
Base.copyto!(dest::DPUVector, x::DpuLazy) =
    (x.consumed = true; copyto!(dest, x.bc))

function Base.copyto!(dest::DPUVector, bc::Base.Broadcast.Broadcasted{DpuStyle})
    e, primary, operands, scalars = _lower_tree(bc)
    length(dest) == length(primary) || throw(DimensionMismatch(
        "destination has $(length(dest)) elements, expression $(length(primary))"))
    Internal._check_program(e, operands)
    retry_on_oom(() -> PolymerPIM.launch_pipeline_into(
        dest.handle, primary.handle, e.ops, _veclist(operands), scalars))
    return dest
end

# A scalar fill still goes through the same path.
Base.copyto!(dest::DPUVector, bc::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}) =
    copyto!(dest, Base.Broadcast.broadcasted(identity, bc.f(bc.args...)))

Base.similar(v::DPUVector) = DPUVector(length(v))
Base.similar(v::DPUVector, ::Type{Int32}) = DPUVector(length(v))
Base.axes(v::DPUVector) = (Base.OneTo(length(v)),)

export select_op

# ---- launch argument helpers ----

function _veclist(vs)
    vs = map(_force, vs)
    l = PolymerPIM.DpuVecList()
    for v in vs
        PolymerPIM.var"veclist_push!"(l, v.handle)
    end
    return l
end

# ---- per-element winner across K vectors ----
#
# `argmin(collection)` is Base's index of the smallest element, so the winning
# lane per position is `argmin.(zip(v1, v2, v3))`, 1-based as Julia's is.

const DpuZip = Base.Iterators.Zip{<:Tuple{_Lane,Vararg{_Lane}}}

# Any other broadcast over a zip would have Base collect it -- one readback
# per element.
Base.broadcastable(::DpuZip) = throw(ArgumentError(
    "only argmin./argmax. are supported over zip(::DPUVector...); another " *
    "broadcast would collect the vectors to the host one element at a time"))

function _lane_program(nlanes::Integer, want_max::Bool)
    lanes = DpuExpr[input(); [operand(j) for j in 1:(nlanes - 1)]]
    label = want_max ? argmax(lanes) : argmin(lanes)
    return lanes, label
end

# Lazy like any other broadcast, so `argmin.(zip(a, b)) .* 3` is one program.
# LaneArg is a marker for the lowering to dispatch on; it is never called.
struct LaneArg{Max} end

Base.broadcasted(::typeof(argmin), z::DpuZip) =
    Base.broadcasted(DpuStyle(), LaneArg{false}(), z.is...)
Base.broadcasted(::typeof(argmax), z::DpuZip) =
    Base.broadcasted(DpuStyle(), LaneArg{true}(), z.is...)

function _find_lanes(vs::AbstractVector{DPUVector}, want_max::Bool)
    isempty(vs) && throw(ArgumentError("need at least one vector"))
    lanes, label = _lane_program(length(vs), want_max)
    value = _best_expr(lanes, want_max)
    if MAX_CHAINS < 2   # room for one chain only; a pass each
        return (Internal.dpu_pipeline(vs[1], value; operands = vs[2:end]),
                Internal.dpu_pipeline(vs[1], label; operands = vs[2:end]))
    end
    values, labels = Internal.dpu_pipeline_multi(
        vs[1], [value, label]; operands = vs[2:end])
    return values, labels
end

"""
    findmin_lanes(vectors) -> (values, labels)
    findmax_lanes(vectors) -> (values, labels)

Per element, the winning value and the 1-based index of the vector it came
from, in one kernel pass. Julia's `findmin.(zip(v1, v2, v3))` is a vector of
tuples, which a DPU cannot hold, so the columns come back unzipped:

    values, labels = findmin_lanes([v1, v2, v3])
    collect(zip(Array(values), Array(labels))) == findmin.(zip(a1, a2, a3))
"""
findmin_lanes(vs::AbstractVector{DPUVector}) = _find_lanes(vs, false)
findmax_lanes(vs::AbstractVector{DPUVector}) = _find_lanes(vs, true)

export findmin_lanes, findmax_lanes

# No min-of-K-vectors opcode, so the value is a select chain.  Strict
# comparison, so ties keep the lowest lane and value and label agree.
function _best_expr(lanes::AbstractVector{DpuExpr}, want_max::Bool)
    best = lanes[1]
    for j in 2:length(lanes)
        best = select(want_max ? best < lanes[j] : lanes[j] < best, lanes[j], best)
    end
    return best
end

"""
    min_squared_distance(cols, query) -> DpuFuture

Minimum over rows of the squared euclidean distance to `query`, where `cols[j]`
holds coordinate `j` of every row. One fused pass over all columns.

`vectordpu.h` declared a C++ `min_squared_distance` but never defined it, so
this is built from the expression API instead.
"""
function min_squared_distance(cols::AbstractVector{DPUVector},
                              query::AbstractVector{<:Integer})
    isempty(cols) && throw(ArgumentError("need at least one column"))
    length(cols) == length(query) || throw(ArgumentError(
        "$(length(cols)) columns but $(length(query)) query coordinates"))
    distance = abs2.(cols[1] .- query[1])
    for j in 2:length(cols)
        distance = distance .+ abs2.(cols[j] .- query[j])
    end
    return minimum(distance)
end

export min_squared_distance

# ---- elementwise comparisons ----

_materialize_broadcast(f, args...) =
    DPUVector(Base.copy(Base.broadcasted(f, args...)))

for f in (:>, :>=, :<=)
    @eval Base.$f(a::DPUVector, b::DPUVector) =
        _materialize_broadcast($f, a, b)
end

Base.:(==)(a::DPUVector, b::DPUVector) = _materialize_broadcast(==, a, b)

for f in (:>, :>=, :<=, :<)
    @eval Base.$f(a::DPUVector, s::Integer) =
        _materialize_broadcast($f, a, s)
end

# ---- local (WRAM) scatter accumulators ----
#
# `bins[idx] .+= v` records an accumulation; nothing launches.  Index and value
# stay lazy, so several updates lower into one program.  Reading a local, or
# `sync()`, flushes them.

"""
    DPULocalVector(n; reduce_op = :sum)

A small per-DPU accumulator array in WRAM. Scatter into it by indexing with a
lazy expression, and read it with `Array`, which merges every DPU's copy with
`reduce_op`:

    bins = DPULocalVector(16)
    bins[(da .* 16) .>> 10] .+= 1        # queued
    Array(bins)                          # flushed, then merged

`reduce_op` is one of `:sum`, `:product`, `:min`, `:max`, and is also the
accumulation each update performs, so `.+=` belongs to a `:sum` local and
`.= min.(...)` to a `:min` one.
"""
mutable struct DPULocalVector
    handle::Any
    len::Int
    reduce_op::Symbol
end

function DPULocalVector(n::Integer; reduce_op::Symbol = :sum)
    reduce_op in LOCAL_REDUCE_OPS || throw(ArgumentError(
        "reduce_op must be one of $(LOCAL_REDUCE_OPS)"))
    idx = findfirst(==(reduce_op), LOCAL_REDUCE_OPS) - 1
    handle = retry_on_oom(() -> PolymerPIM.local_alloc(Int32(n), Int32(idx)))
    return DPULocalVector(handle, Int(n), reduce_op)
end

Base.length(l::DPULocalVector) = l.len

# In the order written: one flush, one program, so updates to different locals
# share a pass.
struct _PendingUpdate
    target::DPULocalVector
    op::UInt8
    index::Any
    value::Any
end

const _PENDING_UPDATES = _PendingUpdate[]

# `bins[idx]` on its own is not a value -- only the target of an accumulation.
struct _LocalSlot
    target::DPULocalVector
    index::Any
end

# `bins[i] .+= v` desugars to `bins[i] .= bins[i] .+ v`: only the assignment
# target goes through dotview, so the read needs the same slot object.
Base.dotview(l::DPULocalVector, index) = _LocalSlot(l, index)
Base.getindex(l::DPULocalVector, index) = _LocalSlot(l, index)

const _ACCUM_OPS = Dict{Any,UInt8}(
    (+) => Internal.Opcodes.OP_SUM, (*) => Internal.Opcodes.OP_PRODUCT,
    min => Internal.Opcodes.OP_MIN, max => Internal.Opcodes.OP_MAX,
)

struct _LocalAccum
    slot::_LocalSlot
    op::UInt8
    value::Any
end

function Base.broadcasted(f, slot::_LocalSlot, value)
    op = get(_ACCUM_OPS, f, nothing)
    op === nothing && throw(ArgumentError(
        "a local accumulator supports .+=, .*=, min and max, not $f"))
    return _LocalAccum(slot, op, value)
end

Base.broadcasted(f, value, slot::_LocalSlot) = Base.broadcasted(f, slot, value)

function Base.materialize!(dest::_LocalSlot, acc::_LocalAccum)
    acc.slot.target === dest.target || throw(ArgumentError(
        "a local accumulation reads and writes the same local vector"))
    acc.op == _local_reduce_opcode(dest.target.reduce_op) || throw(ArgumentError(
        "this accumulation is $(acc.op) but the local vector merges with " *
        ":$(dest.target.reduce_op); they have to agree"))
    push!(_PENDING_UPDATES, _PendingUpdate(dest.target, acc.op,
                                           _capture_scalars(dest.index),
                                           _capture_scalars(acc.value)))
    return dest.target
end

_no_accum() = throw(ArgumentError(
    "a local vector can only be accumulated into: bins[i] .+= v"))
Base.materialize!(::_LocalSlot, ::Any) = _no_accum()
Base.materialize!(::_LocalSlot, ::Base.Broadcast.Broadcasted) = _no_accum()

_lower_operand(x, st::_Lowering) = _lower(x, st)
_lower_operand(x::Base.Broadcast.Broadcasted, st::_Lowering) = _lower(x, st)

# The queued updates as one program, unlaunched: what flush_locals! submits and
# `@code_jitted` shows.
function _pending_program(updates::Vector{_PendingUpdate};
                          consume::Bool = true)
    st = _Lowering()
    locals = DPULocalVector[]
    reductions = _LocalReduce[]
    for u in updates
        index = _lower_operand(u.index, st)
        value = _lower_operand(u.value, st)
        slot = findfirst(l -> l === u.target, locals)
        if slot === nothing
            push!(locals, u.target)
            slot = length(locals)
        end
        push!(reductions, _LocalReduce(slot - 1, u.op, index, value))
    end
    length(locals) <= MAX_LOCAL_SCRATCH_VECTORS || throw(ArgumentError(
        "$(length(locals)) local vectors exceeds MAX_LOCAL_SCRATCH_VECTORS " *
        "($MAX_LOCAL_SCRATCH_VECTORS); WRAM has room for no more, and the " *
        "extras would silently read back as zeros"))
    st.primary === nothing && throw(ArgumentError(
        "a scatter needs a DPUVector in its index or value"))
    # Folded in, so `sync()` must not run them a second time.
    consume && for x in st.inlined
        x.consumed = true
        x.uses += 1
    end
    return _scatter_program(reductions), st.primary, st.operands, st.scalars,
           locals
end

# Alive, unrun, and not folded into anything: the values a caller still holds.
# A step in a larger expression is referenced by its consumer, so not one.
function _dangling_lazies()
    live, keep = DpuLazy[], WeakRef[]
    for wr in _LAZY_REGISTRY
        x = wr.value
        x === nothing && continue
        push!(keep, wr)
        (x.forced === nothing && !x.consumed) && push!(live, x)
    end
    resize!(_LAZY_REGISTRY, 0)
    append!(_LAZY_REGISTRY, keep)
    isempty(live) && return DpuLazy[]

    referenced = Base.IdSet{DpuLazy}()
    for x in live
        _mark_referenced(x.bc, referenced)
    end
    return [x for x in live if !(x in referenced)]
end

function _mark_referenced(bc::Base.Broadcast.Broadcasted, seen)
    for a in bc.args
        _mark_referenced(a, seen)
    end
    return nothing
end
function _mark_referenced(x::DpuLazy, seen)
    x in seen && return nothing
    push!(seen, x)
    _mark_referenced(x.bc, seen)
    return nothing
end
_mark_referenced(::Any, _) = nothing

# `sync()`'s half of the bargain: a kept result gets computed, the steps behind
# it do not.
function _run_dangling_lazies()
    for x in _dangling_lazies()
        DPUVector(x)
    end
    return nothing
end

"""
    flush_locals!()

Launch the scatter program the queued updates describe, if any. Called by
[`sync`](@ref) and by reading a local vector, so it rarely needs calling
directly.
"""
function flush_locals!()
    isempty(_PENDING_UPDATES) && return nothing
    updates = copy(_PENDING_UPDATES)
    empty!(_PENDING_UPDATES)
    program, primary, operands, scalars, locals = _pending_program(updates)
    ll = PolymerPIM.DpuLocalList()
    for l in locals
        PolymerPIM.var"locallist_push!"(ll, l.handle)
    end
    retry_on_oom(() -> PolymerPIM.launch_pipeline_scatter(
        primary.handle, program.ops, _veclist(operands), scalars, ll))
    return nothing
end

function Base.Array(l::DPULocalVector)
    flush_locals!()
    out = Vector{Int32}(undef, l.len)
    retry_on_oom(() -> PolymerPIM.var"local_to_cpu!"(l.handle, out))
    return out
end

export DPULocalVector, flush_locals!
