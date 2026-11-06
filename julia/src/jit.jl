# What the JIT will generate for an expression, without compiling or launching it.

"""
    JittedCode

The C kernel one RPN program compiles to: the opcode stream `ops`, its cache key
`hash`, the `source`, the `path` `jit_compile` writes it to, and the shape it was
lowered from (`nelements`, `noperands`).
"""
struct JittedCode
    ops::Vector{UInt8}
    hash::String
    source::String
    path::String
    nelements::Int
    noperands::Int
    nchains::Int
end

# `source` keeps the shared preamble, so it matches the compiled file byte for
# byte; printing skips to the function.
function _body(c::JittedCode)
    at = findfirst("int k_" * c.hash, c.source)
    return at === nothing ? (c.source, false) : (c.source[first(at):end], true)
end

function Base.show(io::IO, ::MIME"text/plain", c::JittedCode)
    println(io, "JIT kernel k_", c.hash, " -- ", length(c.ops), " opcodes, ",
            c.noperands, " operand", c.noperands == 1 ? "" : "s",
            c.nelements > 0 ? ", $(c.nelements) elements" : "")
    if c.nchains > 1
        println(io, "  horizontally fused: ", c.nchains,
                " chains sharing one pass over the input")
    end
    body, elided = _body(c)
    # Checked here, not at construction: the kernel is written on first launch.
    println(io, "  ", c.path, iscompiled(c) ? " (compiled)" : " (not compiled yet)",
            elided ? "; preamble elided, full text in .source" : "")
    println(io)
    print(io, body)
end

Base.show(io::IO, c::JittedCode) = print(io, "JittedCode(k_", c.hash, ")")

"""
    code_jitted(x)

The generated kernel for a lazy broadcast, a `DpuExpr`, or an opcode stream.
Usually reached through [`@code_jitted`](@ref).
"""
function code_jitted(ops::AbstractVector{UInt8}; nelements = 0, noperands = 0,
                     nchains = 1)
    ops = Vector{UInt8}(ops)
    hash = String(jit_hash(ops))
    return JittedCode(ops, hash, String(jit_source(ops)),
                      joinpath(String(jit_dir()), "k_" * hash * ".c"),
                      Int(nelements), Int(noperands), Int(nchains))
end

code_jitted(e::DpuExpr; kwargs...) = code_jitted(e.ops; kwargs...)

function code_jitted(bc::Base.Broadcast.Broadcasted)
    e, primary, operands, _ = _lower_tree(bc; consume = false)
    return code_jitted(e; nelements = length(primary), noperands = length(operands))
end

# An unrun expression: the program it would submit.  Inspecting must not count
# as a use, or it would change whether `sync()` runs it.
code_jitted(x::DpuLazy) = code_jitted(x.bc)

"""
    iscompiled(c::JittedCode)

Whether the JIT has written this kernel yet. It appears at `c.path` the first
time a program with this `hash` is launched, and is reused from then on.
"""
iscompiled(c::JittedCode) = isfile(c.path)

# Already materialised: it ran a statically compiled kernel (`a + b`, `sum(a)`).
code_jitted(x) = error("""
    a $(typeof(x)) is materialised, so it came from a statically compiled kernel.
    @code_jitted takes broadcasts, reductions over them, and RPN programs:
        @code_jitted a .+ b
        @code_jitted sum(a .+ b)""")

# `sum(a .+ b)`: append the reduction terminal to the lazy chain.
const REDUCERS = (:sum, :prod, :minimum, :maximum)

function _code_jitted_reduce(f, bc::Base.Broadcast.Broadcasted)
    e, primary, operands, _ = _lower_tree(bc; consume = false)
    return code_jitted(f(e); nelements = length(primary),
                       noperands = length(operands))
end

_code_jitted_reduce(f, e::DpuExpr) = code_jitted(f(e))
_code_jitted_reduce(f, x::DpuLazy) = _code_jitted_reduce(f, x.bc)
_code_jitted_reduce(f, x) = code_jitted(x)

# ---- horizontal fusion: independent reductions, one kernel, a chain each ----

const _O = Internal.Opcodes

# A reduction's program with what its opcodes refer to: input 1 is the primary
# (OP_PUSH_INPUT), the rest operands.
struct _Chain
    ops::Vector{UInt8}
    inputs::Vector{DPUArray}
    nscalars::Int
end

function _chain_of(f, bc::Base.Broadcast.Broadcasted)
    e, primary, operands, scalars = _lower_tree(bc; consume = false)
    return _Chain(f(e).ops, DPUArray[primary, operands...], length(scalars))
end

_chain_of(f, v::DPUArray) = _Chain(f(Internal.input()).ops, DPUArray[v], 0)
_chain_of(f, x::DpuLazy) = _chain_of(f, x.bc)
_chain_of(f, x) = error("each reduction must be over a DPUVector or a broadcast " *
                        "over one; got a $(typeof(x))")

# Which input an opcode reads, 0 for none.
function _reads(op::UInt8)
    op == _O.OP_PUSH_INPUT && return 1
    lo = _O.OP_PUSH_OPERAND_0
    return lo <= op < lo + MAX_VFUSE_INPUTS ? Int(op - lo) + 2 : 0
end

_pushes_scalar(op::UInt8) = op == _O.OP_PUSH_SCALAR_VAR ||
    _O.OP_ADD_SCALAR_VAR <= op <= _O.OP_LE_SCALAR_VAR

function _slot!(fused::Vector{DPUArray}, v::DPUArray)
    at = findfirst(x -> x === v, fused)
    at === nothing || return at
    push!(fused, v)
    return length(fused)
end

# Rewrite a chain onto the shared input and scalar tables, as push_op_for() and
# the scalar-slot shift in host/hfuse.cc do.
function _remap(c::_Chain, fused::Vector{DPUArray}, scalar_base::Int)
    slots = [_slot!(fused, v) for v in c.inputs]
    out, i = UInt8[], 1
    while i <= length(c.ops)
        op = c.ops[i]
        n = Int(_O.inline_bytes(op))
        read = _reads(op)
        push!(out, read == 0 ? op :
              slots[read] == 1 ? _O.OP_PUSH_INPUT :
              UInt8(_O.OP_PUSH_OPERAND_0 + slots[read] - 2))
        if n == 1 && _pushes_scalar(op)
            push!(out, UInt8(scalar_base + c.ops[i + 1]))
        else
            append!(out, c.ops[(i + 1):(i + n)])
        end
        i += 1 + n
    end
    return out
end

function _code_jitted_hfused(cs::Vector{_Chain})
    length(cs) <= MAX_CHAINS || error(
        "$(length(cs)) reductions exceeds MAX_HFUSE_CHAINS ($MAX_CHAINS), so the " *
        "queue splits them across passes")
    fused, base, chains = DPUArray[], 0, Internal.DpuExpr[]
    for c in cs
        push!(chains, Internal.DpuExpr(_remap(c, fused, base)))
        base += c.nscalars
    end
    return code_jitted(Internal.chain(chains...);
                       nelements = length(first(fused)),
                       noperands = length(fused) - 1, nchains = length(cs))
end

# `sum(abs, a)` / `mapreduce(abs, +, a)`: trace the function, append the terminal.
_code_jitted_map(terminal, f, v::DPUArray) =
    code_jitted(terminal(_trace(f)); nelements = length(v))

function _code_jitted_mapreduce(f, op, v::DPUArray)
    terminal = get(MAPREDUCE_TERMINALS, op, nothing)
    terminal === nothing && throw(ArgumentError("op must be +, *, min or max"))
    return _code_jitted_map(terminal, f, v)
end

# The arg forms launch as they build, so their result is already materialised;
# rebuild the program the launcher submitted.
const ARG_FORMS = (:argmin, :argmax, :findmin, :findmax)


function _code_jitted_scatter(statement)
    before = length(_PENDING_UPDATES)
    statement()
    length(_PENDING_UPDATES) > before || throw(ArgumentError(
        "that statement queued no local accumulation"))
    queued = _PENDING_UPDATES[(before + 1):end]
    resize!(_PENDING_UPDATES, before)
    program, primary, operands, _, _ = _pending_program(queued; consume = false)
    return code_jitted(program; nelements = length(primary),
                       noperands = length(operands))
end

# Whole-vector arg forms are one pair-valued reduction terminal.
function _code_jitted_arg(name::Symbol, v::DPUArray)
    terminal = name in (:argmax, :findmax) ? Internal._argmax_terminal :
                                            Internal._argmin_terminal
    return code_jitted(terminal(Internal.input()); nelements = length(v))
end

_code_jitted_arg(name::Symbol, x) = error(
    "@code_jitted $name takes a DPUVector or a list of them, got a $(typeof(x))")

# Assignments describe their right-hand side.  Stripped before dispatch, so a
# reduction under one stays visible; nothing is launched, so nothing is assigned.
_host_combined_message(inner) = """
    this reduction is combined on the host, so the result is a value, not a program.
    Inspect the reduction itself:
        @code_jitted $(inner[1])"""

# Reducer calls nested inside something else: `sum(x) + sum(y)` is two programs
# and a host-side add, so no one kernel describes it.
function _reduce_calls(ex)
    ex isa Expr || return Any[]
    out = Any[]
    if ex.head === :call && ex.args[1] isa Symbol &&
       (ex.args[1] in REDUCERS || ex.args[1] in ARG_FORMS)
        push!(out, ex)
    end
    for arg in ex.args
        append!(out, _reduce_calls(arg))
    end
    return out
end


_rhs(x) = x
_rhs(ex::Expr) = (ex.head === :(=) || ex.head === :.=) ? _rhs(ex.args[2]) : ex

# `a .+ b` parses as a call to `.+`, `f.(x)` as Expr(:., f, tuple); both lower to
# materialize(broadcasted(...)).  `broadcasted` keeps the tree lazy instead.
_bcify(x) = x
function _bcify(ex::Expr)
    if ex.head === :call && ex.args[1] isa Symbol &&
       length(string(ex.args[1])) > 1 && string(ex.args[1])[1] == '.'
        op = Symbol(string(ex.args[1])[2:end])
        return Expr(:call, Base.broadcasted, op, map(_bcify, ex.args[2:end])...)
    elseif ex.head === :. && length(ex.args) == 2 &&
           ex.args[2] isa Expr && ex.args[2].head === :tuple
        return Expr(:call, Base.broadcasted, ex.args[1],
                    map(_bcify, ex.args[2].args)...)
    end
    return ex
end

"""
    @code_jitted expr

The C kernel `expr` would be JIT compiled to, without launching it:

    julia> @code_jitted a .+ b .* c

Broadcasts stay lazy, so the whole expression shows as the one kernel it becomes
-- including a reduction over one, `@code_jitted sum(a .+ b)`, which folds the
terminal into the same program. `sum(abs, a)` and `mapreduce(abs, +, a)` work the
same way. The reduction must be written by name: a macro cannot see through a
variable holding `sum`.

`a + b` and `sum(a)` use statically compiled kernels and have no generated
source. `argmin` / `argmax` / `findmin` / `findmax` over a single vector show
their pair-valued reduction kernel.
`argmin.(zip(a, b))` shows the per-element lane kernel.

An assignment describes its right-hand side: `@code_jitted g = sum(a .+ b)` is
`@code_jitted sum(a .+ b)`.
"""
macro code_jitted(ex)
    # A scatter is a statement: queue it, take it back off, show the program.
    if ex isa Expr && ex.head in (:.=, :(.+=), :(.*=)) && ex.args[1] isa Expr &&
       ex.args[1].head === :ref
        return :(_code_jitted_scatter(() -> $(esc(ex))))
    end
    ex = _rhs(ex)
    if ex isa Expr && ex.head === :call && ex.args[1] isa Symbol
        if length(ex.args) == 2 && ex.args[1] in ARG_FORMS
            return :(_code_jitted_arg($(QuoteNode(ex.args[1])),
                                      $(esc(ex.args[2]))))
        elseif length(ex.args) == 2 && ex.args[1] in REDUCERS
            return :(_code_jitted_reduce($(esc(ex.args[1])),
                                         $(esc(_bcify(ex.args[2])))))
        elseif length(ex.args) == 3 && ex.args[1] in REDUCERS
            return :(_code_jitted_map($(esc(ex.args[1])), $(esc(ex.args[2])),
                                      $(esc(ex.args[3]))))
        elseif length(ex.args) == 4 && ex.args[1] === :mapreduce
            return :(_code_jitted_mapreduce($(esc(ex.args[2])),
                                            $(esc(ex.args[3])),
                                            $(esc(ex.args[4]))))
        end
    end
    inner = _reduce_calls(ex)
    if length(inner) > 1
        chains = [:(_chain_of($(esc(c.args[1])), $(esc(_bcify(c.args[2])))))
                  for c in inner]
        return :(_code_jitted_hfused([$(chains...)]))
    elseif length(inner) == 1
        return :(error($(_host_combined_message(inner))))
    end
    return :(code_jitted($(esc(_bcify(ex)))))
end
