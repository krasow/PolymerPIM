# DPUArray -- Julia wrapper around the CxxWrap-managed dpu_vector<int32_t>

"""
    DPUArray(data::AbstractArray{<:Integer})
    DPUArray(dims::Integer...)

An array of up to 3 dimensions stored in UPMEM DPU memory. The DPUs see one
flat column-major vector; the shape lives on the host, so `reshape` moves no
data. `DPUVector` and `DPUMatrix` are the 1-D and 2-D cases.

Construct from a Julia array to transfer data to the DPUs, or pass the
dimensions to allocate an uninitialised array of that shape.

# Examples
```julia
v = DPUVector(Int32[1, 2, 3, 4])
M = DPUArray(rand(Int32, 4, 5))
w = DPUVector(1024)                 # uninitialised, length 1024
T = DPUArray(4, 5, 6)               # 4×5×6
```
"""
mutable struct DPUArray{N}
    handle::PolymerPIM.DpuVectorInt32   # CxxWrap-managed C++ object
    len::Int64
    dims::NTuple{N,Int64}

    function DPUArray{N}(handle::PolymerPIM.DpuVectorInt32,
                         dims::NTuple{N,Integer}) where {N}
        _check_ndims(N)
        return new{N}(handle, prod(Int64.(dims)), Int64.(dims))
    end
end

const DPUVector = DPUArray{1}
const DPUMatrix = DPUArray{2}

_check_ndims(N) = 1 <= N <= 3 || throw(ArgumentError(
    "DPUArray supports 1 to 3 dimensions, got $N"))

DPUArray{1}(handle::PolymerPIM.DpuVectorInt32) =
    DPUArray{1}(handle, (Int64(PolymerPIM.cpp_length(handle)),))

# The same storage under `dims`.
_like(dims::NTuple{N,Integer}, handle) where {N} = DPUArray{N}(handle, dims)
_like(a, handle) = _like(size(a), handle)

# Construct from a Julia array -- transfer to DPU memory.  Handed over as-is:
# it is already the contiguous buffer the wrapper wants, and `collect` would
# copy the lot (~200ms per 512MB) first.  The wrapper fences before returning.
function DPUArray(data::Array{Int32,N}) where {N}
    _check_ndims(N)
    handle = retry_on_oom(() -> PolymerPIM.from_cpu_int32(vec(data)))
    return DPUArray{N}(handle, size(data))
end

# Views, other lazy arrays and other integer types are materialised first.
DPUArray(data::AbstractArray{<:Integer}) = DPUArray(convert(Array{Int32}, data))
DPUArray{N}(data::AbstractArray{<:Integer,N}) where {N} = DPUArray(data)

# Allocate an uninitialised DPU array.  CxxWrap exposes the bound
# `constructor<uint32_t>()` as the type itself, not as a cpp_alloc_* helper.
function DPUArray{N}(dims::Vararg{Integer,N}) where {N}
    _check_ndims(N)
    all(>=(0), dims) || throw(ArgumentError("dimensions must be non-negative, got $dims"))
    n = prod(Int64.(dims))
    handle = retry_on_oom(() -> PolymerPIM.DpuVectorInt32(UInt32(n)))
    return DPUArray{N}(handle, dims)
end
DPUArray(dims::Integer...) = DPUArray{length(dims)}(dims...)
DPUArray(dims::Tuple{Vararg{Integer}}) = DPUArray(dims...)
DPUArray{N}(dims::NTuple{N,Integer}) where {N} = DPUArray{N}(dims...)

"""
    PolymerPIM.zeros(Int32, dims...) -> DpuZeros

Zeros that exist only as the additive identity: `zeros(Int32, n) .+ x` lowers
to the program `x` alone would give, so an accumulator loop can start at its
first element. Nothing is allocated until a `DPUArray` reads them.
"""
struct DpuZeros
    dims::Tuple{Vararg{Int64}}

    function DpuZeros(dims::Integer...)
        _check_ndims(length(dims))
        all(>=(0), dims) || throw(ArgumentError(
            "dimensions must be non-negative, got $dims"))
        return new(Int64.(dims))
    end
end

zeros(::Type{Int32}, dims::Integer...) = DpuZeros(dims...)
zeros(::Type{Int32}, dims::Tuple{Vararg{Integer}}) = DpuZeros(dims...)

Base.length(z::DpuZeros) = prod(z.dims)
Base.size(z::DpuZeros) = z.dims
Base.ndims(z::DpuZeros) = length(z.dims)
Base.eltype(::DpuZeros) = Int32

# Dropped as the tree is built, not during lowering: folding a runtime scalar
# that happens to be 0 would split the JIT program shared across its values.
Base.broadcasted(::typeof(+), z::DpuZeros, x) = _drop_zeros(z, x)
Base.broadcasted(::typeof(+), x, z::DpuZeros) = _drop_zeros(z, x)

function _drop_zeros(z::DpuZeros, x)
    dims = _bcsize(x)
    dims === nothing || dims == size(z) || throw(DimensionMismatch(
        "zeros of size $(size(z)) added to an array of size $dims"))
    return x
end

"""
    PolymerPIM.fill(Int32, dims, value) -> DPUArray

`value` repeated over `dims` (an integer or a tuple), written by the DPUs
rather than staged through a host buffer.
"""
function fill(::Type{Int32}, dims::Tuple{Vararg{Integer}}, value::Integer)
    _check_ndims(length(dims))
    all(>=(0), dims) || throw(ArgumentError("dimensions must be non-negative, got $dims"))
    n = prod(Int64.(dims))
    handle = retry_on_oom(() -> PolymerPIM.fill_int32(n, Int32(value)))
    return _like(dims, handle)
end
fill(::Type{Int32}, n::Integer, value::Integer) = fill(Int32, (n,), value)

# Reading them is what materialises them.
DPUArray(z::DpuZeros) = fill(Int32, z.dims, Int32(0))
DPUArray{N}(z::DpuZeros) where {N} =
    (_check_same_ndims(N, ndims(z)); fill(Int32, NTuple{N,Int64}(z.dims), Int32(0))::DPUArray{N})

# Only `+` can drop the zeros for free, so every other use materialises rather
# than failing on a type that has no storage.
Base.broadcastable(z::DpuZeros) = DPUArray(z)
for f in (:sum, :prod, :minimum, :maximum, :findmax, :findmin, :argmin, :argmax)
    @eval Base.$f(z::DpuZeros) = Base.$f(DPUArray(z))
end

_check_same_ndims(want, have) = want == have || throw(ArgumentError(
    "a $have-dimensional value cannot become a DPUArray{$want}; use DPUArray(x) " *
    "or reshape it"))

export DpuZeros

# ---- Conversions: DPU -> Julia ----

"""
    Array(a::DPUArray) -> Array{Int32}

Transfer DPU array contents back to the host as a Julia array of the same shape.
"""
function Base.Array(a::DPUArray{N}) where {N}
    out = Array{Int32,N}(undef, a.dims)
    retry_on_oom(() -> PolymerPIM.to_cpu!(a.handle, vec(out)))
    return out
end

Base.Vector(v::DPUVector) = Array(v)
Base.Matrix(m::DPUMatrix) = Array(m)
Base.collect(a::DPUArray) = Array(a)

# ---- Basic queries ----

Base.length(a::DPUArray) = a.len
Base.size(a::DPUArray) = a.dims
Base.size(a::DPUArray, d::Integer) = d <= ndims(a) ? a.dims[d] : 1
Base.ndims(::DPUArray{N}) where {N} = N
Base.eltype(::DPUArray) = Int32
Base.axes(a::DPUArray) = map(Base.OneTo, a.dims)

# Scalar indexing (requires full transfer -- use sparingly)
function Base.getindex(a::DPUArray, i::Int)
    @boundscheck 1 <= i <= a.len || throw(BoundsError(a, i))
    return Array(a)[i]
end

function Base.getindex(a::DPUArray{N}, I::Vararg{Int,N}) where {N}
    @boundscheck all(1 .<= I .<= a.dims) || throw(BoundsError(a, I))
    return Array(a)[I...]
end

# ---- Reshaping: the storage is flat, so only the shape changes ----

"""
    reshape(a::DPUArray, dims...) -> DPUArray

The same DPU storage under a new shape of up to 3 dimensions; nothing moves, so
writes through either are seen by both. One dimension may be `:`.
"""
function Base.reshape(a::DPUArray, dims::Tuple{Vararg{Union{Integer,Colon}}})
    dims = _uncolon(dims, a.len)
    prod(dims) == a.len || throw(DimensionMismatch(
        "cannot reshape $(join(a.dims, "×")) into $(join(dims, "×"))"))
    return _like(dims, a.handle)
end
Base.reshape(a::DPUArray, dims::Union{Integer,Colon}...) = reshape(a, dims)

function _uncolon(dims, len)
    ncolon = count(d -> d isa Colon, dims)
    ncolon == 0 && return Int64.(dims)
    ncolon == 1 || throw(DimensionMismatch("only one dimension can be `:`"))
    known = prod(Int64(d) for d in dims if !(d isa Colon); init = Int64(1))
    (known != 0 && len % known == 0) || throw(DimensionMismatch(
        "cannot reshape $len elements into $dims"))
    return map(d -> d isa Colon ? len ÷ known : Int64(d), dims)
end

Base.vec(a::DPUArray) = reshape(a, a.len)
Base.vec(v::DPUVector) = v

"""
    fence(v::DPUArray)

Explicitly synchronize: block until all pending DPU operations on `v` complete.
"""
function fence(v::DPUArray)
    retry_on_oom(() -> PolymerPIM.dpu_fence(v.handle))
end

export fence

"""
    release!(v::DPUArray)

Free `v`'s DPU memory now instead of waiting for the GC to collect it.  The GC
cannot see MRAM pressure, so a loop that allocates a fresh vector each pass can
exhaust the DPUs while the dead ones are still unreclaimed; C++ avoids this by
destroying each `dpu_vector` at scope exit.  Using `v` afterwards is invalid.
"""
function release!(v::DPUArray)
    finalize(v.handle)
    return nothing
end

export release!

# ---- DpuFuture -- a queued reduction whose value has not been read ----

"""
    DpuFuture

A queued reduction, unread.  Leave several unread and they merge into one
kernel pass; `f[]`, `get(f)` or `fetch(f)` forces them.
"""
mutable struct DpuFuture
    handle::PolymerPIM.DpuFutureInt32
    value::Int64
    resolved::Bool

    DpuFuture(handle::PolymerPIM.DpuFutureInt32) = new(handle, 0, false)
end

"""
    get(f::DpuFuture) -> Int64

Read a queued reduction, blocking until the DPUs have produced it.  Kept, so a
second read transfers nothing.
"""
function Base.get(f::DpuFuture)
    f.resolved && return f.value
    f.value = retry_on_oom(() -> PolymerPIM.cpp_get(f.handle))
    f.resolved = true
    return f.value
end

# `f[]` as for any value holder; `fetch` is the Julia future spelling.
Base.getindex(f::DpuFuture) = get(f)
Base.fetch(f::DpuFuture) = get(f)

# A future stands in for its value.  Both operands were queued before the read,
# so this costs no fusion.
Base.convert(::Type{T}, f::DpuFuture) where {T<:Number} = convert(T, get(f))
Base.Int64(f::DpuFuture) = get(f)

for op in (:+, :-, :*, :/, :÷, :%, :^, :min, :max)
    @eval Base.$op(f::DpuFuture, g::DpuFuture) = Base.$op(get(f), get(g))
    @eval Base.$op(f::DpuFuture, x::Number) = Base.$op(get(f), x)
    @eval Base.$op(x::Number, f::DpuFuture) = Base.$op(x, get(f))
end

# Base derives `>` and `>=` from these.
for op in (:(==), :<, :<=, :isless)
    @eval Base.$op(f::DpuFuture, g::DpuFuture) = Base.$op(get(f), get(g))
    @eval Base.$op(f::DpuFuture, x::Number) = Base.$op(get(f), x)
    @eval Base.$op(x::Number, f::DpuFuture) = Base.$op(x, get(f))
end

Base.:-(f::DpuFuture) = -get(f)
Base.abs(f::DpuFuture) = abs(get(f))
# `==` reads, so `hash` must too.
Base.hash(f::DpuFuture, h::UInt) = hash(get(f), h)

export DpuFuture
