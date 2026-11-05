# PolymerPIM.jl

PolymerPIM.jl provides lazy `Int32` vectors backed by UPMEM DPUs. Broadcasts,
reductions, and indexed local updates are fused before execution.

## Build

From the repository root:

```bash
source /usr/upmem_env.sh
make install BACKEND=hw PIPELINE=1 JIT=1
make -C julia test
```

The Julia package requires `PIPELINE=1 JIT=1`. Running `make -C julia` directly
builds and installs that configuration into the repository's `install/`
prefix first. Set `POLYMERPIM_ROOT` to build against another installation.

## Usage

Set the DPU count before the first allocation; the default is 8:

```bash
NR_DPUS=64 julia --project=julia script.jl
```

```julia
using PolymerPIM

a = DPUVector(Int32[1, 2, 3, 4])
b = DPUVector(fill(Int32(10), 4))

result = abs2.(a .- 3) .+ b
mask = a .< b
chosen = ifelse.(mask, a, b)

Array(result)             # read back and block
sum(result)[]             # read a lazy reduction
sync()                    # drain all pending work
```

Supported broadcasts include `+`, `-`, `*`, `div`, `>>`, comparisons, unary
`-`, `abs`, `abs2`, and `ifelse`. Unsupported functions raise an error instead
of falling back to a host loop.

Host scalars are captured when an expression is built but passed at launch, so
different values reuse the same compiled kernel.

## Timing

`@dputime` measures one synchronized execution and returns its value:

```julia
result = @dputime abs2.(a .- 3) .+ b
```

The first call includes JIT compilation; later calls reuse the kernel.

## Tracing

`@trace` records a Perfetto trace of a block and prints the absolute path it
wrote; open it at <https://ui.perfetto.dev>. It needs a library built with
`TRACE=1`.

```julia
@trace begin
    c .= a .+ b
    total = sum(c)
end
@trace filename = "add.perfetto-trace" c .= a .+ b
```

Only one trace records at a time, so `@trace` errors inside another `@trace`
or while `POLYMERPIM_TRACE` is tracing the whole run.

## Reductions

`sum`, `prod`, `minimum`, and `maximum` return futures. Leave independent
futures unread so they can share a pass:

```julia
x = sum(a)
y = maximum(b)
x[], y[]
```

Reductions over broadcasts remain fused, including `sum(abs, a)`,
`mapreduce(abs, +, a)`, and `sum(a .* b)`.

## Local accumulators

`DPULocalVector` keeps small indexed reductions in DPU-local memory until
`Array` or `sync()` flushes them:

```julia
bins = DPULocalVector(256)
bins[(a .* 256) .>> 12] .+= 1
histogram = Array(bins)
```

Use `reduce_op = :product`, `:min`, or `:max` for other accumulation modes.

## Multiple vectors

`argmin.(zip(a, b, c))` and `argmax.(...)` return the winning vector index per
element. `findmin_lanes` and `findmax_lanes` return both values and indices.
`min_squared_distance(cols, query)` performs a fused distance reduction.

## Inspecting JIT code

`@code_jitted` displays the generated kernel without compiling or launching it:

```julia
@code_jitted abs2.(a .- b) .+ 1
@code_jitted sum(a .* b)
```

The returned `JittedCode` exposes `.source`, `.ops`, `.hash`, and `.path`;
`iscompiled(code)` reports whether it is already cached.

## Configuration and tests

```julia
PolymerPIM.versioninfo()
configuration()
installinfo()
ndpus(), ntasklets()
```

The wrapper checks that the loaded C++ library matches the configuration it
was built against.

```bash
make -C julia test
julia --project=julia julia/test/runtests.jl broadcast reductions
```

Raw expression builders and launch helpers live in `PolymerPIM.Internal` and
are not part of the supported public API.
