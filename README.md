# PolymerPIM

PolymerPIM is a C++ and Julia library for programming UPMEM DPUs with vector
expressions. Operations are automatically fused and JIT-compiled into DPU
kernels.

## Build

```bash
source /usr/upmem_env.sh
make install BACKEND=hw PIPELINE=1 JIT=1
make test PIPELINE=1 JIT=1
```

`make` installs the repository-local dependencies on first use. Use
`BACKEND=simulator` when DPU hardware is unavailable.

JIT-compiled kernels are written to `build/jit` under the working directory
and removed at `shutdown()`. Set `POLYMERPIM_JIT_DIR` to put them elsewhere:
each process then uses its own `polymerpim-jit-<pid>` subdirectory and removes
only that.

## C++ API

Include `<polymerpim.h>` and link against `libpolymerpim`:

```cpp
#include <polymerpim.h>

using namespace polymerpim;

init(64);
{
  std::vector<int32_t> host_a{1, 2, 3, 4};
  std::vector<int32_t> host_b{10, 10, 10, 10};
  int32_t centroid = 3;

  DPUVector<int32_t> a(host_a);
  DPUVector<int32_t> b(host_b);
  auto result = sqr(a - centroid) + b;
  auto total = sum(result);             // lazy reduction

  std::vector<int32_t> values = result.to_cpu();
  auto scalar = total.get();
}
shutdown();
```

Expressions remain lazy until a read, `fence(vector)`, or `sync()`. Small indexed
reductions can stay local to each DPU until read:

```cpp
DPULocalVector<int32_t> bins(256);
bins[index] += 1;
std::vector<int32_t> histogram = bins.to_cpu();
```

The public header is installed under `include/polymerpim`; implementation
headers under `host/detail` are private.

## Julia API

PolymerPIM.jl provides the same lazy execution model through Julia arrays,
broadcasts, reductions, and local accumulators:

```julia
using PolymerPIM

a = DPUVector(Int32[1, 2, 3, 4])
b = DPUVector(fill(Int32(10), 4))

result = abs2.(a .- 3) .+ b
total = sum(result)

Array(result)
total[]
```

See the [Julia API guide](julia/README.md) for installation, supported
operations, synchronization, JIT inspection, and tests.

## Benchmarks

The self-contained benchmark suite and runner are documented in
[benchmarks/README.md](benchmarks/README.md).
