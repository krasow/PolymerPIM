module PolymerPIM

using CxxWrap

# Where `make build` installs the wrapper -- the library and the stamps recording
# what it was built against, together.
const INSTALL_DIR = normpath(joinpath(@__DIR__, "..", "lib", "wrapper", "PolymerPIM"))

# Path to the compiled wrapper shared library (without extension)
const _wrapper_lib = joinpath(INSTALL_DIR, "libpolymerpim_wrapper")

if !isfile(_wrapper_lib * ".so")
    error("""
          PolymerPIM wrapper not installed at $INSTALL_DIR.
          Build it:
            make install
          """)
end

# key=value stamps, the same format the C++ build.config uses.
_parse_stamp(text) = Dict(p[1] => p[2] for p in
                          (split(l, '=', limit = 2) for l in eachline(IOBuffer(text)))
                          if length(p) == 2)
_read_stamp(name) = (f = joinpath(INSTALL_DIR, name);
                     isfile(f) ? _parse_stamp(read(f, String)) : Dict{String,String}())

"""
    installinfo() -> Dict{String,String}

Provenance of this install: the PolymerPIM prefix it links against, when and with
what each side was built. Written by `deps/build.jl`.
"""
installinfo() = _read_stamp("install.config")

"""
    configuration() -> Dict{String,String}

The build configuration of the `libpolymerpim` actually loaded, read out of the
library itself. Ground truth; `installinfo()` records how it got there.
"""
configuration() = _parse_stamp(String(build_config()))

# RUNPATH points at an install prefix that can be rebuilt with different flags
# without anything here being relinked, so refuse a library that no longer
# matches the one this wrapper was built against.
function _check_config()
    snapshot = _read_stamp("build.config")
    isempty(snapshot) && return
    live = configuration()
    drift = sort([k for k in union(keys(snapshot), keys(live))
                  if get(snapshot, k, nothing) != get(live, k, nothing)])
    isempty(drift) && return
    error("""
          The libpolymerpim loaded is not the one the wrapper was built against:

          $(join(["  $k: built against $(get(snapshot, k, "-")), loaded $(get(live, k, "-"))"
                  for k in drift], "\n"))

          Rebuild the wrapper against it:
            make -C julia clean build POLYMERPIM_ROOT=$(get(installinfo(), "POLYMERPIM_ROOT", "/path/to/install"))
          """)
end

@wrapmodule(() -> _wrapper_lib)

function __init__()
    @initcxx
    _check_config()
    # PIPELINE=0 / JIT=0 exist so the C++ library can benchmark the
    # alternatives; this package targets the configuration you would actually
    # deploy, and the op set it binds only exists there.  Checked on every load
    # so a relinked library cannot be masked by a precompile cache.
    if !PolymerPIM.built_with_pipeline() || !PolymerPIM.built_with_jit()
        error("""
              PolymerPIM.jl requires libpolymerpim built with PIPELINE=1 JIT=1 \
              (found PIPELINE=$(Int(PolymerPIM.built_with_pipeline())) \
              JIT=$(Int(PolymerPIM.built_with_jit()))).

              Rebuild and reinstall the C++ library, then rebuild the wrapper:
                make PIPELINE=1 JIT=1 BACKEND=hw install
                make -C julia clean build
              """)
    end
    # `@show_log` is part of the package's surface rather than a build option:
    # a library compiled with LOGGING=0 has no log to capture.  Costs nothing
    # when no capture is open -- an int compare per call site.
    if PolymerPIM.log_max_level() < 1
        error("""
              PolymerPIM.jl requires libpolymerpim built with LOGGING >= 1 \
              (found ENABLE_DPU_LOGGING=$(Int(PolymerPIM.log_max_level()))); \
              @show_log has nothing to capture without it.

              Rebuild and reinstall the C++ library, then rebuild the wrapper:
                make LOGGING=3 PIPELINE=1 JIT=1 BACKEND=hw install
                make -C julia clean build
              """)
    end
    atexit() do
        try
            GC.gc(true) # we need to ensure all vectors are destructed before cleanup is called
            PolymerPIM.cleanup()
        catch e # ignore errors during shutdown
        end
    end
end

"""
    ndpus() -> Int

How many DPUs the runtime is using. Set with the `NR_DPUS` environment variable
before the first allocation (default 8); DPUs are claimed on the first
`DPUVector`, and before that this reports the count that will be taken.
"""
ndpus() = Int(num_dpus())

"""
    ntasklets() -> Int

Tasklets per DPU, fixed at library build time by `NR_TASKLETS`.
"""
ntasklets() = Int(num_tasklets())

"""
    sync()

Synchronize all DPUs: blocks until all pending operations on all vectors
complete.
"""
function sync()
    flush_locals!()          # queued scatter updates first, so they inline
    _run_dangling_lazies()   # then values nothing else will run
    PolymerPIM.dpu_sync()
    _release_cut_temps!()    # drained, so the last split's intermediates go
end

"""
    retry_on_oom(f)

Executes function `f`. If a DPU OOM exception is caught, triggers Julia GC
and retries once.
"""
function retry_on_oom(f)
    try
        return f()
    catch e
        # CxxWrap throws exceptions as CxxException.
        # We check the message for "DPU OOM".
        if occursin("DPU OOM", sprint(showerror, e))
            @warn "DPU OOM detected. Syncing and triggering GC..."
            sync()     # Flush event queue and wait for DPUs
            GC.gc(true) # Major GC
            yield()
            return f()
        else
            rethrow(e)
        end
    end
end

# Slot limits read from the library itself, so they cannot drift from the
# configuration it was compiled with.
const MAX_VFUSE_INPUTS = Int(PolymerPIM.limit_operands())
const MAX_PIPELINE_SCALARS = Int(PolymerPIM.limit_scalars())
const MAX_LOCAL_SCRATCH_VECTORS = Int(PolymerPIM.limit_locals())
const MAX_CHAINS = Int(PolymerPIM.limit_chains())

include("types.jl")
include("internal/Internal.jl")
using .Internal: DpuExpr, input, operand, constant, scalar_var
using .Internal: sqr, select
using .Internal: add_var, sub_var, mul_var, divide_var, shr_var
using .Internal: eq_var, lt_var, gt_var, ge_var, le_var
using .Internal: _DpuScalar, LOCAL_REDUCE_OPS, _LocalReduce
using .Internal: _local_reduce_opcode, _scatter_program
include("operations.jl")
include("jit.jl")
include("display.jl")
include("logging.jl")
include("timing.jl")

export DPUVector, DpuFuture, DpuLazy, fence, sync
export MAX_VFUSE_INPUTS, MAX_PIPELINE_SCALARS, MAX_LOCAL_SCRATCH_VECTORS
export MAX_CHAINS
export installinfo, configuration, ndpus, ntasklets
export code_jitted, @code_jitted, iscompiled
export withlog, @show_log
export @dputime

end # module PolymerPIM
