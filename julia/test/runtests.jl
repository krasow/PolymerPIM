using Test
using PolymerPIM

# DPU vectors need at least num_dpus elements to behave, and reductions want
# enough to span every tasklet, so the suites share one comfortable length.
const N = 4096

# One file per concern, under suites/.  Pass substrings to run a subset:
#   julia --project=. test/runtests.jl broadcast internal
const SUITES = ["core", "elementwise", "reductions", "inplace",
                "kary", "scatter", "broadcast", "jit", "logging", "timing",
                "tracing", "internal"]

const SELECTED = isempty(ARGS) ? SUITES :
    filter(s -> any(a -> occursin(a, s), ARGS), SUITES)

isempty(SELECTED) && error("no suite matches $(ARGS); available: $(join(SUITES, ", "))")

@testset verbose = true "PolymerPIM" begin
    for name in SELECTED
        @testset "$name" begin
            include(joinpath(@__DIR__, "suites", name * ".jl"))
        end
    end
end
