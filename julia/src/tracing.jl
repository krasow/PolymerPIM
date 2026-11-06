"""
    @trace [filename = "..."] expr

Record a Perfetto trace of `expr` and print the absolute path it was written
to. Pending work is drained first, and `sync()` after `expr` keeps its lazy DPU
execution inside the trace. The expression's value is returned.

Without `filename`, the trace goes to `polymerpim-<date>-<time>.perfetto-trace`
in the working directory. Open it at https://ui.perfetto.dev. Needs
`libpolymerpim` built with `TRACE=1`.

    @trace begin
        c .= a .+ b
        total = sum(c)
    end
    @trace filename = "add.perfetto-trace" c .= a .+ b
"""
macro trace(args...)
    isempty(args) && throw(ArgumentError(
        "@trace needs an expression: @trace begin ... end"))
    filename = nothing
    for opt in args[1:(end - 1)]
        (opt isa Expr && opt.head === :(=) && opt.args[1] === :filename) ||
            throw(ArgumentError("@trace takes one option: filename = \"...\""))
        filename = esc(opt.args[2])
    end
    return quote
        PolymerPIM._trace_begin($filename)
        Base.@__tryfinally($(esc(args[end])), PolymerPIM._trace_end())
    end
end

function _trace_begin(filename)
    get(configuration(), "TRACE", "0") == "1" || error("""
        @trace needs libpolymerpim built with TRACE=1 (found TRACE=$(get(configuration(), "TRACE", "0"))).

        Rebuild and reinstall the C++ library, then rebuild the wrapper:
          make TRACE=1 PIPELINE=1 JIT=1 BACKEND=hw install
          make -C julia clean build
        """)
    path = filename === nothing ? _default_trace_path() : String(filename)
    sync()
    trace_start(path)
    return nothing
end

function _trace_end()
    try
        sync()
    finally
        println("Trace written to ", String(trace_stop()))
    end
end

function _default_trace_path()
    stem = joinpath(pwd(), "polymerpim-" * Libc.strftime("%Y%m%d-%H%M%S", time()))
    path, n = stem * ".perfetto-trace", 1
    while ispath(path)
        path, n = string(stem, "-", n, ".perfetto-trace"), n + 1
    end
    return path
end
