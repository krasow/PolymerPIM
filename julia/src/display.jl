# REPL display for DPUVector

const _MAX_DISPLAY_ELEMENTS = 10

function Base.show(io::IO, v::DPUVector)
    print(io, "DPUVector{Int32}(", v.len, ")")
end

function Base.show(io::IO, ::MIME"text/plain", v::DPUVector)
    println(io, v.len, "-element DPUVector{Int32}:")
    if v.len == 0
        return
    end

    data = Array(v)

    if v.len <= 2 * _MAX_DISPLAY_ELEMENTS
        # small enough to show everything
        for i in eachindex(data)
            print(io, " ", data[i])
            i < length(data) && println(io)
        end
    else
        # show first and last _MAX_DISPLAY_ELEMENTS entries
        for i in 1:_MAX_DISPLAY_ELEMENTS
            println(io, " ", data[i])
        end
        println(io, " \u22ee")                  # vertical ellipsis
        for i in (v.len - _MAX_DISPLAY_ELEMENTS + 1):v.len
            print(io, " ", data[i])
            i < v.len && println(io)
        end
    end
end

# Only the REPL's display forces; `show(io, f)` and a trailing `;` stay queued.
Base.show(io::IO, f::DpuFuture) =
    print(io, "DpuFuture(", f.resolved ? string(f.value) : "unread", ")")

function Base.show(io::IO, ::MIME"text/plain", f::DpuFuture)
    value = try
        get(f)
    catch err
        # Displaying must not throw.
        print(io, "DpuFuture(unread: ", sprint(showerror, err), ")")
        return
    end
    # Named, so it does not read as a plain Int64.
    print(io, "DpuFuture(", value, ")")
end

_field(io, key, value) = println(io, "  ", rpad(key * ":", 16), value)

"""
    PolymerPIM.versioninfo(io = stdout)

Print the runtime shape, the configuration of the `libpolymerpim` in use, and
where it came from. `configuration()` and `installinfo()` return the same
information as dictionaries.
"""
function versioninfo(io::IO = stdout)
    config, install = configuration(), installinfo()
    cfg(key, fallback = "unknown") = get(config, key, fallback)
    inst(key, fallback = "unknown") = get(install, key, fallback)

    println(io, "PolymerPIM v", pkgversion(@__MODULE__))

    println(io, "Runtime:")
    _field(io, "DPUs", string(ndpus(), runtime_initialized() ? "" : " (not yet claimed)"))
    _field(io, "Tasklets/DPU", ntasklets())
    _field(io, "Block size", string(1 << parse(Int, cfg("BLOCK_SIZE_LOG2", "0")),
                                    " elements/tasklet"))

    println(io, "Library:")
    _field(io, "Backend", cfg("BACKEND"))
    _field(io, "Pipeline/JIT", string("PIPELINE=", cfg("PIPELINE"), " JIT=", cfg("JIT"),
                                      " batch=", cfg("JIT_BATCH_SIZE")))
    _field(io, "Build", string(inst("BUILD_TYPE"), ", ", cfg("CXX_STANDARD"),
                               ", ", inst("CXX")))
    _field(io, "Fusion", string("absorb=", cfg("QUEUE_ABSORB_LIMIT"),
                                " chains=", cfg("MAX_HFUSE_CHAINS"),
                                " ops=", cfg("MAX_VFUSE_OPS")))
    _field(io, "Slots", string(MAX_VFUSE_INPUTS, " operands, ",
                               MAX_PIPELINE_SCALARS, " scalars, ",
                               MAX_LOCAL_SCRATCH_VECTORS, " locals"))

    println(io, "Install:")
    _field(io, "Prefix", inst("POLYMERPIM_ROOT"))
    _field(io, "Installed", string(inst("INSTALL_DATE"), " from ",
                                   inst("GIT_BRANCH"), " ", first(inst("GIT_REV"), 7),
                                   inst("GIT_DIRTY") == "1" ? " (dirty)" : ""))
    _field(io, "Wrapper", string(INSTALL_DIR, ", built ", inst("WRAPPER_BUILT")))
    _field(io, "Julia", string(inst("JULIA_VERSION"),
                               inst("JULIA_VERSION") == string(VERSION) ? "" :
                               " (running $(VERSION))"))
    return nothing
end
