# Captures what @trace prints, returning (value, output).
function _traced(f)
    mktemp() do _, io
        value = redirect_stdout(f, io)
        flush(io)
        seekstart(io)
        return value, read(io, String)
    end
end

_written(output) = match(r"^Trace written to (.+)\n$", output)

if get(configuration(), "TRACE", "0") != "1"
    @testset "scoped tracing needs TRACE=1" begin
        @test_throws ErrorException @trace nothing
    end
elseif haskey(ENV, "POLYMERPIM_TRACE")
    @testset "scoped tracing refused under POLYMERPIM_TRACE" begin
        @test_throws Exception @trace nothing
    end
else
    @testset "scoped tracing" begin
        a = DPUVector(Int32.(collect(1:N)))
        mktempdir() do dir
            file = joinpath(dir, "sub", "add.perfetto-trace")
            result, output = _traced() do
                @trace filename = file a .+ 7
            end
            m = _written(output)
            @test m !== nothing
            @test m[1] == file
            @test filesize(file) > 0
            @test Array(result) == Int32.(collect(1:N)) .+ 7

            # The default name is absolute, in the working directory.
            _, output = cd(dir) do
                _traced(() -> @trace nothing)
            end
            m = _written(output)
            @test m !== nothing && isabspath(m[1]) && dirname(m[1]) == realpath(dir)

            # One trace at a time; a failing block still writes and stops its own.
            inner = joinpath(dir, "inner.perfetto-trace")
            @test_throws Exception redirect_stdout(devnull) do
                @trace filename = joinpath(dir, "outer.perfetto-trace") begin
                    @trace filename = inner nothing
                end
            end
            @test isfile(joinpath(dir, "outer.perfetto-trace"))
            @test !isfile(inner)
            _, output = _traced(() -> @trace filename = inner nothing)
            @test _written(output) !== nothing

            scope = Module(:TraceScopeTest)
            Core.eval(scope, :(using PolymerPIM))
            redirect_stdout(devnull) do
                Core.eval(scope, :(@trace filename = $(joinpath(dir, "s.perfetto-trace")) assigned = 42))
            end
            @test Core.eval(scope, :assigned) == 42
        end
    end
end
