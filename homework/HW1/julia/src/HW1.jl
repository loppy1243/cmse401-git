module HW1
using BenchmarkTools
using Plots; gr()

Plots.default(legend=false)

"""
    exact_pseudocode()

Return the evolved y-values following the provided pseudocode exactly, with the
exception that we only loop through times excluding the start time.
"""
exact_pseudocode() = exact_pseudocode((_...,) -> nothing)
function exact_pseudocode()
    xmin = 0; xmax = 10; nx = 500
    dx = (xmax-xmin)/nx
    x = range(xmin, stop=xmax, length=nx)

    tmin = 0; tmax = 10; nt = 10^6
    dt = (tmax-tmin)/nt
    times = range(tmin, stop=tmax, length=nt)

    y = [exp(-(x[i]-5)^2) for i=1:nx]

    y_dot = [0.0 for i=1:nx]
    y_ddot = [0.0 for i=1:nx]
    γ = 1

    for _ in times[2:end]
        for i = 2:nx-1
            y_ddot[i] = γ*(y[i+1]+y[i-1]-2y[i])/dx^2
        end
        for i = 1:nx
            y[i] = y[i] + y_dot[i]*dt
            y_dot[i] = y_dot[i] + y_ddot[i]*dt
        end
    end

    y
end

standard_y0(x) = exp(-(x-5)^2)

"""
    solve([cb, ]y0, dx, ts; γ, gamma)

Solve the wave equation at times `ts` given the initial displacements `y0`,
space step size `dx`, and parameter `γ`. All initial times derivatives are
assumed to be zero.

Either `γ` or `gamma` may be specified; if both are provided, they must be equal
in the sense of `==`. If the function `cb(j, y, y_dot, y_ddot, t)` is provided,
it is called for each `j`th time `t` in `ts` with the current values of `y` and
its time derivatives.
"""
solve(y0, dx, ts; kws...) = solve((_...,) -> nothing, y0, dx, ts; kws...)
function solve(cb, y0, dx, ts; kws...)
    z = zero(eltype(y0))
    y_dot = fill!(similar(y0), z)
    y_ddot = fill!(similar(y0), z)

    solve!(cb, copy(y0), y_dot, y_ddot, dx, ts; kws...)
end

"""
    solve!([cb ,]y, y_dot, y_ddot, dx, ts; γ, gamma)

Solve the wave equation at times `ts` given the initial displacements `y` with
first and second time derivatives `y_dot` and `y_ddot`, space step size `dx`, and parameter
`\gamma`.

`y`, `y_dot`, and `y_ddot` are used in-place.  Either `γ` or `gamma` may be
specified; if both are provided, they must be equal in the sense of `==`. If
the function `cb(j, y, y_dot, y_ddot, t)` is provided, it is called for each
`j`th time `t` in `ts` with the current values of `y` and its time derivatives.
"""

solve!(y, y_dot, y_ddot, dx, ts; kws...) =
    solve!((_...,) -> nothing, y, y_dot, y_ddot, dx, ts; kws...)
function solve!(cb, y, y_dot, y_ddot, dx, ts; γ, gamma=γ)
    @assert γ == gamma

    nt = length(ts)
    dt(i) = ts[i] - ts[i-1]

    y_left = @view y[1:end-2]
    y_middle = @view y[2:end-1]
    y_right = @view y[3:end]
    y_ddot_middle = @view y_ddot[2:end-1]

    cb(1, y, y_dot, y_ddot, ts[1]) 
    for j in eachindex(ts)[2:end]
        _solve_kernel!(y, y_left, y_middle, y_right,
                       y_dot,
                       y_ddot, y_ddot_middle,
                       γ/dx^2, dt(j))

        cb(j, y, y_dot, y_ddot, ts[j])
    end

    y
end

## Because of compilation boundaries, Julia optimizes the loop in solve! better if we pull out
## the kernel into its own function.
function _solve_kernel!(y, y_left, y_middle, y_right,
                        y_dot,
                        y_ddot, y_ddot_middle,
                        γ_dxsq, dt)
    @. y_ddot_middle = γ_dxsq*(y_right + y_left - 2y_middle)
    @. y += y_dot*dt
    @. y_dot += y_ddot*dt
end

function main()
    xs = range(0, stop=10, length=500); dx = step(xs)
    ts = range(0, stop=10, length=10^6)

    solve(standard_y0.(xs), dx, ts; γ=1)
end

# M[aybe]Integer
MInteger = Union{Nothing, Integer}
"""
    gen_anim(file="anim.gif"; nframes, speed=1, fps)

Save a GIF to `file` of the evolution of the solution to the wave equation sped
up by a factor of `speed` with framerate `fps` and number of frames `nframes`.

Only one of `nframes` and `fps` may be specified.
"""
function gen_anim(file="anim.gif"; nframes::MInteger=nothing, speed=1, fps::MInteger=nothing)
    if nframes === nothing && fps !== nothing
        nframes = trunc(Int, fps*10/speed)
    elseif nframes !== nothing && fps === nothing
        fps = div(nframes, 10/speed) |> Int
    else
        error("Only one of nframes or fps may be specified.")
    end

    anim = Animation(); framenum = 1
    xs = range(0, stop=10, length=500)
    ts = range(0, stop=10, length=10^6)

    println("Running simulation...")
    print("Rendering frame... ")
    solve(standard_y0.(xs), step(xs), ts, γ=1) do i, y, _, _, _
        if framenum <= nframes && (i-1) % div(10^6, nframes) == 0
            print(" $framenum")
            frame(anim, plot(xs, y, ylims=(-1.0, 1.0)))
            framenum += 1
        end
    end
    println()

    println("Generating GIF...")
    gif(anim, file, fps=fps)
    println(" Done.")
end

function bench()
    xs = range(0, stop=10, length=500); dx = step(xs)
    ts = range(0, stop=10, length=10^6)
    init_y() = (standard_y0.(xs), zeros(500), zeros(500))
    γ = 1

    print("Running functions to compile...")
    y1 = exact_pseudocode()
    y2 = solve!(init_y()..., dx, ts; γ=γ)
    println(" Done.")
    print("Testing equality of function calls..")
    # For some reason these are only equivalent to within 1e-3
    @assert all(abs.(y1 .- y2) .< 1e-3)
    println(" Done.")

    ## Yes, this is no a fair comparison because exact_pseudocode() allocates its arrays. But:
    ## (1) this is negligible, and (2) this sort of is a comparison between the
    ## exact_pseudocode() and something smarter.
    println("Benchmarking...")
    println()
    println("exact_pseudocode():")
    println("--------------------")
    show(stdout, "text/plain", @benchmark exact_pseudocode())
    println(); println()
    println("solve():")
    println("--------------------")
    y, y_dot, y_ddot = init_y()
    show(stdout, "text/plain", @benchmark solve!($y, $y_dot, $y_ddot, $dx, $ts; γ=$γ))
end

end # module HW1
