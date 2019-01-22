This is a solution in Julia-1.0 of the wave-equation

$$\ddot y = \gamma\frac{\mathrm d^2y}{\mathrm dx^2}.$$

# Installation
```
(shell)# git clone https://github.com/loppy1243/cmse401-git
(shell)# cd cmse401-git/homework/HW1/julia
```

To install julia, see <https://julialang.org>. To install dependencies, open a Julia REPL

```
(shell)# julia
```

and type

```Julia
julia> import Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
```

# Running
To run for $500$ positions between $x=0$ and $x=500$ for $10^6$ times between $t=0$ and $t=10$
with $\gamma=1$ and

$$y_0(x) = e^(-(x-5)^2), y_0\dot(x) = 0, y_0\ddot(x) = 0,$$

in your julia prompt type

```Julia
julia> import Pkg
julia> Pkg.activate(".")
julia> import HW1
julia> HW1.main()
```

# Benchmarking
To bench using [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl), call the
function `HW1.bench()`:

```Julia
julia> import Pkg
julia> Pkg.activate(".")
julia> import HW1
julia> HW1.bench()
Running functions to compile... Done.
Testing equality of function calls.. Done.
Benchmarking...

exact_pseudocode():
--------------------
BenchmarkTools.Trial:
  memory estimate:  12.19 KiB
  allocs estimate:  3
  --------------
  minimum time:     1.606 s (0.00% GC)
  median time:      1.611 s (0.00% GC)
  mean time:        1.622 s (0.00% GC)
  maximum time:     1.660 s (0.00% GC)
  --------------
  samples:          4
  evals/sample:     1

solve():
--------------------
BenchmarkTools.Trial:
  memory estimate:  192 bytes
  allocs estimate:  4
  --------------
  minimum time:     349.568 ms (0.00% GC)
  median time:      350.281 ms (0.00% GC)
  mean time:        353.001 ms (0.00% GC)
  maximum time:     367.050 ms (0.00% GC)
  --------------
  samples:          15
  evals/sample:     1
```

Output is from my HP Spectre x360 with 8GB of RAM and Intel Core i7-5500U running Gentoo
Linux (kernel version 4.14.65, custom configuration), with Julia

```Julia
julia> versioninfo()
Julia Version 1.0.2
Commit d789231e99 (2018-11-08 20:11 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Core(TM) i7-5500U CPU @ 2.40GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.0 (ORCJIT, broadwell)
```

# Improvements
This code could possibly be sped up with parallelism; specifically, the the updates to the
position and its derivatives could be done in chunks on separate cores (for large position
sample size).
