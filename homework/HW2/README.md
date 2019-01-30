## Transpose Timings

# Building
The code herein is written in C, and the provided Makefile assumes GCC. An
alternative compiler may be specified like
```sh
make CC=<compiler> [...]
```
To build the test output of my `transpose` and `transpose_blocked(16)`, run `make`.

To build the benchmarks,
run `make bench`. This will generate the files `transpose.bench.x` and
`provided.bench.x` which benchmark my transpose functions and the provided
ones, respectively. Each of these must be run as, e.g.,
```sh
./transpose.bench.x <N>
```
with outputted timeings averaged over `N` samples.

To generate a plot from a (suitably formatted) file `timings.tbl`, run `make
plot`. This generates `timings.pdf`.

# Provided Functions
There were two functions provided in the Jupyter notebook, `transpose` and
`transposeBase`. It would seem that `tranposeBase` is the "cache-unaware"
function referenced (and mostly equivalent to my `transpose`), and the provided
`tranpose` is a hybrid between this and a "cache-aware" algorithm.

# Provided Output
A (hand-curated) table `timings.tbl` is provided with outputs from
`./transpose.bench.x 10` and `./provided.bench.x 10`. If you want to generate
your own timings and have them work with the provided gnuplot script, follow
`timings.tbl` as an example.

The timings were produced on one core of the HPCC development node
`dev-intel14`:

- OS: Linux with a version string of "3.10.0-693.21.1.el7.x86\_64"
- CPU: Intel Xeon E5-2670, nominal 2.50GHz, with actual clockspeed between
  1.2GHz and 3.3GHz. As of 2019-01-29 17:05 (NOT when I generated my timings)
  the speed is ~2.9GHz. Supposedly this changes based on workload.
- RAM: 251GB total.
- Compiler: GCC 6.4.0

# Results
There are two subplots in `timings.pdf`. The first is a log-log plot of average
time vs. matrix size. The second is a chart of `log(speed) = log(t_min/t)` for
each matrix size, with `speed` relative to the fastest method for that matrix
size. The fit lines are least-squares fits of the form

$\log t = A\log n + B

for time $t$ and matrix size $n$. My `transpose` is faster than
`tranpose_blocked(16)` until matrix size of 5000x5000; from there
`transpose_blocked(16)` is ~1.5 times faster, and even 2.4 time faster for a
matrix size of 40000x40000.

We see that the provided `transpose` function is always the fastest; this
is consistent with it being a hybrid algorithm. We also see that `transposeBase`
is mostly on par with my `transpose`, which is also consistent with them being
essentially the same algorithm.

The fit lines are also consistent with what we expect. The fit for the provided
`transpose` is the worst, since it is a hybrid method; the fit for my
`transpose` and `transposeBase` are very similar; and the fit for my `transpose`
starts off below that of `transpose_blocked(16)`, and then past a matrix size
of 5000x5000 they cross.

# Improvements
Clearly my `transpose` could be improved by following in the steps of the
provided `transpose` and switching to the `transpose_blocked` when the matrix
size reaches a certain threshold.

The fits give us a way to estimate when to switch over. In particular, we have

$(2.18\cdot10^{-10})n_{best}^(2.52) = (3.34\cdot10^{-9}n_{best}^(2.19)$

$\implies n_{best} \approx 3900$
