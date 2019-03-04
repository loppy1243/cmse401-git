# Major Lesson Learned
Excessive text processing of timing data leads to an awful and crippled implementation of a
relational database written in AWK and taped together with shell scripting. I would have had a
much better time with this timing data if I had realized this, but hindsight is 20/20 and
foresight is often non-existent.

# Branches and Tags
You'll notice there are a couple of different git branches and tags; I planned on giving the
version of the code that led to each timing set its own tag, but towards the end I fell off
from this. It's also annoying that the data exploration scripts I wrote are tied to these as
well; what I should have done was make each timing set its own branch and rebased the changes
in the scripts onto each of these, but oh well.

You can `git tag` and `git branch` to list tags and branches, and `git checkout
<tag-or-branch>` to load that version of the code. A simple `make -B bench` will guarantee a
benchmarking build of that code.

# Chronological Comments on Timing Data
Most comparisons were done using `images/earth.png`, since it is the largest and so most
intensive work load. In the following, I provide commentary on the changes I made along with
commands using my [tooling](#Tooling) that produces the relevant data. Descriptions of the
timing sets can be found in [Timing Data](#Timing-Data).

## orig vs. row-major
The first thing I noticed with the original code was that the image loops were in column-major
order, but the images were row-major; fixing this gave about a 3.5s improvement, or 23%.
```
./compare-timings.sh earth orig row-major
```

## row-major vs. less-pointing
I then thought to switch all image arrays to be flat, 1D arrays instead of
arrays-of-pointers-to-arrays. This provides a overall speedup of 0.42s (3.6%), but note
that there is a very significant decrease in the threshholding time of 0.00735 (11%).
```
./compare-timings.sh earth row-major less-pointing
```

## less-pointing vs. O3
To not much surprise, add `-O3` to `gcc` improves things markedly by 8.1s (72%).
```
./compare-timings.sh earth less-pointing O3
```

## O3 vs. unrolled-grad
The gradient filtering is done with (small) fixed-sized filters `xfilter` and `yfilter`, so I
thought perhaps removing the arrays and inling the calculation could produce some speedup.
This appears to have worked, with the gradient filtering sped up by 0.15s (54%), though the
overall difference is less significant at 0.26s (8.3%). This is clearly due to the average
filtering being the largest time sink at 41% cumulative time, and the gradient filtering
dropping from 8.9% to 4.4%. At this point, I would say that the average filter has become the
only significant time sink.
```
./compare-timings.sh earth O3 unrolled-grad
```

## unrolled-grad vs. direct-count
The first thing I noticed in the average filter loop is that the count things averaged over is
generated on-the-fly, rather than from the averaging loop bounds. Changing this provided a
mostly insignificant speedup (average filter: 0.084 (7.1%), t=1.20 | cumulative: 0.16 (5.5%),
t=1.67), which could be due to random variation.
```
./compare-timings.sh earth unrolled-grad direct-count
```

## direct-count vs. omp-avg-rloop and omp-avg-rcloop
Adding a simple `#pragma omp parallel for` to the outer average filter loop on my 4-core
machine did very well, giving a speedup of 0.57s (52%) to the average filter and 1.0s (38%)
overall.
```
./compare-timings.sh earth unrolled-grad omp-avg-rloop
```
However, collapsing the two average filter loops with `#pragma omp parallel for collapse(2)`
produces no difference (if anything, the program is 1% slower):
```
./compare-timings.sh earth omp-avg-rloop omp-avg-rcloop
```
I elected to only put parallelize the outer loop, the reasoning being that the work for each
pixel is too miniscule, while the work for each row is sizeable. If the size of the average
filter were increased, this may change, but I have only considered the default size.

## omp-avg-rloop vs. omp-avg-\<scheduling\>
There were no significant difference when considering other scheduling methods (with default
chunk sizes), so I opted for static. (This final version is called `omp-avg`.)
```
./compare-timing-pairs.sh earth omp-avg-rloop omp-avg-static omp-avg-dynamic omp-avg-guided
```

## omp-avg vs. omp-avg-grad-\<scheduling\>
Parallelizing the gradient filter has a general benefit of about 54% speedup, or 12% speedup
total. Comparing static and dynamic scheduling to guided, though not highly significant
the guided scheduling is consistently slower across all timing segments. Static and dynamic
scheduling have no significant difference; I opted for static scheduling because of its
simplicity. (This final version is called `omp-avg-grad`.)
```
./compare-timing-pairs.sh omp-avg omp-avg-grad-static omp-avg-grad-dynamic omp-avg-grad-guided
```

## omp-avg-grad vs. omp-avg-grad-thresh-\<scheduling\>
Parallelizing the threshholding does not produce a very large speedup for the earth image, if
at all; for the MSUstadium image, there appears to be a ~50% speedup, but in both cases there
is huge variation across trials and the differences are mostly insignificant. Comparing the
scheduling methods against each other doesn't reveal anything new, so I opted for static
scheduling, since likely if the image were very, very large there would be some sort of
advantage.
```
./compare-timing-pairs.sh omp-avg-grad omp-avg-grad-thresh-static omp-avg-grad-thresh-dynamic omp-avg-grad-thresh-guided
```

## HPCC Many-thread Comparisons
It is at this point that I decided the effect of number of threads and chunk size needed to be
examined, so I moved over to the HPCC and created the SLURM script `timings.sb`. This
generates timing sets (on one node) of the form
```
omp-<nthreads>-<scheduling>[-<chunk-size>]
```
for `nthreads=1,2,4,8,16,32` and `chunk-size=1,2,4,8,16,32,64,128` (specified only for dynamic
scheduling). I then examined these with
```
./lowest-times.sh
```
which give the three fastest timing sets for each segment. Doing various comparisons based off
this, I determined that the best configuration was likely
- average filtering: maximum threads, dynamic scheduling w/ chunk-size=8
- gradient filtering: maximum threads, dynamic scheduling w/ chunk-size=64
- threshholding: 8 threads, dynamic scheduling w/ chunk-size=16

Explaining these results seems difficult, but it is what it is.

The final version with these considerations is called `final`, and is the version avaiable on
the `master`. It's timing set is generated by the SLURM script `final-timings.sb`.
`./get-timings.sh earth final` is reproduced here for posterity:

```
final
Name            Reps  tot_time   mean_time  stddev     %err
TOT_file_read   40    7.998e+00  1.999e-01  4.604e-03  5.15
CUMULATIVE      40    1.875e+01  4.688e-01  2.121e-02  6.61
average_filter  40    3.501e+00  8.753e-02  1.041e-02  40.20
threshholding   40    5.255e-01  1.314e-02  4.855e-03  322.37
TOT_processing  40    5.378e+00  1.344e-01  1.126e-02  22.84
filtering       40    1.350e+00  3.376e-02  8.018e-03  129.27
```

# Timing Data
The timing data I took can be found in `timings.tar.gz`. Extract in current directory with
```
tar -xzvf timings.tar.gz
```
and explore with the tools described below.

The timing sets found therein that were run on my local machine:
- `orig`: Original version of the program (with timing statements added).
- `row-major`: All loops over images switched to row-major order.
- `less-pointing`: Change 2D array representation during image processing from
  array-of-pointers-to-arrays to strided 1D array.
- `O3`: Adding the `-O3` optimization flag to the compiler command line.
- `unrolled-grad`: Unrolling the gradient filter by inline `xfilter` and `yfilter`.
- `direct-count`: In the averaging filter, don't count the total for the average by
  incrementing a counter, instead calculate it directly from the loop bounds.
- `omp-avg-rloop`: Add simple OpenMP directive to the outer loop of the average filter.
- `omp-avg-rcloop`: Add OpenMP collapse directive to divide work over both average filter
  loops.
- `omp-avg-<scheduling>`: OpenMP directive on just the outer average filter loop, but with
  the specified scheduling.
- `omp-avg`: Final version of parallelized average filter.
- `omp-avg-grad-<scheduling>`: OpenMP directive on outer average and gradient filter loops,
  with specified scheduling on the gradient loop.
- `omp-avg-grad`: Final version of parallelized gradient filter.
- `omp-avg-grad-thresh-<scheduling>.`: OpenMP directive on outer average, gradient, and
  threshhold filter loops, with specified scheduling on the threshhold loop.

The sets found therein that were run on the HPCC with the `timings.sb` submission script:
- `omp-<nthreads>-<scheduling>[-<chunk-size>]`: Run with `nthreads` OMP threads and
  all (outer) loops scheduled with `scheduling` and chunk size `chunk-size`.
- `final`: The final version of the program.

# Tooling
- `Makefile`
  - Before building, ensure that `libpng` and `openmp` are available.
    - On the MSU HPCC, run
      ```
      module purge
      module load GNU/7.3.0 libpng/1.6.34
      ```
  - `make [all]` will build `./process` (with no timing information).
  - `make bench` will build `./process` so that it outputs timing information. Timing
    segments (in the order they tend to appear from `./get-timings.sh`):
    - `TOT_file_read`: Time spent reading in the image. I did not end up trying to improve
      this.
    - `CUMULATIVE`: Sum of all timing segments.
    - `average_filter`: Time spent applying the averaging filter to the input image.
    - `thresholding`: Time spent producing the final image from converting the gradient to
      black-and-white.
    - `TOT_processing`: Total time spent processing the input image.
    - `filtering`: (unfortunate name due to history) Time spent applying the gradient filter
      to the averaged image.
  - NOTE: To force a rebuild, run `make -B [target]`. E.g., `make -B bench` will run all rules
    required to make `bench` regardless of whether their outputs exist or not.
- `./gen-timings.sh <timing_set> [reps=10]`
  - Run `./process` on each image in `images/` `reps` number of times (outputting to
    `out.png`), recording standard output in `timings/raw/<timing_set>/<image>/<n>.out`
    for `n=1,...,reps`. Then `./compile-timings.sh` is called.
  - NOTE that this is useless unless you build `./process` with `make bench`.
- `./compile-timings.sh <timing_set>`
  - Specifing `all` as `timing_set` compiles all timings.
  - Compiles the timing set generated by `./gen-timings.sh` into a table for each image (e.g.
    `timings/compiled/<timing_set>/earth.dat`) with number of repetitions, total time, mean
    time, standard deviation, and percent standard error in the mean time.
- `./get-timings.sh <image> <set1> <set2> ...`
  - Display the specified timing set tables as generated by `./compile-timings.sh`.
- `./compare-timings.sh <image> <timing_set1> <timing_set2>`
  - Generate a comparison table for the two specified timing sets giving for each segment the
    fraction of time spent there, percent difference mean time relative to the first, percent
    difference relative to the second, as well as the absolute difference with associated
    t-values (difference divided by error-in-difference).
- `./compare-timing-pairs.sh <image> <timing_set1> <timing_set2> ...`
  - Run `./compare-timings.sh` for each pair of timing sets specified.
- `./lowest-times.sh`
  - For each image, generate a table giving the three fastest timing sets for each segment
    with their mean times and percent standard errors.
- `sbatch timings.sb`
  - SLURM script to generate timings (with 40 repetitions) for all combinations of:
    - 1, 2, 4, 8, 16, 32 threads
    - Scheduling:
      - static (default chunk size)
      - dynamic with chunk sizes 1, 2, 4, 8, 16, 32, 64, 128
      - guided
  - Timings set are named `omp-<nthreads>-<scheduling>[-<chunk-size>]`
- `./run-thread-group.sh <nthreads>`
  - Part of the implementation of `timings.sb`.
- `sbatch final-timings.sb`
  - SLURM script to generate timings for the final version of the program.
