# Building
```bash
# Make serial and MPI binaries
. load-modules.sh
make

# Make binaries with timing output
. load-modules.sh
make BENCH=1

# Make binaries with debugging output
. load-modules.sh
make DEBUG=1

# Compile timings from program output in timings/raw/
. load-modules.sh
make timings

# Make fine-serial timing plot and MPI scaling plot
. load-gnuplot-modules.sh
make all-plots

# Make visualization `rumors.gif` of program output. Assumes images are in ./images
make gif
```

# Running
The serial version is run as `./mill`, and outputs to the directory `./images`.

The MPI version is run al `./mill.mpi <prefix> <x_size> <y_size>`, where `prefix` is the
output directory and `x_size`, `y_size` is the size of the simulation.

If build with `BENCH=1`, both output timing information to stderr.

# Timing
To generate the serial timing data:
```
./mill 2>timings/raw/fine-serial.out
```
On dev-intel14, this took an average of 

To generate the MPI timings for 1, 2, 4, 16, 32, 64 threads and `x_size` = 1000, `y_size` =
500, 1000, 2000, 4000, 8000, 16000:
```
./launch-mpi-timings.sh
```
