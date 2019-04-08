#!/bin/bash

for i in $(seq 0 6); do
  threads=$((2**$i))
  sbatch --ntasks=$threads --job-name="mill-$threads" mpi-timings.sb
done
