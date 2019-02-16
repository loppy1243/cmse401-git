#!/bin/bash

if (( $# > 2 )); then
  echo "Too many arguments: expected at most 2, found $#"
  return 1
fi

outprefix=${1:-job-out}
reps=${2:-10}

for i in {0..6}; do
  n=$(( 2**$i ))
  echo "Launching job array for $n threads"
  sbatch -n $n -a "1-$reps" -o "$outprefix/${n}threads_timing%a.out" \
         runBwaAln_slurm_giab_noRg.sb --no-copy-scratch --delete-scratch
done
