OMP_SCHEDULE=static ./gen-timings.sh omp-$1-static 40
for i in {0..7}; do
  k=$(( 2**$i ))
  OMP_SCHEDULE=dynamic,$k ./gen-timings.sh omp-$1-dynamic-$k 40
  OMP_SCHEDULE=guided ./gen-timings.sh omp-$1-guided 40
done
