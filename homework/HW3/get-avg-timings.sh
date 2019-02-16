nthreads=($(awk -e 'FNR != 1 { print $1 }' timings.txt | uniq))

echo "Threads AvgTime"
for n in ${nthreads[@]}; do
  ts=$(awk -e '$1 == '"$n"' { print $2 }' timings.txt)
  sum=$(paste -sd + <<<"$ts")
  ntimes=$(wc -l <<<"$ts")
  avg_time=$(bc <<<"scale=5; ($sum)/$ntimes")
  echo "$n $avg_time"
done
