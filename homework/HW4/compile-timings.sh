#!/bin/bash

if (( $# < 1 )); then
  echo "Usage: $0 <timing_set>" >&2
  exit 1
fi

script=$(cat <<'HERE'
BEGIN { print "Name", "Reps", "tot_time", "mean_time", "stddev", "%err" }

match($0, /^CLOCK (.*):/, name) {
  if (!(name[1] in times)) {
    times[name[1]] = 0.0
    times2[name[1]] += 0.0
    ntimes[name[1]] = 0
  }

  times[name[1]] += $3
  times2[name[1]] += $3^2
  ntimes[name[1]] += 1
}

END {
  for (n in times) {
    mean = times[n]/ntimes[n]
    sd = sqrt(times2[n]/ntimes[n] - mean^2)
    print n,
          ntimes[n],
          sprintf("%.3e", times[n]),
          sprintf("%.3e", mean),
          sprintf("%.3e", sd),
          sprintf("%.2f", sd/mean^(3.0/2.0)*100)
  }
}
HERE
)

echo -n "Compiling $1 timings..."
if [[ $1 == all ]]; then
  for set in $(ls timings/raw/); do
    mkdir -p timings/compiled/"$set"
    for img_f in images/*; do
      img=$(basename -s.png "$img_f")
      awk -e "$script" timings/raw/"$set"/"$img"/* | column -t \
          >timings/compiled/"$set"/"$img".dat
    done
  done
else
  mkdir -p timings/compiled/"$1"
  for img_f in images/*; do
    img=$(basename -s.png "$img_f")
    awk -e "$script" timings/raw/"$1"/"$img"/* | column -t \
        >"timings/compiled/$1/$img.dat"
  done
fi
echo " Done."
