#!/bin/bash

if (( $# < 3 )); then
  echo "Usage: $0 <image> <timing_set1> <timing_set2> ..."
  exit 1
fi

img="$1"
shift

for i in $(seq 1 $#); do
  for j in $(seq $(( $i+1 )) $#); do
    ./compare-timings.sh "$img" "${!i}" "${!j}"
    echo
  done
done
