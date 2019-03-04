#!/bin/bash

if (( $# < 2 )); then
  echo "Usage: $0 <image> <set1> <set2> ..." >&2
  exit 1
fi

img="$1"
shift

for set in "$@"; do
  echo "$set"
  cat timings/compiled/"$set"/"$img".dat
  echo
done
