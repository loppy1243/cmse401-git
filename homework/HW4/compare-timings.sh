#!/bin/sh

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <image> <timing_set1> <timing_set2> [timing_set3 ...]"
  exit 1
fi

img="$1"
shift

files=""
for s in "$@"; do
  files+=" timings/compiled/$s/$img.dat"
done

paste -d '|' $files | column -ts '|' -o ' | '
