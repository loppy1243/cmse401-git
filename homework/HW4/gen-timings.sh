#!/bin/sh

if [ \( "$#" -lt 1 \) -o \( "$#" -gt 2 \) ]; then
  echo "Usage: $0 <timing_set> [reps=10]" >&2
  exit 1
fi

reps=10
if [ "$#" -eq 2 ]; then
  reps=$2
fi

for img_f in images/*; do
  img="$(basename -s.png "$img_f")"
  mkdir -p "timings/raw/$1/$img"

  echo -n "Timing $1/$img..."
  for i in $(seq $reps); do
    echo -n " $i"
    ./process "$img_f" out.png >"timings/raw/$1/$img/$i.out"
  done
  echo " Done."
done

. ./compile-timings.sh "$1"
