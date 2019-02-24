#!/bin/sh

if ! [ "$#" -eq 1 ]; then
  echo "Usage: $0 <timing_set>" >&2
  exit 1
fi

for img_f in images/*; do
  img="$(basename -s.png "$img_f")"
  mkdir -p "timings/raw/$1/$img"

  echo -n "Timing $1/$img..."
  for i in $(seq 10); do
    echo -n " $i"
    ./process "$img_f" out.png >"timings/raw/$1/$img/$i.out"
  done
  echo " Done."
done

echo -n "Compiling $1 timings..."
mkdir -p "timings/compiled/$1"
for img_f in images/*; do
  img="$(basename -s.png "$img_f")"
  awk -f get-timings.awk "timings/raw/$1/$img/"* >"timings/compiled/$1/$img.dat"
done
echo " Done."
