#!/bin/sh

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <timing_set>" >&2
  exit 1
fi

echo -n "Compiling $1 timings..."
mkdir -p "timings/compiled/$1"
for img_f in images/*; do
  img="$(basename -s.png "$img_f")"
  awk -f get-timings.awk "timings/raw/$1/$img/"* | column -t >"timings/compiled/$1/$img.dat"
done
echo " Done."
