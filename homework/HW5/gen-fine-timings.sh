#!/bin/bash

prefix=timings/raw/"$2"
mkdir -p "$prefix"
file="$prefix"/"$HOSTNAME".out
echo -n >"$file"

ntrials=50
echo -n Doing $ntrials trials...
for i in $(seq 1 $ntrials); do
  echo -n " $i"
  ./"$1" 2>>"$file" 1>/dev/null
done
echo
