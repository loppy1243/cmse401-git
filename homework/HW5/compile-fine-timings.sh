#!/bin/bash

for x in "$@"; do
  mkdir -p timings/compiled/"$x"
  for host in dev-intel14-k20 dev-intel16-k80; do
    awk -f compile-fine-timings.awk timings/raw/"$x"/"$host".out >timings/compiled/"$x"/"$host".dat
  done
done
