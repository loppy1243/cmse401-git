#!/bin/bash

./hw2.bench.x >timings.tbl
./provided.bench.x >>timings.tbl

mailme timings.tbl
