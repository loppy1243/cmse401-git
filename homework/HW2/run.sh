#!/bin/bash

./hw2.bench.x 100 >timings.tbl
./provided.bench.x 100 >>timings.tbl

mailme timings.tbl
