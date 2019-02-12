#!/bin/bash

url_base='ftp://ftp-trace.ncbi.nih.gov/giab/ftp/data/NA12878/Garvan_NA12878_HG001_HiSeq_Exome'
urls[0]="$url_base/NIST7035_TAAGGCGA_L001_R1_001.fastq.gz"
urls[1]="$url_base/NIST7035_TAAGGCGA_L001_R2_001.fastq.gz"

TIMEFORMAT=$'Time: %Rs'
time {
  for i in ${!urls[@]}; do
    time wget -nv -P sequences "${urls[$i]}"
  done
TIMEFORMAT=$'\nTotal Time: %Rs'
}
