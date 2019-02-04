#!/bin/bash

url_base='ftp://ftp-trace.ncbi.nih.gov/giab/ftp/data/NA12878/Garvan_NA12878_HG001_HiSeq_Exome'
urls[0]="$url_base/NIST7035_TAAGGCGA_L001_R1_001.fastq.gz"
urls[1]="$url_base/NIST7035_TAAGGCGA_L001_R2_001.fastq.gz"

getdata() {
  wget -P sequences
  wget -P sequences Garvan_NA12878_HG001_HiSeq_Exome/NIST7035_TAAGGCGA_L001_R2_001.fastq.gz
}

tot_time=$(TIME='%E' time for url in "${urls[@]}"; do
  wget -nv --show-progress -P "$url"
done)

echo "Total time: $tot_time"
