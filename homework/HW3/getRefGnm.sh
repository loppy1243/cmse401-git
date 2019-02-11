#!/bin/bash

url='ftp://ftp.ensembl.org/pub/release-94/fasta/homo_sapiens/dna'
url+='/Homo_sapiens.GRCh38.dna.toplevel.fa.gz'

TIMEFORMAT=$'\nTotal Time: %R' time wget -nv --show-progress -P sequences "$url"
