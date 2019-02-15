#!/bin/bash

url='ftp://ftp.ensembl.org/pub/release-95/fasta/caenorhabditis_elegans/dna'
url+='/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz'

wget -P sequences "$url";
