BEGIN {
  column_cmd = "column -t"
  trial = 0
  print "trial", "total", "init", "sim", "edge_comm", "file_io" | column_cmd
}

/^BENCHMARKING/ {
  ++trial;
  getline; getline

  printf "%d ", trial | column_cmd
  print | column_cmd
}
