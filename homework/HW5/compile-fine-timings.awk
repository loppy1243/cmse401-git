BEGIN {
  column_cmd = "column -t"
  print "TOTAL", "setup", "file_io", "simulation" | column_cmd
}

FNR % 3 == 0 { print | column_cmd }
