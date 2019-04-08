BEGIN {
  column_cmd = "column -t"
  trial = 0
  print "Trial", "Time" | column_cmd
}

/^real/ {
  ++trial
  split($2, times, "[ms]")
  print trial, (times[1] + 0)*60.0 + (times[2] + 0) | column_cmd
}
