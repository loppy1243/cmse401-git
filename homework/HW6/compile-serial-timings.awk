BEGIN {
  column_cmd = "column -t"
  trial = 0
  avg = 0; stderr = 0;
  print "Trial", "Time", "err_Time" | column_cmd
}

/^real/ {
  ++trial
  split($2, times, "[ms]")
  t = (times[1] + 0)*60.0 + (times[2] + 0)
  avg += t
  stderr += t^2
  print trial, t | column_cmd
}

END {
  avg /= trial
  stderr = sqrt(stdev/trial - avg^2)

  printf "AVG " | column_cmd
  print avg, stderr | column_cmd
}
