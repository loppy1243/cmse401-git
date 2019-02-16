BEGIN {
  if (ofieldwidth == "") ofieldwidth=9
  printf "%-*s%s%-*s%s",
         ofieldwidth, "Threads", OFS,
         ofieldwidth, "Time", ORS
}

/^THREADS/ {
  split($1, thread_field, "=")
  split($2, time_field, "=")
  printf "%*d%s%*.3f%s",
         ofieldwidth, thread_field[2], OFS,
         ofieldwidth, time_field[2], ORS
}
