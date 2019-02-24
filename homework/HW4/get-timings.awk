BEGIN { print "Name", "Reps", "tot_time", "avg_time" }

match($0, /^CLOCK (.*):/, name) {
  if (!(name[1] in times)) {
    times[name[1]] = 0.0
    ntimes[name[1]] = 0
  }

  times[name[1]] += $3
  ntimes[name[1]] += 1
}

END {
  for (n in times) {
    print n, ntimes[n], times[n], times[n]/ntimes[n]
  }
}
