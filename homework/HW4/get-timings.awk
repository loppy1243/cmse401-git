BEGIN { print "Name", "Reps", "tot_time", "mean_time", "stddev" }

match($0, /^CLOCK (.*):/, name) {
  if (!(name[1] in times)) {
    times[name[1]] = 0.0
    times2[name[1]] += 0.0
    ntimes[name[1]] = 0
  }

  times[name[1]] += $3
  times2[name[1]] += $3^2
  ntimes[name[1]] += 1
}

END {
  for (n in times) {
    mean = times[n]/ntimes[n]
    sd = sqrt(times2[n]/ntimes[n] - mean^2)
    print n, ntimes[n], times[n], mean, sd
  }
}
