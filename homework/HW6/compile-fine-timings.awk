BEGIN {
  column_cmd = "column -t"
  trial = 0
  split("total init sim edge_comm file_io", cats, " ")

  printf "trial" | column_cmd
  for (i in cats) {
    mean[cats[i]] = 0
    stderr[cats[i]] = 0

    printf " %s", cats[i] | column_cmd
  }
  for (i in cats) {
    printf " err_%s", cats[i] | column_cmd
  }
  printf "\n" | column_cmd
}

/^BENCHMARKING/ {
  ++trial;
  getline; getline

  for (i in cats) {
    mean[cats[i]] += $i
    stderr[cats[i]] += ($i + 0)^2
  }

  printf "%d ", trial | column_cmd
  print | column_cmd
}

END {
  printf "AVG" | column_cmd
  for (i in cats) {
    mean[cats[i]] /= trial
    stderr[cats[i]] = sqrt((stderr[cats[i]]/trial - mean[cats[i]]^2)/trial)
    
    printf " %.3e", mean[cats[i]] | column_cmd
  }
  for (i in cats) {
    printf " %.3e", stderr[cats[i]] | column_cmd
  }
  printf "\n"
}
