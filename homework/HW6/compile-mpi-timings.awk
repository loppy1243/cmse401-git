BEGIN {
  column_cmd = "column -t"
  print "trial", "threads", "xsize", "ysize", "total", "init", "sim", "edge_comm", "file_io",
        "err_total", "err_init", "err_sim", "err_edge_comm", "err_file_io"
}
BEGINFILE {
  split(FILENAME, parts, "[-x.]")

  xsize = parts[2] + 0
  ysize = parts[3] + 0
  threads = parts[4] + 0
  trial = 0
  avg["total"] = 0; avg["init"] = 0; avg["sim"] = 0; avg["edge_comm"] = 0; avg["file_io"] = 0
  stderr["total"] = 0; stderr["init"] = 0; stderr["edge_comm"] = 0; stderr["file_io"] = 0
}

/^BENCHMARKING/ {
  ++trial;
  getline; getline
  avg["total"] += $1; avg["init"] += $2; avg["sim"] += $3; avg["edge_comm"] += $4
  avg["file_io"] += $5

  stderr["total"] += ($1+0)^2; stderr["init"] += ($2+0)^2; stderr["sim"] += ($3+0)^2;
  stderr["edge_comm"] += ($4+0)^2; stderr["file_io"] += ($5+0)^2

  printf "%d %d %d %d ", trial, threads, xsize, ysize
  print
}

ENDFILE {
  for (k in avg) {
    avg[k] /= trial

    stderr[k] /= trial
    stderr[k] -= avg[k]^2
    stderr[k] = sqrt(stderr[k]/trial)
  }
  printf "AVG %d %d %d %e %e %e %e %e %e %e %e %e %e\n",
         threads, xsize, ysize,
         avg["total"], avg["init"], avg["sim"], avg["edge_comm"], avg["file_io"],
         stderr["total"], stderr["init"], stderr["sim"], stderr["edge_comm"],
         stderr["file_io"]
  print ""
  print ""
}
