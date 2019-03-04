#!/bin/bash

script=$(cat <<'EOF'
FNR != 1 {
  names[$1] = 0

  if (min[$1] == "" || $4 < min[$1]) {
    prev2_min_set[$1] = prev_min_set[$1]
    prev2_min[$1] = prev_min[$1]
    prev2_min_err[$1] = prev_min_err[$1]

    prev_min_set[$1] = min_set[$1]
    prev_min[$1] = min[$1]
    prev_min_err[$1] = min_err[$1]

    split(FILENAME, fname_comps, "/")
    l = length(fname_comps)
    min_set[$1] = fname_comps[l-1]
    min[$1] = $4
    min_err[$1] = $6
  }
}

END {
  column_pipe = "column -t"

  printf "%s\n", "Name 3rd_min val %err 2nd_min val %err min val %err" | column_pipe
  for (n in names) {
    printf "%s %s %.3e %.2f %s %.3e %.2f %s %.3e %.2f\n",
           n,
           prev2_min_set[n], prev2_min[n], prev2_min_err[n],
           prev_min_set[n], prev_min[n], prev_min_err[n],
           min_set[n], min[n], min_err[n] \
    | column_pipe
  }
  close(column_pipe)
}
EOF
)

for img_file in images/*; do
  img=$(basename -s .png "$img_file")
  echo "$img"
  echo '----------'
  awk -e "$script" timings/compiled/*/"$img".dat
  echo
done
