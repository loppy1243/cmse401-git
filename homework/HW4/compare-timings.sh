#!/bin/bash


if (( $# != 3 )); then
  echo "Usage: $0 <image> <timing_set1> <timing_set2>"
  exit 1
fi

img="$1"
shift

files=()
for s in "$@"; do
  files+=("timings/compiled/$s/$img.dat")
done

tmpdir=$(mktemp -d)
cleanup() {
  rm -rf "$tmpdir"
  (( $# != 0 )) && kill "-SIG$1" $$
}
trap cleanup EXIT
for sig in HUP TERM INT; do trap "cleanup $sig" $sig; done

script1=$(cat <<EOF
FNR == 1 {
  printf "%s", "$1"
  for (i=0; i < NF; ++i) printf " ."
  printf " |\n"
  print ".", \$0, "|"
}
FNR != 1 { print ".", \$0, "|" | "sort -bk 1,1" }
EOF
)

script2=$(cat <<EOF
FNR == 1 {
  printf "%s", "$2"
  for (i=0; i < NF; ++i) printf " ."
  printf "\n"
  print ".", \$0
}
FNR != 1 { print ".", \$0 | "sort -bk 1,1" }
EOF
)

awk -e "$script1" "${files[0]}" >"$tmpdir/table1"
awk -e "$script2" "${files[1]}" >"$tmpdir/table2"
join --header -j2 -a1 -a2 "$tmpdir/table1" "$tmpdir/table2" >"$tmpdir/joined"

#cat "$tmpdir/joined" | column -t
#echo

tot_times=($(awk -e '/^CUMULATIVE/ { print $4 }' "${files[@]}"))
script3=$(cat <<EOF
BEGIN {
  tot_left = ${tot_times[0]}
  tot_right = ${tot_times[1]}

  print "Name", "%$1", "%$2", "%diff/1", "%diff/2", "diff", "tval"
}

FNR >= 3 {
  no_l = \$5 == ""
  no_r = \$12 == ""

  print \$1,
        no_l ? "N/A" : sprintf("%.2f", \$5/tot_left*100),
        no_r ? "N/A" : sprintf("%.2f", \$12/tot_right*100),
        no_l || no_r ? "N/A" : sprintf("%.2f", (\$5-\$12)/\$5*100),
        no_l || no_r ? "N/A" : sprintf("%.2f", (\$5-\$12)/\$12*100),
        no_l || no_r ? "N/A" : sprintf("%.3e", \$5-\$12),
        no_l || no_r ? "N/A" : sprintf("%.2f", (\$5-\$12)/sqrt(\$6^2/\$5+\$13^2/\$11))
}
EOF
)
awk -e "$script3" "$tmpdir/joined" | column -t
