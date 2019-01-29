set terminal pdfcairo size 8, 6

set key left top
set logscale

array func_names[4]
func_names[1] = 'transpose'
func_names[2] = 'transpose\_blocked(16)'
func_names[3] = 'transpose(provided)'
func_names[4] = 'transposeBase(provided)'

sed_just_matsizes = "sed -e '/^#/ d; /^$/,$ d' <timings.tbl"
nmatsizes = system(sed_just_matsizes." | wc -l")
getmatsize(n) = \
    system(sed_just_matsizes." | awk 'FNR == ".n."{ print $1 }'")

do for [n=1:nmatsizes] {
    matsize = getmatsize(n)
    set xrange [matsize:matsize]
    stats 'timings.tbl' using 1:4 name 'INDEX_'.matsize nooutput
}

set output 'timings.pdf'
set title 'Transpose Algorithms Comparison'
set xlabel 'Matrix size'
set ylabel 'time / 10 samples (s)'
set autoscale x
plot for [i=1:4] 'timings.tbl' \
     index '['.func_names[i].']' \
     using 1:4 \
     title func_names[i]

set output 'rel_timings.pdf'
set ylabel 'rel time / 10 samples'
unset logscale x
plot [n=1:nmatsizes] for [i=1:4] 'timings.tbl' \
     index '['.func_names[i].']' \
     using ($0+0.5) \
           :(column(4)/value('INDEX_'.getmatsize(strcol(0)).'_min_y')) \
           :(1.0-i/6.0) \
           :xtic(getmatsize(strcol(0))) \
     title func_names[i] \
     with boxes fill solid
