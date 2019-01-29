set terminal pdfcairo size 8,6 font 'Times,12'

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
minmatsize = getmatsize(1)
maxmatsize = getmatsize(nmatsizes)

do for [n=1:nmatsizes] {
    matsize = getmatsize(n)
    stats [matsize:matsize] 'timings.tbl' using 1:4 name 'INDEX_'.matsize nooutput
}

set output 'timings.pdf'
set multiplot
set origin 0.0,0.4
set size 1.0,0.6

set key left top
set title 'Transpose Algorithms Comparison' font 'Times,20'
set ylabel 'Time / 10 samples (s)'
set autoscale x
set tics scale 2.0,1.0,0.5
set xtics font 'Times,10'
do for [n=1:nmatsizes] {
    set xtics add (getmatsize(n) getmatsize(n) 2)
}
xmin = minmatsize-0.1*minmatsize
xmax = maxmatsize+0.1*maxmatsize
plot [xmin:xmax] for [i=1:4] 'timings.tbl' \
     index '['.func_names[i].']' \
     using 1:4 \
     title func_names[i]
#set multiplot prev
#clear
#set multiplot prev

set origin 0.0,0.0
set size 1.0,0.4

relspeed(col) = value('INDEX_'.getmatsize(strcol(col)+1).'_max_y')/column(4)
set key center top
set xlabel 'Matrix Sizes'
set ylabel 'Speed wrt. Slowest'
set grid front mytics linewidth 2.5
unset title
unset logscale x
set xtics auto scale 0.0,0.0 font 'Times,12'
set ytics add (2 2, 3 3)
plot [][0.9:4.0] for [i=1:4] 'timings.tbl' \
        index '['.func_names[i].']' \
        using ($0+0.5) \
              :(relspeed(0)) \
              :(1.0-i/6.0) \
              :xtic(getmatsize(strcol(0)+1)) \
        title func_names[i] \
        with boxes fill solid
