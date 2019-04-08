set terminal pdfcairo size 14,10 noenhanced font "Times,10"
set output 'plots/mpi-scaling.pdf'

datafile = 'timings/compiled/mpi-avg.dat'

array headings[5]
do for [i=5:9] {
    headings[i-4] = system('awk -e "FNR == 1 { print \$'.i.' }" '.datafile)
}

set multiplot layout 6,5

do for [j=0:5] {
    if (j == 5) { set xlabel 'Threads' font "Times,12" }
    do for [i=5:9] {
        if (i == 5) { set ylabel 'Average Time' font "Times,12" } else { unset ylabel }
        if (j == 0) { set title headings[i-4] font "Times,14" } else { unset title }
        plot datafile index j using 2:i:(column(i+4)) with yerrorbars notitle
    }
}
