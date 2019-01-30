set terminal pdfcairo size 8,6 font 'Times,12'

stats 'timings.tbl' nooutput
nfuncs = STATS_blocks
stats 'timings.tbl' index 0 nooutput
nmatsizes = STATS_records

getfuncname(n) = \
    system("sed -rne 's/^#\\[(.+)\\]#$/\\1/ p' <timings.tbl | sed -ne '".sprintf('%d', n)." p'")
getmatsize(n) = \
    system("sed -e '/^#/ d; /^\\s*$/,$ d' <timings.tbl " \
           ."| awk 'FNR == ".sprintf('%d', n)." { print $1 }'")

array funcnames[nfuncs]
array a[nfuncs]
array b[nfuncs]
fit_func(a, b, x) = a*x + b
do for [n=1:nfuncs] {
    funcnames[n] = getfuncname(n)
    eval 'a'.n.' = 2.8'
    eval 'b'.n.' = -11.0'
    eval "fit fit_func(a".n.", b".n.", x) 'timings.tbl' index ".(n-1)." using (log10($1)):(log10($4)) via a".n.",b".n
    eval 'a['.n.'] = a'.n
    eval 'b['.n.'] = b'.n
}
f(n, x) = 10**fit_func(a[n], b[n], log10(x))

array matsizes[nmatsizes]
array maxtimes_by_matsize[nmatsizes]
do for [n=1:nmatsizes] {
    matsize = getmatsize(n)
    matsizes[n] = matsize
    stats [matsize:matsize] 'timings.tbl' using 1:4 nooutput
    maxtimes_by_matsize[n] = STATS_max_y
}
minmatsize = matsizes[1]
maxmatsize = matsizes[nmatsizes]

set output 'timings.pdf'
set multiplot
set origin 0.0,0.4
set size 1.0,0.6

funcname_with_eq(n) = \
    funcnames[n].' -- ('.sprintf("%.2e", 10**b[n]).')n^{'.sprintf("%.2f", a[n]).'}'
set key left top
set title 'Transpose Algorithms Comparison' font 'Times,20'
set ylabel 'Average Time (s)'
set logscale
set autoscale x
set tics scale 2.0,1.0,0.5
set xtics font 'Times,10'
do for [n=1:nmatsizes] {
    matsize = getmatsize(n)
    set xtics add (matsize matsize 2)
}
xmin = minmatsize-0.1*minmatsize
xmax = maxmatsize+0.1*maxmatsize
plot [xmin:xmax] for [i=0:nfuncs-1] 'timings.tbl' \
        index i \
        using 1:4 \
        linestyle i+1 \
        title funcname_with_eq(i+1), \
     for [n=1:nfuncs] f(n, x) \
        linestyle n \
        notitle 

set origin 0.0,0.0
set size 1.0,0.4

relspeed = 'maxtimes_by_matsize[column(0)+1]/column(4)'
box_width(i) = 1.0-(i+1)/5.0
getsign(i) = i == 1 ? 1 : i == 2 ? 1 : i == 3 ? -1 : -1
unset title
unset logscale x
set key center top
set xlabel 'Matrix Size'
set ylabel 'Speed wrt. Slowest'
set grid mytics linewidth 2.0
set xtics auto scale 0.0,0.0 font 'Times,12'
set ytics add (2 2, 3 3)
plot [-0.5:nmatsizes-0.5][0.9:4.0] \
     for [i=0:nfuncs-1] 'timings.tbl' \
        index i \
        using 0 \
              :(@relspeed) \
              :(box_width(i)) \
              :xtic(getmatsize(column(0)+1)) \
        title getfuncname(i+1) \
        with boxes fill solid, \
     for [i=0:nfuncs-1] '' \
        index i \
        using ($0 + getsign(i)*0.5*box_width(i)) \
              :(@relspeed*1.05) \
              :(abs(@relspeed) - 1.0 < 0.1 ? '' : sprintf("%.1f", @relspeed)) \
        notitle \
        with labels
