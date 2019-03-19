set terminal pdfcairo
set output 'serial_hist.pdf'

plot 'timings/compiled/serial/dev-intel14-k20.dat' using 2 with histogram, \
     'timings/compiled/serial/dev-intel14-k20.dat' using 3 with histogram, \
     'timings/compiled/serial/dev-intel16-k80.dat' using 2 with histogram, \
     'timings/compiled/serial/dev-intel16-k80.dat' using 3 with histogram
