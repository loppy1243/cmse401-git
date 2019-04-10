ifeq ($(origin DEBUG), undefined)
DEBUG := 0
endif
ifeq ($(origin BENCH), undefined)
BENCH := 0
endif

ifneq ($(DEBUG), 0)
CFLAGS := $(CFLAGS) -DDEBUG
endif
ifneq ($(BENCH), 0)
CFLAGS := $(CFLAGS) -DBENCH
endif

# PFX --> PreFiX
# TSET --> Timing SET
TSET_PFX := timings
TSET_RAW_PFX := $(TSET_PFX)/raw
TSET_COMP_PFX := $(TEST_PFX)/compiled
TSETS := serial fine-serial mpi mpi-avg

PLOT_PFX := plots

.PHONY: all gif clean all-plots timings

all: mill mill.mpi

mill: rumor_mill.c
	mpicc $(CFLAGS) -O3 -l png16 -o $@ $<

mill.mpi: rumor_mill.mpi.c
	mpicc $(CFLAGS) -O3 -l png16 -o $@ $<

timings: $(TSETS:%=%(TSET_COMP_PFX)/%.dat)
$(TSET_COMP_PFX)/%.dat:
	awk -f compile-$(@*F)-timings.awk $(TEST_RAW_PFX)/serial.out >$@
$(TEST_COMP_PFX)/mpi-avg.dat: $(TEST_COMP_PFX)/mpi.dat
	sed -ne '1p;/^AVG/p' $< \
		| sort -n -k4,4 -k2,2 \
		| awk -e 'p != "" && p < $2 { printf "\n\n" } { print; p = $2 }' >$@

all-plots: $(PLOT_PFX)/fine-serial.pdf $(PLOT_PFX)/mpi-scaling.pdf
$(PLOT_PFX)/fine-serial.pdf: $(TEST_COMP_PFX)/fine-serial.pdf
	python hist.py -o $@ -x trial -p $(TSET_COMP) $(@*F)
$(PLOT_PFX)/mpi-scaling.pdf: mpi-timings.plt $(TSET_COMP_PFX)/mpi-avg.dat | $(PLOT_PFX)
	gnuplot $<

$(PLOT_PFX):
	mkdir -p plots

gif:
	convert -delay 20 -loop 0 ./images/file*.png rumors.gif

clean:
	@rm mill 
	@rm rumors.gif
