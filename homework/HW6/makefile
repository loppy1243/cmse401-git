.PHONY: all gif clean

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

all: mill mill.mpi

mill: rumor_mill.c
	mpicc $(CFLAGS) -O3 -l png16 -o $@ $<

mill.mpi: rumor_mill.mpi.c
	mpicc $(CFLAGS) -O3 -l png16 -o $@ $<

gif:
	convert -delay 20 -loop 0 ./images/file*.png rumors.gif

clean:
	@rm mill 
	@rm rumors.gif
