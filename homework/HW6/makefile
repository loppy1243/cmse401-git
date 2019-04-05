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

mill: rumor_mill.c | images
	mpicc $(CFLAGS) -O3 -l png16 -o $@ $<

mill.mpi: rumor_mill.mpi.c | images
	mpicc $(CFLAGS) -O3 -l png16 -o $@ $<

gif: images
	convert -delay 20 -loop 0 ./images/file*.png rumors.gif

clean:
	@rm -rf images
	@rm mill 
	@rm rumors.gif

images:
	mkdir -p $@
