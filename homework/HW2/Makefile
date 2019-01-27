CC := gcc
WARN := -Wall
CFLAGS := $(CFLAGS) $(WARN)
BENCH_CFLAGS := -DBENCH $(CFLAGS)

NAMES := hw2 provided
BENCH_NAMES := hw2

EXE_SUFFIX := .x
BENCH_SUFFIX := .bench

EXES := $(NAMES:%=%$(EXE_SUFFIX))
BENCH_EXES := $(BENCH_NAMES:%=%$(BENCH_SUFFIX)$(EXE_SUFFIX))

all: $(EXES)

bench: $(BENCH_EXES)

%$(EXE_SUFFIX) : %.c
	$(CC) $(CFLAGS) -o $@ $<

%$(BENCH_SUFFIX)$(EXE_SUFFIX) : %.c
	$(CC) $(BENCH_CFLAGS) -o $@ $<
