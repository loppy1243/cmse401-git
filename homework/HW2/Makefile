CC := gcc
WARN := -Wall
CFLAGS := $(CFLAGS) $(WARN)
BENCH_CFLAGS := -DBENCH -DSAMPLES=$(SAMPLES) $(CFLAGS)

NAME := hw2
EXENAME := $(NAME).x
BENCHNAME := $(NAME).bench.x

SRC := hw2.c

all: $(SRC)
	$(CC) $(CFLAGS) -o $(EXENAME) $<

bench: $(SRC)
	$(CC) $(BENCH_CFLAGS) -o $(BENCHNAME) $<
