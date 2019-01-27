CC := gcc
WARN := -Wall
CFLAGS := $(CFLAGS) $(WARN)
BENCH_CFLAGS := -DBENCH -DSAMPLES=$(SAMPLES) $(CFLAGS)

NAME := hw2
EXENAME := $(NAME).x
BENCHNAME := $(NAME).bench.x

WORKDIR = build
SRCS := hw2.c
OBJS := $(SRCS:%.c=$(WORKDIR)/%.o)
BENCH_OBJS := $(OBJS:%.o=%.bench.o)

all: $(OBJS)
	$(CC) $(CFLAGS) -o $(EXENAME) $^

bench: $(BENCH_OBJS)
	$(CC) $(BENCH_CFLAGS) -o $(BENCHNAME) $^

$(WORKDIR) :
	mkdir $(WORKDIR)

$(WORKDIR)/%.o: %.c $(WORKDIR)
	$(CC) $(CFLAGS) -c -o $@ $<
$(WORKDIR)/%.bench.o: %.c $(WORKDIR)
	$(CC) $(BENCH_CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -rf $(WORKDIR)
