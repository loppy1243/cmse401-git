#ifndef _BENCH_H
#define _BENCH_H

#include <time.h>

#ifdef BENCH
#define INIT_CLOCK(label) \
    struct timespec label##_start, label##_end; \
    double label##_time = 0.0;
#define START_CLOCK(label) \
    clock_gettime(CLOCK_MONOTONIC_RAW, &label##_start)
#define STOP_CLOCK(label) \
    clock_gettime(CLOCK_MONOTONIC_RAW, &label##_end); \
    label##_time += (double) (label##_end.tv_sec - label##_start.tv_sec) \
                    + (double) (label##_end.tv_nsec - label##_start.tv_nsec)*1e-9;
#else
#define INIT_CLOCK(label)
#define START_CLOCK(label)
#define STOP_CLOCK(label)
#endif

#endif // _BENCH_H defined
