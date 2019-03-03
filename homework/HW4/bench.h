#ifndef _BENCH_H
#define _BENCH_H

#ifdef BENCH
    #include <stdio.h>
    #include <omp.h>

    #define IFDEF_BENCH(x) x
    double bench_total_clock = 0.0;
#else
    #define IFDEF_BENCH(x)
#endif

#define _START_CLOCK(name) \
    double name = omp_get_wtime();
#define START_CLOCK(name) IFDEF_BENCH(_START_CLOCK(name))

#define _STOP_CLOCK(name) \
    name = omp_get_wtime() - name; \
    bench_total_clock += name;
#define STOP_CLOCK(name) IFDEF_BENCH(_STOP_CLOCK(name))

#define _PRINT_CLOCK(name) \
    printf("CLOCK %s: %gs\n", #name, (name));
#define PRINT_CLOCK(name) IFDEF_BENCH(_PRINT_CLOCK(name))

#define STOP_PRINT_CLOCK(name) STOP_CLOCK(name); PRINT_CLOCK(name);

#define _PRINT_TOTAL_CLOCK \
    printf("CLOCK CUMULATIVE: %gs\n", bench_total_clock);
#define PRINT_TOTAL_CLOCK IFDEF_BENCH(_PRINT_TOTAL_CLOCK)

#endif // _BENCH_H defined
