#ifndef _BENCH_H
#define _BENCH_H

#ifdef BENCH
    #include <stdio.h>
    #include <time.h>

    #define IFDEF_BENCH(x) x
    clock_t bench_total_clock = 0;
#else
    #define IFDEF_BENCH(x)
#endif

#define _START_CLOCK(name) \
    clock_t name = clock();
#define START_CLOCK(name) IFDEF_BENCH(_START_CLOCK(name))

#define _STOP_CLOCK(name) \
    name = clock() - name; \
    bench_total_clock += name;
#define STOP_CLOCK(name) IFDEF_BENCH(_STOP_CLOCK(name))

#define _PRINT_CLOCK(name) \
    printf("CLOCK %s: %gs\n", #name, (name)/(double) CLOCKS_PER_SEC)
#define PRINT_CLOCK(name) IFDEF_BENCH(_PRINT_CLOCK(name))

#define STOP_PRINT_CLOCK(name) STOP_CLOCK(name); PRINT_CLOCK(name);

#define _PRINT_TOTAL_CLOCK \
    printf("CLOCK CUMULATIVE: %gs\n", bench_total_clock/(double) CLOCKS_PER_SEC);
#define PRINT_TOTAL_CLOCK IFDEF_BENCH(_PRINT_TOTAL_CLOCK)

#endif // _BENCH_H defined
