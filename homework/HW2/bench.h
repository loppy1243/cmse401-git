#ifndef _INCLUDE_BENCH_H
#define _INCLUDE_BENCH_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define IDX2D(mat, stride, i, j) (mat)[(i)*(stride) + (j)]

size_t size_t_max(size_t a, size_t b) {
    return a >= b ? a : b;
}

size_t max_strlen(char **strs, size_t n) {
    size_t ret = 0;
    for (size_t i=0; i < n; ++i)
        ret = size_t_max(ret, strlen(strs[i]));
    return ret;
}

typedef void (*transpose_func_ptr)(double *, size_t, size_t, double *);

static size_t sizes[8] = {200, 1000, 5000, 8000, 10000, 15000, 20000, 40000};

void init_mat(double *mat, size_t rows, size_t cols) {
    for (size_t i=0; i < rows; ++i) {
        for (size_t j=0; j < cols; ++j)
            IDX2D(mat, cols, i, j) = (double) i - (double) j;
    }
}

void bench_transpose(int samples, transpose_func_ptr *funcs, char **func_names, size_t nfuncs) {
    time_t start;
    double time;
    size_t name_width = max_strlen(func_names, nfuncs);

    printf("SAMPLES=%d\n", samples);
    printf("%-5s   %-*s   %-9s   %-9s\n", "size", (int) name_width, "function", "tot", "avg");
    puts(  "--------------------------------------------------------------------------------");
    fflush(stdout);
    for (int i=0; i < 8; ++i) {
        size_t size = sizes[i];
        double *mat   = malloc((sizeof (double))*size*size);
        double *mat_T = malloc((sizeof (double))*size*size);
        init_mat(mat, size, size);

        for (size_t j=0; j < nfuncs; ++j) {
            start = clock();
            for (int n=1; n < samples; ++n)
                (*funcs[j])(mat, size, size, mat_T);
            time = (clock() - start)/(double) CLOCKS_PER_SEC;
            printf("%5d   %-*s   %.3e   %.3e\n",
                   (unsigned) size, (int) name_width, func_names[j], time, time/(double) samples);
            fflush(stdout);
        }

        free(mat_T);
        free(mat);
    }
    fflush(stdout);
}

#endif // _INCLUDE_BENCH_H defined
