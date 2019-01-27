#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#ifdef BENCH
#include <time.h>
#include <str

#if !(SAMPLES + 0)
#error SAMPLES must be a positive integer when benchmarking.
#endif

#endif

#define IDX2D(mat, stride, i, j) (mat)[(i)*(stride) + (j)]

void transpose(double *mat, size_t rows, size_t cols, double *mat_T) {
    for (size_t i=0; i < rows; ++i) {
        for (size_t j=0; j < cols; ++j)
            IDX2D(mat_T, rows, j, i) = IDX2D(mat, cols, i, j);
    }
}

size_t size_t_min(size_t a, size_t b) {
    return a <= b ? a : b;
}

void transpose_blocked(size_t blksize, double *mat, size_t rows, size_t cols, double* mat_T) {
    for (size_t i=0; i < rows; i += blksize) {
        for (size_t j=0; j < cols; j += blksize) {
            for (size_t i1=i; i1 < size_t_min(i+blksize, rows); ++i1) {
                for (size_t j1=j; j1 < size_t_min(j+blksize, cols); ++j1)
                    IDX2D(mat_T, rows, j1, i1) = IDX2D(mat, cols, i1, j1);
            }
        }
    }
}

void print_mat(double *mat, size_t rows, size_t cols) {
    for (size_t i=0; i < rows; ++i) {
        printf("%f", IDX2D(mat, cols, i, 0));
        for (size_t j=1; j < cols; ++j)
            printf(", %f", IDX2D(mat, cols, i, j));
        putchar('\n');
    }
}

void init_mat(double *mat, size_t rows, size_t cols) {
    for (size_t i=0; i < rows; ++i) {
        for (size_t j=0; j < cols; ++j)
            IDX2D(mat, cols, i, j) = (double) i - (double) j;
    }
}

#ifdef BENCH
size_t size_t_max(size_t a, size_t b) {
    return a >= b ? a : b;
}

size_t max_strlen(char **strs, size_t n) {
    size_t ret = 0;
    for (size_t i=0; i < n; ++i)
        ret = size_t_max(ret, strlen(strs[i]));
    return ret;
}

void transpose_blocked_16(double *mat, size_t rows, size_t cols, double* mat_T) {
    transpose_blocked(16, mat, rows, cols, mat_T);
}

static void (*funcs[2])(double *, size_t, size_t, double *) =
    {&transpose, &transpose_blocked_16};
static void *func_names[2] = {"transpose", "transpose_blocked(16)"};
static size_t sizes[8] = {200, 1000, 5000, 8000, 10000, 15000, 20000, 40000};

void bench() {
    time_t start;
    double time;
    size_t name_width = max_strlen(func_names, 2);

    printf("SAMPLES=%d\n", SAMPLES);
    printf("%-3s   %-*s   %-9s   %-9s\n", "size", name_width, "function", "tot", "avg");
    puts(  "--------------------------------------------------------------------------------");
    for (int i=0; i < 8; ++i) {
        size_t size = sizes[i];
        double *mat   = malloc((sizeof (double))*size*size);
        double *mat_T = malloc((sizeof (double))*size*size);
        init_mat(mat, size, size);

        for (size_t j=0; i < NFUNCS; ++j) {
            start = clock();
            for (int n=1; n < SAMPLES; ++n)
                (*funcs[j])(mat, size, size, mat_T);
            time = (clock() - start)/(double) CLOCKS_PER_SEC;
            printf("%5d   %-*s   %.3e   %.3e\n",
                   (unsigned) size, name_width, func_names[j], time, time/(double) SAMPLES);
        }

        free(mat_T);
        free(mat);
    }
}
#endif

#ifdef BENCH
int main() {
    bench();
    return 0;
}
#else
#define ROWS 5
#define COLS 8
int main() {
    double mat[ROWS][COLS], mat_T[COLS][ROWS];
    init_mat((double *) mat, ROWS, COLS);

    print_mat((double *) mat, ROWS, COLS);
    putchar('\n');

    transpose((double *) mat, ROWS, COLS, (double *) mat_T);
    print_mat((double *) mat_T, COLS, ROWS);
    putchar('\n');

    transpose_blocked(16, (double *) mat, ROWS, COLS, (double *) mat_T);
    print_mat((double *) mat_T, COLS, ROWS);

    return 0;
}
#endif
