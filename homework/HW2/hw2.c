#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "bench.h"

#define ERR_TOO_MANY_ARGS 1

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

#ifdef BENCH
void transpose_blocked_16(double *mat, size_t rows, size_t cols, double* mat_T) {
    transpose_blocked(16, mat, rows, cols, mat_T);
}

#define NFUNCS 2
static void (*funcs[2])(double *, size_t, size_t, double *) =
    {&transpose, &transpose_blocked_16};
static char *func_names[2] = {"transpose", "transpose_blocked(16)"};

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Expected 1 argument, found %d. Usage:\n", argc);
        fprintf(stderr, "    %s <samples>\n", argv[0]);
        return ERR_TOO_MANY_ARGS;
    }
    int samples = atoi(argv[1]);

    bench_transpose(samples, funcs, func_names, NFUNCS);

    return 0;
}
#else // ifdef BENCH
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
#endif // BENCH defined
