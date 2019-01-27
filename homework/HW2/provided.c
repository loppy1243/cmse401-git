#include <stdlib.h>
#include "bench.h"

#define ERR_TOO_MANY_ARGS 1

/* Transpose, assuming column-major (Fortran) storage order (allows direct
   comparison with Fortran routines) */
void transpose( double *a, int ndra, int nr, int nc, double *b, int ndrb )
{
    if (nr < 32) {
        /* perform transpose */
        int i, j, ii;
        double *bb=b;
        const double *aa=a;
        for (j=0; j<nc; j++) {
            ii = 0;
            for (i=0; i<nr; i++) {
	            /* b[j+i*ndrb] = a[i+j*ndra]; */
	            bb[ii] = aa[i];
	            /* Strength reduction */
	            ii += ndrb;
            }
            aa += ndra; 
            bb ++;
        }
    }
    else {
        /* subdivide the long side */
        if (nr > nc) {
            transpose( a, ndra, nr/2, nc, b, ndrb );
            transpose( a + nr/2 ,ndra, nr-nr/2, nc, b+(nr/2)*ndrb, ndrb );
        }
        else {
            transpose( a, ndra, nr, nc/2, b, ndrb );
            transpose( a+ndra*(nc/2), ndra, nr, nc-nc/2, b+nc/2, ndrb );
        }
    }
}

void transposeBase( double *a, int ndra, int nr, int nc, double *b, int ndrb )
{
    /* perform transpose */
    int i, j, ii;
    double *bb=b;
    const double *aa=a;
    for (j=0; j<nc; j++) {
        ii = 0;
        for (i=0; i<nr; i++) {
            /* b[j+i*ndrb] = a[i+j*ndra]; */
            bb[ii] = aa[i];
            /* Strength reduction */
            ii += ndrb;
        }
        aa += ndra; 
        bb ++;
    }
}

#ifdef BENCH

void transpose_wrapper(double *mat, size_t rows, size_t cols, double *mat_T) {
    transpose(mat, rows, rows, cols, mat_T, cols);
}
void transposeBase_wrapper(double *mat, size_t rows, size_t cols, double *mat_T) {
    transposeBase(mat, rows, rows, cols, mat_T, cols);
}

#define NFUNCS 2
static void (*funcs[2])(double *, size_t, size_t, double *) =
    {&transpose_wrapper, &transposeBase_wrapper};
static char *func_names[2] = {"transpose(provided)", "transposeBase(provided)"};

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
#endif // BENCH defined
