#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <errno.h>

#define XMIN 0.0
#define XMAX 10.0
#define NX 500

#define TMIN 0.0
#define TMAX 10
#define NT 1000000

#define GAMMA 1.0

#define CHECK_MEM(ptr) \
     if ((ptr) == NULL) { \
          printf("Failed to allocate memory, error number: %d\n", errno); \
          exit(1); \
     }

void init_ys(double *ys, size_t nx, double dx) {
     for (size_t i=0; i < nx; ++i) {
          double xm5 = XMIN + i*dx - 5.0;
          double y_sqrt = exp(-xm5*xm5);

          ys[i] = y_sqrt*y_sqrt;
     }
}

void solve(double *ys, size_t nx, double dx, double dt) {
     double y_dots[nx]; double y_ddots[nx];

     for (size_t i=0; i < nx; ++i)
          y_dots[i] = y_ddots[i] = 0.0;

     for (size_t j=0; j < NT; ++j) {
          for (size_t i=2; i < nx-1; ++i)
               y_ddots[i] = GAMMA/(dx*dx)*(ys[i+1]+ys[i-1]-2.0*ys[i]);
          for (size_t i=0; i < nx; ++i) {
               ys[i] = ys[i] + y_dots[i]*dt;
               y_dots[i] = y_dots[i] + y_ddots[i]*dt;
          }
     }
}

void bench(size_t reps) {
     double dx = (XMAX-XMIN)/((double) NX);
     double dt = (TMAX-TMIN)/((double) NT);
     double ys[NX];
     clock_t tot = 0.0, tot_sq = 0.0, diff;

     for (size_t i=0; i < reps; ++i) {
          init_ys(ys, NX, dx);
          clock_t start = clock();
          solve(ys, NX, dx, dt);
          diff = clock() - start;
          tot += diff;
          tot_sq += diff*diff;
     }

     double mean = tot/(((double) reps)*CLOCKS_PER_SEC);
     double sdm = sqrt((tot_sq / (((double) reps)*CLOCKS_PER_SEC*CLOCKS_PER_SEC)
                        - mean*mean)
                       / ((double) reps));

     printf("Average elapsed time: %f +- %f seconds.\n", mean, sdm);
}

#ifdef BENCH
int main() {
     #ifndef REPS
     puts("Must specify number of repitions for benchmark as REPS");
     return 1;
     #else
     bench(REPS);
     return 0;
     #endif
}
#else
int main() {
     double dx = (XMAX-XMIN)/((double) NX);
     double dt = (TMAX-TMIN)/((double) NT);
     double ys[NX]; init_ys(ys, NX, dx);
     
     solve(ys, NX, dx, dt);
     for (size_t i=0; i < NX; ++i)
          printf("%.4f ", ys[i]);
     putchar('\n');

     return 0;
}
#endif
