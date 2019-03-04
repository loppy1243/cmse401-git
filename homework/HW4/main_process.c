#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <omp.h>
#include "png_util.h"
#include "bench.h"

#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))

// Indexing 1D array as 2D given image_size_t sz
#define IDX2D(arr, sz, i, j) (arr)[((i)*sz.width+(j))]

void abort_(const char * s, ...)
{
        va_list args;
        va_start(args, s);
        vfprintf(stderr, s, args);
        fprintf(stderr, "\n");
        va_end(args);
        abort();
}

char *process_img(char *img, char *output, image_size_t sz, int halfwindow, double thresh)
{
START_CLOCK(average_filter);
    //Average Filter 
    #pragma omp parallel for schedule(dynamic, 8)
    for (int r=0; r < sz.height; ++r)
        for (int c=0 ; c < sz.width; ++c) {
            const int rw_min = max(0, r - halfwindow);
            const int rw_max = min(sz.height, r + halfwindow + 1);
            const int cw_min = max(0, c - halfwindow);
            const int cw_max = min(sz.width, c + halfwindow + 1);
            const double count = (rw_max - rw_min)*(cw_max - cw_min);

            double tot = 0;
            for(int rw = rw_min; rw < rw_max; ++rw)
                for(int cw = cw_min; cw < cw_max; ++cw) {
                    tot += (double) IDX2D(img, sz, rw, cw);
                }
            IDX2D(output, sz, r, c) = (int) (tot/count);
        }
STOP_PRINT_CLOCK(average_filter);

    //write debug image
    //write_png_file("after_smooth.png",output,sz);

    double *gradient = (double *) malloc(sz.width*sz.height*sizeof(double));

START_CLOCK(filtering);
    // Gradient filter
    #pragma omp parallel for schedule(dynamic, 64)
    for(int r=1;r<sz.height-1;r++)
        for(int c=1;c<sz.width-1;c++) {
            // _GradientTerm
            #define _GT(rw, cw) IDX2D(output, sz, r+(rw)-1, c+(cw)-1)
            const double Gx =
                -   _GT(0, 0) +   _GT(0, 2)
                - 2*_GT(1, 0) + 2*_GT(1, 2)
                -   _GT(2, 0) +   _GT(2, 2);
            const double Gy =
                -  _GT(0, 0) + -2*_GT(0, 1) -  _GT(0, 2)
                +  _GT(2, 0) +  2*_GT(2, 1) +  _GT(2, 2);
            IDX2D(gradient, sz, r, c) = sqrt(Gx*Gx+Gy*Gy);
        }
STOP_PRINT_CLOCK(filtering);

START_CLOCK(threshholding);
    // thresholding
    #pragma omp parallel for num_threads(8) schedule(dynamic, 16)
    for(int r=0;r<sz.height;r++)
        for(int c=0;c<sz.width;c++) {
            if (IDX2D(gradient, sz, r, c) > thresh)
                IDX2D(output, sz, r, c) = 255;
            else
                IDX2D(output, sz, r, c) = 0;
        }
STOP_PRINT_CLOCK(threshholding);
}

int main(int argc, char **argv)
{
    //Code currently does not support more than one channel (i.e. grayscale only)
    int channels=1; 
    double thresh = 50;
    int halfwindow = 3;

    //Ensure at least two input arguments
    if (argc < 3)
        abort_("Usage: process <file_in> <file_out> <halfwindow=3> <threshold=50>");

    //Set optional window argument
    if (argc > 3)
        halfwindow = atoi(argv[3]);

    //Set optional threshold argument
    if (argc > 4)
        thresh = (double) atoi(argv[4]);

    fprintf(stderr, "Number of available threads: %d\n", omp_get_max_threads());

    //Allocate memory for images
    image_size_t sz = get_image_size(argv[1]);
    char * s_img = (char *) malloc(sz.width*sz.height*channels*sizeof(char));
    char * o_img = (char *) malloc(sz.width*sz.height*channels*sizeof(char));

    //Read in serial 1D memory
START_CLOCK(TOT_file_read);
    read_png_file(argv[1],s_img,sz);
STOP_PRINT_CLOCK(TOT_file_read);
    #ifdef BENCH
    putchar('\n');
    #endif

START_CLOCK(TOT_processing);
    //Run the main image processing function
    process_img(s_img,o_img,sz,halfwindow,thresh);
STOP_PRINT_CLOCK(TOT_processing);
    #ifdef BENCH
    putchar('\n');
    #endif

        //Write out output image using 1D serial pointer
    write_png_file(argv[2],o_img,sz);

    PRINT_TOTAL_CLOCK;

    return 0;
}
