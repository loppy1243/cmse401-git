#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include "png_util.h"
#include "bench.h"

#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))

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
    for(int r=0;r<sz.height;r++)
        for(int c=0;c<sz.width;c++) {
            double count = 0;
            double tot = 0;
            for(int rw=max(0,r-halfwindow); rw<min(sz.height,r+halfwindow+1); rw++)
                for(int cw=max(0,c-halfwindow); cw<min(sz.width,c+halfwindow+1); cw++) {
                    count++;
                    tot += (double) IDX2D(img, sz, rw, cw);
                }
            IDX2D(output, sz, r, c) = (int) (tot/count);
        }
STOP_PRINT_CLOCK(average_filter);

    //write debug image
    //write_png_file("after_smooth.png",output,sz);

    //Sobel Filters
    double xfilter[3][3] =
        {-1, 0, 1,
         -2, 0, 2,
         -1, 0, 1};
    double yfilter[3][3] =
        {-1, -2, -1,
          0,  0,  0,
          1,  2,  1};

    double * gradient = (double *) malloc(sz.width*sz.height*sizeof(double));

START_CLOCK(filtering);
    // Gradient filter
    for(int r=1;r<sz.height-1;r++)
        for(int c=1;c<sz.width-1;c++) {
            double Gx = 0;
            double Gy = 0;
            for(int rw=0; rw<3; rw++)
                for(int cw=0; cw<3; cw++) {
                    Gx += ((double) IDX2D(output, sz, r+rw-1, c+cw-1))*xfilter[rw][cw];
                    Gy += ((double) IDX2D(output, sz, r+rw-1, c+cw-1))*yfilter[rw][cw];
                }
            IDX2D(gradient, sz, r, c) = sqrt(Gx*Gx+Gy*Gy);
        }
STOP_PRINT_CLOCK(filtering);

START_CLOCK(threshholding);
    // thresholding
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
