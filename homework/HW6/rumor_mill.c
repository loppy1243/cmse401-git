#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <mpi.h>
#include "debug.h"

#define IDX2D_2(i, j) ((i)*IDX2D_STRIDE + (j))
#define IDX2D_3(i, j, s) ((i)*(s) + (j))
#define IDX2D_GET_MACRO(_1, _2, _3, macro, ...) macro
#define IDX2D(...) IDX2D_GET_MACRO(__VA_ARGS__, IDX2D_3, IDX2D_2)(__VA_ARGS__)

void writeworld(char *filename, char *my_world, int sz_x, int sz_y) {
   int width = sz_x+2;
   int height = sz_y+2;

   FILE *fp = fopen(filename, "wb");
   if (!fp) abort();

   png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   if (!png) abort();

   png_infop info = png_create_info_struct(png);
   if (!info) abort();

   if (setjmp(png_jmpbuf(png))) abort();

   png_init_io(png, fp);

   png_set_IHDR(
     png,
     info,
     width, height,
     8,
     PNG_COLOR_TYPE_RGB,
     PNG_INTERLACE_NONE,
     PNG_COMPRESSION_TYPE_DEFAULT,
     PNG_FILTER_TYPE_DEFAULT
   );
   png_write_info(png, info);

   png_bytep row = (png_bytep) malloc(3 * width * sizeof(png_byte));
   IF_DEBUG(printf("writing RGB %d, %d\n",width, height);)

   int full_size_with_borders = (sz_x+2)*(sz_y+2);

   for (int r=0; r < height; r++) {
        for (int i=0; i < width; i++) {
           switch (my_world[IDX2D(r, i, width)]) {
               case 1:
                    row[(i*3)+0] = 255;
                    row[(i*3)+1] = 255;
                    row[(i*3)+2] = 255;
                    break;
               case 2:
                    row[(i*3)+0] = 255;
                    row[(i*3)+1] = 0;
                    row[(i*3)+2] = 0;
                    break;
               case 3:
                    row[(i*3)+0] = 0;
                    row[(i*3)+1] = 255;
                    row[(i*3)+2] = 0;
                    break;
               case 4:
                    row[(i*3)+0] = 0;
                    row[(i*3)+1] = 0;
                    row[(i*3)+2] = 255;
                    break;
               case 5:
                    row[(i*3)+0] = 255;
                    row[(i*3)+1] = 255;
                    row[(i*3)+2] = 0;
                    break;
               case 6:
                    row[(i*3)+0] = 255;
                    row[(i*3)+1] = 0;
                    row[(i*3)+2] = 255;
                    break;
               case 7:
                    row[(i*3)+0] = 0;
                    row[(i*3)+1] = 255;
                    row[(i*3)+2] = 255;
                    break;
               case 8:
                    row[(i*3)+0] = 233;
                    row[(i*3)+1] = 131;
                    row[(i*3)+2] = 0;
                    break;
                default:
                    row[(i*3)+0] = 0;
                    row[(i*3)+1] = 0;
                    row[(i*3)+2] = 0;
                    break;
           }
        }   
        png_write_row(png, row);
    }
  png_write_end(png, NULL);
  IF_DEBUG(printf("done writing file\n");)

  free(row);
  fclose(fp);
}

#define IDX2D_STRIDE (sz_x+2)
void make_world(char **my_world, int sz_x, int sz_y) {
    int full_size_with_borders = (sz_x+2)*(sz_y+2);
    my_world[0] = (char *) malloc(2*full_size_with_borders*sizeof(char));
    my_world[1] = my_world[0] + full_size_with_borders;
    IF_DEBUG(puts("After Allocate");)

    //Inicialize Random World
    for (int r=0; r < sz_y+2; r++)
       for (int c=0; c < sz_x+2; c++) {
       float rd = ((float) rand())/(float)RAND_MAX;
           if (rd < pop_prob)
            my_world[which][IDX2D(r, c)] = 1;
    }
    IF_DEBUG(puts("After Initialize");)

    //Pick Rumor Starting location
    for (int u=0; u < NUM_RUMORS; u++) {
        int r = (int) ((float)rand())/(float)RAND_MAX*(sz_y+2);
        int c = (int) ((float)rand())/(float)RAND_MAX*(sz_x+2);
        my_world[which][IDX2D(r, c)] = u+2;
        IF_DEBUG(printf("Starting a Rumor %d, %d = %d\n", r, c, u+2);)
    }
    IF_DEBUG(puts("After Start location picked");)
}
void free_world(char **my_world) { free(my_world[0]); }

#define IDX2D_STRIDE (sz_x+2)
int main(int argc, char **argv) {
    srand(0);

    //Simulation Parameters
    int sz_x = 1000;
    int sz_y = 500;
    char filename[sizeof "./images/file00000.png"];
    int img_count = 0;

    float pop_prob = 0.75;
    float rumor_prob = 0.25;
    int num_loops = 1000;
    int NUM_RUMORS = 7;

    int which = 0;
    char *my_world[2]; make_world(&my_world, sz_x, sz_y);
//    writeworld("start.png", my_world[which], sz_x, sz_y);

    //Main Time loop
    for(int t=0; t<num_loops;t++) {
        //Communicate Edges

        for (int c=1; c<sz_x+1; c++) {
           my_world[which][IDX2D(0, c)] = my_world[which][IDX2D(sz_y, c)];
           my_world[which][IDX2D(sz_y+1, c)] = my_world[which][IDX2D(1, c)];
        }
        for (int r=1; r<sz_y+1; r++) {
           my_world[which][IDX2D(r, 0)] = my_world[which][IDX2D(r, sz_x)];
           my_world[which][IDX2D(r, sz_x+1)] = my_world[which][IDX2D(r, 1)];
        }

        IF_DEBUG(printf("Step %d\n",t);)
        int rumor_counts[NUM_RUMORS+2];
        for (int r=1; r<sz_y+1; r++) {
            for (int c=1; c<sz_x+1; c++) {
                my_world[!which][IDX2D(r, c)] = my_world[which][IDX2D(r, c)];
                if (my_world[which][IDX2D(r, c)] >0) {
                    for(int n=0;n<NUM_RUMORS+2;n++)
                        rumor_counts[n] = 0;
                    rumor_counts[my_world[which][IDX2D(r-1, c)]]++;
                    rumor_counts[my_world[which][IDX2D(r+1, c)]]++;
                    rumor_counts[my_world[which][IDX2D(r, c-1)]]++;
                    rumor_counts[my_world[which][IDX2D(r, c+1)]]++;

                    float rd = ((float) rand())/(float)RAND_MAX;
                    float my_prob = 0;
                    for(int n=2;n<NUM_RUMORS+2;n++) {
                        if (rumor_counts[n] > 0) {
                            my_prob += rumor_prob*rumor_counts[n];
                            if(rd <= my_prob) {
//                                IF_DEBUG(printf(".");)
                                my_world[!which][IDX2D(r, c)] = n;
                                break;
                            }
                        }
                    }
                }
            }
        }
        which = !which;
        if (t%10 == 0) {
            //Send everything back to master for saving.
            sprintf(filename, "./images/file%05d.png", img_count);
            writeworld(filename, my_world[0], sz_x, sz_y);
            img_count++;
        }
    }

    //Write out output image using 1D serial pointer
    //writeworld("end.png", world_mem, sz_x, sz_y);
    IF_DEBUG(printf("After Loop\n");)
    free_world(&my_world);
    IF_DEBUG(printf("After Clean up\n");)

    return 0;
}
