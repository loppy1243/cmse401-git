#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <mpi.h>

#include <sys/types.h>
#include <unistd.h>

#include "debug.h"
#include "bench.h"


#define IDX2D_2(i, j) ((i)*IDX2D_STRIDE + (j))
#define IDX2D_3(i, j, s) ((i)*(s) + (j))
#define IDX2D_GET_MACRO(_1, _2, _3, macro, ...) macro
#define IDX2D(...) IDX2D_GET_MACRO(__VA_ARGS__, IDX2D_3, IDX2D_2)(__VA_ARGS__)

#define TOP_TO_BOT_TAG 1
#define BOT_TO_TOP_TAG 2
#define OUT_TAG        3
#define INIT_TAG       4

int MY_MPI_RANK;

void writeworld(char *filename, char *my_world, int sz_x, int sz_y) {
   int width = sz_x+2;
   int height = sz_y;

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
   IF_DEBUG(printf("Thread %d writing RGB %d, %d\n", MY_MPI_RANK, width, height);)

   for (int r=0; r < height; r++) {
        for (int i=1; i < width-1; i++) {
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
  IF_DEBUG(printf("Thread %d: done writing file\n", MY_MPI_RANK);)

  free(row);
  fclose(fp);
}

#define IDX2D_STRIDE (sz_x+2)
#define POP_PROB 0.75
char *make_world(int sz_x, int sz_y, int num_rumors) {
    char *full_world = malloc(sz_y*(sz_x+2)*sizeof(char));

    //Initialize Random World
    for (int r=0; r < sz_y; r++)
        for (int c=0; c < sz_x+2; c++) {
            float rd = ((float) rand())/(float)RAND_MAX;
            if (rd < POP_PROB) full_world[IDX2D(r, c)] = 1;
        }

    //Pick Rumor Starting location
    for (int u=0; u < num_rumors; u++) {
        int r = rand()%sz_y;
        int c = rand()%sz_x + 1;
        full_world[IDX2D(r, c)] = u+2;
        IF_DEBUG(printf("Thread %d: Starting a Rumor %d, %d = %d\n", MY_MPI_RANK, r, c, u+2);)
    }

    return full_world;
}

/*
 * Store `size` random integers in the range `lo` to `hi` in ascending order into the array
 * `out`. Uses Knuth's algorithm for generating ascending random numbers.
 */
void ascending_rand(int *out, size_t size, int lo, int hi) {
    const int mx = hi-lo+1;
    size_t j = 0;
    for (int i = 0, j = 0; i < mx && j < size; ++i) {
        if (rand() % (mx - i) < size - j)
            out[j++] = i + lo;
    }
}

#define IDX2D_STRIDE (sz_x+2)
int main(int argc, char **argv) {
    IF_DEBUG(puts("Program starting");)
    MPI_Init(&argc, &argv);

    INIT_CLOCK(total); INIT_CLOCK(init); INIT_CLOCK(edge_comm); INIT_CLOCK(sim);
    INIT_CLOCK(file_io);

    START_CLOCK(total); START_CLOCK(init);

    MPI_Comm_rank(MPI_COMM_WORLD, &MY_MPI_RANK);
    int mpi_world_size; MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

#ifdef DEBUG
    printf("Thread %d: After MPI initialization\n", MY_MPI_RANK);
    char hostname[HOST_NAME_MAX]; gethostname(hostname, HOST_NAME_MAX);
    int pid = getpid();
    printf("Thread %d: %s:%d\n", MY_MPI_RANK, hostname, pid);
#endif

    //Simulation Parameters
    const int sz_x = 1000;
    const int sz_y = 500;
    char filename[sizeof "./images/file00000.png"];
    int img_count = 0;

    const float rumor_prob = 0.25;
    const int num_loops = 1000;
    const int NUM_RUMORS = 7;

    if (mpi_world_size > sz_y) {
        fprintf(stderr,
                "Too many threads! Please set the number of threads to be at most sz_y (%d)."
                "\n",
                sz_y);

        MPI_Finalize();
        return 1;
    }

    // Seed the random number generator on each thread with something unique.
    const int rand_spread = 100;
    const int rand_max = rand_spread*mpi_world_size;
    int rand_seeds[mpi_world_size];
    if (MY_MPI_RANK == 0)
        ascending_rand(rand_seeds, mpi_world_size, 0, rand_max);
    MPI_Bcast(rand_seeds, mpi_world_size, MPI_INT, 0, MPI_COMM_WORLD);
    srand(rand_seeds[MY_MPI_RANK]);

    const int normal_sz_y = (sz_y-1)/mpi_world_size+1;
    const int last_sz_y = normal_sz_y - (normal_sz_y*mpi_world_size - sz_y);

    int my_sz_y = normal_sz_y;
    if (MY_MPI_RANK == mpi_world_size - 1) {
        my_sz_y = last_sz_y;
    }

    const int normal_world_chunk_size = normal_sz_y*(sz_x+2);
    const int last_world_chunk_size = last_sz_y*(sz_x+2);
    const int my_world_chunk_size = my_sz_y*(sz_x+2);
    const int my_world_size = (my_sz_y+2)*(sz_x+2);

    char *full_world = NULL;
    char *my_world[2]; my_world[0] = malloc(2*my_world_size*sizeof(char));
                       my_world[1] = my_world[0] + my_world_size;
    if (MY_MPI_RANK == 0) {
        full_world = make_world(sz_x, sz_y, NUM_RUMORS);

        char *cur_loc = full_world;
        memcpy(&my_world[0][IDX2D(1, 0)], cur_loc, my_world_chunk_size*sizeof(char));
        cur_loc += my_world_chunk_size;

        for (int i = 1; i < mpi_world_size-1; ++i) {
            IF_DEBUG(
                printf("Thread %d: Sending initial world to thread %d\n", MY_MPI_RANK, i);
            )
            MPI_Send(cur_loc, normal_world_chunk_size, MPI_CHAR, i, INIT_TAG, MPI_COMM_WORLD);
            IF_DEBUG(printf("Thread %d: Initial world sent to thread %d\n", MY_MPI_RANK, i);)
            cur_loc += normal_world_chunk_size;
        }
        if (mpi_world_size > 1) {
            IF_DEBUG(
                printf(
                    "Thread %d: Sending initial world to thread %d\n",
                    MY_MPI_RANK, mpi_world_size-1
                );
            )
            MPI_Send(
                cur_loc, last_world_chunk_size, MPI_CHAR, mpi_world_size-1, INIT_TAG,
                MPI_COMM_WORLD
            );
            IF_DEBUG(
                printf(
                    "Thread %d: Initial world sent to thread %d\n",
                    MY_MPI_RANK, mpi_world_size-1
                );
            )
        }
    }
    else if (mpi_world_size > 1) {
        IF_DEBUG(printf("Thread %d: Receiving initial world\n", MY_MPI_RANK);)
        MPI_Recv(
            &my_world[0][IDX2D(1, 0)], my_world_chunk_size, MPI_CHAR, 0, INIT_TAG, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
        IF_DEBUG(printf("Thread %d: Initial world received\n", MY_MPI_RANK);)
    }

    int which = 0;
    int rumor_counts[NUM_RUMORS+2];

    MPI_Request prev_send_reqs[2];

    STOP_CLOCK(init);

    const int next_thread = (MY_MPI_RANK + 1)%mpi_world_size;
    const int prev_thread = (mpi_world_size + MY_MPI_RANK - 1)%mpi_world_size;
    //Main Time loop
    for(int t=0; t<num_loops;t++) {
        //Communicate Edges

        START_CLOCK(edge_comm);

        for (int r=1; r<my_sz_y+1; r++) {
            my_world[which][IDX2D(r, 0)] = my_world[which][IDX2D(r, sz_x)];
            my_world[which][IDX2D(r, sz_x+1)] = my_world[which][IDX2D(r, 1)];
        }

        IF_DEBUG(printf("Thread %d: prev_thread=%d,next_thread=%d\n", MY_MPI_RANK, prev_thread, next_thread));

        MPI_Isend(
            &my_world[which][IDX2D(my_sz_y, 1)], sz_x, MPI_CHAR, next_thread, BOT_TO_TOP_TAG,
            MPI_COMM_WORLD, &prev_send_reqs[0]
        );
        MPI_Isend(
            &my_world[which][IDX2D(1, 1)], sz_x, MPI_CHAR, prev_thread, TOP_TO_BOT_TAG,
            MPI_COMM_WORLD, &prev_send_reqs[1]
        );

        MPI_Recv(
            &my_world[which][IDX2D(0, 1)], sz_x, MPI_CHAR, prev_thread, BOT_TO_TOP_TAG,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        MPI_Recv(
            &my_world[which][IDX2D(my_sz_y+1, 1)], sz_x, MPI_CHAR, next_thread,
            TOP_TO_BOT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        MPI_Waitall(2, prev_send_reqs, MPI_STATUSES_IGNORE);

        STOP_CLOCK(edge_comm);

        START_CLOCK(sim);
        IF_DEBUG(printf("Thread %d: Step %d\n", MY_MPI_RANK, t);)
        for (int r=1; r<my_sz_y+1; r++) {
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
                                my_world[!which][IDX2D(r, c)] = n;
                                break;
                            }
                        }
                    }
                }
            }
        }

        STOP_CLOCK(sim);
        if (t%10 == 0) {
            START_CLOCK(file_io);
            if (MY_MPI_RANK != 0) {
                MPI_Send(
                    &my_world[which][IDX2D(1, 0)], my_world_chunk_size, MPI_CHAR, 0,
                    OUT_TAG, MPI_COMM_WORLD
                );
            }
            else {
                char *cur_loc = full_world;
                memcpy(
                    cur_loc, &my_world[which][IDX2D(1, 0)],
                    my_world_chunk_size*sizeof(char)
                );
                cur_loc += my_world_chunk_size;
                for (int i = 1; i < mpi_world_size-1; ++i) {
                    MPI_Recv(
                        cur_loc, normal_world_chunk_size, MPI_CHAR, i, OUT_TAG,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE
                    );
                    cur_loc += normal_world_chunk_size;
                }
                if (mpi_world_size > 1)
                    MPI_Recv(
                        cur_loc, last_world_chunk_size, MPI_CHAR, mpi_world_size-1, OUT_TAG,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE
                    );

                //Send everything back to master for saving.
                sprintf(filename, "./images/file%05d.png", img_count);
                writeworld(filename, full_world, sz_x, sz_y);
                img_count++;
                STOP_CLOCK(file_io);
            }
        }

        which = !which;
    }
    IF_DEBUG(printf("Thread %d: After Loop\n", MY_MPI_RANK);)

    free(my_world[0]);
    free(full_world);
    IF_DEBUG(printf("Thread %d: After Clean up\n", MY_MPI_RANK);)

    STOP_CLOCK(total);

#ifdef BENCH
    double times[4] = {total_time, edge_comm_time, file_io_time, sim_time};
    double avg_times[4];
    MPI_Reduce(times, avg_times, 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (MY_MPI_RANK == 0) {
        fputs("BENCHMARKING\ntotal init edge_comm file_io\n", stderr);
        for (size_t i = 0; i < 4; ++i) {
            avg_times[i] /= mpi_world_size;
            fprintf(stderr, "%s%.3e", i == 0 ? "" : " ", avg_times[i]);
        }
        fputc('\n', stderr);
    }
#endif

    MPI_Finalize();
    return 0;
}
