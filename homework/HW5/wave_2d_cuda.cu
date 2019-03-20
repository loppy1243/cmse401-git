#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "png_util.h"
#include "bench.h"

#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))
#define IDX2D(a, i, stride, j) ((a)[(i)*(stride) + (j)])
#define CHKERR(test, message) \
    if ((test)) fprintf(stderr, "%s:%d: ERROR: %s\n", __FILE__, __LINE__, (message));
#define CUDA_CHKERR(x) { \
    cudaError_t cuda_error__ = (x); \
    if (cuda_error__) \
        fprintf(stderr, "%s:%d: CUDA ERROR: %d returned \"%s\"\n", \
                __FILE__, __LINE__, cuda_error__, cudaGetErrorString(cuda_error__)); \
}
#define DEFINE_CUDA_MEM_STRUCT(ty) \
    struct ty##_cuda_mem { \
        ty *host_ptr; \
        ty *device_ptr; \
        size_t size;
    }; typedef struct ty##_cuda_mem ty##_cuda_mem;


DEFINE_CUDA_MEM_STRUCT(void);
DEFINE_CUDA_MEM_STRUCT(double);
DEFINE_CUDA_MEM_STRUCT(char);

void_cuda_mem cuda_mem_malloc(size_t size) {
    void_cuda_mem ret;

    ret.host_ptr = malloc(size);
    CHKERR(ret.host_ptr == NULL, "Failed to allocate host memory in cuda_mem_malloc()");
    CUDA_CHKERR(cudaMalloc(&ret.device_ptr, size));
    ret.size = size;

    return ret;
}

void cuda_mem_free(void_cuda_mem mem) {
    free(ret.host_ptr);
    CUDA_CHKERR(cudaFree(ret.device_ptr));
}

void cuda_mem_to_device(void_cuda_mem mem) {
    CUDA_CHKERR(cudaMemcpy(mem.device_ptr, mem.host_ptr, mem.size, cudaMemcpyHostToDevice));
}

void cuda_mem_to_host(void_cuda_mem mem) {
    CUDA_CHKERR(cudaMemcpy(mem.host_ptr, mem.device_ptr, mem.size, cudaMemcpyDeviceToHost));
}

#define BLKSIZE 32
#define TILESIZE 8
__global__ void sim_kernel(double *z, double *v, double *a, int nx, int ny, double dx2inv, double dy2inv, double dt) {
    // TEST adding 1 to last dimension
    __static__ double z_tile[BLKSIZE][BLKSIZE];
    __static__ double v_tile[BLKSIZE][BLKSIZE];
    __static__ double a_tile[BLKSIZE][BLKSIZE];
    __static__ char output_tile[BLKSIZE][BLKSIZE];

    int block_thrd_y = TILESIZE*threadIdx.y;
    int mesh_x = blockIdx.y*BLKSIZE + threadIdx.x;
    int mesh_y = blockIdx.x*BLKSIZE + block_thrd_y;

    for (int k = 0; k < TILESIZE; ++k) {
        z_tile[threadIdx.x][block_thrd_y+k] = IDX2D(z, mesh_x, nx, mesh_y+k);
        v_tile[threadIdx.x][block_thrd_y+k] = IDX2D(v, mesh_x, nx, mesh_y+k);
        a_tile[threadIdx.x][block_thrd_y+k] = IDX2D(a, mesh_x, nx, mesh_y+k);
    }

    __syncthreads();

//    // Only for threads in a warp :(
//    __shfl_down_sync()

    const int n = (gridDim.y - blockIdx.y)/gridDim.y;
    const int m = (blockIdx.y + 1)/gridDim.y;
#define STERM(a, b, c) ((a)[block_thrd_y+(b)][threadIdx.x+(c)])
#define GTERM(a, b, c) IDX2D((a), threadIdx.x+(b), nx, block_thrd_y+(c))
    if (0 < mesh_x && mesh_x < nx-1 && 0 < mesh_y && mesh_y < ny-1) {
        if (0 < threadIdx.x && threadIdx.x < BLKSIZE-1) {
            for (int k = n; k < TILESIZE-m; ++k) {
                const double ax =
                    dx2inv*(STERM(z_tile, k+1, 0) + STERM(z_tile, k-1, 0)
                            - 2.0*STERM(z_tile, k, 0));
                const double ay =
                    dy2inv*(STERM(z_tile, k, 1) + STERM(z_tile, k, -1)
                            - 2.0*STERM(z_tile, k, 0));
                STERM(a_tile, k, 0) = (ax + ay)/2.0;
            }
        }
        else {
            for (int k = n; k < TILESIZE-m; ++k) {
                const double ax =
                    dx2inv*(STERM(z_tile, k+1, 0) + STERM(z_tile, k-1, 0)
                            - 2.0*STERM(z_tile, k, 0));
                const double ay =
                    dy2inv*(GTERM(z, k, 1) + GTERM(z, k, -1) - 2.0*STERM(z, k, 0));

                STERM(a_tile, 0, 0) = (ax + ay)/2.0;
            }
        }

#define TERM(a, b, c) ((a)[threadIdx.x+(b)][block_thrd_y+k+(c)])
        for (int k = 0; k < TILESIZE; ++k) {
            STERM(v_tile, k, 0) += dt*STERM(a_tile, k, 0);
            STERM(z_tile, k, 0) += dt*STERM(v_tile, k, 0);
        }

        for (int k = 0; k < TILESIZE; ++k) {
            IDX2D(z, mesh_x, nx, mesh_y+k) = z_tile[threadIdx.x][block_thrd_y+k];
            IDX2D(v, mesh_x, nx, mesh_y+k) = v_tile[threadIdx.x][block_thrd_y+k];
            IDX2D(a, mesh_x, nx, mesh_y+k) = a_tile[threadIdx.x][block_thrd_y+k];
        }
    }
}

int main(int argc, char ** argv) {
    INIT_CLOCK(setup); INIT_CLOCK(simulation); INIT_CLOCK(file_io); INIT_CLOCK(total);

    START_CLOCK(total); START_CLOCK(setup);
    const int nx = 500;
    const int ny = 500;
    const int mesh_size = nx*ny;
    const int nt = 10000; 
    //int nt = 1000000;
    int frame = 0;
    // fps = 1/(dt*frame_skip)
    int frame_skip = 100;
    int r,c,it;
    double dx,dy,dt;
    double max,min;
    double tmax;
    double dx2inv, dy2inv;
    char filename[sizeof "./images/file00000.png"];

    image_size_t sz; 
    sz.width=nx;
    sz.height=ny;

    const size_t double_mesh_size = mesh_size*sizeof(double);
    const size_t char_mesh_size = mesh_size*sizeof(char);

    //make mesh
    double_cuda_mem z = cuda_mem_malloc(double_mesh_size);
    //Velocity
    double_cuda_mem v = cuda_mem_malloc(double_mesh_size);
    //Accelleration
    double_cuda_mem a = cuda_mem_malloc(double_mesh_size);
    //output image
    char_cuda_mem output = cuda_mem_malloc(char_mesh_size);

    max=10.0;
    min=0.0;
    dx = (max-min)/(double)(nx-1);
    dy = (max-min)/(double)(ny-1);
    
    tmax=20.0;
    dt = (tmax-0.0)/(double)(nt-1);

    double x,y; 
    for (r=0;r<ny;r++) {
        for (c=0;c<nx;c++) {
        x = min+(double)c*dx;
        y = min+(double)r*dy;
            IDX2D(z.host_ptr, r, nx, c) = exp(-(sqrt((x-5.0)*(x-5.0)+(y-5.0)*(y-5.0))));
            IDX2D(v.host_ptr, r, nx, c) = 0.0;
            IDX2D(a.host_ptr, r, nx, c) = 0.0;
        }
    }

    cuda_mem_to_device(z);
    cuda_mem_to_device(a);
    cuda_mem_to_device(v);
    cuda_mem_to_device(output);

    STOP_CLOCK(setup);

    printf("nt=%d, dt=%g, frame_skip=%d, fps=%g\n", nt, dt, frame_skip, 1/(dt*frame_skip));

    START_CLOCK(simulation);
    dx2inv = 1.0/(dx*dx);
    dy2inv = 1.0/(dy*dy);

    for(it=0;it<nt-1;it++) {
    //printf("%d\n",it);
        for (r=1;r<ny-1;r++)  
            for (c=1;c<nx-1;c++) {
                const double z_val =    IDX2D(z, r,   nx, c);
                const double z_x_high = IDX2D(z, r+1, nx, c);
                const double z_x_low =  IDX2D(z, r-1, nx, c);
                const double z_y_high = IDX2D(z, r,   nx, c+1);
                const double z_y_low =  IDX2D(z, r,   nx, c-1);
                const double ax = (z_x_high+z_x_high-2.0*z_val)*dx2inv;
                const double ay = (z_y_high+z_y_low-2.0*z_val)*dy2inv;
                IDX2D(a, r, nx, c) = (ax+ay)/2;
            }
        for (r=1; r<ny-1; r++)  
            for (c=1;c<nx-1;c++) {
                IDX2D(v, r, nx, c) = IDX2D(v, r, nx, c) + dt*IDX2D(a, r, nx, c);
                IDX2D(z, r, nx, c) = IDX2D(z, r, nx, c) + dt*IDX2D(v, r, nx, c);
            }

        if (it % frame_skip == 0) {
            double mx,mn;
            mx = -999999;
            mn = 999999;
            for (size_t k = 0; k < mesh_size; ++k) {
                mx = max(mx, z[k]);
                mn = min(mn, z[k]);
            }
            for (size_t k=0; k < mesh_size; ++k)
                output[k] = (char) round((z[k]-mn)/(mx-mn)*255);

            STOP_CLOCK(simulation);
            START_CLOCK(file_io);

            sprintf(filename, "./images/file%05d.png", frame);
            printf("Writing %s\n",filename);    
            write_png_file(filename,output,sz);

            STOP_CLOCK(file_io);
            START_CLOCK(simulation);

            frame+=1;
        }
    }
    
    double mx,mn;
    mx = -999999;
    mn = 999999;
    for (size_t k = 0; k < mesh_size; ++k) {
        mx = max(mx, z[k]);
        mn = min(mn, z[k]);
    }

    printf("%f, %f\n", mn, mx);

    for (size_t k = 0; k < mesh_size; ++k)
        output[k] = (char) round((z[k]-mn)/(mx-mn)*255);
    STOP_CLOCK(simulation);

    START_CLOCK(file_io);
    sprintf(filename, "./images/file%05d.png", it);
    printf("Writing %s\n",filename);    
    //Write out output image using 1D serial pointer
    write_png_file(filename,output,sz);
    STOP_CLOCK(file_io); STOP_CLOCK(total);

#ifdef BENCH
    fputs("BENCHMARKING\nTOTAL setup file_io simulation\n", stderr);
    fprintf(stderr, "%.3e %.3e %.3e %.3e\n", total_time, setup_time, file_io_time, simulation_time);
#endif

    return 0;
}
