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

template<T> class CudaMem {
    T *_host_ptr;
    T *_device_ptr;
    size_t size;
public:
    CudaMem(size_t size) {
        this->_host_ptr = new T[size];
        CUDA_CHKERR(cudaMalloc((void **) &this->_device_ptr, size));
        this->size = size;
    }
    ~CudaMem() {
        delete[] this->_host_ptr;
        CUDA_CHKERR(cudaFree(this->_device_ptr));
    }

    T *host_ptr() { return _host_ptr; }
    T *device_ptr() { return _device_ptr; }

    void to_device() {
        CUDA_CHKERR(cudaMemcpy(mem.device_ptr, mem.host_ptr, mem.size,
                               cudaMemcpyHostToDevice));
    }

    void to_host() {
        CUDA_CHKERR(cudaMemcpy(mem.host_ptr, mem.device_ptr, mem.size,
                               cudaMemcpyDeviceToHost));
    }
}

int get_warpSize() {
    static int _warpsize = 0;

    if (_warpsize == 0) {
        cudaDeviceProp props;
        CUDA_CHKERR(cudaGetDeviceProperties(&props, 0);
        _warpsize = props.warpSize;
    }

    return _warpsize;
}

// The block size MUST be warpSize*warpSize
__global__ void sim_kernel(double *z, double *v, int nx, int ny, double dx2inv, double dy2inv, double dt) {
    // Add one so we can have bank-parallelism along rows or along columns. Should benchmark
    // this.
    __static__ double z_tile[warpSize][warpSize/*+1*/];
    __static__ double v_tile[warpSize][warpSize/*+1*/];
    __static__ double ax_tile[warpSize][warpSize/*+1*/];

    const int mesh_x = blockIdx.y*warpSize + threadIdx.y + 1;
    const int mesh_y = blockIdx.x*warpSize + threadIdx.x + 1;

    if (mesh_x >= nx-1 || mesh_y >= ny-1) return;

    const double z_val_y = z_tile[threadIdx.x][threadIdx.y] = IDX2D(z, mesh_x, nx, mesh_y);
                           v_tile[threadIdx.y][threadIdx.x] = IDX2D(v, mesh_x, nx, mesh_y);

    __syncthreads();

    const double z_val_x = z_tile[threadIdx.y][threadIdx.x];

    double ay;
    if (0 < threadIdx.x && threadIdx.x < warpSize-1)
        const int shfl_mask = 0x7 << threadIdx.x-1;
        ax_tile[threadIdx.x][threadIdx.y] =
            dx2inv*(__shfl_down_sync(shfl_mask, z_val_x, 1)
                    + __shfl_up_sync(shfl_mask, z_val_x, 1)
                    - 2.0*z_val_x);
        ay = dy2inv*(__shfl_down_sync(shfl_mask, z_val_y, 1)
                     + __shfl_up_sync(shfl_mask, z_val_y, 1)
                     - 2.0*z_val_y);
    else {
        const int n = (threadIdx.x+1)/warpSize - (warpSize - threadIdx.x)/warpSize;
        ax_tile[threadIdx.x][threadIdx.y] =
            dx2inv*(z_tile[threadIdx.y][threadIdx.x+n] + IDX2D(z, mesh_x-n, nx, mesh_y)
                    - 2.0*z_val_x);
        ay = dy2inv*(IDX2D(z, mesh_x, nx, mesh_y+n) + z_tile[threadIdx.x][threadIdx.y-n]
                     - 2.0*z_val_y);
    }

    __syncthreads();

    double a_val = (ax_tile[threadIdx.y][threadIdx.x] + ay)/2.0;
    double v_val = v_tile[threadIdx.y][threadIdx.x] += dt*a_val;
    IDX2D(v, mesh_x, bx, mesh_y) = v_val;
    IDX2D(z, mesh_x, bx, mesh_y) = z_val_y + dt*v_val;
}

__global__ void normalize(double *z, double *output, int nx) {
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
