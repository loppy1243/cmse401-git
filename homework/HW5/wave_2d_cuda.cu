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
protected:
    T *_host_ptr;
    T *_device_ptr;
    size_t size;

public:
    CudaMem(size_t size) {
        this->_host_ptr = new T[size];
        CUDA_CHKERR(cudaMalloc((void **) &this->_device_ptr, size));
        this->size = size;
    }
    CudaMem(size_t size, size_t align) {
        const size_t x = size % align;
        size = x == 0 ? size : size - x + align;

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

    void to_device(size_t off, size_t n) {
        CUDA_CHKERR(cudaMemcpy(mem.device_ptr+off, mem.host_ptr+off, n*sizeof(T),
                               cudaMemcpyHostToDevice));
    }
    void to_device() { this->to_device(0, this->size); }

    void to_host(size_t off, size_t n) {
        CUDA_CHKERR(cudaMemcpy(mem.host_ptr+off, mem.device_ptr+off, n*sizeof(T),
                               cudaMemcpyDeviceToHost));
    }
    void to_host() { this->to_host(0, this->size); }

    void device_copy(CudaMem<T> other, size_t n) {
        CUDA_CHKERR(cudaMemcpy(this->device_ptr, other.device_ptr, n*sizeof(T),
                               cudaMemcpyDeviceToDevice));
    }

    void device_copy(CudaMem<T> other) {
        const size_t n = this->size > other.size ? other.size : this->size;
        this->device_copy(other, n);
    }
}

cudaDeviceProp get_deviceProps() {
    static cudaDeviceProp props;
    static bool gotten = false;

    if (!gotten) CUDA_CHKERR(cudaGetDeviceProperties(&props, 0));

    return props;
}

int get_warpSize() {
    return get_deviceProps().warpSize;
}

int get_maxGridSize() {
    int *sizes = get_deviceProps().maxGridSize;

    return sizes[0]*size[1]*sizes[2];
}

__device__ inline double warp_accel(double z, double d_2inv) {
    return d_2inv*(__shfl_down_sync(shfl_mask, z, 1) + __shfl_up_sync(shfl_mask, z, 1) - 2.0*z);
}

// The block size MUST be warpSize by warpSize.
__global__ void sim_kernel(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
    // Add one so we can have bank-parallelism along rows or along columns. Should benchmark
    // this.
    __static__ double z_tile[warpSize][warpSize/*+1*/];
    __static__ double ax_tile[warpSize][warpSize/*+1*/];

    const int mesh_x = blockIdx.y*warpSize + threadIdx.y + 1;
    const int mesh_y = blockIdx.x*warpSize + threadIdx.x + 1;

    if (mesh_x >= nx-1 || mesh_y >= ny-1) return;

    const double z_val_y = z_tile[threadIdx.y][threadIdx.x] = IDX2D(z, mesh_x, ny, mesh_y);

    __syncthreads();

    const double z_val_x = z_tile[threadIdx.x][threadIdx.y];

    double ay;
    if (0 < threadIdx.x && threadIdx.x < warpSize-1)
        const int shfl_mask = 0x7 << threadIdx.x-1;
        ax_tile[threadIdx.y][threadIdx.x] = warp_accel(z_val_x, dx2inv);
        ay = warp_accel(z_val_y, dy2inv);
    else {
        const int n = (threadIdx.x+1)/warpSize - (warpSize - threadIdx.x)/warpSize;
        ax_tile[threadIdx.y][threadIdx.x] =
            dx2inv*(z_tile[threadIdx.x+n][threadIdx.y] + IDX2D(z, mesh_x-n, ny, mesh_y)
                    - 2.0*z_val_x);
        ay = dy2inv*(IDX2D(z, mesh_x, ny, mesh_y-n) + z_tile[threadIdx.y][threadIdx.x+n]
                     - 2.0*z_val_y);
    }

    __syncthreads();

    const double a_val = (ax_tile[threadIdx.x][threadIdx.y] + ay)/2.0;
    const double v_val = IDX2D(v, mesh_x, ny, mesh_y) += dt*a_val;
    IDX2D(z, mesh_x, bx, mesh_y) = z_val_y + dt*v_val;
}

void cuda_sim(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
    const int warpsize = get_warpSize();
    const dim3 blocksize(warpsize, warpsize);
    const dim3 gridsize((ny-1)/blocksize+1, (nx-1)/blocksize+1);

    sim_kernel<<<gridsize, blocksize>>>(z, b, nx, ny, dx2inv, dy2inv, dt);
}

// Return value is defined only for the first thread of each warp.
// `blockDim.x` must be a multiple of `warpSize`.
__device__ double warp_min_max(double *m_ptr, double *M_ptr) {
    double m = *m_ptr, M = *M_ptr;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        const int mask = (1 << (2*offset)) - 1;
        const double other_m = __shfl_down_sync(mask, m, offset);
        const double other_M = __shfl_down_sync(mask, M, offset);
        if (other_m < m) m = other_m;
        if (other_M > M) M = other_M;
    }

    *m_ptr = m; *M_ptr = M;
}

// `blockDim.x` must be a multiple of `warpSize` less than `warpSize**2`.
__device__ double block_min_max(double *m_ptr, double *M_ptr) {
    __shared__ double warp_share_m[warpSize];
    __shared__ double warp_share_M[warpSize];

    const int warp_lane = threadIdx.x % warpsize;
    const int warp_idx = threadIdx.x/warpSize;

    if (warp_idx == 0) {
        warp_share_m[warp_lane] = *m_ptr;
        warp_share_M[warp_lane] = *M_ptr;
    }

    warp_min_max(m_ptr, M_ptr);
    if (warp_lane == 0) {
        warp_share_m[warp_idx] = *m_ptr;
        warp_share_M[warp_idx] = *M_ptr;
    }

    *m_ptr = warp_share_m[warp_lane];
    *M_ptr = warp_share_M[warp_lane];
    if (warp_idx == 0) val = warp_min_max(m_ptr, M_ptr);
}

// Reduce `in_m`/`in_M` to `gridDim.x` candidates in `out_m`/`out_M` for its minimim/maximum.
// Assumes
//     1D grid and blocks.
//     `gridDim.x = k*warpSize` for `1 <= k <= warpSize`.
//     `in_m` and `in_M` must have size `>= gridDim.x*blockDim.x`.
//     `out_m` and `out_M` must have size `>= gridDim.x`.
__global__ block_min_max_kernel(double *in_m, double *in_M, int size, double *out_m, double *out_M) {
    const double grid_size = blockDim.x*gridDim.x;

    double m = in_m[i]; double M = in_M[i];
    for (int i = idx; i < size; i += grid_size) {
        if (in_m[i] < m) m = other;
        if (in_M[i] > M) M = other;
    }

    block_min_max(&m, &M);

    if (threadIdx.x == 0) {
        out_m[blockIdx.x] = m; out_M[blockIdx.x] = M;
    }
}

#define CUDA_MAX_MAX_BLOCKS 1024
// Compute the min and max of `z` in `scratch_m[0]` and `scratch_M[0]`, respectively.
// `scratch_m` and `scratch_M` must have size at least gridsize.
void cuda_min_max(double *z, size_t size, double *scratch_m, double *scratch_M) {
    const int warpsize = get_warpSize();

    const int blocksize = min(size - size%warpsize, get_maxThreadsPerBlock());
    const int gridsize = min(size/blocksize, CUDA_MAX_MAX_BLOCKS);
    const int gridsize2 = gridsize - gridsize%warpsize;

    block_max_kernel<<<gridsize, blocksize>>>(z, z, size, scratch_m, scratch_M);
    block_max_kernel<<<1, gridsize2>>>(scratch_m, scratch_M, gridsize, scratch_m, scratch_M);
}

__global__ void grayscale_kernel(double *z, char *output, size_t size, double z_min, double z_max) {
    const double grid_size = blockDim.x*gridDim.x;
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;

    for (int i = idx; i < size; i += grid_size)
        output[i] = (char) round((z[i]-z_min)/(z_max-z_min)*255);
}

#define CUDA_GRAYSCALE_MAX_BLOCKS 1024
// `z` and `output` can alias.
void cuda_grayscale(double *z, double *output, size_t size, double z_min. double z_max) {
    const int blocksize = min(size, get_maxThreadsPerBlock());
    const int gridsize = min(size/blocksize, CUDA_GRAYSCALE_MAX_BLOCKS);

    grayscale_kernel<<<gridsize, blocksize>>>(z, output, size, z_min, z_max);
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

    CudaMem<double> z(nx*ny), v(nx*ny), a(nx*ny);
    CudaMem<char> output(nx*ny);

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

    z.to_device(); a.to_device(); v.to_device();

    STOP_CLOCK(setup);

    printf("nt=%d, dt=%g, frame_skip=%d, fps=%g\n", nt, dt, frame_skip, 1/(dt*frame_skip));

    START_CLOCK(simulation);
    dx2inv = 1.0/(dx*dx);
    dy2inv = 1.0/(dy*dy);

    for(it=0;it<nt-1;it++) {
        cuda_sim(z.device_ptr(), v.device_ptr(), a.device_ptr(), nx, ny, dx2inv, dy2inv, dt);

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

            /*
             * cuda_min_max(z.device_ptr(), z.size, min_scratch, max_scratch);
             * min_scratch.to_host(0, 1); max_scratch.to_host(0, 1);
             * const double mn = min_scratch.host_ptr()[0];
             * const double mx = max_scratch.host_ptr()[0];
             * cuda_grayscale(z.device_ptr(), output.device_ptr(), nx*ny, mn, mx);
             */

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

    /*
     * cuda_min_max(z.device_ptr(), z.size, min_scratch, max_scratch);
     * min_scratch.to_host(0, 1); max_scratch.to_host(0, 1);
     * const double mn = min_scratch.host_ptr()[0];
     * const double mx = max_scratch.host_ptr()[0];
     * cuda_grayscale(z.device_ptr(), output.device_ptr(), nx*ny, mn, mx);
     */

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
