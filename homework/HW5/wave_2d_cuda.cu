#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "png_util.h"
#include "cuda_util.h"
#include "CudaMem.h"
#include "cuda_props.h"
#include "bench.h"

//#define min(X,Y) ((X) < (Y) ? (X) : (Y))
//#define max(X,Y) ((X) > (Y) ? (X) : (Y))
#define IDX2D(a, i, stride, j) ((a)[(i)*(stride) + (j)])
#define CHKERR(test, message) \
    if ((test)) fprintf(stderr, "%s:%d: ERROR: %s\n", __FILE__, __LINE__, (message));

__device__ inline double warp_accel(double z, double d_2inv, int shfl_mask) {
    return d_2inv*(__shfl_down_sync(shfl_mask, z, 1) + __shfl_up_sync(shfl_mask, z, 1) - 2.0*z);
}

/************************************
 * I can't get this version working.
 ***********************************/
//// The block size MUST be warpSize by warpSize.
//__global__ void sim_kernel(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
//    // Add one so we can have bank-parallelism along rows or along columns. Should benchmark
//    // this.
//    __shared__ double z_tile[WARP_SIZE][WARP_SIZE/*+1*/];
//    __shared__ double ax_tile[WARP_SIZE][WARP_SIZE/*+1*/];
//
//    const int block_mesh_x = blockIdx.x*warpSize;
//    const int block_mesh_y = blockIdx.y*warpSize;
//    const int mesh_x = block_mesh_x + threadIdx.x;
//    const int mesh_y = block_mesh_y + threadIdx.y;
//    const int t_mesh_x = mesh_y - block_mesh_y + block_mesh_x;
//    const int t_mesh_y = mesh_x - block_mesh_x + block_mesh_y;
//
//    if (mesh_x <= 0 || mesh_x >= nx-1 || mesh_y <= 0 || mesh_y >= ny-1) return;
//
//    const double z_val_y = z_tile[threadIdx.y][threadIdx.x] = IDX2D(z, mesh_y, nx, mesh_x);
//
//    __syncthreads();
//
//    const double z_val_x = z_tile[threadIdx.x][threadIdx.y];
//
//    double ay;
//    if (0 < threadIdx.x && threadIdx.x < warpSize-1) {
//        const int shfl_mask = 0x7 << threadIdx.x-1;
//        ax_tile[threadIdx.y][threadIdx.x] = warp_accel(z_val_x, dx2inv, shfl_mask);
//        ay = warp_accel(z_val_y, dy2inv, shfl_mask);
//    }
//    else {
//        const int n = (threadIdx.x+1)/warpSize - (warpSize - threadIdx.x)/warpSize;
//        ax_tile[threadIdx.y][threadIdx.x] =
//            dx2inv*(z_tile[threadIdx.x-n][threadIdx.y] + IDX2D(z, t_mesh_y+n, nx, t_mesh_x)
//                    - 2.0*z_val_x);
//        ay = dy2inv*(z_tile[threadIdx.y][threadIdx.x-n] + IDX2D(z, mesh_y, nx, mesh_x+n)
//                     - 2.0*z_val_y);
//    }
//
//    __syncthreads();
//
//    const double a_val = (ax_tile[threadIdx.x][threadIdx.y] + ay)/2.0;
//    const double v_val = IDX2D(v, mesh_y, nx, mesh_x) += dt*a_val;
//    IDX2D(z, mesh_y, nx, mesh_x) = z_val_y + dt*v_val;
//}

__global__ void sim_kernel_naive(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
    const int mesh_x = blockIdx.x*warpSize + threadIdx.x + 1;
    const int mesh_y = blockIdx.y*warpSize + threadIdx.y + 1;

    if (mesh_x >= nx-1 || mesh_y >= ny-1) return;

    const double z_val = IDX2D(z, mesh_y, nx, mesh_x);
    const double ax = dx2inv*(IDX2D(z, mesh_y, nx, mesh_x-1) + IDX2D(z, mesh_y, nx, mesh_x+1)
                              - 2.0*z_val);
    const double ay = dy2inv*(IDX2D(z, mesh_y-1, nx, mesh_x) + IDX2D(z, mesh_y+1, nx, mesh_x)
                              - 2.0*z_val);

    double const v_val = IDX2D(v, mesh_y, nx, mesh_x) += dt*(ax + ay)/2;
    IDX2D(z, mesh_y, nx, mesh_x) += dt*v_val;
}

// This doesn't work for some reason. WTF???
//void cuda_sim(CudaMem<double> z, CudaMem<double> v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
//    const int warpsize = get_warpSize();
//    const dim3 blocksize(warpsize, warpsize);
//    const dim3 gridsize((nx-1)/warpsize+1, (ny-1)/warpsize+1);
//
//    sim_kernel_naive<<<gridsize, blocksize>>>(z.device_ptr(), v.device_ptr(), nx, ny, dx2inv, dy2inv, dt);
//}
void cuda_sim(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
    const int warpsize = get_warpSize();
    const dim3 blocksize(warpsize, warpsize);
    const dim3 gridsize((nx-1)/warpsize+1, (ny-1)/warpsize+1);

    sim_kernel_naive<<<gridsize, blocksize>>>(z, v, nx, ny, dx2inv, dy2inv, dt);
}

// Return value is defined only for the first thread of each warp.
// `blockDim.x` must be a multiple of `warpSize`.
__device__ void warp_min_max(double *m_ptr, double *M_ptr) {
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
__device__ void block_min_max(double *m_ptr, double *M_ptr) {
    __shared__ double warp_share_m[WARP_SIZE];
    __shared__ double warp_share_M[WARP_SIZE];

    const int warp_lane = threadIdx.x % warpSize;
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
    if (warp_idx == 0) warp_min_max(m_ptr, M_ptr);
}

// Reduce `in_m`/`in_M` to `gridDim.x` candidates in `out_m`/`out_M` for its minimim/maximum.
// Assumes
//     1D grid and blocks.
//     `gridDim.x = k*warpSize` for `1 <= k <= warpSize`.
//     `in_m` and `in_M` must have size `>= gridDim.x*blockDim.x`.
//     `out_m` and `out_M` must have size `>= gridDim.x`.
__global__ void block_min_max_kernel(double *in_m, double *in_M, size_t size, double *out_m, double *out_M) {
    const double grid_size = blockDim.x*gridDim.x;
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;

    double m = in_m[0]; double M = in_M[0];
    for (int i = idx; i < size; i += grid_size) {
        const double other_m = in_m[i], other_M = in_M[i];
        if (other_m < m) m = other_m;
        if (other_M > M) M = other_M;
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

    const int blocksize = min(int(size - size%warpsize), get_maxThreadsPerBlock());
    const int gridsize = min(int(size/blocksize), CUDA_MAX_MAX_BLOCKS);
    const int gridsize2 = gridsize - gridsize%warpsize;

    block_min_max_kernel<<<gridsize, blocksize>>>(z, z, size, scratch_m, scratch_M);
    block_min_max_kernel<<<1, gridsize2>>>(scratch_m, scratch_M, gridsize, scratch_m, scratch_M);
}

__global__ void grayscale_kernel(double *z, unsigned char *output, size_t size, double z_min, double z_max) {
    const double grid_size = blockDim.x*gridDim.x;
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;

    for (int i = idx; i < size; i += grid_size)
        output[i] = (char) round((z[i]-z_min)/(z_max-z_min)*255);
}

#define CUDA_GRAYSCALE_MAX_BLOCKS 1024
// `z` and `output` can alias.
void cuda_grayscale(double *z, unsigned char *output, size_t size, double z_min, double z_max) {
    const int blocksize = min(int(size), get_maxThreadsPerBlock());
    const int gridsize = min(int(size/blocksize), CUDA_GRAYSCALE_MAX_BLOCKS);

    grayscale_kernel<<<gridsize, blocksize>>>(z, output, size, z_min, z_max);
}

int main(int argc, char ** argv) {
    assert(WARP_SIZE == get_warpSize());

    INIT_CLOCK(setup); INIT_CLOCK(simulation); INIT_CLOCK(file_io); INIT_CLOCK(total);

    START_CLOCK(total); START_CLOCK(setup);
    const int nx = 500;
    const int ny = 500;
    const int mesh_size = nx*ny;
    const int nt = 10000; 
    //int nt = 1000000
    //                                ;
    int frame = 0;
    // fps = 1/(dt*frame_skip)
    int frame_skip = 100;
    int r,c,it;
    double dx,dy,dt;
    double xy_max, xy_min;
    double tmax;
    double dx2inv, dy2inv;
    char filename[sizeof "./images/file00000.png"];

    image_size_t sz; 
    sz.width=nx;
    sz.height=ny;

    CudaMem<double> z(nx*ny); CUDA_CHKERR(z.init_device());
    CudaMem<double> v(nx*ny); CUDA_CHKERR(v.init_device());
    CudaMem<unsigned char> output(nx*ny); CUDA_CHKERR(output.init_device());

    xy_max=10.0;
    xy_min=0.0;
    dx = (xy_max-xy_min)/(double)(nx-1);
    dy = (xy_max-xy_min)/(double)(ny-1);
    
    tmax=20.0;
    dt = (tmax-0.0)/(double)(nt-1);

    double x,y; 
    for (r=0;r<ny;r++) {
        for (c=0;c<nx;c++) {
            x = xy_min+(double)c*dx;
            y = xy_min+(double)r*dy;
            IDX2D(z.host_ptr(), r, nx, c) = exp(-(sqrt((x-5.0)*(x-5.0)+(y-5.0)*(y-5.0))));
            IDX2D(v.host_ptr(), r, nx, c) = 0.0;
        }
    }

    CUDA_CHKERR(z.to_device()); CUDA_CHKERR(v.to_device());

    STOP_CLOCK(setup);

    printf("nt=%d, dt=%g, frame_skip=%d, fps=%g\n", nt, dt, frame_skip, 1/(dt*frame_skip));

    START_CLOCK(simulation);
    dx2inv = 1.0/(dx*dx);
    dy2inv = 1.0/(dy*dy);

    for(it=0;it<nt-1;it++) {
        cuda_sim(z.device_ptr(), v.device_ptr(), nx, ny, dx2inv, dy2inv, dt);
        CUDA_CHKERR(cudaPeekAtLastError());
//        CUDA_CHKERR(cudaDeviceSynchronize());
        CUDA_CHKERR(z.to_host());

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
             * const double mn = min_scratch.[0];
             * const double mx = max_scratch.[0];
             * cuda_grayscale(z.device_ptr(), output.device_ptr(), mesh_size mn, mx);
             */

            STOP_CLOCK(simulation);
            START_CLOCK(file_io);

            sprintf(filename, "./images/file%05d.png", frame);
            printf("Writing %s\n",filename);    
            write_png_file(filename,output.host_ptr(),sz);

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
     * cuda_grayscale(z.device_ptr(), output.device_ptr(), mesh_size, mn, mx);
     */

    STOP_CLOCK(simulation);

    START_CLOCK(file_io);
    sprintf(filename, "./images/file%05d.png", it);
    printf("Writing %s\n",filename);    
    //Write out output image using 1D serial pointer
    write_png_file(filename,output.host_ptr(),sz);
    STOP_CLOCK(file_io); STOP_CLOCK(total);

#ifdef BENCH
    fputs("BENCHMARKING\nTOTAL setup file_io simulation\n", stderr);
    fprintf(stderr, "%.3e %.3e %.3e %.3e\n", total_time, setup_time, file_io_time, simulation_time);
#endif

    return 0;
}
