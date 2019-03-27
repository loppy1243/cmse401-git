#include "cuda_props.h"
#include "cuda_kernels.h"

#define IDX2D(a, i, stride, j) ((a)[(i)*(stride) + (j)])

__device__ inline double warp_accel_shfl(double z, double d_2inv, int shfl_mask) {
    return d_2inv*(__shfl_down_sync(shfl_mask, z, 1) + __shfl_up_sync(shfl_mask, z, 1) - 2.0*z);
}

// THIS DOES NOT WORK.
// The block size MUST be warpSize by warpSize.
//#define SIM_BLKSIZE WARP_SIZE
__global__ void sim_kernel_shfl(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
    __shared__ double z_tile[WARP_SIZE][WARP_SIZE];
    __shared__ double ay_tile[WARP_SIZE][WARP_SIZE];

    const int block_mesh_x = warpSize*blockIdx.x + 1;
    const int block_mesh_y = warpSize*blockIdx.y + 1;

    const int mesh_xx = block_mesh_x + threadIdx.x;
    const int mesh_xy = block_mesh_y + threadIdx.y;
    const int mesh_yx = block_mesh_x + threadIdx.y;
    const int mesh_yy = block_mesh_y + threadIdx.x;

    const double z_val_x = z_tile[threadIdx.y][threadIdx.x] = IDX2D(z, mesh_xy, nx, mesh_xx);

    if (mesh_xx >= nx-1 || mesh_xy >= ny-1 /*|| mesh_yx > nx-1 || mesh_yy >= ny-1*/)
        return;

    __syncthreads();

    const double z_val_y = z_tile[threadIdx.x][threadIdx.y];

//    const int shfl_mask = 0x7 << (threadIdx.x - 1);
    const int shfl_mask = 0x7 << (threadIdx.x - 1);

    double ax = warp_accel_shfl(z_val_x, dx2inv, shfl_mask);
    double ay = warp_accel_shfl(z_val_y, dy2inv, shfl_mask);
    if (threadIdx.x == 0 || threadIdx.x == warpSize-1) {
        const int n = threadIdx.x == 0 ? -1 : +1;
        ax = dx2inv*(IDX2D(z, mesh_xy, nx, mesh_xx+n) + z_tile[threadIdx.y][threadIdx.x-n]
                     - 2.0*z_val_x);
        ay = dy2inv*(IDX2D(z, mesh_yy+n, nx, mesh_yx) + z_tile[threadIdx.x-n][threadIdx.y]
                     - 2.0*z_val_y);
    }

    ay_tile[threadIdx.x][threadIdx.x] = ay;
    __syncthreads();
    ay = ay_tile[threadIdx.y][threadIdx.x];

    const double v_val = (IDX2D(v, mesh_xy, nx, mesh_xx) += (ax+ay)/2.0*dt);
    IDX2D(z, mesh_xy, nx, mesh_xx) += dt*v_val;
}

__global__ void sim_kernel_tiled(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
    extern __shared__ double z_tile[];

    const int block_mesh_x = blockDim.x*blockIdx.x + 1;
    const int block_mesh_y = blockDim.y*blockIdx.y + 1;

    const int mesh_xx = block_mesh_x + threadIdx.x;
    const int mesh_xy = block_mesh_y + threadIdx.y;

    // We have to read into the tile BEFORE dropping threads so that it's actually fully
    // initialized!
    const double z_val = IDX2D(z_tile, threadIdx.y, blockDim.x, threadIdx.x)
                       = IDX2D(z, mesh_xy, nx, mesh_xx);

    if (mesh_xx >= nx-1 || mesh_xy >= ny-1)
        return;

    __syncthreads();

    double ax, ay;
    if (1 <= threadIdx.x && threadIdx.x <= blockDim.x-2)
        ax = dx2inv*(IDX2D(z_tile, threadIdx.y, blockDim.x, threadIdx.x-1)
                     + IDX2D(z_tile, threadIdx.y, blockDim.x, threadIdx.x+1)
                     - 2.0*z_val);
    else {
        const int n = threadIdx.x == 0 ? -1 : +1;
        ax = dx2inv*(IDX2D(z, mesh_xy, nx, mesh_xx+n)
                     + IDX2D(z_tile, threadIdx.y, blockDim.x, threadIdx.x-n)
                     - 2.0*z_val);
    }

    if (1 <= threadIdx.y && threadIdx.y <= blockDim.y-2)
        ay = dy2inv*(IDX2D(z_tile, threadIdx.y-1, blockDim.x, threadIdx.x)
                     + IDX2D(z_tile, threadIdx.y+1, blockDim.x, threadIdx.x)
                     - 2.0*z_val);
    else {
        const int n = threadIdx.y == 0 ? -1 : +1;
        ay = dx2inv*(IDX2D(z, mesh_xy+n, nx, mesh_xx)
                     + IDX2D(z_tile, threadIdx.y-n, blockDim.x, threadIdx.x)
                     - 2.0*z_val);
    }

    const double v_val = IDX2D(v, mesh_xy, nx, mesh_xx) += (ax+ay)/2.0*dt;
    IDX2D(z, mesh_xy, nx, mesh_xx) += dt*v_val;
}

__global__ void sim_kernel_naive(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
    const int mesh_x = blockIdx.x*blockDim.x + threadIdx.x + 1;
    const int mesh_y = blockIdx.y*blockDim.y + threadIdx.y + 1;

    if (mesh_x >= nx-1 || mesh_y >= ny-1) return;

    const double z_val = IDX2D(z, mesh_y, nx, mesh_x);
    const double ax = dx2inv*(IDX2D(z, mesh_y, nx, mesh_x-1) + IDX2D(z, mesh_y, nx, mesh_x+1)
                              - 2.0*z_val);
    const double ay = dy2inv*(IDX2D(z, mesh_y-1, nx, mesh_x) + IDX2D(z, mesh_y+1, nx, mesh_x)
                              - 2.0*z_val);

    double const v_val = IDX2D(v, mesh_y, nx, mesh_x) + dt*(ax + ay)/2;
    IDX2D(v, mesh_y, nx, mesh_x) = v_val;
    IDX2D(z, mesh_y, nx, mesh_x) += dt*v_val;
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

__global__ void grayscale_kernel(double *z, unsigned char *output, size_t size, double z_min, double z_max) {
    const double grid_size = blockDim.x*gridDim.x;
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;

    for (int i = idx; i < size; i += grid_size)
        output[i] = (char) round((z[i]-z_min)/(z_max-z_min)*255);
}
