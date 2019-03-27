#include "cuda_kernel_interface.h"
#include "cuda_kernels.h"
#include "cuda_props.h"
#include "debug.h"

#ifdef DEBUG
#include <iostream>
#endif

// This doesn't work for some reason. WTF???
//void cuda_sim(CudaMem<double> z, CudaMem<double> v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
//    const int warp_size = get_warpSize();
//    const dim3 blocksize(warp_size, warp_size);
//    const dim3 gridsize((nx-1)/warp_size+1, (ny-1)/warp_size+1);
//
//    sim_kernel_tiled<<<gridsize, blocksize>>>(z.device_ptr(), v.device_ptr(), nx, ny, dx2inv, dy2inv, dt);
//}
void cuda_sim_shfl(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
    const int warp_size = get_warpSize();
    const dim3 blocksize(warp_size, warp_size);
    const dim3 gridsize((nx-1)/warp_size+1, (ny-1)/warp_size+1);

    sim_kernel_shfl<<<gridsize, blocksize>>>(z, v, nx, ny, dx2inv, dy2inv, dt);
}

CudaKernelParams cuda_sim_tiled_params(size_t nx, size_t ny) {
    const size_t warp_size = get_warpSize();
    const size_t shmem_max = get_sharedMemPerBlock();

    const int largest_dim = max(((int(nx)-1)/warp_size+1)*warp_size,
                                ((int(ny)-1)/warp_size+1)*warp_size);
    const int smallest_dim_max = min(get_maxThreadsDim(0), get_maxThreadsDim(1));
    const int shmem_max_doubles_sqrt = (int(sqrt(shmem_max/sizeof(double)))/warp_size-1)*warp_size;

    int blocksize_1D = min(shmem_max_doubles_sqrt, min(largest_dim, smallest_dim_max));

    dim3 blocksize(blocksize_1D, blocksize_1D);
    dim3 gridsize((nx-1)/blocksize_1D+1, (ny-1)/blocksize_1D+1);
    size_t shmem_size = blocksize_1D*blocksize_1D*sizeof(double);

    return CudaKernelParams(blocksize, gridsize, shmem_size);
}

void cuda_sim_tiled(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv,
                    double dt, CudaKernelParams kp)
{
    sim_kernel_tiled<<<kp.grid_size, kp.block_size, kp.shared_mem_size>>>
                    (z, v, nx, ny, dx2inv, dy2inv, dt);
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

#define CUDA_GRAYSCALE_MAX_BLOCKS 1024
// `z` and `output` can alias.
void cuda_grayscale(double *z, unsigned char *output, size_t size, double z_min, double z_max) {
    const int blocksize = min(int(size), get_maxThreadsPerBlock());
    const int gridsize = min(int(size/blocksize), CUDA_GRAYSCALE_MAX_BLOCKS);

    grayscale_kernel<<<gridsize, blocksize>>>(z, output, size, z_min, z_max);
}
