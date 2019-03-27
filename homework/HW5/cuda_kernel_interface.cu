/*******************************************************************************
 * Many of the functions in this file could potentially be inline.
 ******************************************************************************/

#include "cuda_kernel_interface.h"
#include "cuda_kernels.h"
#include "cuda_props.h"
#include "debug.h"

// This doesn't work for some reason. WTF???
//void cuda_sim(CudaMem<double> z, CudaMem<double> v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
//    const int warp_size = get_warpSize();
//    const dim3 blocksize(warp_size, warp_size);
//    const dim3 gridsize((nx-1)/warp_size+1, (ny-1)/warp_size+1);
//
//    sim_kernel_tiled<<<gridsize, blocksize>>>(z.device_ptr(), v.device_ptr(), nx, ny, dx2inv, dy2inv, dt);
//}

CudaKernelParams sim_kernel_shfl_params(size_t nx, size_t ny) {
    const int warp_size = get_warpSize();
    const dim3 blocksize(warp_size, warp_size);
    const dim3 gridsize((nx-1)/warp_size+1, (ny-1)/warp_size+1);

    return CudaKernelParams(gridsize, blocksize);
}
CudaKernelParams sim_kernel_params(size_t nx, size_t ny) {
    const size_t warp_size = get_warpSize();
    const size_t shmem_max = get_sharedMemPerBlock();

    const int largest_dim = max(((int(nx)-1)/warp_size+1)*warp_size,
                                ((int(ny)-1)/warp_size+1)*warp_size);
    const int smallest_dim_max = min(get_maxThreadsDim(0), get_maxThreadsDim(1));
    // Without the -1 CUDA still claims an invalid configuration.
    const int shmem_max_doubles_sqrt = (int(sqrt(shmem_max/sizeof(double)))/warp_size-1)*warp_size;

    int blocksize_1D = min(shmem_max_doubles_sqrt, min(largest_dim, smallest_dim_max));

    dim3 blocksize(blocksize_1D, blocksize_1D);
    dim3 gridsize((nx-1)/blocksize_1D+1, (ny-1)/blocksize_1D+1);
    size_t shmem_size = blocksize_1D*blocksize_1D*sizeof(double);

    return CudaKernelParams(gridsize, blocksize, shmem_size);
}
void launch_sim_kernel_tiled(double *z, double *v, size_t nx, size_t ny,
                             double dx2inv, double dy2inv, double dt,
                             CudaKernelParams kp)
{
    sim_kernel_tiled<<<kp.grid_size, kp.block_size, kp.shared_mem_size>>>
                    (z, v, nx, ny, dx2inv, dy2inv, dt);
}

// Compute the min and max of `z` in `scratch_m[0]` and `scratch_M[0]`, respectively.
// `scratch_m` and `scratch_M` must have size at least gridsize.
CudaKernelParams min_max_kernel_params(size_t size) {
    const int warpsize = get_warpSize();

    const int blocksize = min(int(size - size%warpsize), get_maxThreadsPerBlock());
    const int gridsize = min(int(size/blocksize), CUDA_MAX_MAX_BLOCKS);

    return CudaKernelParams(gridsize, blocksize);
}
void launch_min_max_kernel(double *z, size_t size, double *scratch_m, double *scratch_M,
                           CudaKernelParams kp)
{
    const int warp_size = get_warpSize();
    const int grid_size2 = kp.grid_size.x - kp.grid_size.x%warp_size;

    block_min_max_kernel<<<kp.grid_size, kp.block_size>>>(z, z, size, scratch_m, scratch_M);
    block_min_max_kernel<<<1, grid_size2>>>(scratch_m, scratch_M, kp.grid_size.x,
                                            scratch_m, scratch_M);
}

CudaKernelParams grayscale_kernel_params(size_t size) {
    const int blocksize = min(int(size), get_maxThreadsPerBlock());
    const int gridsize = min(int(size/blocksize), CUDA_GRAYSCALE_MAX_BLOCKS);

    return CudaKernelParams(gridsize, blocksize);
}
void launch_grayscale_kernel(double *z, unsigned char *output, size_t size,
                             double z_min, double z_max, CudaKernelParams kp)
{
    grayscale_kernel<<<kp.grid_size, kp.block_size>>>(z, output, size, z_min, z_max);
}
