#ifndef _CUDA_KERNEL_INTERFACE_H
#define _CUDA_KERNEL_INTERFACE_H

// It has come to my attention that the correct name is "execution context". I don't really
// care right now, so I'm not changing it.
struct CudaKernelParams {
    dim3 grid_size;
    dim3 block_size;
    size_t shared_mem_size;

    CudaKernelParams(dim3 grid_size, dim3 block_size, size_t shared_mem_size)
        : grid_size(grid_size), block_size(block_size), shared_mem_size(shared_mem_size) {}
    CudaKernelParams(dim3 grid_size, dim3 block_size) :
        CudaKernelParams(grid_size, block_size, 0) {}
    CudaKernelParams(int grid_size, int block_size, size_t shared_mem_size)
        : CudaKernelParams(dim3(grid_size), dim3(block_size), shared_mem_size) {}
    CudaKernelParams(int grid_size, int block_size)
        : CudaKernelParams(grid_size, block_size, 0) {}
};

CudaKernelParams sim_kernel_shfl_params(size_t, size_t);
CudaKernelParams sim_kernel_params(size_t, size_t);
CudaKernelParams min_max_kernel_params(size_t);
CudaKernelParams grayscale_kernel_params(size_t);

void launch_sim_kernel_shfl(double *, double *, size_t, size_t, double, double, double, CudaKernelParams);
void launch_sim_kernel_tiled(double *, double *, size_t, size_t, double, double, double, CudaKernelParams);
void launch_min_max_kernel(double *, size_t, double *, double *, CudaKernelParams);
void launch_grayscale_kernel(double *, unsigned char *, size_t, double, double, CudaKernelParams);

#endif // _CUDA_KERNEL_INTERFACE_H defined
