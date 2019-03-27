#ifndef _CUDA_KERNEL_INTERFACE_H
#define _CUDA_KERNEL_INTERFACE_H

struct CudaKernelParams {
    dim3 block_size;
    dim3 grid_size;
    size_t shared_mem_size;

    CudaKernelParams(dim3 block_size, dim3 grid_size, size_t shared_mem_size)
        : block_size(block_size), grid_size(grid_size), shared_mem_size(shared_mem_size) {}
    CudaKernelParams(dim3 block_size, dim3 grid_size) :
        CudaKernelParams(block_size, grid_size, 0) {}
    CudaKernelParams(int block_size, int grid_size, size_t shared_mem_size)
        : CudaKernelParams(dim3(block_size), dim3(grid_size), shared_mem_size) {}
    CudaKernelParams(int block_size, int grid_size)
        : CudaKernelParams(block_size, grid_size, 0) {}
};

//void cuda_sim(CudaMem<double>, CudaMem<double>, size_t, double, double, double);
void cuda_sim_shfl(double *, double *, size_t, size_t, double, double, double);
CudaKernelParams cuda_sim_tiled_params(size_t, size_t);
void cuda_sim_tiled(double *, double *, size_t, size_t, double, double, double, CudaKernelParams);
void cuda_min_max(double *, size_t, double *, double *);
void cuda_grayscale(double *, unsigned char *, size_t, double, double);

#endif // _CUDA_KERNEL_INTERFACE_H defined
