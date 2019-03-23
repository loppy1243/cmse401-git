#include "cuda_props.h"
#include "cuda_util.h"

cudaDeviceProp get_deviceProps() {
    static cudaDeviceProp props;
    static bool gotten = false;

    if (!gotten) CUDA_CHKERR(cudaGetDeviceProperties(&props, 0));

    return props;
}

// For use on host
int get_warpSize() {
    return get_deviceProps().warpSize;
}

int get_maxThreadsPerBlock() {
    return get_deviceProps().maxThreadsPerBlock;
}
