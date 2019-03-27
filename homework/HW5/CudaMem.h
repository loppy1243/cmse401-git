#ifndef _CUDA_MEM_H
#define _CUDA_MEM_H

#include "debug.h"

template<class T> class CudaMem {
    T *_host_ptr;
    T *_device_ptr;
    size_t _size;

public:
    CudaMem(size_t size)
        : _host_ptr(new T[size]), _device_ptr(nullptr), _size(size)
    {
        CUDA_CHKERR(cudaMalloc((void **) &this->_device_ptr, size*sizeof(T)));

        // Hey, yeah, so, `cudaMalloc` can be asynchronous. This causes problems because
        // during construction `this` can point to an intermediate version of the object being
        // construct; `cudaMalloc` attempts to writes the device pointer to this intermediary
        // AFTER THIS CONSTRUCTOR HAS ENDED and it never gets copied over to the object
        // proper.
        // 
        // This line of code is the crystallization of several hours of my life.
        CUDA_CHKERR(cudaDeviceSynchronize());
    }
    ~CudaMem() {
        delete[] this->_host_ptr;
        CUDA_CHKERR(cudaFree(this->_device_ptr));
    }

    T *host_ptr() { return _host_ptr; }
    T *device_ptr() { return _device_ptr; }
    size_t size() { return this->_size; }

    T &operator[](size_t i) { return this->_host_ptr[i]; }
    const T &operator[](size_t i) const { return this->_host_ptr[i]; }

    cudaError_t to_device(size_t off, size_t n) {
        return cudaMemcpy(_device_ptr+off, _host_ptr+off, n*sizeof(T),
                          cudaMemcpyHostToDevice);
    }
    cudaError_t to_device() { return this->to_device(0, this->_size); }

    cudaError_t to_host(size_t off, size_t n) {
        return cudaMemcpy(this->_host_ptr+off, this->_device_ptr+off, n*sizeof(T),
                          cudaMemcpyDeviceToHost);
    }
    cudaError_t to_host() { return this->to_host(0, this->_size); }
};

#endif // CUDA_MEM_H defined
