#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "png_util.h"
#include "CudaMem.h"
#include "cuda_props.h"
#include "cuda_kernel_interface.h"
#include "bench.h"
#include "debug.h"

//#define min(X,Y) ((X) < (Y) ? (X) : (Y))
//#define max(X,Y) ((X) > (Y) ? (X) : (Y))
#define IDX2D(a, i, stride, j) ((a)[(i)*(stride) + (j)])
#define CHKERR(test, message) \
    IF_DEBUG(if ((test)) fprintf(stderr, "%s:%d: ERROR: %s\n", __FILE__, __LINE__, (message));)

int main(int argc, char ** argv) {
    assert(WARP_SIZE == get_warpSize());

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
    double xy_max, xy_min;
    double tmax;
    double dx2inv, dy2inv;
    char filename[sizeof "./images/file00000.png"];

    image_size_t sz; 
    sz.width=nx;
    sz.height=ny;

    CudaMem<double> z(mesh_size);
    CudaMem<double> v(mesh_size);
    CudaMem<unsigned char> output(mesh_size);
//    unsigned char *output = new unsigned char[mesh_size];

    CudaMem<double> min_scratch(CUDA_MAX_MAX_BLOCKS);
    CudaMem<double> max_scratch(CUDA_MAX_MAX_BLOCKS);

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

    CudaKernelParams sim_params = sim_kernel_params(nx, ny);
    CudaKernelParams min_max_params = min_max_kernel_params(mesh_size);
    CudaKernelParams grayscale_params = grayscale_kernel_params(mesh_size);

    CUDA_CHKERR(z.to_device()); CUDA_CHKERR(v.to_device());

    STOP_CLOCK(setup);

    printf("nt=%d, dt=%g, frame_skip=%d, fps=%g\n", nt, dt, frame_skip, 1/(dt*frame_skip));

    START_CLOCK(simulation);
    dx2inv = 1.0/(dx*dx);
    dy2inv = 1.0/(dy*dy);

    for(it=0;it<nt-1;it++) {
        launch_sim_kernel_tiled(z.device_ptr(), v.device_ptr(), nx, ny, dx2inv, dy2inv, dt,
                                sim_params);
        IF_DEBUG(CUDA_CHKERR(cudaGetLastError());)
        IF_DEBUG(CUDA_CHKERR(cudaDeviceSynchronize());)

        if (it % frame_skip == 0) {
            launch_min_max_kernel(z.device_ptr(), z.size(),
                                  min_scratch.device_ptr(), max_scratch.device_ptr(),
                                  min_max_params);
            IF_DEBUG(CUDA_CHKERR(cudaGetLastError());)
            IF_DEBUG(CUDA_CHKERR(cudaDeviceSynchronize());)
            min_scratch.to_host(0, 1); max_scratch.to_host(0, 1);
            const double mn = min_scratch[0];
            const double mx = max_scratch[0];

            launch_grayscale_kernel(z.device_ptr(), output.device_ptr(), mesh_size, mn, mx,
                                    grayscale_params);
            IF_DEBUG(CUDA_CHKERR(cudaGetLastError());)
            IF_DEBUG(CUDA_CHKERR(cudaDeviceSynchronize());)
//            CUDA_CHKERR(z.to_host());
            CUDA_CHKERR(output.to_host());

//            for (size_t k=0; k < mesh_size; ++k)
//                output[k] = (char) round((z[k]-mn)/(mx-mn)*255);

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
    
    launch_min_max_kernel(z.device_ptr(), z.size(),
                          min_scratch.device_ptr(), max_scratch.device_ptr(),
                          min_max_params);
    IF_DEBUG(CUDA_CHKERR(cudaGetLastError());)
    IF_DEBUG(CUDA_CHKERR(cudaDeviceSynchronize());)
    min_scratch.to_host(0, 1); max_scratch.to_host(0, 1);
    const double mn = min_scratch[0];
    const double mx = max_scratch[0];

    launch_grayscale_kernel(z.device_ptr(), output.device_ptr(), mesh_size, mn, mx,
                            grayscale_params);
    IF_DEBUG(CUDA_CHKERR(cudaGetLastError());)
    IF_DEBUG(CUDA_CHKERR(cudaDeviceSynchronize());)
//    CUDA_CHKERR(z.to_host());
    CUDA_CHKERR(output.to_host());

//    for (size_t k = 0; k < mesh_size; ++k)
//        output[k] = (char) round((z[k]-mn)/(mx-mn)*255);

    printf("%f, %f\n", mn, mx);

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
