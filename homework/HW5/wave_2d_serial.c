#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "png_util.h"
#include "debug.h"
#include "bench.h"
#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))

#define IDX2D(a, i, stride, j) ((a)[(i)*(stride) + (j)])

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

    //make mesh
    double *z = (double *) malloc(mesh_size*sizeof(double));
    //Velocity
    double *v = (double *) malloc(mesh_size*sizeof(double));
    //Accelleration
    double *a = (double *) malloc(mesh_size*sizeof(double));
    //output image
    unsigned char *output = (unsigned char *) malloc(mesh_size*sizeof(unsigned char));

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
            IDX2D(z, r, nx, c) = exp(-(sqrt((x-5.0)*(x-5.0)+(y-5.0)*(y-5.0))));
            IDX2D(v, r, nx, c) = 0.0;
            IDX2D(a, r, nx, c) = 0.0;
        }
    }
    STOP_CLOCK(setup);

    IF_DEBUG(printf("nt=%d, dt=%g, frame_skip=%d, fps=%g\n", nt, dt, frame_skip, 1/(dt*frame_skip));)

    START_CLOCK(simulation);
    dx2inv = 1.0/(dx*dx);
    dy2inv = 1.0/(dy*dy);

    for(it=0;it<nt-1;it++) {
    //printf("%d\n",it);
        for (r=1;r<ny-1;r++)  
            for (c=1;c<nx-1;c++) {
                const double z_val =    IDX2D(z, r,   nx, c);
                const double z_x_high = IDX2D(z, r,   nx, c+1);
                const double z_x_low =  IDX2D(z, r,   nx, c-1);
                const double z_y_high = IDX2D(z, r+1, nx, c);
                const double z_y_low =  IDX2D(z, r-1, nx, c);
                const double ax = (z_x_high+z_x_low-2.0*z_val)*dx2inv;
                const double ay = (z_y_high+z_y_low-2.0*z_val)*dy2inv;
                IDX2D(a, r, nx, c) = (ax+ay)/2;
            }
        for (r=1; r<ny-1; r++)  
            for (c=1;c<nx-1;c++) {
                IDX2D(v, r, nx, c) = IDX2D(v, r, nx, c) + dt*IDX2D(a, r, nx, c);
                IDX2D(z, r, nx, c) = IDX2D(z, r, nx, c) + dt*IDX2D(v, r, nx, c);
            }

        if (it % frame_skip == 0) {
            double mx,mn;
            mx = -999999;
            mn = 999999;
            for (size_t k = 0; k < mesh_size; ++k) {
                mx = max(mx, z[k]);
                mn = min(mn, z[k]);
            }
            for (size_t k=0; k < mesh_size; ++k)
                output[k] = (unsigned char) round((z[k]-mn)/(mx-mn)*255);

            STOP_CLOCK(simulation);
            START_CLOCK(file_io);

            sprintf(filename, "./images/file%05d.png", frame);
            IF_DEBUG(printf("Writing %s\n",filename);)
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
        output[k] = (unsigned char) round((z[k]-mn)/(mx-mn)*255);
    STOP_CLOCK(simulation);

    START_CLOCK(file_io);
    sprintf(filename, "./images/file%05d.png", it);
    IF_DEBUG(printf("Writing %s\n",filename);)
    //Write out output image using 1D serial pointer
    write_png_file(filename,output,sz);
    STOP_CLOCK(file_io); STOP_CLOCK(total);

#ifdef BENCH
    fputs("BENCHMARKING\nTOTAL setup file_io simulation\n", stderr);
    fprintf(stderr, "%.3e %.3e %.3e %.3e\n", total_time, setup_time, file_io_time, simulation_time);
#endif

    return 0;
}
