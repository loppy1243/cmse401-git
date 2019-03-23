CFLAGS := $(CFLAGS) -std=c++11 -O3

.PHONY: all bench bench_serial bench_cuda images video gif test clean

all: serial cuda

bench: CFLAGS := $(CFLAGS) -DBENCH
bench: serial cuda

bench_serial: CFLAGS := $(CFLAGS) -DBENCH
bench_serial: serial

bench_cuda: CFLAGS := $(CFLAGS) -DBENCH
bench_cuda: cuda

wave_2d_serial.o: wave_2d_serial.c
	$(CC) $(CFLAGS) -c wave_2d_serial.c 

wave_2d_cuda.o: wave_2d_cuda.cu
	$(CC) $(CFLAGS) -c wave_2d_cuda.cu 

cuda_props.o: cuda_props.cu
	$(CC) $(CFLAGS) -c cuda_props.cu

png_util.o: png_util.c
	$(CC) $(CFLAGS) -l lpng16 -c png_util.c

serial: CC := gcc
serial: wave_2d_serial.o png_util.o
	mkdir -p images
	$(CC) $(CFLAGS) -o serial -lm -l png16 wave_2d_serial.o png_util.o

cuda: CC := nvcc
cuda: wave_2d_cuda.o png_util.o cuda_props.o
	mkdir -p images
	$(CC) $(CFLAGS) -o cuda -lm -l png16 wave_2d_cuda.o png_util.o cuda_props.o

images: serial
	./serial

video: images 
	ffmpeg -i ./images/file%05d.png -pix_fmt yuv420p wave2d.mp4
	ffmpeg -i ./images/cuda%05d.png -pix_fmt yuv420p cuda2d.mp4

gif: images
	convert -delay 20 -loop 0 ./images/file*.png wave2d.gif

test: video

clean:
	rm *.o
	rm serial cuda 
	rm wave2d.mp4 wave2d.gif
	rm -r ./images
