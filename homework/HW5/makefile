CFLAGS := $(CFLAGS) -std=c++11 -O3

ifeq ($(origin BENCH), undefined)
BENCH := 0
endif
ifeq ($(origin DEBUG), undefined)
DEBUG := 0
endif

ifneq ($(BENCH), 0)
CFLAGS := $(CFLAGS) -DBENCH
endif
ifneq ($(DEBUG), 0)
CFLAGS := $(CFLAGS) -DDEBUG
endif

.PHONY: all bench bench_serial bench_cuda images video gif test clean

all: serial cuda

%.o : %.c
	$(CC) $(CFLAGS) -c $<
%.o : %.cu
	$(CC) $(CFLAGS) -c $<

serial: CC := gcc
serial: wave_2d_serial.o png_util.o
	mkdir -p images
	$(CC) $(CFLAGS) -o serial -lm -lpng16 $^

cuda: CC := nvcc
cuda: wave_2d_cuda.o png_util.o cuda_props.o cuda_kernels.o cuda_kernel_interface.o
	mkdir -p images
	$(CC) $(CFLAGS) -o cuda -lm -lpng16 $^

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
