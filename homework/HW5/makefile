.PHONY: all hpcc_modules images video gif test clean

include config.mk

ifneq ($(origin HPCC), undefined)
	HPCC := 0
endif
ifneq ($(HPCC), 0)
	HPCC := 1
	HPCC_RULE := hpcc_modules
endif

all: serial cuda

ifeq ($(HPCC), 1)
hpcc_modules:
	@module purge 
	@module load GNU/7.3.0 libpng/1.6.34
endif

wave_2d_serial.o: wave_2d_serial.c
	gcc -c wave_2d_serial.c 

wave_2d_cuda.o: wave_2d_cuda.cu
	nvcc -c wave_2d_cuda.cu 

png_util.o: png_util.c
	gcc -l lpng16 -c png_util.c

serial: wave_2d_serial.o png_util.o $(HPCC_RULE)
	mkdir -p images
	gcc -o serial -lm -l png16 wave_2d_serial.o png_util.o

cuda: wave_2d_cuda.o png_util.o $(HPCC_RULE)
	mkdir -p images
	nvcc -o cuda -lm -l png16 wave_2d_cuda.o png_util.o

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
