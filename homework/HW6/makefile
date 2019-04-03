all : mill 

mill: rumor_mill.c
	mkdir -p images
	mpicc -O3 -l png16 -o mill rumor_mill.c 

gif: images
	convert -delay 20 -loop 0 ./images/file*.png rumors.gif

clean:
	rm -rf images
	rm mill 
	rm rumors.gif
