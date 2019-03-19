#ifndef __PNG_UTIL__
#define __PNG_UTIL__

#ifdef __cplusplus
extern "C" {
#endif

#include <png.h>

typedef struct image_size_t {
  int width;
  int height;
} image_size_t;

image_size_t get_image_size(char*);

void read_png_file(char*, unsigned char*, image_size_t);

void write_png_file(char*, unsigned char*, image_size_t);

#ifdef __cplusplus
}
#endif


#endif
