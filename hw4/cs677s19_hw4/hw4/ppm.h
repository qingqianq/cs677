#ifndef _PPM_H_
#define _PPM_H_
__global__ void ppmKernel2(unsigned int *input, unsigned int *output, int width, int height, int thresh);
__global__ void ppmKernel(unsigned int *input, unsigned int *output, int width, int height, int thresh);
#endif
