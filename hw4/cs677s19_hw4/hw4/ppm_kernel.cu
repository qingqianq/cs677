#ifndef _PPM_KERNEL_H_
#define _PPM_KERNEL_H_
#include <stdio.h>
#include <math.h>
#include "ppm.h"
#define TILESIZE 16

__global__ void ppmKernel2(unsigned int *input, unsigned int *output, int width, int height, int thresh){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int index = col + width * row;
    if(col < width && row < height){
        if(col > 0 && row > 0 && col < width - 1 && row < height - 1){
            int sum1 = input[(row-1)*width+col+1] - input[(row-1) * width + col-1] + 2*input[row*width+col+1] - 2*input[row*width+col-1] + input[(row+1)*width+col+1 ]- input[(row+1)*width+col-1];
            int sum2 = input[(row-1)*width+col-1] + 2*input[(row-1)*width+col] + input[(row-1)*width+col+1] - input[(row+1)*width+col-1] - 2*input[(row+1)*width+col] - input[(row+1)*width+col+1];
            int magnitude = sum1 * sum1 + sum2 * sum2;
            if(magnitude > thresh)
                output[index] = 255;
            else
                output[index] = 0;
        }else
            output[index] = 0;
    }else
        output[index] = 0;
}
__global__ void ppmKernel(unsigned int *input, unsigned int *output, int width, int height, int thresh){
    int idx = threadIdx.x + 1;
    int idy = threadIdx.y + 1;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int bx = blockIdx.x * TILESIZE;
    int by = blockIdx.y * TILESIZE;
    int index = col + width * row;
    /* __shared__ int s_data[TILESIZE + 1][TILESIZE + 1]; */
    /* s_data[tx][ty] = input[idx]; */
    __shared__ int s_data[TILESIZE + 2][TILESIZE + 2];
    if(threadIdx.y < 2){
        s_data[threadIdx.x][threadIdx.y * (TILESIZE + 1)] = input[(by - 1) + threadIdx.y * (TILESIZE + 1) * width + col - 1];
        s_data[threadIdx.y * (TILESIZE + 1)][threadIdx.x + 1] = input[(by + threadIdx.x) * width + (bx - 1) + threadIdx.y * (TILESIZE + 1)];
    }
    /*
      need 68 points
      4 points may missing
     */
    s_data[threadIdx.x + 1][threadIdx.y + 1] = input[index];
    __syncthreads();
    if(col < width && row < height){
        if(col < width - 1 && row < height - 1 && col > 0 && row > 0){
            int sum1 = s_data[idx - 1][idy + 1] - s_data[idx - 1][idy - 1] + 2 * s_data[idx][idy+1] - 2 * s_data[idx][idy - 1] + s_data[idx + 1][idy + 1] - s_data[idx + 1][idy - 1];
            int sum2 = s_data[idx - 1][idy - 1] + s_data[idx - 1][idy] + s_data[idx - 1][idy + 1] - s_data[idx + 1][idy - 1] - 2 * s_data[idx + 1][idy] - s_data[idx + 1][idy + 1];
            int magnitude = sum1 * sum1 + sum2 * sum2;
            if(magnitude > thresh)
                output[index] = 255;
            else
                output[index] = 0;
        }else
            output[index] = 0;
    }
}
#endif
