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
    /* int idx = threadIdx.x + 1; */
    /* int idy = threadIdx.y + 1; */
    /* int col = threadIdx.x + TILESIZE * blockIdx.x; */
    /* int row = threadIdx.y + TILESIZE * blockIdx.y; */
    /* int index = col + width * row; */
    /* __shared__ int s_data[TILESIZE + 2][TILESIZE + 2]; */
    /* if(threadIdx.y < 2 && blockIdx.x > 0 && blockIdx.y > 0){ */
    /*     /\* s_data[threadIdx.x][threadIdx.y*(TILESIZE+1)] = (blockIdx.y > 0)?input[((blockIdx.y*TILESIZE - 1)+(TILESIZE+1)*threadIdx.y)*width + col - 1]:0; *\/ */
    /*     s_data[threadIdx.x][threadIdx.y*(TILESIZE+1)] = input[((blockIdx.y*TILESIZE - 1)+(TILESIZE+1)*threadIdx.y)*width + col - 1]; */
    /*     /\* s_data[threadIdx.y * (TILESIZE + 1)][idx] = (blockIdx.x>0)?input[(blockIdx.x*TILESIZE-1)+threadIdx.y*(TILESIZE+1)+ width*(blockIdx.y * TILESIZE + threadIdx.x)]:0; *\/ */
    /*     s_data[threadIdx.y * (TILESIZE + 1)][idx] = input[(blockIdx.x*TILESIZE-1)+threadIdx.y*(TILESIZE+1)+ width*(blockIdx.y * TILESIZE + threadIdx.x)]; */
    /*     if(threadIdx.x < 2 ){ */
    /*         s_data[threadIdx.x+TILESIZE][threadIdx.y*(TILESIZE+1)] = input[((TILESIZE + 1)*threadIdx.y+blockIdx.y*TILESIZE-1)*width + TILESIZE+threadIdx.x]; */
    /*     } */
    /* } */
    /* s_data[threadIdx.x + 1][threadIdx.y + 1] = input[index]; */
    /* __syncthreads(); */
    /* if(col < width && row < height){ */
    /*     if(col < width - 1 && row < height - 1 && col > 0 && row > 0){ */
    /*         int sum1 = s_data[idx - 1][idy + 1] - s_data[idx - 1][idy - 1] + 2 * s_data[idx][idy+1] - 2 * s_data[idx][idy - 1] + s_data[idx + 1][idy + 1] - s_data[idx + 1][idy - 1]; */
    /*         int sum2 = s_data[idx - 1][idy - 1] + s_data[idx - 1][idy] + s_data[idx - 1][idy + 1] - s_data[idx + 1][idy - 1] - 2 * s_data[idx + 1][idy] - s_data[idx + 1][idy + 1]; */
    /*         int magnitude = sum1 * sum1 + sum2 * sum2; */
    /*         if(magnitude > thresh) */
    /*             output[index] = 255; */
    /*         else */
    /*             output[index] = 1; */
    /*     }else */
    /*         output[index] = 2; */
    /* } */
    output[0] = 100;
}
__global__ void test(void){
    /* output[0] = 1; */
    if (threadIdx.x == 0)
        {
            printf("Hello World from GPU thread %d!\n", threadIdx.x);
        }
}
#endif
