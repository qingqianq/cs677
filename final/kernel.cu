#ifndef _KMEANS_KERNEL_H_
#define _KMEANS_KERNEL_H_

#include <stdio.h>
/*
  if the data is consequent, use the add reduction, here I use atomic add.
 */
__global__ void cal_center_total(float *dev_obj_data, int *dev_cluster_info, float *dev_cluster_data, int dimension, int obj_num){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(col < dimension && row < obj_num){
        int idx = dimension * col + row;
        int cluster_idx = dev_cluster_idx[row];
        atomicAdd(&dev_cluser_data[cluster_idx * dimension + col], dev_obj_data[idx]);
    }
}

__global__ void count_cluster_num(int *dev_cluster_info, int *dev_cluster_count, int dimension, int obj_num){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < obj_num){
        int cluster_idx = dev_cluster_info[idx];
        atomicAdd((int*)&dev_cluster_count[cluster_idx], 1);
    }
}
__global__ void cal_center_mean(float *dev_cluster_data, int *dev_cluster_count, int dimension, int cluster_num){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(col < dimension && row < cluster_num){
        int idx = row * dimension + col;
        dev_cluster_data[idx] /= (float)dev_cluster_count[row];
    }
}
#endif
