#ifndef _KMEANS_KERNEL_H_
#define _KMEANS_KERNEL_H_

#include <stdio.h>
#define BLOCKSIZE_16 16
#define DIMENSION 8
/*
  if the data is consequent, use the add reduction, here I use atomic add.
 */
__global__ void cal_center_total(float *dev_obj_data, int *dev_cluster_info, float *dev_center_data, int dimension, int obj_num){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(col < dimension && row < obj_num){
        int idx = dimension * row + col;
        int cluster_idx = dev_cluster_info[row];
        atomicAdd(&dev_center_data[cluster_idx * dimension + col], dev_obj_data[idx]);
    }
}

__global__ void count_cluster_num(int *dev_cluster_info, int *dev_cluster_count, int dimension, int obj_num){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < obj_num){
        int cluster_idx = dev_cluster_info[idx];
        atomicAdd((int*)&dev_cluster_count[cluster_idx], 1);
    }
}
__global__ void cal_center_mean(float *dev_center_data, int *dev_cluster_count, int dimension, int cluster_num){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(col < dimension && row < cluster_num){
        int idx = row * dimension + col;
        dev_center_data[idx] /= (float)dev_cluster_count[row];
    }
}
__device__ float distance(float *dev_obj_data, float *dev_center_data, int dimension){
    float distance = 0;
    for (int i = 0; i < dimension; ++i) {
        float tmp = dev_obj_data[i] - dev_center_data[i];
        distance = distance + tmp * tmp;
    }
    return distance;
}
/*
  without shared memory
 */
__global__ void cal_distance(float *dev_obj_data,float *dev_center_data, float *dev_disntance, int obj_num, int cluster_num, int dimension){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(row < obj_num && col < cluster_num){
        int idx = col + row * cluster_num;
        dev_disntance[idx] = distance(dev_obj_data + row * dimension, dev_center_data + col * dimension, dimension);
    }
}

__global__ void cal_distance_share(float *dev_obj_data, float *dev_center_data, float *dev_distance, int obj_num, int cluster_num, int dimension){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    /* const int size = dimension * BLOCKSIZE_16; */
    const int size = DIMENSION * BLOCKSIZE_16;
    __shared__ float cluster_center_share[BLOCKSIZE_16][size];
    __shared__ float obj_data_share[BLOCKSIZE_16][size];
    /* extern __shared__ float cluster_center_share[]; */
    /* extern __shared__ float obj_data_share[]; */
    if(row < obj_num){
        float *obj_data = &dev_obj_data[dimension * blockDim.y * blockIdx.y];
        float *center_data = &dev_center_data[dimension * blockDim.x * blockIdx.x];
        for (int xidx = threadIdx.x; xidx < dimension; xidx += BLOCKSIZE_16) {
            int idx = dimension * threadIdx.y + xidx;
            obj_data_share[threadIdx.y][xidx] = obj_data[idx];
            cluster_center_share[threadIdx.y][xidx] = center_data[idx];
        }
        __syncthreads();
    }
    if(col < cluster_num && row < obj_num){
        dev_distance[row * cluster_num + col] = distance(obj_data_share[threadIdx.y], cluster_center_share[threadIdx.x], dimension);
    }
}
__global__ void update_cluster_index(float *dev_distance, int *dev_cluster_info, int obj_num, int cluster_num){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < obj_num){
        float *p = &dev_distance[index * cluster_num];
        int tmp_idx = 0;
        float tmp_distance = p[0];
        for (int i = 1; i < cluster_num; ++i) {
            if(tmp_distance >= p[i]){
                tmp_distance = p[i];
                tmp_idx = i;
            }
        }
        dev_cluster_info[index] = tmp_idx;
    }
}
#endif
