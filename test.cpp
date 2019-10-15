#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

#define DIMENSION 5
#define CLUSTERS_NUM 3
using namespace std;
int main(int argc, char *argv[])
{
    
    return 0;
}
struct Tuple{
    int label;
    float *attributes;
};
float get_distance(Tuple t1, Tuple t2){
    float distance = 0;
    for (int i = 0; i < DIMENSION; ++i) {
        distance += (t1.attributes[i] - t2.attributes[i]) * (t1.attributes[i] - t2.attributes[i]);
    }
    return distance;
}
int get_cluster_index(Tuple centers[], Tuple tuple){
    float distance = get_distance(centers[0], tuple);
    float tmp;
    int cluster_index = 0;
    for (int i = 1; i < CLUSTERS_NUM; ++i) {
        tmp = get_distance(centers[i], tuple);
        if(tmp < distance){
            tmp = distance;
            cluster_index = i;
        }
    }
    return cluster_index;
}
Tuple find_center(vector<Tuple> clusters){
    int num = clusters.size();
    Tuple center;
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < center.dimension; ++j) {
            center.attributes[j] += clusters[i].attributes[j];
        }
    }
    for (int i = 0; i < center.dimension; ++i) {
        center.attributes[i] /= num;
    }
    return center;
}
float total_distance(vector<Tuple> clusters[], Tuple centers[]){
    float total_distance = 0;
    for (int i = 0; i < CLUSTERS_NUM; ++i) {
        vector<Tuple> cluster = clusters[i];
        for (int j = 0; j < t.size(); ++j) {
            total_distance += get_distance(cluster[j], centers[i]);
        }
    }
    return total_distance;
}
void k_means(vector<Tuple> tuples){
    vector<Tuple> clusters[k];
    //指针可能出问题
    Tuple centers[k];
    int i;
    for(i = 0; i < k; ++i){
        for (int j = 0; j < DIMENSION; ++i) {
            centers[i].attributes[j] = tuples[i].attributes[j];
        }

    }
}
// __global__ void reduction_more(float *g_data, int n, float *result){
//     int tid = threadIdx.x;
//     int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
//     __shared__ float partialSum[512];
//     if(index < n)
//         partialSum[tid] = g_data[index] + g_data[index + blockDim.x];
//     for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//         __syncthreads();
//         if(tid < stride)
//             partialSum[tid] += partialSum[tid + stride];
//     }
//     __syncthreads();
//     if(tid == 0)
//         result[blockIdx.x] = partialSum[0];
// }

// __global__ void cluster_distance(float *x, float *y, float *center_x, float *center_y, float *resulet,){
//     int index = blockIdx.x * blockDim + threadIdx.x;
//     int tid = threadIdx.x;
//     result[index] = (x[index] - center_x[index]) * (x[index] - center_x[index]) +  (y[index] - center_y) *  (y[index] - center_y);
// }



// __global__ void find_cluster(int k, int size, float *center_x, float *center_y, float *tuple_x, float tuple_y, int *cluster_info){
//     int index = blockDim.x * blockIdx.x + threadIdx.x;
//     if(index < size){
//         int index = 0;
//         float dist, min_dist, i;
//         min_dist = (center_x[0] - tuple_x[index]) * (center_x[0] - tuple_x[index]) +
//             (center_y[0] - tuple_y[index]) * (center_y[0] - tuple_y[index]);
//         for (i = 1; i < k; ++i) {
//             dist = (center_x[i] - tuple_x[index]) * (center_x[i] - tuple_x[index]) +
//                 (center_y[i] - tuple_y[index]) * (center_y[i] - tuple_y[index]);
//             if(dist < min_dist){
//                 min_dist = dist;
//                 cluster_info[index] = i;
//             }
//         }

//     }
// }
