#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include "kernel.cu"

#define DEFAULT_CLUSTER_NUM 3
#define BLOCKSIZE_16 16
#define BLOCKSIZE_32 32
#define MAX_ITERATION 10000
typedef struct obj_params params;
struct obj_params{
    int obj_num;
    int obj_dimension;
    int cluster_num;
    int max_iterator;
    float* data;
};
params read_data(char *filename){
    if(!filename || filename[0] == '\0'){
        fprintf(stderr,"No file\n");
        exit(EXIT_FAILURE);
    }
    FILE *fp;
    fprintf(stderr, "read %s \n", filename);
    fp = fopen(filename, "r");
    if(!fp){
        fprintf(stderr, "fopen fail\n");
        exit(EXIT_FAILURE);
    }
    params params;
    int width = 1, height = 0;
    char *line = NULL;
    size_t size;
    ssize_t read_num;
    while((read_num = getline(&line, &size, fp)) != -1){
        if(line[0] == '%' || line[0] == '\n')
            continue;
        if((strcmp(line, "@data\n")) == 0){
            height++;
            continue;
        }
        if(height){
            if(height == 1){
                char *token = NULL;
                if((token = strtok(line, " ,\t")) == NULL){
                    fprintf(stderr,"read err with strtok\n");
                    exit(EXIT_FAILURE);
                }
                while((token = strtok(NULL, " ,\t")) != NULL)
                    width++;
            }
            height++;
        }
    }
    height--;
    fprintf(stderr,"width: %d, height: %d\n" ,width, height);
    if((params.data = (float *)malloc(height * width * sizeof(float))) == NULL){
        fprintf(stderr, "malloc fail\n");
        exit(EXIT_FAILURE);
    }
    params.obj_dimension = width;
    params.obj_num = height;
    fseek(fp,0,SEEK_SET);
    height = 0;
    int i = 0;
    while((read_num = getline(&line, &size, fp)) != -1){
        if(line[0] == '%' || line[0] == '\n')
            continue;
        if((strcmp(line, "@data\n")) == 0){
            height++;
            continue;
        }
        if(height){
            char *token = NULL;
            if((token = strtok(line, " ,\t")) == NULL){
                fprintf(stderr,"read err with strtok\n");
                exit(EXIT_FAILURE);
            }
            params.data[i] = atof(token);
            i++;
            while((token = strtok(NULL, " ,\t")) != NULL){
                params.data[i] = atof(token);
                ++i;
            }
        }
    }
    if(line)
        free(line);
    return params;
}
float* init_cluster_center(params params){
    int dimension = params.obj_dimension;
    int k = params.cluster_num;
    float *cluster_centers = (float*)malloc(sizeof(float) * k * dimension);
    if(!cluster_centers){
        fprintf(stderr, "init cluster_center fail\n");
        return NULL;
    }
    for (int i = 0; i < k; ++i){
        for (int j = 0; j < dimension; ++j)
            cluster_centers[i * dimension + j] = params.data[i * dimension + j];
    }
    return cluster_centers;
}

void update_cluter_info(params params, float *cluster_centers, int *cluster_info){
    int k = params.cluster_num;
    int dimension = params.obj_dimension;
    float distance;
    float tmp;
    int idx;
    float *host_distance = (float*)malloc(sizeof(float) * k * params.obj_num);
    for (int i = 0 ; i < params.obj_num; ++i) {
        idx = 0;
        distance = FLT_MAX;
        for (int j = 0; j < k; ++j) {
            tmp = 0;
            for (int m = 0; m < dimension; ++m) {
                tmp += (params.data[i * dimension + m] - cluster_centers[j * dimension + m]) *
                    (params.data[i * dimension + m] - cluster_centers[j * dimension + m]);
            }
            host_distance[i * k + j] = tmp;
            if(tmp <= distance){ //have to use equal 
                idx = j;
                distance = tmp;
            }
        }
        cluster_info[i] = idx;
    }
    /* printf("Host_distance:\n"); */
    /* for (int i = 0; i < params.obj_num; ++i) { */
    /*     for (int j = 0; j < k; ++j) { */
    /*         printf("%f ", host_distance[i * k + j]); */
    /*     } */
    /*     printf("\n"); */
    /* } */

}
int update_cluster_center(params params, float *cluster_centers_old, int *cluster_info){
    int flag = 0;
    int cluster_idx;
    int count[params.cluster_num];
    memset(count, 0, sizeof(count));
    float *new_center = (float*)calloc(1, sizeof(float) * params.obj_dimension * params.cluster_num);
    if(!new_center){
        fprintf(stderr, "calloc new_center error");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < params.obj_num; ++i) {
        cluster_idx = cluster_info[i];
        count[cluster_idx]++;
        for (int j = 0; j < params.obj_dimension; ++j) {
            new_center[cluster_idx * params.obj_dimension + j] += params.data[i * params.obj_dimension + j];
        }
    }
    for (int i = 0; i < params.cluster_num; ++i) {
        if(count[i] == 0){
            fprintf(stderr,"cluster_center %d has no element.\n",i);
            exit(EXIT_FAILURE);
        }
        /* printf("cluster %d: %d\n",i, count[i]); */
        for (int j = 0; j < params.obj_dimension; ++j)
            new_center[i * params.obj_dimension + j] /= count[i];
    }
    for (int i = 0; i < params.cluster_num * params.obj_dimension; ++i) {
        if(cluster_centers_old[i] != new_center[i]){
            flag = 1;
            cluster_centers_old[i] = new_center[i];
        }
    }
    /* free(new_center); */
    return flag;
}
/*
  dev_cluster_info
  dev_data
  dev_center_data
  dev_distance
  dev_cluster_count

*/
void kmeans_gpu(params params, int *cluster_info){
    /* cudaEvent_t mcp_start, mcp_end; */
    /* float mcp_time; */
    /* cudaEventCreate(&mcp_start); */
    /* cudaEventCreate(&mcp_end); */
    /* cudaEventRecord(mcp_start,0); */

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float *dev_data, *dev_center_data;
    cudaMalloc((void**)&dev_data, params.obj_num * params.obj_dimension * sizeof(float));
    cudaMalloc((void**)&dev_center_data, params.cluster_num * params.obj_dimension * sizeof(float));
    cudaMemcpy(dev_data, params.data, params.obj_num * params.obj_dimension * sizeof(float), cudaMemcpyHostToDevice);
    int *dev_cluster_info;
    cudaMalloc((void**)&dev_cluster_info, params.obj_num * sizeof(int));
    cudaMemcpy(dev_cluster_info, cluster_info, sizeof(int) * params.obj_num, cudaMemcpyHostToDevice);
    float *dev_distance;
    cudaMalloc((void**)&dev_distance, params.obj_num * params.cluster_num * sizeof(float));
    int *dev_cluster_count;
    cudaMalloc((void**)&dev_cluster_count, params.cluster_num * sizeof(int));

    /* cudaEventRecord(mcp_end,0); */
    /* cudaEventSynchronize(mcp_end); */
    /* cudaEventElapsedTime(&mcp_time, mcp_start, mcp_end); */
    /* fprintf(stderr, "gpu memory copy time: %f ms\n", mcp_time); */
    /*
      int *host_cluster_info = (int*)malloc(sizeof(int) * params.obj_num);
      cudaMemcpy(host_cluster_info, dev_cluster_info, sizeof(int)* params.obj_num, cudaMemcpyDeviceToHost);
      for (int i = 0; i < params.obj_num; ++i) {
      printf("%d ", host_cluster_info[i]);
      }
      float *data = (float*)malloc(sizeof(float) * params.obj_dimension * params.obj_num);
      cudaMemcpy(data, dev_data, params.obj_dimension * params.obj_num * sizeof(float), cudaMemcpyDeviceToHost);
      for (int i = 0; i < params.obj_num; ++i) {
      for (int j = 0; j < params.obj_dimension; ++j) {
      printf("%f ",data[i * params.obj_dimension + j]);
      }
      printf("\n");
      }
    */
    dim3 block1d_16(BLOCKSIZE_16 * BLOCKSIZE_16);
    dim3 block1d_32(BLOCKSIZE_32 * BLOCKSIZE_32);
    dim3 grid1d_16((params.obj_num + BLOCKSIZE_16 * BLOCKSIZE_16 - 1) / BLOCKSIZE_16 * BLOCKSIZE_16);
    dim3 grid1d_32((params.obj_num + BLOCKSIZE_32 * BLOCKSIZE_32 - 1) / BLOCKSIZE_32 * BLOCKSIZE_32);
    dim3 block2d_16(BLOCKSIZE_16,BLOCKSIZE_16);
    dim3 grid2d_cluster((params.obj_dimension + BLOCKSIZE_16 - 1) / BLOCKSIZE_16, (params.cluster_num + BLOCKSIZE_16 -1) / BLOCKSIZE_16);
    dim3 grid2d_dimension_obj((params.obj_dimension + BLOCKSIZE_16 - 1) / BLOCKSIZE_16, (params.obj_num + BLOCKSIZE_16 - 1) / BLOCKSIZE_16);
    dim3 grid2d_cluster_obj((params.cluster_num + BLOCKSIZE_16 - 1) / BLOCKSIZE_16, (params.obj_num + BLOCKSIZE_16 - 1) / BLOCKSIZE_16);



    for(int k = 0; k < params.max_iterator; k++){
    cudaMemset(dev_center_data, 0, params.obj_dimension * params.cluster_num * sizeof(float));
    cudaMemset(dev_cluster_count, 0, params.cluster_num * sizeof(int));

    cudaMemset(dev_distance, 0, params.obj_num * params.cluster_num * sizeof(float));

    cal_center_total<<<grid2d_dimension_obj, block2d_16>>>(dev_data, dev_cluster_info, dev_center_data, params.obj_dimension, params.obj_num);
    count_cluster_num<<<grid1d_16, block1d_16>>>(dev_cluster_info, dev_cluster_count, params.obj_dimension, params.obj_num);
    /* count_cluster_num<<<grid1d_32, block1d_32>>>(dev_cluster_info, dev_cluster_count, params.obj_dimension, params.obj_num); */
    cal_center_mean<<<grid2d_cluster, block2d_16>>>(dev_center_data, dev_cluster_count, params.obj_dimension, params.cluster_num);
    cal_distance<<<grid2d_cluster_obj, block2d_16>>>(dev_data, dev_center_data, dev_distance, params.obj_num, params.cluster_num, params.obj_dimension);
    /* cal_distance_share<<<grid2d_cluster_obj, block2d_16, params.obj_dimension * BLOCKSIZE_16 * sizeof(float) * 2>>>(dev_data, dev_center_data, dev_distance, params.obj_num, params.cluster_num, params.obj_dimension); */
    update_cluster_index<<<grid1d_16,block1d_16>>>(dev_distance, dev_cluster_info, params.obj_num, params.cluster_num);
    }

    float *host_center_data;
    host_center_data = (float*)malloc(params.obj_num * params.obj_dimension * sizeof(float));
    cudaMemcpy(host_center_data, dev_center_data, params.obj_dimension * params.cluster_num * sizeof(float), cudaMemcpyDeviceToHost);
    /* float *host_distance = (float*)malloc(params.cluster_num * params.obj_num * sizeof(float)); */
    /* cudaMemcpy(host_distance, dev_distance, params.cluster_num * params.obj_num * sizeof(float), cudaMemcpyDeviceToHost); */
    /* int *host_count = (int*)malloc(params.cluster_num * sizeof(int)); */
    /* cudaMemcpy(host_count, dev_cluster_count, params.cluster_num * sizeof(int), cudaMemcpyDeviceToHost); */

    /* for (int i = 0; i < params.cluster_num; ++i) { */
    /*     printf("cluster%d count: %d\n", i, host_count[i]); */
    /* } */

    /* for (int i = 0; i < params.obj_num; ++i) { */
    /*     for (int j = 0; j < params.cluster_num; ++j) { */
    /*         printf("%f ", host_distance[i * params.cluster_num + j]); */
    /*     } */
    /*     printf("\n"); */
    /* } */

    cudaMemcpy(cluster_info, dev_cluster_info, params.obj_num * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    fprintf(stderr, "gpu calculate time: %f ms\n", time);


    /*
    for (int i = 0; i < params.cluster_num; ++i) {
        for(int j = 0; j < params.obj_dimension; ++j){
            fprintf(stderr, "gpu center: %f ", host_center_data[i * params.obj_dimension + j]);
        }
        fprintf(stderr, "\n");
    }
    */


    /* for (int i = 0 ; i < params.obj_num; ++i) { */
    /*     printf("obj%d: %d\n", i, cluster_info[i]); */
    /* } */

    cudaFree(dev_data);
    cudaFree(dev_cluster_info);
    cudaFree(dev_distance);
    cudaFree(dev_center_data);
    cudaFree(dev_cluster_count);
}
void kmeans(params params){
    clock_t start, end;
    double cpu_time_used;

    start = clock();

    float *cluster_centers = NULL;
    if((cluster_centers = init_cluster_center(params)) == NULL)
        exit(EXIT_FAILURE);

    params.max_iterator = MAX_ITERATION;
    int *cluster_info = (int*)malloc(sizeof(int) * params.obj_num);
    if(!cluster_info){
        fprintf(stderr, "calloc cluster_info error\n");
        exit(EXIT_FAILURE);
    }
    int n = 0;
    int *host_cluster_info = (int*)malloc(sizeof(int) * params.obj_num);
    if(!host_cluster_info){
        fprintf(stderr, "malloc host_cluster_info error\n");
        exit(EXIT_FAILURE);
    }

    update_cluter_info(params, cluster_centers, cluster_info);
    for (int i = 0; i < params.obj_num; ++i)
        host_cluster_info[i] = cluster_info[i];
    update_cluster_center(params, cluster_centers, cluster_info);

    for (int i = 0; i < params.max_iterator; ++i) {
    /* for (int i = 1; i < 2; ++i) { */
        update_cluter_info(params, cluster_centers, cluster_info);
        n++;
        if(update_cluster_center(params, cluster_centers, cluster_info) == 0)
            break;
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    fprintf(stderr,"iterator %d times\n",n);
    fprintf(stderr, "cpu_time: %f ms\n", cpu_time_used * 1000);
    params.max_iterator = n;


    /*

    for (int i = 0; i < params.cluster_num; ++i) {
        fprintf(stderr,"centers%d: ",i);
        for (int j=0; j < params.obj_dimension; ++j) {
            printf("%f ", cluster_centers[i * params.obj_dimension + j]);
        }
        printf("\n");
    }

    */

    /* for (int i = 0; i < params.obj_num; ++i) { */
    /*     printf("obj%d is in cluster%d\n",i,cluster_info[i]); */
    /* } */

    /* for (int i = 0; i < params.obj_num; ++i) */
    /*     printf("obj%d: %d\n",i,cluster_info[i]); */
    /* kmeans_gpu(params, host_cluster_info); */
    kmeans_gpu(params, host_cluster_info);
}
int main(int argc, char *argv[]){
    params params;
    params = read_data(argv[1]);
    if (argc == 3 && atoi(argv[2]) > 1)
        params.cluster_num = atoi(argv[2]);
    else
        params.cluster_num = DEFAULT_CLUSTER_NUM;
    /* printf("cluster_num: %d\n",params.cluster_num); */
    /* for (int i = 0; i < params.obj_dimension * params.obj_num; ++i) { */
    /*     printf("%f ",params.data[i]); */
    /*     if((i - params.obj_dimension + 1) % (params.obj_dimension) == 0) */
    /*         printf("\n"); */
    /* } */
    kmeans(params);
    free(params.data);
    exit(EXIT_SUCCESS);
}
