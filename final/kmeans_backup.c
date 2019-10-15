#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define DEFAULT_CLUSTER_NUM 3
typedef struct obj_params params;
typedef struct obj_cluster_center cluster_center;
struct obj_params{
    int obj_num;
    int obj_dimension;
    int cluster_num;
    int max_iterator;
    float* data;
};
struct obj_cluster_center{
    int dimension;
    int cluster_idx;
    float *center;
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
cluster_center* init_cluster_center(params params,cluster_center *cluster_centers_new){
    int dimension = params.obj_dimension;
    int k = params.cluster_num;
    cluster_centers_new = (cluster_center*)malloc(sizeof(cluster_center) * k);
    if(!cluster_centers_new){
        fprintf(stderr, "init cluster_center fail\n");
        return NULL;
    }
    for (int i = 0; i < k; ++i) {
        cluster_centers_new[i].cluster_idx = i;
        cluster_centers_new[i].dimension = dimension;
        cluster_centers_new[i].center = (float*)malloc(sizeof(float) * dimension);
        if(!cluster_centers_new[i].center){
            fprintf(stderr, "%d cluster center init err\n", i);
            return NULL;
        }
        for (int j = 0; j < params.obj_dimension; ++j)
            cluster_centers_new[i].center[j] = params.data[i * dimension + j];
    }
    return cluster_centers_new;
}

void update_cluter_info(params params, cluster_center *cluster_centers_new, int *cluster_info){
    int k = params.cluster_num;
    int dimension = params.obj_dimension;
    float distance;
    float tmp;
    int idx;
    for (int i = 0 ; i < params.obj_num; ++i) {
        idx = 0;
        distance = FLT_MAX;
        for (int j = 0; j < k; ++j) {
            tmp = 0;
            for (int m = 0; m < dimension; ++m) {
                tmp += (params.data[i * dimension + m] - cluster_centers_new[j].center[m]) *
                    (params.data[i * dimension + m] - cluster_centers_new[j].center[m]);
                /* printf("%f %f\n", params.data[i * dimension + m], cluster_centers_new[j].center[m]); */
            }
            if(tmp < distance){
                idx = j;
                distance = tmp;
            }
        }
        cluster_info[i] = idx;
    }
}
int update_cluster_center(params params, cluster_center *cluster_centers_new, int *cluster_info){
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
    for (int i = 0; i < params.cluster_num; ++i) {
        for (int j = 0; j < params.obj_dimension; ++j) {
            if(cluster_centers_new[i].center[j] != new_center[i * params.obj_dimension + j]){
                cluster_centers_new[i].center[j] = new_center[i * params.obj_dimension + j];
                flag = 1;
            }
        }
    }
    return flag;
}
void kmeans_gpu(params params, int *cluster_info){
    float *dev_data, *dev_center;
    cudaMalloc((void**)&dev_data, params.obj_dimension * params.obj_dimension * sizeof(float));
    cudaMalloc((void**)&dev_center, params.cluster_num * params.obj_num * sizeof(float));
}
void kmeans(params params){
    cluster_center *cluster_centers_new = NULL;
    if((cluster_centers_new = init_cluster_center(params, cluster_centers_new)) == NULL)
        exit(EXIT_FAILURE);

    params.max_iterator = 10000;
    int *cluster_info = (int*)malloc(sizeof(int) * params.obj_num);
    if(!cluster_info){
        fprintf(stderr, "calloc cluster_info error\n");
        exit(EXIT_FAILURE);
    }
    int n = 0;
    for (int i = 0; i < params.max_iterator; ++i) {
        n++;
        update_cluter_info(params, cluster_centers_new, cluster_info);
        if(update_cluster_center(params, cluster_centers_new, cluster_info) == 0)
            break;
    }
    fprintf(stderr,"iterator %d times\n",n);
    for (int i = 0; i < params.cluster_num; ++i) {
        fprintf(stderr,"centers%d: ",i);
        for (int j=0; j < params.obj_dimension; ++j) {
            printf("%f ", cluster_centers_new[i].center[j]);
        }
        printf("\n");
    }
    kmeans_gpu(params, cluster_info);
    /* for (int i = 0; i < params.obj_num; ++i) { */
    /*     printf("obj%d is in cluster%d\n",i,cluster_info[i]); */
    /* } */

    /* for (int i = 0; i < params.obj_num; ++i) */
    /*     printf("Index %d : %d\n",i,cluster_info[i]); */
}
int main(int argc, char *argv[]){
    params params;
    params = read_data(argv[1]);
    if (argc == 3 && atoi(argv[2]) > 1)
        params.cluster_num = atoi(argv[2]);
    else
        params.cluster_num = DEFAULT_CLUSTER_NUM;
    /* for (int i = 0; i < params.obj_dimension * params.obj_num; ++i) { */
    /*     printf("%f ",params.data[i]); */
    /*     if((i - params.obj_dimension + 1) % (params.obj_dimension) == 0) */
    /*         printf("\n"); */
    /* } */
    kmeans(params);
    free(params.data);
    exit(EXIT_SUCCESS);

}
