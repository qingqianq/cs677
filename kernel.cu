__global__ updateClusterChanged(float **tuples, float **centers, int *cluster_info, int *cluster_change, int n, int dimension){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int changed[BLOCKSIZE];
	changed[threadIdx.x] = 0
	if(tid < n){
		int i,j,index,distance;
		float min = 0
		for(i = 0; i < dimension; ++i){
			min += (tuples[tid][i] - centers[0][i]) * (tuples[tid][i] - centers[0][i]);
		}
		for(i = 1; i < k; ++i){
			distance = 0;
			for(j =0; j < dimension; ++j){
				distance += ((tuples[tid][j] - centers[i][j]) * (tuples[tid][j] - centers[i][j]));
			}
			if(distance < min){
				min = distance;
				index = i;
			}
		}
		if(cluster_info[tid] != index){
			changed[threadIdx.x] = 1;
			cluster_info[tid] = index;
		}
		__syncthreads();
		for (unsigned int stride = blockDim.x/2;
	       stride > 0;  stride >>= 1){
     	if (threadIdx.x < stride)
			changed[threadIdx.x] += changed[threadIdx.x+stride]; 
		 __syncthreads();
		}
		if(threadIdx.x == 0)
			cluster_change[blockIdx.x] = changed[0];
	}
}

class sParameter
{
 public:
    int objNum; // 样本数
    int objLength; // 样本维度
    int clusterNum; // 聚类数
    int minClusterNum; // 最少的聚类数
    int minObjInClusterNum; // 每个聚类中的最少样本数
    int maxKmeansIter; // 最大迭代次数
};
