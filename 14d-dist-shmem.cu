#include <cstdio>
#include <chrono>
#include <iostream>
#include <cassert>
#include <cooperative_groups.h>
using namespace std;
using namespace std::chrono;
namespace cg = cooperative_groups;

#include "./common.hpp"

const int CLUSTER_SIZE = 2;
const int BLOCK_SIZE = 16;

const int N = CLUSTER_SIZE * BLOCK_SIZE;

int* ptrData0 = nullptr;
int* ptrData1 = nullptr;

int* devPtrData0 = nullptr;
int* devPtrData1 = nullptr;

// CUDA kernel function, with cluster support

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) kernel_dist_shmem( int* pData0, int* pData1 ) {
	extern __shared__ int s_data[];
	cg::thread_block block = cg::this_thread_block();
	cg::cluster_group cluster = cg::this_cluster();
	int tx = block.thread_rank(); // threadIdx.x
	int bx = cluster.block_rank(); // blockIdx.x
	int gx = bx * block.size() + tx; // blockDim.x
	// generate s_data values
	if (bx == 0) { // first block: 0, 1, 2, 3, . . .
		s_data[tx] = tx;
	} else { // second block: 0, 100, 200, 300, . . .
		s_data[tx] = tx * 100;
	}
	cluster.sync();
	// write my values to the buffer 0
	pData0[gx] = s_data[tx];
	// access to the neighbor's shared memory
	int nbx = (cluster.block_rank() + 1) % cluster.num_blocks(); // neighbor's bx
	int *n_s_data = cluster.map_shared_rank( s_data, nbx ); // neighbor's s_data
	// write added values to the buffer 1
	pData1[gx] = s_data[tx] + n_s_data[tx]; // my value + neighbor's value
}

int main() {
	// check for cooperative group extension support
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 9) {
		printf("Error: this program can be executed SM 9.0 and higher.\n");
		printf("Error: this system's SM version = %d.%02d\n", prop.major, prop.minor);
        return 0;
    }
	// show settings
	printf("CLUSTER_SIZE = %d\n", CLUSTER_SIZE);
	printf("BLOCK_SIZE = %d\n", BLOCK_SIZE);
	printf("N = total num elem = %d\n", N);
	// GPU part: warm up
	CUDA_WARM_UP();
	// CPU part: memory allocation
	cudaMallocHost( &ptrData0, N * sizeof(int) );
	cudaMallocHost( &ptrData1, N * sizeof(int) );
	CUDA_CHECK_ERROR();
	// GPU part: memory allocation
	cudaMalloc((void**)&devPtrData0, N * sizeof(int));
	cudaMalloc((void**)&devPtrData1, N * sizeof(int));
	CUDA_CHECK_ERROR();
	// GPU kernel configure
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(CLUSTER_SIZE, 1, 1);
	CUDA_PRINT_CONFIG(N);
	// GPU kernel action
    int shmem_nbytes = BLOCK_SIZE * sizeof(int);
    kernel_dist_shmem<<<dimGrid, dimBlock, shmem_nbytes>>>(devPtrData0, devPtrData1);
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();
	// device to host copy
	cudaMemcpy(ptrData0, devPtrData0, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(ptrData1, devPtrData1, N * sizeof(int), cudaMemcpyDeviceToHost);
	printf("original s_data values:\n");
	printf("TB0 = ");
	printArray(BLOCK_SIZE, ptrData0);
	printf("TB1 = ");
	printArray(BLOCK_SIZE, ptrData0 + BLOCK_SIZE);
	printf("after addition, s_data values:\n");
	printf("TB0 = ");
	printArray(BLOCK_SIZE, ptrData1);
	printf("TB1 = ");
	printArray(BLOCK_SIZE, ptrData1 + BLOCK_SIZE);
	// done
	fflush(stdout);
    return 0;
}

// HISTORY: dist-shmem.cu
static const char rcsid[] __attribute__((used)) = "$Id: 14d-dist-shmem.cu,v 1.1 2026/10/17 03:30:48 wayfarecru Exp $";

