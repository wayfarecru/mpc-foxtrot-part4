#include <cstdio>
#include <chrono>
#include <cassert>
#include <climits>
#include <algorithm>
#include <functional>
using namespace std;
using namespace std::chrono;

#include "./common.hpp"

// CLUSTER-based processing: data = NUM_CLUSTER x CLUSTER
const int BLOCK_SIZE = 1024; // number of threads in a TB (thread block)
const int NUM_BLOCK = 64; // total number of blocks in a Grid
const int NUM_BLOCK_PER_CLUSTER = 2; // number of blocks per cluster
const int NUM_CLUSTER = NUM_BLOCK / NUM_BLOCK_PER_CLUSTER;
const int BOUND = 1000 * 1000; // numbers will be ranged in [0..bound)

const int N = NUM_BLOCK * BLOCK_SIZE; // total number of elements
const int M = NUM_BLOCK_PER_CLUSTER * BLOCK_SIZE; // number of elements for a TB

int* ptrSrc; // source data to be sorted
int* ptrData0; // data area for C++ std::sort results
int* ptrData; // data area for sorting

int* devPtrData; // device data area for sorting

enum {
	DECREASING = 0,
	INCREASING = 1,
};
const unsigned direction = INCREASING;
// const unsigned direction = DECREASING;

// bitonic sort, block-size, CPU

template <unsigned dir, typename TYPE>
__device__ __host__ inline void comparator( TYPE& lhs, TYPE& rhs ) {
	// if dir == 0: decreasing case, we want to make (lhs >= rhs)
	// if dir == 1: increasing case, we want to make (lhs < rhs)
	if (dir == (lhs > rhs)) { // simple swap
		TYPE t = lhs;
		lhs = rhs;
		rhs = t;
	}
}

template <int dir>
void cpu_bitonic_merge( int* pData, int num ) {
	if (num > 1) {
		int half = num / 2;
		for (int i = 0; i < half; ++i) {
			comparator<dir>( pData[i], pData[i + half] );
		}
		cpu_bitonic_merge<dir>( pData, half );
		cpu_bitonic_merge<dir>( pData + half, half );
	}
}

template <int dir>
void cpu_bitonic_sort( int* pData, int num ) {
	if (num > 1) {
		int half = num / 2;
		cpu_bitonic_sort<dir>( pData, half );
		cpu_bitonic_sort<1-dir>( pData + half, half ); // reverse direction
		cpu_bitonic_merge<dir>( pData, num );
	}
}

// index-based bitonic sort, block-size, CPU

template <int dir>
void cpu_index_based_bitonic_sort( int* pData, int num ) {
	for (int k = 2; k <= num; k <<= 1) { // k works as the length of the target list
		for (int j = k / 2; j > 0; j >>= 1) { // j works as the stride
			for (int i = 0; i < num; ++i) {
				int ij = i ^ j;
				if ((ij) > i) {
					if ((i & k) == 0) {
						comparator<dir>( pData[i], pData[ij] );
					} else {
						comparator<1-dir>( pData[i], pData[ij] );
					}
				}
			}
		}
	}
}

// bitonic sort, block-size, GPU

template <typename TYPE>
__device__ __host__ inline void comparator_dir( TYPE& lhs, TYPE& rhs, int dir ) {
	// if dir == 0: decreasing case, we want to make (lhs >= rhs)
	// if dir == 1: increasing case, we want to make (lhs < rhs)
	if (dir == (lhs > rhs)) { // simple swap
		TYPE t = lhs;
		lhs = rhs;
		rhs = t;
	}
}

// CUDA kernel function
template <int dir>
__global__ void kernel_bitonic_sort_block( int* devPtrData ) {
	__shared__ int sData[M]; // M = NUM_BLOCK_PER_CLUSTER * BLOCK_SIZE
	assert(NUM_BLOCK_PER_CLUSTER == 2);
	int block_offset = NUM_BLOCK_PER_CLUSTER * blockIdx.x * blockDim.x;
	// copy from global memory to shared mem, (2 * blockDim.x) elements
	int tx = threadIdx.x;
	sData[tx] = devPtrData[block_offset + tx];
	sData[tx + blockDim.x] = devPtrData[block_offset + tx + blockDim.x];
	__syncthreads();
	// bitonic sort: 1st phase. halfsize is half of the target sorting range
	for (int halfsize = 1; halfsize < blockDim.x; halfsize <<= 1) {
		// comparison steps
		int cur_dir = dir ^ ((tx & halfsize) != 0);
		for (int stride = halfsize; stride > 0; stride >>= 1) {
			int pos = 2 * tx - (tx & (stride - 1));
			comparator_dir( sData[pos], sData[pos + stride], cur_dir );
			__syncthreads();
		}
	}
	// bitonic merge step
	for (int stride = blockDim.x; stride > 0; stride >>= 1) {
		int pos = 2 * tx - (tx & (stride - 1));
		comparator<dir>( sData[pos], sData[pos + stride] );
		__syncthreads();
	}
	// copy from shared mem to global mem
	devPtrData[block_offset + tx] = sData[tx];
	devPtrData[block_offset + tx + blockDim.x] = sData[tx + blockDim.x];
}

// CUDA kernel function
template <int dir>
__global__ void kernel_index_based_bitonic_sort_block( int* devPtrData ) {
	__shared__ int sData[M]; // M = NUM_BLOCK_PER_CLUSTER * BLOCK_SIZE
	assert(NUM_BLOCK_PER_CLUSTER == 2);
	int block_offset = NUM_BLOCK_PER_CLUSTER * blockIdx.x * blockDim.x;
	// copy from global memory to shared mem, (2 * blockDim.x) elements
	int tx1 = threadIdx.x;
	int tx2 = threadIdx.x + blockDim.x;
	sData[tx1] = devPtrData[block_offset + tx1];
	sData[tx2] = devPtrData[block_offset + tx2];
	__syncthreads();
	// main loop
	for (int k = 2; k <= 2 * blockDim.x; k <<= 1) {
		for (int j = k / 2; j > 0; j >>= 1) {
			int ij = tx1 ^ j;
			if (tx1 < ij) {
				comparator_dir( sData[tx1], sData[ij], ((tx1 & k) == 0) ? dir : 1 - dir );
			}
			__syncthreads();
			ij = tx2 ^ j;
			if (tx2 < ij) {
				comparator_dir( sData[tx2], sData[ij], ((tx2 & k) == 0) ? dir : 1 - dir );
			}
			__syncthreads();
		}
	}
	// done, copy back the data
	devPtrData[block_offset + tx1] = sData[tx1];
	devPtrData[block_offset + tx2] = sData[tx2];
}


int main(int argc, const char* argv[]) {
	srand(time(NULL));
	printf("BLOCK_SIZE = %d\n", BLOCK_SIZE);
	printf("NUM_BLOCK = %d\n", NUM_BLOCK);
	printf("NUM_BLOCK_PER_CLUSTER = %d\n", NUM_BLOCK_PER_CLUSTER);
	printf("NUM_CLUSTER = %d\n", NUM_CLUSTER);
	printf("BOUND = %d\n", BOUND);
	printf("N = total num elem = %d\n", NUM_BLOCK * BLOCK_SIZE);
	printf("M = num elements for a TB = %d\n", NUM_BLOCK_PER_CLUSTER * BLOCK_SIZE);
	// CPU part: set values
	printf("generating %d random numbers: ", N);
	fflush(stdout);
	ptrSrc = (int*)malloc(N * sizeof(int));
	ptrData0 = (int*)malloc(N * sizeof(int));
	ptrData = (int*)malloc(N * sizeof(int));
	assert(ptrSrc != nullptr);
	setRand(N, ptrSrc, BOUND);
	printf("done\n");
	fflush(stdout);
	// CLUSTER-based processing: data = NUM_CLUSTER x CLUSTER
	printf("Src[%d][%d] = ", 0, M);
	printArray(M, ptrSrc + 0 * M);
	printf("Src[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrSrc + (NUM_CLUSTER - 1) * M);
	fflush(stdout);
	// CPU processing: C++ sort
	memcpy(ptrData0, ptrSrc, N * sizeof(int));
	steady_clock::time_point start = steady_clock::now();
	if (direction == INCREASING) {
		for (int i = 0; i < NUM_CLUSTER; ++i) {
			std::sort( ptrData0 + i * M, ptrData0 + (i + 1) * M );
		}
	} else {
		for (int i = 0; i < NUM_CLUSTER; ++i) {
			std::sort( ptrData0 + i * M, ptrData0 + (i + 1) * M, std::greater<int>() );
		}
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU std::sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData0 + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData0 + (NUM_CLUSTER - 1) * M);
	fflush(stdout);
	// CPU processing: bitonic sort
	memcpy(ptrData, ptrSrc, N * sizeof(int));
	start = steady_clock::now();
	for (int i = 0; i < NUM_CLUSTER; ++i) {
		cpu_bitonic_sort<direction>( ptrData + i * M, M );
	}
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU bitonic sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData + (NUM_CLUSTER - 1) * M);
	bool check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
	// CPU processing: index-based bitonic sort
	memcpy(ptrData, ptrSrc, N * sizeof(int));
	start = steady_clock::now();
	for (int i = 0; i < NUM_CLUSTER; ++i) {
		cpu_index_based_bitonic_sort<direction>( ptrData + i * M, M );
	}
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU index-based bitonic sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData + (NUM_CLUSTER - 1) * M);
	check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: memory allocation
	cudaMalloc((void**)&devPtrData, N * sizeof(int));
	CUDA_CHECK_ERROR();
	// host to device copy
	cudaMemcpy(devPtrData, ptrSrc, N * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// GPU kernel action: bitonic sort block
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((N + NUM_BLOCK_PER_CLUSTER * dimBlock.x - 1) / (NUM_BLOCK_PER_CLUSTER * dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG(N);
	start = steady_clock::now();
	kernel_bitonic_sort_block<direction> <<< dimGrid, dimBlock>>>( devPtrData );
	cudaDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU bitonic sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// check the result
	cudaMemcpy(ptrData, devPtrData, N * sizeof(int), cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData + (NUM_CLUSTER - 1) * M);
	check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	// host to device copy
	cudaMemcpy(devPtrData, ptrSrc, N * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// GPU kernel action: index-based bitonic sort block
	start = steady_clock::now();
	kernel_index_based_bitonic_sort_block<direction> <<< dimGrid, dimBlock>>>( devPtrData );
	cudaDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU index-based bitonic sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// check the result
	cudaMemcpy(ptrData, devPtrData, N * sizeof(int), cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData + (NUM_CLUSTER - 1) * M);
	check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: bitonic-block.cu
static const char rcsid[] __attribute__((used)) = "$Id: 12b-bitonic-block.cu,v 1.1 2026/10/17 03:28:54 wayfarecru Exp $";
