#include "hip/hip_runtime.h"
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

// merge sort, increasing-order only, block-size, CPU

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

void cpu_merge( int* pData, int num ) { // increasing order only
	static int buf[2 * BLOCK_SIZE];
	int half = num / 2;
	int* lhs = pData;
	int* rhs = pData + half;
	int i = 0;
	int j = 0;
	int k = 0;
	while (i < half && j < half) {
		if (lhs[i] <= rhs[j]) {
			buf[k++] = lhs[i++];
		} else {
			buf[k++] = rhs[j++];
		}
	}
	while (i < half) {
		buf[k++] = lhs[i++];
	}
	while (j < half) {
		buf[k++] = rhs[j++];
	}
	memcpy( pData, buf, num * sizeof(int) );
}

void cpu_merge_sort( int* pData, int num ) { // increasing order only
	if (num > 1) {
		int half = num / 2;
		cpu_merge_sort( pData, half );
		cpu_merge_sort( pData + half, half );
		cpu_merge( pData, num );
	}
}

// counting merge sort, increasing-order only, block-size, GPU

__device__ inline int bsearch_inclusive( int val, const int* pData, int bound, int stride ) {
	if (bound == 0) {
		return 0;
	}
	int pos = 0;
	for (; stride > 0; stride >>= 1) {
		int newPos = min(pos + stride, bound);
		if (pData[newPos - 1] <= val) {
			pos = newPos;
		}
	}
	return pos;
}

__device__ inline int bsearch_exclusive( int val, const int* pData, int bound, int stride ) {
	if (bound == 0) {
		return 0;
	}
	int pos = 0;
	for (; stride > 0; stride >>= 1) {
		int newPos = min(pos + stride, bound);
		if (pData[newPos - 1] < val) {
			pos = newPos;
		}
	}
	return pos;
}

// CUDA kernel function
__global__ void kernel_count_merge_sort_block( int* devPtrData ) {
	__shared__ int sData[M]; // M = NUM_BLOCK_PER_CLUSTER * BLOCK_SIZE
	assert(NUM_BLOCK_PER_CLUSTER == 2);
	int block_offset = NUM_BLOCK_PER_CLUSTER * blockIdx.x * blockDim.x;
	// copy from global memory to shared mem, (2 * blockDim.x) elements
	int tx = threadIdx.x;
	sData[tx] = devPtrData[block_offset + tx];
	sData[tx + blockDim.x] = devPtrData[block_offset + tx + blockDim.x];
	__syncthreads();
	// merge sort main loop
	for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {
		int pos = tx & (stride - 1);
		int* base = sData + 2 * (tx - pos);
		int keyA = base[pos + 0];
		int keyB = base[pos + stride];
		int posA = bsearch_exclusive(keyA, base + stride, stride, stride) + pos;
		int posB = bsearch_inclusive(keyB, base + 0, stride, stride) + pos;
		__syncthreads();
		base[posA] = keyA;
		base[posB] = keyB;
		__syncthreads();
	}
	// copy from shared mem to global mem
	devPtrData[block_offset + tx] = sData[tx];
	devPtrData[block_offset + tx + blockDim.x] = sData[tx + blockDim.x];
}

// counting merge sort, block-size, GPU

template <int dir>
__device__ inline int bsearch_incl( int val, const int* pData, int bound, int stride ) {
	if (bound == 0) {
		return 0;
	}
	int pos = 0;
	for (; stride > 0; stride >>= 1) {
		int newPos = min(pos + stride, bound);
		if ((dir && (pData[newPos - 1] <= val)) || (!dir && (pData[newPos - 1] >= val))) {
			pos = newPos;
		}
	}
	return pos;
}

template <int dir>
__device__ inline int bsearch_excl( int val, const int* pData, int bound, int stride ) {
	if (bound == 0) {
		return 0;
	}
	int pos = 0;
	for (; stride > 0; stride >>= 1) {
		int newPos = min(pos + stride, bound);
		if ((dir && (pData[newPos - 1] < val)) || (!dir && (pData[newPos - 1] > val))) {
			pos = newPos;
		}
	}
	return pos;
}

// CUDA kernel function
template <int dir>
__global__ void kernel_merge_sort_block( int* devPtrData ) {
	__shared__ int sData[M]; // M = NUM_BLOCK_PER_CLUSTER * BLOCK_SIZE
	assert(NUM_BLOCK_PER_CLUSTER == 2);
	int block_offset = NUM_BLOCK_PER_CLUSTER * blockIdx.x * blockDim.x;
	// copy from global memory to shared mem, (2 * blockDim.x) elements
	int tx = threadIdx.x;
	sData[tx] = devPtrData[block_offset + tx];
	sData[tx + blockDim.x] = devPtrData[block_offset + tx + blockDim.x];
	__syncthreads();
	// merge sort main loop
	for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {
		int pos = tx & (stride - 1);
		int* base = sData + 2 * (tx - pos);
		int keyA = base[pos + 0];
		int keyB = base[pos + stride];
		int posA = bsearch_excl<dir>(keyA, base + stride, stride, stride) + pos;
		int posB = bsearch_incl<dir>(keyB, base + 0, stride, stride) + pos;
		__syncthreads();
		base[posA] = keyA;
		base[posB] = keyB;
		__syncthreads();
	}
	// copy from shared mem to global mem
	devPtrData[block_offset + tx] = sData[tx];
	devPtrData[block_offset + tx + blockDim.x] = sData[tx + blockDim.x];
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
	// CPU processing: merge sort
	memcpy(ptrData, ptrSrc, N * sizeof(int));
	start = steady_clock::now();
	for (int i = 0; i < NUM_CLUSTER; ++i) {
		cpu_merge_sort( ptrData + i * M, M );
	}
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU merge sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData + (NUM_CLUSTER - 1) * M);
	bool check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: memory allocation
	hipMalloc((void**)&devPtrData, N * sizeof(int));
	CUDA_CHECK_ERROR();
	// host to device copy
	hipMemcpy(devPtrData, ptrSrc, N * sizeof(int), hipMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// GPU kernel action: counting merge sort block
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((N + NUM_BLOCK_PER_CLUSTER * dimBlock.x - 1) / (NUM_BLOCK_PER_CLUSTER * dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG(N);
	start = steady_clock::now();
	kernel_count_merge_sort_block <<< dimGrid, dimBlock>>>( devPtrData );
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU counting merge sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// check the result
	hipMemcpy(ptrData, devPtrData, N * sizeof(int), hipMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData + (NUM_CLUSTER - 1) * M);
	check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	// host to device copy
	hipMemcpy(devPtrData, ptrSrc, N * sizeof(int), hipMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// GPU kernel action: counting merge sort block
	start = steady_clock::now();
	kernel_merge_sort_block<direction> <<< dimGrid, dimBlock>>>( devPtrData );
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU new counting merge sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// check the result
	hipMemcpy(ptrData, devPtrData, N * sizeof(int), hipMemcpyDeviceToHost);
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

// HISTORY: count-merge-block.cu
static const char rcsid[] __attribute__((used)) = "$Id: 12c-count-merge-block.cu,v 1.1 2026/10/17 03:28:54 wayfarecru Exp $";
