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
int* ptrData; // data area for sorting

int* devPtrData; // device data area for sorting

enum {
	DECREASING = 0,
	INCREASING = 1,
};
const unsigned direction = INCREASING;
// const unsigned direction = DECREASING;


// compare functions for qsort

int compLess( const void* lhs, const void* rhs ) {
	int lval = *(static_cast<const int*>(lhs));
	int rval = *(static_cast<const int*>(rhs));
	return (lval - rval);
}

int compGreater( const void* lhs, const void* rhs ) {
	int lval = *(static_cast<const int*>(lhs));
	int rval = *(static_cast<const int*>(rhs));
	return (rval - lval);
}

// bubble sort, block-size, CPU

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

template <unsigned dir>
void cpu_bubble_sort( int* data, int num ) {
	for (int i = 0; i < num - 1; ++i) {
		for (int j = 0; j < num - i - 1; ++j) {
			comparator<dir>( data[j], data[j + 1] );
		}
	}
}

// even-odd sort, block-size, CPU

template <unsigned dir>
void cpu_even_odd_sort( int* data, int num ) {
	for (int i = 0; i < num / 2; ++i) {
		// even phase
		for (int j = 0; j < num - 1; j += 2) {
			comparator<dir>( data[j], data[j + 1] );
		}
		// odd phase
		for (int j = 1; j < num - 1; j += 2) {
			comparator<dir>( data[j], data[j + 1] );
		}
	}
}

// even-odd sort, block-size, GPU

template <unsigned dir>
__global__ void kernel_even_odd_sort_block( int* devPtrData ) {
	__shared__ int sData[M]; // M = NUM_BLOCK_PER_CLUSTER * BLOCK_SIZE
	assert(NUM_BLOCK_PER_CLUSTER == 2);
	int block_offset = NUM_BLOCK_PER_CLUSTER * blockIdx.x * blockDim.x;
	// copy from global memory to shared mem, (2 * blockDim.x) elements
	int tx = threadIdx.x;
	sData[tx] = devPtrData[block_offset + tx];
	sData[tx + blockDim.x] = devPtrData[block_offset + tx + blockDim.x];
	__syncthreads();
	// outer-loop
	for (int i = 0; i < blockDim.x; ++i) {
		// even phase
		comparator<dir>( sData[2 * tx], sData[2 * tx + 1] );
		__syncthreads();
		// odd phase
		if (tx != 0) {
			comparator<dir>( sData[2 * tx - 1], sData[2 * tx] );
		}
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
	// CPU processing: qsort in C
	memcpy(ptrData, ptrSrc, N * sizeof(int));
	int (*compar)(const void*, const void*) = (direction == INCREASING) ? compLess : compGreater;
	steady_clock::time_point start = steady_clock::now();
	for (int i = 0; i < NUM_CLUSTER; ++i) {
		qsort( ptrData + i * M, M, sizeof(int), compar );
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU qsort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData + (NUM_CLUSTER - 1) * M);
	fflush(stdout);
	// CPU processing: C++ sort
	memcpy(ptrData, ptrSrc, N * sizeof(int));
	start = steady_clock::now();
	if (direction == INCREASING) {
		for (int i = 0; i < NUM_CLUSTER; ++i) {
			std::sort( ptrData + i * M, ptrData + (i + 1) * M );
		}
	} else {
		for (int i = 0; i < NUM_CLUSTER; ++i) {
			std::sort( ptrData + i * M, ptrData + (i + 1) * M, std::greater<int>() );
		}
	}
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU std::sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData + (NUM_CLUSTER - 1) * M);
	fflush(stdout);
	// CPU processing: bubble sort
	memcpy(ptrData, ptrSrc, N * sizeof(int));
	start = steady_clock::now();
	for (int i = 0; i < NUM_CLUSTER; ++i) {
		cpu_bubble_sort<direction>( ptrData + i * M, M );
	}
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU bubble sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData + (NUM_CLUSTER - 1) * M);
	fflush(stdout);
	// CPU processing: even-odd sort
	memcpy(ptrData, ptrSrc, N * sizeof(int));
	start = steady_clock::now();
	for (int i = 0; i < NUM_CLUSTER; ++i) {
		cpu_even_odd_sort<direction>( ptrData + i * M, M );
	}
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU even-odd sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData + (NUM_CLUSTER - 1) * M);
	fflush(stdout);
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: memory allocation
	hipMalloc((void**)&devPtrData, N * sizeof(int));
	CUDA_CHECK_ERROR();
	// host to device copy
	hipMemcpy(devPtrData, ptrSrc, N * sizeof(int), hipMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// GPU kernel action: even-odd sort
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((N + NUM_BLOCK_PER_CLUSTER * dimBlock.x - 1) / (NUM_BLOCK_PER_CLUSTER * dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG(N);
	start = steady_clock::now();
	kernel_even_odd_sort_block<direction> <<< dimGrid, dimBlock>>>( devPtrData );
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU even-odd sort for block: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// check the result
	hipMemcpy(ptrData, devPtrData, N * sizeof(int), hipMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();
	printf("Data[%d][%d] = ", 0, M);
	printArray(M, ptrData + 0 * M);
	printf("Data[%d][%d] = ", (NUM_CLUSTER - 1), M);
	printArray(M, ptrData + (NUM_CLUSTER - 1) * M);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: even-odd-block.cu
static const char rcsid[] __attribute__((used)) = "$Id: 12a-even-odd-block.cu,v 1.1 2026/10/17 03:28:54 wayfarecru Exp $";
