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
const int NUM_BLOCK = 512; // total number of blocks in a Grid
const int BOUND = 1000 * 1000; // numbers will be ranged in [0..bound)

const int N = NUM_BLOCK * BLOCK_SIZE; // total number of elements

int* ptrSrc; // source data to be sorted
int* ptrData; // data area for sorting

int* devPtrData; // device data area for sorting

enum {
	DECREASING = 0,
	INCREASING = 1,
};
const unsigned direction = INCREASING;
// const unsigned direction = DECREASING;


// global even-odd sort, GPU

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
__global__ void kernel_even_odd_sort_even_phase( int* devPtrData, int N ) {
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	if (2 * gx + 1 < N) {
		comparator<dir>( devPtrData[2 * gx], devPtrData[2 * gx + 1] );
	}
}

template <int dir>
__global__ void kernel_even_odd_sort_odd_phase( int* devPtrData, int N ) {
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gx != 0 && 2 * gx < N) {
		comparator<dir>( devPtrData[2 * gx - 1], devPtrData[2 * gx] );
	}
}

void printFunc(void* data) {
	printf("%s:\n", (char*)data);
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
}

int main(int argc, const char* argv[]) {
	srand(time(NULL));
	printf("BLOCK_SIZE = %d\n", BLOCK_SIZE);
	printf("NUM_BLOCK = %d\n", NUM_BLOCK);
	printf("BOUND = %d\n", BOUND);
	printf("N = total num elem = %d\n", NUM_BLOCK * BLOCK_SIZE);
	// GPU part: warm up
	CUDA_WARM_UP();
	// CPU part: set values
	printf("generating %d random numbers: ", N);
	fflush(stdout);
	hipHostMalloc( &ptrSrc, N * sizeof(int) );
	hipHostMalloc( &ptrData, N * sizeof(int) );
	assert(ptrSrc != nullptr);
	assert(ptrData != nullptr);
	setRand(N, ptrSrc, BOUND);
	printf("done\n");
	printf("Src[%d] = ", N);
	printArray(N, ptrSrc);
	memset(ptrData, 0, N * sizeof(int));
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
	fflush(stdout);
	// prepare a stream
	hipStream_t stream1;
	hipStreamCreate( &stream1 );
	// GPU part: memory allocation
	hipMalloc((void**)&devPtrData, N * sizeof(int));
	CUDA_CHECK_ERROR();
	// GPU kernel configure
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
	CUDA_PRINT_CONFIG(N);
	// host to device copy
	printf("before start: Data[%d] = %d\n", N - 1, ptrData[N - 1]);
	printf("start using stream1 ***\n");
	hipMemcpyAsync(devPtrData, ptrSrc, N * sizeof(int), hipMemcpyHostToDevice, stream1);
	// GPU kernel action: even-odd sort
	steady_clock::time_point start = steady_clock::now();
	for (int i = 0; i < N / 2; ++i) {
		kernel_even_odd_sort_even_phase<direction> <<< dimGrid, dimBlock, 0, stream1>>>( devPtrData, N );
		kernel_even_odd_sort_odd_phase<direction> <<< dimGrid, dimBlock, 0, stream1>>>( devPtrData, N );
	}
	steady_clock::time_point end = steady_clock::now();
	hipMemcpyAsync(ptrData, devPtrData, N * sizeof(int), hipMemcpyDeviceToHost, stream1);
	printf("after memcpy: Data[%d] = %d\n", N - 1, ptrData[N - 1]);
	hipLaunchHostFunc(stream1, printFunc, (void*)"stream1 host func");
	hipStreamSynchronize(stream1);
	printf("after stream sync: Data[%d] = %d\n", N - 1, ptrData[N - 1]);
	printf("end using stream1 ***\n");
	// cleaning
	hipStreamDestroy(stream1);
	CUDA_CHECK_ERROR();
	// show the results
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
	fflush(stdout);
	// print out
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU even-odd sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: stream.cu
static const char rcsid[] __attribute__((used)) = "$Id: 14a-stream.cu,v 1.1 2026/10/17 03:30:48 wayfarecru Exp $";
