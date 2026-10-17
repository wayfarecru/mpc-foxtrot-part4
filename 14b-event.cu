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

int* ptrSrc0; // source data to be sorted
int* ptrSrc1; // source data to be sorted
int* ptrData0; // data area for sorting
int* ptrData1; // data area for sorting

int* devPtrData0; // device data area for sorting
int* devPtrData1; // device data area for sorting

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
	cudaMallocHost( &ptrSrc0, N * sizeof(int) );
	cudaMallocHost( &ptrSrc1, N * sizeof(int) );
	cudaMallocHost( &ptrData0, N * sizeof(int) );
	cudaMallocHost( &ptrData1, N * sizeof(int) );
	assert(ptrSrc0 != nullptr);
	assert(ptrSrc1 != nullptr);
	assert(ptrData0 != nullptr);
	assert(ptrData1 != nullptr);
	setRand(N, ptrSrc0, BOUND);
	setRand(N, ptrSrc1, BOUND);
	printf("done\n");
	printf("Src0[%d] = ", N);
	printArray(N, ptrSrc0);
	printf("Src1[%d] = ", N);
	printArray(N, ptrSrc1);
	memset(ptrData0, 0, N * sizeof(int));
	memset(ptrData1, 0, N * sizeof(int));
	printf("Data0[%d] = ", N);
	printArray(N, ptrData0);
	printf("Data1[%d] = ", N);
	printArray(N, ptrData1);
	fflush(stdout);
	// prepare a stream
	cudaStream_t stream1; // processing stream
	cudaStream_t stream2; // copy-back stream
	cudaStreamCreate( &stream1 );
	cudaStreamCreate( &stream2 );
	// prepare events
	cudaEvent_t event0;
	cudaEvent_t event1;
	cudaEvent_t event2;
	cudaEventCreate( &event0 );
	cudaEventCreate( &event1 );
	cudaEventCreate( &event2 );
	// GPU part: memory allocation
	cudaMalloc((void**)&devPtrData0, N * sizeof(int));
	cudaMalloc((void**)&devPtrData1, N * sizeof(int));
	CUDA_CHECK_ERROR();
	// GPU kernel configure
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
	CUDA_PRINT_CONFIG(N);
	printf("start using stream1 ***\n");
	// GPU kernel action: even-odd sort
	cudaMemcpyAsync(devPtrData0, ptrSrc0, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
	cudaEventRecord(event0, stream1);
	for (int i = 0; i < N / 2; ++i) {
		kernel_even_odd_sort_even_phase<direction> <<< dimGrid, dimBlock, 0, stream1>>>( devPtrData0, N );
		kernel_even_odd_sort_odd_phase<direction> <<< dimGrid, dimBlock, 0, stream1>>>( devPtrData0, N );
	}
	cudaEventRecord(event1, stream1);
	// GPU kernel action: even-odd sort, again
	cudaMemcpyAsync(devPtrData1, ptrSrc1, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
	for (int i = 0; i < N / 2; ++i) {
		kernel_even_odd_sort_even_phase<direction> <<< dimGrid, dimBlock, 0, stream1>>>( devPtrData1, N );
		kernel_even_odd_sort_odd_phase<direction> <<< dimGrid, dimBlock, 0, stream1>>>( devPtrData1, N );
	}
	cudaEventRecord(event2, stream1);
	// event1, event2 processing
	bool startCopy1 = false;
	bool startCopy2 = false;
	while ((!startCopy1) || (!startCopy2)) {
		if ((!startCopy1) && (cudaEventQuery(event1) == cudaSuccess)) {
			cudaMemcpyAsync(ptrData0, devPtrData0, N * sizeof(int), cudaMemcpyDeviceToHost, stream2);
			startCopy1 = true;
		}
		if ((!startCopy2) && (cudaEventQuery(event2) == cudaSuccess)) {
			cudaMemcpyAsync(ptrData1, devPtrData1, N * sizeof(int), cudaMemcpyDeviceToHost, stream2);
			startCopy2 = true;
		}
	}
	// GPU kernel done
	cudaStreamSynchronize(stream1);
	printf("end using stream1 ***\n");
	cudaStreamSynchronize(stream2);
	// show the elapsed time
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, event0, event1 );
	printf("elasped time between event 0 and event 1: %f msec\n", elapsedTime);
	cudaEventElapsedTime( &elapsedTime, event1, event2 );
	printf("elasped time between event 1 and event 2: %f msec\n", elapsedTime);
	// cleaning
	cudaEventDestroy(event0);
	cudaEventDestroy(event1);
	cudaEventDestroy(event2);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	CUDA_CHECK_ERROR();
	// show the results
	printf("Data0[%d] = ", N);
	printArray(N, ptrData0);
	printf("Data1[%d] = ", N);
	printArray(N, ptrData1);
	fflush(stdout);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: event.cu
static const char rcsid[] __attribute__((used)) = "$Id: 14b-event.cu,v 1.1 2026/10/17 03:30:48 wayfarecru Exp $";
