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
#if defined(__NVCC__)
const int NUM_BLOCK = 1 * 1024; // total number of blocks in a Grid
#else
const int NUM_BLOCK = 1 * 1024; // total number of blocks in a Grid
#endif
const int NUM_BLOCK_PER_CLUSTER = 2; // number of blocks per cluster
const int NUM_CLUSTER = NUM_BLOCK / NUM_BLOCK_PER_CLUSTER;
const int BOUND = 1000 * 1000; // numbers will be ranged in [0..bound)

const int N = NUM_BLOCK * BLOCK_SIZE; // total number of elements
const int M = NUM_BLOCK_PER_CLUSTER * BLOCK_SIZE; // number of elements for a TB

const int SAMPLE_STRIDE = BLOCK_SIZE / 2;

int* ptrSrc; // source data to be sorted
int* ptrData0; // data area for sorting
int* ptrData; // data area for sorting

int* devPtrData0; // device data area for sorting
int* devPtrData1; // device data area for sorting

enum {
	DECREASING = 0,
	INCREASING = 1,
};
const unsigned direction = INCREASING;
// const unsigned direction = DECREASING;


// counting merge sort, GPU

template <int dir>
__device__ inline unsigned bsearch_incl( int val, const int* pData, int bound, int stride ) {
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
__device__ inline unsigned bsearch_excl( int val, const int* pData, int bound, int stride ) {
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

template <int dir>
__global__ void kernel_merge_sort_step( int* dst, int* src, int stride ) {
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int pos = gx & (stride - 1);
	// read from src
	int* base = src + 2 * (gx - pos);
	int keyA = base[pos + 0];
	int keyB = base[pos + stride];
	int posA = bsearch_excl<dir>(keyA, base + stride, stride, stride) + pos;
	int posB = bsearch_incl<dir>(keyB, base + 0, stride, stride) + pos;
	// write to dst
	base = dst + 2 * (gx - pos);
	base[posA] = keyA;
	base[posB] = keyB;
}

// new counting merge sort, GPU

template <int dir>
__global__ void kernel_new_merge_sort_block( const int* src, int* dst ) { // src to dst !
	__shared__ int sData[M]; // M = NUM_BLOCK_PER_CLUSTER * BLOCK_SIZE
	assert(NUM_BLOCK_PER_CLUSTER == 2);
	int block_offset = NUM_BLOCK_PER_CLUSTER * blockIdx.x * blockDim.x;
	// copy from global memory to shared mem, (2 * blockDim.x) elements
	int tx = threadIdx.x;
	sData[tx] = src[block_offset + tx];
	sData[tx + blockDim.x] = src[block_offset + tx + blockDim.x];
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
	dst[block_offset + tx] = sData[tx];
	dst[block_offset + tx + blockDim.x] = sData[tx + blockDim.x];
}

template <int dir>
__global__ void kernel_get_rank( const int* pData, int* rankA, int* rankB, int segment_size ) {
	int bx = blockIdx.x; // target block number
	int tx = threadIdx.x; // sample number in a block
	int gx = tx + bx * blockDim.x;
	if (bx % 2 == 0) { // segment A
		rankA[gx] = tx * SAMPLE_STRIDE;
		const int* pSegA = pData + bx * segment_size; // start address of segment A
		const int* pSegB = pData + (bx + 1) * segment_size; // start address of segment B
		int sampleValue = pSegA[tx * SAMPLE_STRIDE];
		rankB[gx] = bsearch_excl<dir>( sampleValue, pSegB, segment_size, segment_size );
	} else { // segment B
		rankB[gx] = tx * SAMPLE_STRIDE;
		const int* pSegA = pData + (bx - 1) * segment_size; // start address of segment A
		const int* pSegB = pData + bx * segment_size; // start address of segment B
		int sampleValue = pSegB[tx * SAMPLE_STRIDE];
		rankA[gx] = bsearch_incl<dir>( sampleValue, pSegA, segment_size, segment_size );
	}
}

template <int dir>
__device__ inline void merge_intervals( int* dst, const int* srcA, const int* srcB,
                                        int lenA, int strideA, int lenB, int strideB ) {
	// assumes 'dst', 'srcA', 'srcB' have no overalps at all.
	int tx = threadIdx.x;
	int keyA, keyB, dstPosA, dstPosB;
	if (tx < lenA) {
		keyA = srcA[tx];
		dstPosA = bsearch_excl<dir>(keyA, srcB, lenB, strideB) + tx;
	}
	if (tx < lenB) {
		keyB = srcB[tx];
		dstPosB = bsearch_incl<dir>(keyB, srcA, lenA, strideA) + tx;
	}
	if (tx < lenA) {
		dst[dstPosA] = keyA;
	}
	if (tx < lenB) {
		dst[dstPosB] = keyB;
	}
}

template <int dir>
__global__ void kernel_merge_segments( const int* src, int* dst,
                                       const int* limitA, const int* limitB, int segment_size ) {
	__shared__ int s_src[2 * SAMPLE_STRIDE];
	__shared__ int s_dst[2 * SAMPLE_STRIDE];
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int gx = tx + bx * blockDim.x;
	// get the segment base address
	int segBase = (gx / (2 * segment_size)) * (2 * segment_size);
	src += segBase;
	dst += segBase;
	__shared__ int startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;
	if (threadIdx.x == 0) {
		int num_samples = (2 * segment_size / SAMPLE_STRIDE);
		startSrcA = limitA[bx];
		startSrcB = limitB[bx];
		int endSrcA = (bx % num_samples == num_samples - 1) ? segment_size : limitA[bx + 1];
		int endSrcB = (bx % num_samples == num_samples - 1) ? segment_size : limitB[bx + 1];
		lenSrcA = endSrcA - startSrcA;
		lenSrcB = endSrcB - startSrcB;
		startDstA = startSrcA + startSrcB;
		startDstB = startDstA + lenSrcA;
#if 0
		if (segment_size == 2048 && bx < 16) {
			printf("bx = %d: segBase = %d startSrcA,B=%d,%d lenA,B=%d,%d startDstA,B=%d,%d\n",
			       bx, segBase, startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB);
		}
#endif
	}
	__syncthreads();
	// get the data
	if (tx < lenSrcA) {
		s_src[tx] = src[startSrcA + tx];
	}
	if (tx < lenSrcB) {
		s_src[tx + SAMPLE_STRIDE] = src[segment_size + startSrcB + tx];
	}
	__syncthreads();
	// merge two intervals into one
	merge_intervals<dir>( s_dst, s_src, s_src + SAMPLE_STRIDE, lenSrcA, SAMPLE_STRIDE, lenSrcB, SAMPLE_STRIDE );
	__syncthreads();
	// store the data
	if (tx < lenSrcA) {
		dst[startDstA + tx] = s_dst[tx];
	}
	if (tx < lenSrcB) {
		dst[startDstB + tx] = s_dst[lenSrcA + tx];
	}
}

int* rankA = nullptr;
int* rankB = nullptr;
int* limitA = nullptr;
int* limitB = nullptr;
int* vecC = nullptr;

int* dev_rankA = nullptr;
int* dev_rankB = nullptr;
int* dev_limitA = nullptr;
int* dev_limitB = nullptr;
int* dev_vecC = nullptr;

void gpu_new_merge_sort_prepare( int TOTAL_NUM ){
	// CPU part: allocate memory
	rankA = new int[TOTAL_NUM / SAMPLE_STRIDE];
	rankB = new int[TOTAL_NUM / SAMPLE_STRIDE];
	limitA = new int[TOTAL_NUM / SAMPLE_STRIDE];
	limitB = new int[TOTAL_NUM / SAMPLE_STRIDE];
	vecC = new int[TOTAL_NUM];
	// GPU part: allocate memory
	hipMalloc( (void**)&dev_rankA, (TOTAL_NUM / SAMPLE_STRIDE) * sizeof(int) );
	hipMalloc( (void**)&dev_rankB, (TOTAL_NUM / SAMPLE_STRIDE) * sizeof(int) );
	hipMalloc( (void**)&dev_limitA, (TOTAL_NUM / SAMPLE_STRIDE) * sizeof(int) );
	hipMalloc( (void**)&dev_limitB, (TOTAL_NUM / SAMPLE_STRIDE) * sizeof(int) );
	hipMalloc( (void**)&dev_vecC, TOTAL_NUM * sizeof(int) );
	CUDA_CHECK_ERROR();
}

void gpu_new_merge_sort( int TOTAL_NUM, int* dev_vecA, int* dev_vecB ) {
	// phase 1: sort each cluster of (2 * BLOCK_SIZE) elements
	dim3 dimBlock(BLOCK_SIZE, 1, 1); // only one block
	dim3 dimGrid((N + 2 * dimBlock.x - 1) / (2 * dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG( TOTAL_NUM );
	kernel_new_merge_sort_block<direction> <<< dimGrid, dimBlock>>>( dev_vecA, dev_vecB );
#if 0 // for some compilers
	if (direction) {
		kernel_new_merge_sort_block<INCREASING> <<< dimGrid, dimBlock>>>( dev_vecA, dev_vecB );
	} else {
		kernel_new_merge_sort_block<DECREASING> <<< dimGrid, dimBlock>>>( dev_vecA, dev_vecB );
	}
#endif
	CUDA_CHECK_ERROR();
	// phase 2: 'segment_size' is size of each sorted segment
	for (int segment_size = 2 * BLOCK_SIZE; segment_size < TOTAL_NUM; segment_size <<= 1) {
		// step 1. generate rank
		dimBlock = dim3( segment_size / SAMPLE_STRIDE, 1, 1 );
		dimGrid = dim3( TOTAL_NUM / segment_size, 1, 1 );
#if 0 // only for debug
		CUDA_PRINT_CONFIG( TOTAL_NUM );
		printf("rank: dimGrid  = %d * %d * %d, dimBlock = %d * %d * %d\n",
		         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
#endif
		kernel_get_rank<direction> <<< dimGrid, dimBlock>>>( dev_vecB, dev_rankA, dev_rankB, segment_size );
#if 0 // for some compilers
		if (direction) {
			kernel_get_rank<INCREASING> <<< dimGrid, dimBlock>>>( dev_vecB, dev_rankA, dev_rankB, segment_size );
		} else {
			kernel_get_rank<DECREASING> <<< dimGrid, dimBlock>>>( dev_vecB, dev_rankA, dev_rankB, segment_size );
		}
#endif
		CUDA_CHECK_ERROR();
#if 0 // only for debug
		hipMemcpy( rankA, dev_rankA, (segment_size / SAMPLE_STRIDE) * sizeof(unsigned), hipMemcpyDeviceToHost );
		hipMemcpy( rankB, dev_rankB, (segment_size / SAMPLE_STRIDE) * sizeof(unsigned), hipMemcpyDeviceToHost );
		printf("rankA = ");
		printArray(segment_size / SAMPLE_STRIDE, (int*)rankA);
		printf("rankB = ");
		printArray(segment_size / SAMPLE_STRIDE, (int*)rankB);
#endif
		// step 2. calculate limit: actually sort two ranks as a cluster
		dimBlock = dim3( segment_size / SAMPLE_STRIDE, 1, 1 ); // merge sort gets two segments
		dimGrid = dim3( TOTAL_NUM / (2 * segment_size), 1, 1 );
#if 0 // only for debug
		CUDA_PRINT_CONFIG( TOTAL_NUM );
		printf("limit: dimGrid  = %d * %d * %d, dimBlock = %d * %d * %d\n",
		         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
#endif
		kernel_new_merge_sort_block<direction> <<< dimGrid, dimBlock>>>( dev_rankA, dev_limitA );
		kernel_new_merge_sort_block<direction> <<< dimGrid, dimBlock>>>( dev_rankB, dev_limitB );
#if 0 // for some compilers
		if (direction) {
			kernel_new_merge_sort_block<INCREASING> <<< dimGrid, dimBlock>>>( dev_rankA, dev_limitA );
			kernel_new_merge_sort_block<INCREASING> <<< dimGrid, dimBlock>>>( dev_rankB, dev_limitB );
		} else {
			kernel_new_merge_sort_block<DECREASING> <<< dimGrid, dimBlock>>>( dev_rankA, dev_limitA );
			kernel_new_merge_sort_block<DECREASING> <<< dimGrid, dimBlock>>>( dev_rankB, dev_limitB );
		}
#endif
#if 0 // only for debug
		hipMemcpy( limitA, dev_limitA, (segment_size / SAMPLE_STRIDE) * sizeof(unsigned), hipMemcpyDeviceToHost );
		hipMemcpy( limitB, dev_limitB, (segment_size / SAMPLE_STRIDE) * sizeof(unsigned), hipMemcpyDeviceToHost );
		printf("limitA = ");
		printArray(segment_size / SAMPLE_STRIDE, (int*)limitA);
		printf("limitB = ");
		printArray(segment_size / SAMPLE_STRIDE, (int*)limitB);
#endif
		// step 3. merge two segments
		int* src = dev_vecB;
		int* dst = dev_vecC;
		dimBlock = dim3( SAMPLE_STRIDE, 1, 1 ); // merge gets two segments
		dimGrid = dim3( TOTAL_NUM / (1 * SAMPLE_STRIDE), 1, 1 );
#if 0 // only for debug
		CUDA_PRINT_CONFIG( TOTAL_NUM );
		printf("merge: dimGrid  = %d * %d * %d, dimBlock = %d * %d * %d\n",
		         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
#endif
		kernel_merge_segments<direction> <<< dimGrid, dimBlock>>>( src, dst, dev_limitA, dev_limitB, segment_size );
#if 0 // for some compilers
		if (direction) {
			kernel_merge_segments<INCREASING> <<< dimGrid, dimBlock>>>( src, dst, dev_limitA, dev_limitB, segment_size );
		} else {
			kernel_merge_segments<DECREASING> <<< dimGrid, dimBlock>>>( src, dst, dev_limitA, dev_limitB, segment_size );
		}
#endif
#if 0 // only for debug
		hipMemcpy( vecC, dev_vecC, TOTAL_NUM * sizeof(unsigned), hipMemcpyDeviceToHost );
#endif
		hipMemcpy( dev_vecB, dev_vecC, TOTAL_NUM * sizeof(unsigned), hipMemcpyDeviceToDevice );
		// partially done
		hipDeviceSynchronize();
		CUDA_CHECK_ERROR();
	}
	// result is in dev_vecB
}

void gpu_new_merge_sort_release( void ) {
	hipFree( dev_rankA );
	hipFree( dev_rankB );
	hipFree( dev_limitA );
	hipFree( dev_limitB );
	hipFree( dev_vecC );
	CUDA_CHECK_ERROR();
	delete[] rankA;
	delete[] rankB;
	delete[] limitA;
	delete[] limitB;
}

int main(const int argc, const char* argv[]) {
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
	printf("Src[%d] = ", N);
	printArray(N, ptrSrc);
	fflush(stdout);
	// CPU processing: C++ sort
	memcpy(ptrData0, ptrSrc, N * sizeof(int));
	steady_clock::time_point start = steady_clock::now();
	if (direction == INCREASING) {
		std::sort( ptrData0, ptrData0 + N );
	} else {
		std::sort( ptrData0, ptrData0 + N, std::greater<int>() );
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU std::sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d] = ", N);
	printArray(N, ptrData0);
	fflush(stdout);
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: memory allocation
	hipMalloc((void**)&devPtrData0, N * sizeof(int));
	hipMalloc((void**)&devPtrData1, N * sizeof(int));
	CUDA_CHECK_ERROR();
	// host to device copy
	hipMemcpy(devPtrData0, ptrSrc, N * sizeof(int), hipMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// GPU kernel action: counting merge sort, naive
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((N + NUM_BLOCK_PER_CLUSTER * dimBlock.x - 1) / (NUM_BLOCK_PER_CLUSTER * dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG(N);
	int dst = 0;
	start = steady_clock::now();
	for (int stride = 1; stride < N; stride <<= 1) {
		dst = 1 - dst; // current destination buffer
		if (dst == 1) {
			kernel_merge_sort_step<direction> <<< dimGrid, dimBlock>>>( devPtrData1, devPtrData0, stride );
		} else {
			kernel_merge_sort_step<direction> <<< dimGrid, dimBlock>>>( devPtrData0, devPtrData1, stride );
		}
	}
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU counting merge sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// check the result
	if (dst == 0) {
		hipMemcpy(ptrData, devPtrData0, N * sizeof(int), hipMemcpyDeviceToHost);
	} else {
		hipMemcpy(ptrData, devPtrData1, N * sizeof(int), hipMemcpyDeviceToHost);
	}
	CUDA_CHECK_ERROR();
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
	bool check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
	// host to device copy
	hipMemcpy(devPtrData0, ptrSrc, N * sizeof(int), hipMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// GPU kernel action: counting merge sort
	start = steady_clock::now();
	// phase 1: sort each cluster of (2 * BLOCK_SIZE) elements
	kernel_merge_sort_block<direction> <<< dimGrid, dimBlock>>>( devPtrData0 );
	// pahse 2: merge on the global memory
	dst = 0;
	for (int stride = M; stride < N; stride <<= 1) {
		dst = 1 - dst; // current destination buffer
		if (dst == 1) {
			kernel_merge_sort_step<direction> <<< dimGrid, dimBlock>>>( devPtrData1, devPtrData0, stride );
		} else {
			kernel_merge_sort_step<direction> <<< dimGrid, dimBlock>>>( devPtrData0, devPtrData1, stride );
		}
	}
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU counting merge sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// check the result
	if (dst == 0) {
		hipMemcpy(ptrData, devPtrData0, N * sizeof(int), hipMemcpyDeviceToHost);
	} else {
		hipMemcpy(ptrData, devPtrData1, N * sizeof(int), hipMemcpyDeviceToHost);
	}
	CUDA_CHECK_ERROR();
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
	check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
	// host to device copy
	hipMemcpy(devPtrData0, ptrSrc, N * sizeof(int), hipMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// GPU kernel action: new counting merge sort
	gpu_new_merge_sort_prepare( N );
	start = steady_clock::now();
	gpu_new_merge_sort( N, devPtrData0, devPtrData1 );
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU new counting merge sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	gpu_new_merge_sort_release();
	// check the result
	hipMemcpy(ptrData, devPtrData1, N * sizeof(int), hipMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
	check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: new-merge.cu
static const char rcsid[] __attribute__((used)) = "$Id: 13c-new-merge.cu,v 1.1 2026/10/17 03:29:23 wayfarecru Exp $";
