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

// global even-odd sort, CPU

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

// global even-odd sort, GPU

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

// bitonic sort, CPU

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
		cpu_bitonic_sort < 1 - dir > ( pData + half, half ); // reverse direction
		cpu_bitonic_merge<dir>( pData, num );
	}
}

// bitonic sort, GPU

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

__global__ void kernel_bitonic_first_phase_step( int* devPtrData, int halfsize, int stride, int dir) {
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int cur_dir = dir ^ ((gx & halfsize) != 0);
	int pos = 2 * gx - (gx & (stride - 1));
	comparator_dir( devPtrData[pos], devPtrData[pos + stride], cur_dir );
}

__global__ void kernel_bitonic_second_phase_step( int* devPtrData, int stride, int dir) {
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int pos = 2 * gx - (gx & (stride - 1));
	comparator_dir( devPtrData[pos], devPtrData[pos + stride], dir );
}

// merge sort, CPU

template <class TYPE, class Compare>
void cpu_merge( TYPE* pData, int num, Compare comp ) {
	static TYPE buf[N]; // big size buffer for N
	int half = num / 2;
	TYPE* lhs = pData;
	TYPE* rhs = pData + half;
	int i = 0;
	int j = 0;
	int k = 0;
	while (i < half && j < half) {
		if (comp(lhs[i], rhs[j])) {
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
	memcpy( pData, buf, num * sizeof(TYPE) );
}

template <class TYPE, class Compare>
void cpu_merge_sort( TYPE* pData, int num, Compare comp ) {
	if (num > 1) {
		int half = num / 2;
		cpu_merge_sort( pData, half, comp );
		cpu_merge_sort( pData + half, half, comp );
		cpu_merge( pData, num, comp );
	}
}

template <class TYPE, int dir>
void cpu_merge_sort( TYPE* pData, int num ) {
	if (dir == INCREASING) {
		cpu_merge_sort( pData, num, std::less_equal<TYPE>());
	} else {
		cpu_merge_sort( pData, num, std::greater_equal<TYPE>());
	}
}

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
	printf("Src[%d] = ", N);
	printArray(N, ptrSrc);
	fflush(stdout);
	// CPU processing: qsort in C
	memcpy(ptrData0, ptrSrc, N * sizeof(int));
	int (*compar)(const void*, const void*) = (direction == INCREASING) ? compLess : compGreater;
	steady_clock::time_point start = steady_clock::now();
	qsort( ptrData0, N, sizeof(int), compar );
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU qsort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d] = ", N);
	printArray(N, ptrData0);
	fflush(stdout);
	// CPU processing: C++ sort
	memcpy(ptrData, ptrSrc, N * sizeof(int));
	start = steady_clock::now();
	if (direction == INCREASING) {
		std::sort( ptrData, ptrData + N );
	} else {
		std::sort( ptrData, ptrData + N, std::greater<int>() );
	}
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU std::sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
	bool check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
#if 0
	// CPU processing: even-odd sort
	memcpy(ptrData, ptrSrc, N * sizeof(int));
	start = steady_clock::now();
	cpu_even_odd_sort<direction>( ptrData, N );
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU even-odd sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
	check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
#endif
	// CPU processing: bitonic sort
	memcpy(ptrData, ptrSrc, N * sizeof(int));
	start = steady_clock::now();
	cpu_bitonic_sort<direction>( ptrData, N );
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU bitonic sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
	check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
	// CPU processing: merge sort
	memcpy(ptrData, ptrSrc, N * sizeof(int));
	start = steady_clock::now();
	cpu_merge_sort<int, direction>( ptrData, N );
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU merge sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
	check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: memory allocation
	cudaMalloc((void**)&devPtrData0, N * sizeof(int));
	cudaMalloc((void**)&devPtrData1, N * sizeof(int));
	CUDA_CHECK_ERROR();
	// host to device copy
	cudaMemcpy(devPtrData0, ptrSrc, N * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// GPU kernel action: even-odd sort
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((N + NUM_BLOCK_PER_CLUSTER * dimBlock.x - 1) / (NUM_BLOCK_PER_CLUSTER * dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG(N);
	start = steady_clock::now();
	// phase 1: sort each cluster of (2 * BLOCK_SIZE) elements
	kernel_even_odd_sort_block<direction> <<< dimGrid, dimBlock>>>( devPtrData0 );
	// pahse 2: merge on the global memory
	for (int i = M / 2; i < N / 2; ++i) {
		kernel_even_odd_sort_even_phase<direction> <<< dimGrid, dimBlock>>>( devPtrData0, N );
		kernel_even_odd_sort_odd_phase<direction> <<< dimGrid, dimBlock>>>( devPtrData0, N );
	}
	cudaDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU even-odd sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// check the result
	cudaMemcpy(ptrData, devPtrData0, N * sizeof(int), cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
	check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
	// host to device copy
	cudaMemcpy(devPtrData0, ptrSrc, N * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// GPU kernel action: bitonic sort
	start = steady_clock::now();
	// phase 1: sort each cluster of (2 * BLOCK_SIZE) elements
	kernel_bitonic_sort_block<direction> <<< dimGrid, dimBlock>>>( devPtrData0 );
	// pahse 2: merge on the global memory
	int dst = 0;
	for (int stride = M / 2; stride < N; stride <<= 1) {
		dst = 1 - dst; // current destination buffer
		if (dst == 1) {
			kernel_merge_sort_step<direction> <<< dimGrid, dimBlock>>>( devPtrData1, devPtrData0, stride );
		} else {
			kernel_merge_sort_step<direction> <<< dimGrid, dimBlock>>>( devPtrData0, devPtrData1, stride );
		}
	}
	cudaDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU bitonic sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// check the result
	cudaMemcpy(ptrData, devPtrData0, N * sizeof(int), cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();
	printf("Data[%d] = ", N);
	printArray(N, ptrData);
	check = std::equal(ptrData0, ptrData0 + N, ptrData);
	printf("check difference: %s\n", (check ? "OK" : "ERROR"));
	fflush(stdout);
	// host to device copy
	cudaMemcpy(devPtrData0, ptrSrc, N * sizeof(int), cudaMemcpyHostToDevice);
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
	cudaDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU counting merge sort: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// check the result
	if (dst == 0) {
		cudaMemcpy(ptrData, devPtrData0, N * sizeof(int), cudaMemcpyDeviceToHost);
	} else {
		cudaMemcpy(ptrData, devPtrData1, N * sizeof(int), cudaMemcpyDeviceToHost);
	}
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

// HISTORY: global-two-layer.cu
static const char rcsid[] __attribute__((used)) = "$Id: 13b-global-two-layer.cu,v 1.1 2026/10/17 03:29:23 wayfarecru Exp $";
