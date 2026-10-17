#include "hip/hip_runtime.h"
#include <cstdio>
#include <chrono>
#include <cassert>
using namespace std;
using namespace std::chrono;

#if defined(__NVCC__)
#include <hipblas.h>
#endif

#if defined(__HIP__)
#include <hipblas/hipblas.h>
#endif

#include "./common.hpp"

int N = 16 * 1024 * 1024;

float* ptrA; // host memory (main memory)
float* ptrX; // host memory (main memory)
float* ptrY; // host memory (main memory)
float* ptrZ0; // host memory (main memory)
float* ptrZ; // host memory (main memory)

float* devPtrA; // device memory (graphics memory)
float* devPtrX; // device memory (graphics memory)
float* devPtrY; // device memory (graphics memory)
float* devPtrZ; // device memory (graphics memory)

// CPU kernel function
__host__ void kernel_cpu_axpy(int idx, float* Z, const float* a, const float* X, const float* Y) {
	int i = idx;
	Z[i] = (*a) * X[i] + Y[i];
}

// GPU kernel function
__global__ void kernel_axpy(float* Z, const float* a, const float* X, const float* Y, const int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		Z[i] = (*a) * X[i] + Y[i];
	}
}

int main(int argc, const char* argv[]) {
	srand(time(NULL));
	N = parseNum(argc, argv, N);
	printf("N = %d\n", N);
	// CPU part: set values
	printf("generating %d random numbers: ", N);
	fflush(stdout);
	ptrA = (float*)malloc(1 * sizeof(float));
	ptrX = (float*)malloc(N * sizeof(float));
	ptrY = (float*)malloc(N * sizeof(float));
	ptrZ0 = (float*)malloc(N * sizeof(float));
	ptrZ = (float*)malloc(N * sizeof(float));
	assert(ptrA != nullptr);
	assert(ptrX != nullptr);
	assert(ptrY != nullptr);
	assert(ptrZ0 != nullptr);
	assert(ptrZ != nullptr);
	setRand(1, ptrA);
	setRand(N, ptrX);
	setRand(N, ptrY);
	printf("done\n");
	printf("A = %f\n", (*ptrA));
	printf("X[%d] = ", N);
	printArray(N, ptrX);
	printf("Y[%d] = ", N);
	printArray(N, ptrY);
	fflush(stdout);
	// CPU kernel
	steady_clock::time_point start = steady_clock::now();
	for (int i = 0; i < N; ++i) {
		kernel_cpu_axpy(i, ptrZ0, ptrA, ptrX, ptrY);
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("CPU's Z[%d] = ", N);
	printArray(N, ptrZ0);
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: memory allocation
	hipMalloc((void**)&devPtrA, 1 * sizeof(float));
	hipMalloc((void**)&devPtrX, N * sizeof(float));
	hipMalloc((void**)&devPtrY, N * sizeof(float));
	hipMalloc((void**)&devPtrZ, N * sizeof(float));
	CUDA_CHECK_ERROR();
	// host to device copy
	hipMemcpy(devPtrA, ptrA, 1 * sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(devPtrX, ptrX, N * sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(devPtrY, ptrY, N * sizeof(float), hipMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// kernel action
	start = steady_clock::now();
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
	kernel_axpy <<< dimGrid, dimBlock>>>(devPtrZ, devPtrA, devPtrX, devPtrY, N);
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// device to host copy
	hipMemcpy(ptrZ, devPtrZ, N * sizeof(float), hipMemcpyDeviceToHost);
	printf("GPU's Z[%d] = ", N);
	printArray(N, ptrZ);
	float rmserr = getRMS(ptrZ0, ptrZ, N, false);
	printf("RMS error = %f\n", rmserr);
	fflush(stdout);
	CUDA_CHECK_ERROR();
	// cuBLAS test for AXPY
	hipblasHandle_t handle;
	hipblasCreate( &handle );
	hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
#if 0 // these lines are optional, since they are already in the device memory
	hipblasSetVector(1, sizeof(float), ptrA, 1, devPtrA, 1);
	hipblasSetVector(N, sizeof(float), ptrX, 1, devPtrX, 1);
	hipblasSetVector(N, sizeof(float), ptrY, 1, devPtrY, 1);
#endif
	start = steady_clock::now();
	hipblasSaxpy(handle, N, devPtrA, devPtrX, 1, devPtrY, 1);
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("cuBLAS AXPY: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	fflush(stdout);
	CUDA_CHECK_ERROR();
	// device to host copy
hipblasGetVector(N, sizeof(float), devPtrY, 1, ptrY, 1);
/*
	cudaMemcpy(ptrY, devPtrY, N * sizeof(float), cudaMemcpyDeviceToHost);
*/
	printf("cuBALS's Y[%d] = ", N);
	printArray(N, ptrY);
	rmserr = getRMS(ptrZ0, ptrY, N, false);
	printf("RMS error = %f\n", rmserr);
	fflush(stdout);
	CUDA_CHECK_ERROR();
	hipblasDestroy(handle);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: cublas-axpy.cu
static const char rcsid[] __attribute__((used)) = "$Id: 14e-cublas-axpy.cu,v 1.1 2026/10/17 03:30:48 wayfarecru Exp $";
