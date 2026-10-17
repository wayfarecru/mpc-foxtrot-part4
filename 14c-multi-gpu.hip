#include "hip/hip_runtime.h"
#include <cstdio>
#include <chrono>
#include <cassert>
using namespace std;
using namespace std::chrono;

#include "./common.hpp"

int N = 1 * 1 * 1024;

float* ptrA; // host memory (main memory)
float* ptrB;

float* devPtrA; // device memory (graphics memory)
float* devPtrB;

// CPU kernel function
__host__ void kernel_cpu_adj_diff(int i, float* B, const float* A) {
	if (i == 0) {
		B[i] = A[i] - 0.0F;
	} else {
		B[i] = A[i] - A[i-1];
	}
}

// CUDA kernel function
__global__ void kernel_adj_diff(float* B, const float* A, int N) {
	extern __shared__ float s_Buf[]; // size not fixed at compile time
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	if (i < N) {
		s_Buf[tx] = A[i];
		__syncthreads();
		if (tx > 0) {
			B[i] = s_Buf[tx] - s_Buf[tx - 1];
		} else if (i > 0) {
			B[i] = s_Buf[tx] - A[i - 1];
		} else { // i == 0
			B[i] = s_Buf[i] - 0.0F;
		}
	}
}


int main(const int argc, const char* argv[]) {
	srand(time(NULL));
	N = parseNum(argc, argv, N);
	printf("N = %d\n", N);
	// CPU part: set values
	ptrA = (float*)malloc(N * sizeof(float));
	ptrB = (float*)malloc(N * sizeof(float));
	assert(ptrA != nullptr);
	assert(ptrB != nullptr);
	setRand(N, ptrA);
	printf("A[%d] = ", N);
	printArray(N, ptrA);
	fflush(stdout);
	// CPU kernel
	steady_clock::time_point start = steady_clock::now();
	for (int i = 0; i < N; ++i) {
		kernel_cpu_adj_diff(i, ptrB, ptrA);
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("CPU's B[%d] = ", N);
	printArray(N, ptrB);
	memset(ptrB, 0, N * sizeof(float));
	// GPU part: cuda availability check
	int deviceCount = 0;
	hipGetDeviceCount( &deviceCount );
	CUDA_CHECK_ERROR();
	if (deviceCount == 0) {
		printf("%s: no available CUDA device found\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	printf("device count = %d\n", deviceCount);
	// GPU part: set to device #0
	hipSetDevice(0);
	CUDA_CHECK_ERROR();
	// check the compute mode
	int computeMode = 0;
	hipDeviceGetAttribute( &computeMode, hipDeviceAttributeComputeMode, 0 );
	printf("compute mode = %d, ", computeMode);
	switch (computeMode) {
	case hipComputeModeDefault:
		// Multiple threads can use cudaSetDevice() with this devic
		printf("default\n");
		break;
	case hipComputeModeExclusive:
		// Only one thread in one process will be able to use cudaSetDevice() with this device.
		printf("exclusive\n");
		break;
	case hipComputeModeProhibited:
		// No threads can use cudaSetDevice() with this device.
		printf("prohibited\n");
		break;
	case hipComputeModeExclusiveProcess:
		// Many threads in one process will be able to use cudaSetDevice() with this device.
		printf("exclusive process\n");
		break;
	default:
		printf("unknown\n");
		break;
	}
	// get device properties
	hipDeviceProp_t deviceProp;
	int driverVersion = 0;
	int runtimeVersion = 0;
	hipGetDeviceProperties( &deviceProp, 0 ); // 0 is the default CUDA device
	hipDriverGetVersion( &driverVersion );
	hipRuntimeGetVersion( &runtimeVersion );
	printf("CUDA device \"%s\": driver ver %d.%d, runtime ver %d.%d, capability ver %d.%d\n",
	       deviceProp.name,
	       driverVersion / 1000, (driverVersion % 100) / 10,
	       runtimeVersion / 1000, (runtimeVersion % 100) / 10,
	       deviceProp.major, deviceProp.minor);
	printf("  max num of threads per block = %d\n", deviceProp.maxThreadsPerBlock);
	printf("  max dim size of a thread block (x,y,z) = (%d, %d, %d)\n",
	       deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
	       deviceProp.maxThreadsDim[2]);
	printf("  max dim size of a grid size (x,y,z) = (%d, %d, %d)\n",
	       deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
	       deviceProp.maxGridSize[2]);
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: memory allocation
	hipMalloc((void**)&devPtrA, N * sizeof(float));
	hipMalloc((void**)&devPtrB, N * sizeof(float));
	CUDA_CHECK_ERROR();
	// host to device copy
	hipMemcpy(devPtrA, ptrA, N * sizeof(float), hipMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// kernel action
	int blockSize = deviceProp.maxThreadsPerBlock;
	dim3 dimBlock(blockSize, 1, 1);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
	int bufSizeInByte = dimBlock.x * sizeof(float);
	CUDA_PRINT_CONFIG(N);
	printf("buf size in byte = %d\n", bufSizeInByte);
	start = steady_clock::now();
	kernel_adj_diff <<< dimGrid, dimBlock, bufSizeInByte>>>(devPtrB, devPtrA, N);
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// device to host copy
	hipMemcpy(ptrB, devPtrB, N * sizeof(float), hipMemcpyDeviceToHost); // ptrC <- devPtrC
	printf("GPU's B[%d] = ", N);
	printArray(N, ptrB);
	CUDA_CHECK_ERROR();
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: multi-gpu.cu
static const char rcsid[] __attribute__((used)) = "$Id: 14c-multi-gpu.cu,v 1.1 2026/10/17 03:30:48 wayfarecru Exp $";
