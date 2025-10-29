#include "cudaMiner.h"

#ifdef CUDA_MINING

#include <cuda_runtime.h>

__global__ void kernel()
{
	printf("Hello, World!\n");
}


void cudaMiner::doCuda()
{
	kernel<<<1,1>>>();
	cudaDeviceSynchronize();
}

#endif //CUDA_MINING