#include "device_launch_parameters.h" 
#include <cuda_runtime.h> 
#include <stdlib.h> 
#include <stdio.h>

__global__ void VectorAdd(int *a, int *b, int *c, int size) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	c[tid] = a[tid] + b[tid];
}

int main() {
	const int size = 512 * 65535;
	const int BufferSize = size * sizeof(int);

	int* a;
	int* b;
	int* c;

	a = (int*)malloc(BufferSize);
	b = (int*)malloc(BufferSize);
	c = (int*)malloc(BufferSize);

	int i = 0;

	for (i = 0; i < size; i++) {
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	} 

	int* d_a;
	int* d_b;
	int* d_c;

	cudaMalloc((void**)&d_a, BufferSize);
	cudaMalloc((void**)&d_b, BufferSize);
	cudaMalloc((void**)&d_c, BufferSize);

	cudaMemcpy(d_a, a, BufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, a, BufferSize, cudaMemcpyHostToDevice);

	VectorAdd <<<65535, 512 >>> (d_a, d_b, d_c, size);

	cudaMemcpy(c, d_c, BufferSize, cudaMemcpyDeviceToHost);

	for (i = 0; i < 5; i++) {
		printf(" Result[%d] : %d\n", i, c[i]);
	}
	printf("......\n");
	for (i = size - 5; i < size; i++) {
		printf(" Result[%d] : %d\n", i, c[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(a);
	free(b);
	free(c);

	return 0;
}