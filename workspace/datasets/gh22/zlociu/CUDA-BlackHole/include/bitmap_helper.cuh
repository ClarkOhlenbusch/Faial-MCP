#pragma once
#include "cuda_runtime.h"
#include <curand.h>
#include <curand_kernel.h>

#include "math_helper.cuh"
#include "constants.h"

/*
__device__ __constant__ float aaKernel[49];

float aaGauss[] = {				
	0.000020f, 0.000239f, 0.001072f, 0.001768f, 0.001072f, 0.000239f, 0.000020f,
	0.000239f, 0.002915f, 0.013064f, 0.021539f, 0.013064f, 0.002915f, 0.000239f,
	0.001072f, 0.013064f, 0.058550f, 0.096532f, 0.058550f, 0.013064f, 0.001072f,
	0.001768f, 0.021539f, 0.096532f, 0.159155f, 0.096532f, 0.021539f, 0.001768f,
	0.001072f, 0.013064f, 0.058550f, 0.096532f, 0.058550f, 0.013064f, 0.001072f,
	0.000239f, 0.002915f, 0.013064f, 0.021539f, 0.013064f, 0.002915f, 0.000239f,
	0.000020f, 0.000239f, 0.001072f, 0.001768f, 0.001072f, 0.000239f, 0.000020f };

__global__ void Antialiasing(float* bitmap)
{
	int gpu_x = threadIdx.x + blockIdx.x * blockDim.x;
	int gpu_y = threadIdx.y + blockIdx.y * blockDim.y;
	int gpu_offset = gpu_x + gpu_y * blockDim.x * gridDim.x;

	vector3 gaussed = { 0.f, 0.f, 0.f };

	for (int k = gpu_y - 3; k <= gpu_y + 3; k++)
		for (int i = gpu_x - 3; i <= gpu_x + 3; i++)
			if (i >= 0 && i < DIM_X && k >= 0 && k < DIM_Y)
			{
				int kernIdx = (i - gpu_x + 3) + (k - gpu_y + 3) * 7;
				gaussed.x += bitmap[3 * (i + k * DIM_X) + 0] * aaKernel[kernIdx];
				gaussed.y += bitmap[3 * (i + k * DIM_X) + 1] * aaKernel[kernIdx];
				gaussed.z += bitmap[3 * (i + k * DIM_X) + 2] * aaKernel[kernIdx];
			}

	__syncthreads();

	gaussed.x = fmaf(gaussed.x, 0.2f, bitmap[3 * gpu_offset + 0]);
	gaussed.y = fmaf(gaussed.y, 0.2f, bitmap[3 * gpu_offset + 1]);
	gaussed.z = fmaf(gaussed.z, 0.2f, bitmap[3 * gpu_offset + 2]);


	bitmap[3 * gpu_offset + 0] = Clip<float>(gaussed.x, 0.f, 1.f);
	bitmap[3 * gpu_offset + 1] = Clip<float>(gaussed.y, 0.f, 1.f);
	bitmap[3 * gpu_offset + 2] = Clip<float>(gaussed.z, 0.f, 1.f);
}

*/