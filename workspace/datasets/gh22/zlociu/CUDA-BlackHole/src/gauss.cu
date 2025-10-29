#pragma once
#include "../include/gauss.cuh"


#ifdef GAUSSIAN
__device__ __constant__ float convolveKernel[GAUSS_KERNEL_SIZE];

__global__ void GaussianConvolve(const float* __restrict__ image, unsigned char* __restrict__ gaussImage)
{
	int gpu_x = threadIdx.x + blockIdx.x * blockDim.x;
	int gpu_y = threadIdx.y + blockIdx.y * blockDim.y;
	int gpu_offset = gpu_x + gpu_y * blockDim.x * gridDim.x;

	vector3 gaussed = { 0.f, 0.f, 0.f };

	for (int k = gpu_y - GAUSS_SIZE; k <= gpu_y + GAUSS_SIZE; k++)
		for (int i = gpu_x - GAUSS_SIZE; i <= gpu_x + GAUSS_SIZE; i++)
			if (i >= 0 && i < DIM_X && k >= 0 && k < DIM_Y)
			{
				int kernIdx = (i - gpu_x + GAUSS_SIZE) + (k - gpu_y + GAUSS_SIZE) * (2 * GAUSS_SIZE + 1);
				gaussed.x += image[3 * (i + k * DIM_X) + 0] * convolveKernel[kernIdx];
				gaussed.y += image[3 * (i + k * DIM_X) + 1] * convolveKernel[kernIdx];
				gaussed.z += image[3 * (i + k * DIM_X) + 2] * convolveKernel[kernIdx];
			}

	gaussed.x = fmaf(gaussed.x, 0.2f, image[3 * gpu_offset + 0]);
	gaussed.y = fmaf(gaussed.y, 0.2f, image[3 * gpu_offset + 1]);
	gaussed.z = fmaf(gaussed.z, 0.2f, image[3 * gpu_offset + 2]);

	gaussed.x = Clip<float>(gaussed.x, 0.f, 1.f);
	gaussed.y = Clip<float>(gaussed.y, 0.f, 1.f);
	gaussed.z = Clip<float>(gaussed.z, 0.f, 1.f);

	gaussed = RGBtosRGB(gaussed);

	gaussed = gaussed * 255.f;

	gaussImage[4 * gpu_offset + 0] = (unsigned char)Clip<float>(gaussed.x, 0.f, 255.f);
	gaussImage[4 * gpu_offset + 1] = (unsigned char)Clip<float>(gaussed.y, 0.f, 255.f);
	gaussImage[4 * gpu_offset + 2] = (unsigned char)Clip<float>(gaussed.z, 0.f, 255.f);
	gaussImage[4 * gpu_offset + 3] = 255;
}

/*
// only used with bloom airy convolve
__device__ float airyDisk(float x)
{
	float res = 2.f * j1f(x) / (x);
	return (res * res);
}
*/

/// <summary>
/// Generate gauss kernel used in convolution
/// </summary>
/// <param name="kernel">- memory</param>
/// <param name="sigmaSqr">- squared width of gauss curve</param>
/// <returns></returns>
/*
__global__ void GenerateGaussianKernel(float* kernel, float sigmaSqr = 64.f)
{
	int gpu_x = blockIdx.x * blockDim.x;
	int gpu_y = blockIdx.y * blockDim.y;
	int gpu_offset = gpu_x + gpu_y * blockDim.x * gridDim.x;

	int i = gpu_x - GAUSS_SIZE;
	int k = gpu_y - GAUSS_SIZE;
	//float tmp = airyDisk(((sqrtf(k50 * k50 + i50 * i50) + 0.000001f) / scale));
	float tmp = expf(-(i * i + k * k) / (2.f * sigmaSqr)) / (2.f * M_PI * sigmaSqr);
	kernel[gpu_offset] = tmp;
	//sum over all kernel values should be 1.f; value > 0.99f is good enough
}
*/


__global__ void GenerateGaussianKernel(float* kernel, float sigmaSqr = 64.f)
{
	int gpu_x = blockIdx.x * blockDim.x;
	int gpu_y = blockIdx.y * blockDim.y;
	//int gpu_offset = gpu_x + gpu_y * blockDim.x * gridDim.x;

	int i = gpu_x - GAUSS_SIZE;
	int k = gpu_y - GAUSS_SIZE;
	//float tmp = airyDisk(((sqrtf(k50 * k50 + i50 * i50) + 0.000001f) / scale));
	float tmp = expf(-(i * i + k * k) / (2.f * sigmaSqr)) / (2.f * M_PI * sigmaSqr);

	kernel[gpu_x + gpu_y * (2 * GAUSS_SIZE + 1)] = tmp;
	kernel[((2 * GAUSS_SIZE) - gpu_x) + gpu_y * (2 * GAUSS_SIZE + 1)] = tmp;
	kernel[gpu_x + (2 * GAUSS_SIZE + 1) * ((2 * GAUSS_SIZE) - gpu_y)] = tmp;
	kernel[((2 * GAUSS_SIZE) - gpu_x) + (2 * GAUSS_SIZE + 1) * ((2 * GAUSS_SIZE) - gpu_y)] = tmp;
}

// 0 1 2   1 0
// 1 2 3   2 1
// 2 3 4   3 2

// 1 2 3 2 1
// 0 1 2 1 0

#endif // !GAUSSIAN_ROW_COL