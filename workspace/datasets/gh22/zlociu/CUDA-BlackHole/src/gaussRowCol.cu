#include <math.h>

// Values from: https://github.com/zlociu/CUDA-BlackHole/blob/97d74b6764cb3cc87993069a20f82c78290761ec/include/config.h#L21
#define DIM_X 3840
#define DIM_Y 2160
#define GAUSS_SIZE 600
#define GAUSS_SIZE_SMALL 100
#define GAUSS_KERNEL_SIZE ((2 * GAUSS_SIZE + 1) * (2 * GAUSS_SIZE + 1))
#define vector3 float3
__device__ __constant__ float convolveKernel1D[2 * GAUSS_SIZE + 1];
__device__ __constant__ float convolveKernelSmall1D[2 * GAUSS_SIZE_SMALL + 1];

template<typename T>
	__device__ float Clip(float, float, float);

__global__ void GaussianConvolveRow(const float* __restrict__ input, float* __restrict__ output)
{
	int gpu_x = threadIdx.x + blockIdx.x * blockDim.x;
	int gpu_y = threadIdx.y + blockIdx.y * blockDim.y;
	int gpu_offset = gpu_x + gpu_y * blockDim.x * gridDim.x;

	vector3 gaussed = { 0.f, 0.f, 0.f };

	/*
	const int blockSize = 16;


	// [          |         |           ]
	// [          |         |           ]
	// [          |         |           ]
	// [y-axis][x-axis]
	__shared__ float sharedGauss[blockSize][3 * (blockSize + 2 * GAUSS_SIZE)];

	//sharedGauss[threadIdx.y][3 * (threadIdx.x + GAUSS_SIZE) + 0] = input[3 * gpu_offset + 0];
	//sharedGauss[threadIdx.y][3 * (threadIdx.x + GAUSS_SIZE) + 1] = input[3 * gpu_offset + 1];
	//sharedGauss[threadIdx.y][3 * (threadIdx.x + GAUSS_SIZE) + 2] = input[3 * gpu_offset + 2];

	if (threadIdx.x == 0)
	{
		for (int i = -GAUSS_SIZE; i <= GAUSS_SIZE; i++)
		{
			int gpu_off = (gpu_x + i) + gpu_y * blockDim.x * gridDim.x;
			sharedGauss[threadIdx.y][3 * (i + GAUSS_SIZE) + 0] = (gpu_x + i >= 0 && gpu_x + i < DIM_X) ? input[3 * gpu_off + 0] : 0.f;
			sharedGauss[threadIdx.y][3 * (i + GAUSS_SIZE) + 1] = (gpu_x + i >= 0 && gpu_x + i < DIM_X) ? input[3 * gpu_off + 1] : 0.f;
			sharedGauss[threadIdx.y][3 * (i + GAUSS_SIZE) + 2] = (gpu_x + i >= 0 && gpu_x + i < DIM_X) ? input[3 * gpu_off + 2] : 0.f;
		}
	}

	__syncthreads();


	for (int k = - GAUSS_SIZE; k <= GAUSS_SIZE; k++)
	{
		gaussed.x += sharedGauss[threadIdx.y][3 * (k + threadIdx.x + GAUSS_SIZE) + 0] * convolveKernel1D[k + GAUSS_SIZE];
		gaussed.y += sharedGauss[threadIdx.y][3 * (k + threadIdx.x + GAUSS_SIZE) + 1] * convolveKernel1D[k + GAUSS_SIZE];
		gaussed.z += sharedGauss[threadIdx.y][3 * (k + threadIdx.x + GAUSS_SIZE) + 2] * convolveKernel1D[k + GAUSS_SIZE];
	}	*/

	for (int k = max(gpu_x - GAUSS_SIZE, 0); k <= min(gpu_x + GAUSS_SIZE, DIM_X - 1); k++)
	{
		gaussed.x = fmaf(input[3 * (k + gpu_y * DIM_X) + 0], convolveKernel1D[(k - gpu_x + GAUSS_SIZE)], gaussed.x);
		gaussed.y = fmaf(input[3 * (k + gpu_y * DIM_X) + 1], convolveKernel1D[(k - gpu_x + GAUSS_SIZE)], gaussed.y);
		gaussed.z = fmaf(input[3 * (k + gpu_y * DIM_X) + 2], convolveKernel1D[(k - gpu_x + GAUSS_SIZE)], gaussed.z);
	}

	output[3 * gpu_offset + 0] = gaussed.x;
	output[3 * gpu_offset + 1] = gaussed.y;
	output[3 * gpu_offset + 2] = gaussed.z;
}

__global__ void GaussianConvolveCol(const float* __restrict__ input, float* __restrict__ image)
{
	int gpu_x = threadIdx.x + blockIdx.x * blockDim.x;
	int gpu_y = threadIdx.y + blockIdx.y * blockDim.y;
	int gpu_offset = gpu_x + gpu_y * blockDim.x * gridDim.x;

	vector3 gaussed = { 0.f, 0.f, 0.f };

	/*
	const int blockSize = 16;

	// [          ]
	// [          ]
	// [          ]
	//
	// |          |
	//
	// [          ]
	// [          ]
	// [          ]
	// [y-axis][x-axis]

	__shared__ float sharedGauss[3 * (blockSize + 2 * GAUSS_SIZE)][blockSize];

	//sharedGauss[3 * (threadIdx.y + GAUSS_SIZE) + 0][threadIdx.x] = input[3 * gpu_offset + 0];
	//sharedGauss[3 * (threadIdx.y + GAUSS_SIZE) + 1][threadIdx.x] = input[3 * gpu_offset + 1];
	//sharedGauss[3 * (threadIdx.y + GAUSS_SIZE) + 2][threadIdx.x] = input[3 * gpu_offset + 2];

	if (threadIdx.y == 0)
	{
		for (int i = -GAUSS_SIZE; i <= GAUSS_SIZE; i++)
		{
			int gpu_off = gpu_x + (gpu_y + i) * blockDim.x * gridDim.x;
			sharedGauss[3 * (i + GAUSS_SIZE) + 0][threadIdx.x] = (gpu_y + i >= 0 && gpu_y + i < DIM_Y) ? input[3 * gpu_off + 0] : 0.f;
			sharedGauss[3 * (i + GAUSS_SIZE) + 1][threadIdx.x] = (gpu_y + i >= 0 && gpu_y + i < DIM_Y) ? input[3 * gpu_off + 1] : 0.f;
			sharedGauss[3 * (i + GAUSS_SIZE) + 2][threadIdx.x] = (gpu_y + i >= 0 && gpu_y + i < DIM_Y) ? input[3 * gpu_off + 2] : 0.f;
		}
	}

	__syncthreads();

	for (int k = -GAUSS_SIZE; k <= GAUSS_SIZE; k++)
	{
		gaussed.x += sharedGauss[3 * (k + threadIdx.y + GAUSS_SIZE) + 0][threadIdx.x] * convolveKernel1D[k + GAUSS_SIZE];
		gaussed.y += sharedGauss[3 * (k + threadIdx.y + GAUSS_SIZE) + 1][threadIdx.x] * convolveKernel1D[k + GAUSS_SIZE];
		gaussed.z += sharedGauss[3 * (k + threadIdx.y + GAUSS_SIZE) + 2][threadIdx.x] * convolveKernel1D[k + GAUSS_SIZE];
	}*/

	for (int k = max(gpu_y - GAUSS_SIZE, 0); k <= min(gpu_y + GAUSS_SIZE, DIM_Y - 1); k++)
	{
		gaussed.x = fmaf(input[3 * (gpu_x + k * DIM_X) + 0], convolveKernel1D[(k - gpu_y + GAUSS_SIZE)], gaussed.x);
		gaussed.y = fmaf(input[3 * (gpu_x + k * DIM_X) + 1], convolveKernel1D[(k - gpu_y + GAUSS_SIZE)], gaussed.y);
		gaussed.z = fmaf(input[3 * (gpu_x + k * DIM_X) + 2], convolveKernel1D[(k - gpu_y + GAUSS_SIZE)], gaussed.z);
	}

	gaussed.x = fmaf(gaussed.x, 0.3f, image[3 * gpu_offset + 0]);
	gaussed.y = fmaf(gaussed.y, 0.3f, image[3 * gpu_offset + 1]);
	gaussed.z = fmaf(gaussed.z, 0.3f, image[3 * gpu_offset + 2]);

	gaussed.x = Clip<float>(gaussed.x, 0.f, 1.f);
	gaussed.y = Clip<float>(gaussed.y, 0.f, 1.f);
	gaussed.z = Clip<float>(gaussed.z, 0.f, 1.f);

	image[3 * gpu_offset + 0] = gaussed.x;
	image[3 * gpu_offset + 1] = gaussed.y;
	image[3 * gpu_offset + 2] = gaussed.z;
}

__global__ void GaussianConvolveRowSmall(const float* __restrict__ input, float* __restrict__ output)
{
	int gpu_x = threadIdx.x + blockIdx.x * blockDim.x;
	int gpu_y = threadIdx.y + blockIdx.y * blockDim.y;
	int gpu_offset = gpu_x + gpu_y * blockDim.x * gridDim.x;

	vector3 gaussed = { 0.f, 0.f, 0.f };

	for (int k = max(gpu_x - GAUSS_SIZE_SMALL, 0); k <= min(gpu_x + GAUSS_SIZE_SMALL, DIM_X - 1); k++)
	{
		gaussed.x = fmaf(input[3 * (k + gpu_y * DIM_X) + 0], convolveKernelSmall1D[(k - gpu_x + GAUSS_SIZE_SMALL)], gaussed.x);
		gaussed.y = fmaf(input[3 * (k + gpu_y * DIM_X) + 1], convolveKernelSmall1D[(k - gpu_x + GAUSS_SIZE_SMALL)], gaussed.y);
		gaussed.z = fmaf(input[3 * (k + gpu_y * DIM_X) + 2], convolveKernelSmall1D[(k - gpu_x + GAUSS_SIZE_SMALL)], gaussed.z);
	}

	output[3 * gpu_offset + 0] = gaussed.x;
	output[3 * gpu_offset + 1] = gaussed.y;
	output[3 * gpu_offset + 2] = gaussed.z;
}

__global__ void GaussianConvolveColSmall(const float* __restrict__ input, float* __restrict__ image)
{
	int gpu_x = threadIdx.x + blockIdx.x * blockDim.x;
	int gpu_y = threadIdx.y + blockIdx.y * blockDim.y;
	int gpu_offset = gpu_x + gpu_y * blockDim.x * gridDim.x;

	vector3 gaussed = { 0.f, 0.f, 0.f };

	for (int k = max(gpu_y - GAUSS_SIZE_SMALL, 0); k <= min(gpu_y + GAUSS_SIZE_SMALL, DIM_Y - 1); k++)
	{
		gaussed.x = fmaf(input[3 * (gpu_x + k * DIM_X) + 0], convolveKernelSmall1D[(k - gpu_y + GAUSS_SIZE_SMALL)], gaussed.x);
		gaussed.y = fmaf(input[3 * (gpu_x + k * DIM_X) + 1], convolveKernelSmall1D[(k - gpu_y + GAUSS_SIZE_SMALL)], gaussed.y);
		gaussed.z = fmaf(input[3 * (gpu_x + k * DIM_X) + 2], convolveKernelSmall1D[(k - gpu_y + GAUSS_SIZE_SMALL)], gaussed.z);
	}

	gaussed.x = fmaf(gaussed.x, 0.3f, image[3 * gpu_offset + 0]);
	gaussed.y = fmaf(gaussed.y, 0.3f, image[3 * gpu_offset + 1]);
	gaussed.z = fmaf(gaussed.z, 0.3f, image[3 * gpu_offset + 2]);

	gaussed.x = Clip<float>(gaussed.x, 0.f, 1.f);
	gaussed.y = Clip<float>(gaussed.y, 0.f, 1.f);
	gaussed.z = Clip<float>(gaussed.z, 0.f, 1.f);

	image[3 * gpu_offset + 0] = gaussed.x;
	image[3 * gpu_offset + 1] = gaussed.y;
	image[3 * gpu_offset + 2] = gaussed.z;
}

/*
// only used with bloom airy convolve
__device__ float airyDisk(float x)
{
	float res = 2.f * j1f(x) / (x);
	return (res * res);
}
*/

__global__ void GenerateGaussianKernel1D(float* kernel, float sigmaSqr)
{
    __requires (blockDim.x == 1);
	int gpu_x = blockIdx.x;

	int i = gpu_x - GAUSS_SIZE;
	//float tmp = airyDisk(((sqrtf(k50 * k50 + i50 * i50) + 0.000001f) / scale));
	float tmp = expf(-(i * i) / (2.f * sigmaSqr)) / sqrtf(2.f * M_PI * sigmaSqr);

	kernel[gpu_x] = tmp;
	kernel[(2 * GAUSS_SIZE + 1) - gpu_x] = tmp;
}

__global__ void GenerateGaussianKernelSmall1D(float* kernel, float sigmaSqr)
{
	__requires (blockDim.x == 1);
	int gpu_x = blockIdx.x;

	int i = gpu_x - GAUSS_SIZE_SMALL;
	//float tmp = airyDisk(((sqrtf(k50 * k50 + i50 * i50) + 0.000001f) / scale));
	float tmp = expf(-(i * i) / (2.f * sigmaSqr)) / sqrtf(2.f * M_PI * sigmaSqr);

	kernel[gpu_x] = tmp;
	kernel[(2 * GAUSS_SIZE_SMALL + 1) - gpu_x] = tmp;
}
