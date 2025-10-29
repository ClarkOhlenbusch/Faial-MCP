#pragma once
#include "../include/star_sky.cuh"


__global__ void PrepareRandom(curandStateXORWOW_t* state)
{
	int offset = threadIdx.x + blockIdx.x * blockDim.x;

	//curand_init(66743015, (offset), 0, &state[offset]);
	curand_init(offset, 667, 0, &state[offset]);
}

__global__ void PrepareBackground(unsigned char* bitmap)
{
	int gpu_x = threadIdx.x + blockIdx.x * blockDim.x;
	int gpu_y = threadIdx.y + blockIdx.y * blockDim.y;
	int gpu_offset = gpu_x + gpu_y * blockDim.x * gridDim.x;

	bitmap[4 * gpu_offset + 0] = 0;
	bitmap[4 * gpu_offset + 1] = 0;
	bitmap[4 * gpu_offset + 2] = 0x15;
	bitmap[4 * gpu_offset + 3] = 0xFF;
}

__host__ __device__ int AlphaBlending(int background, int color, float alpha)
{
	return (int)(1.f * color * alpha) + (int)(1.f * background * (1.f - alpha));
}

__global__ void GenerateSky(curandStateXORWOW_t* state, Star_t* stars)
{
	int offset = threadIdx.x + blockIdx.x * blockDim.x;

	curandStateXORWOW_t localState = state[offset];
	float type = curand_uniform(&localState);
	Star_t newOne;
	if (type < 0.7645f)
	{
		newOne.radius = curand_uniform(&localState) * 0.7f; //*0.9f
		newOne.temperature = fmaf(curand_uniform(&localState), 1.3f, 2.4f);
	}
	else if (type < (0.7645f + 0.121f))
	{
		newOne.radius = fmaf(curand_uniform(&localState), 0.26f, 0.7f); // * 0.26f + 0.9f
		newOne.temperature = fmaf(curand_uniform(&localState), 1.5f, 3.7f);
	}
	else if (type < (0.7645f + 0.121f + 0.076f))
	{
		newOne.radius = fmaf(curand_uniform(&localState), 0.19f, 0.96f); // * 0.19f + 1.16f
		newOne.temperature = fmaf(curand_uniform(&localState), 0.8f, 5.2f);
	}
	else if (type < (0.7645f + 0.121f + 0.076f + 0.03f))
	{
		newOne.radius = fmaf(curand_uniform(&localState), 0.25f, 1.15f); // * 0.25f + 1.35f
		newOne.temperature = fmaf(curand_uniform(&localState), 1.5f, 6.f);
	}
	else if (type < (0.7645f + 0.121f + 0.076f + 0.03f + 0.007f))
	{
		newOne.radius = fmaf(curand_uniform(&localState), 0.4f, 1.4f); // * 0.4f + 1.6f
		newOne.temperature = fmaf(curand_uniform(&localState), 2.5f, 7.5f);
	}
	else
	{
		newOne.radius = fmaf(curand_uniform(&localState), 3.2f, 1.8f); // * 4.8f + 2.f
		newOne.temperature = fmaf(curand_uniform(&localState), 20.0f, 10.f);
	}

	newOne.x = (int)(curand_uniform(&localState) * (DIM_X_N(BITMAP_MULTIPIER)));
	newOne.y = (int)(curand_uniform(&localState) * (DIM_Y_N(BITMAP_MULTIPIER)));
	newOne.z = 1.3f;
	//newOne.z = 2.f;
	//STARS
	stars[offset] = newOne;

	//STATE
	state[offset] = localState;
}

__device__ void DrawStar(unsigned char* bitmap, int x, int y, float radius, int colorIdx)
{
	if (radius <= 0) return;
	y = DIM_Y_N(BITMAP_MULTIPIER) - y;
	int radius_2i = (int)(2.f * radius);

	for (int j = y - radius_2i - 1; j <= y + radius_2i + 1; j++)
	{
		for (int i = x - radius_2i - 1; i <= x + radius_2i + 1; i++)
		{
			if (i >= 0 && i < DIM_X_N(BITMAP_MULTIPIER) && j >= 0 && j < DIM_Y_N(BITMAP_MULTIPIER))
			{
				int offset = (i + j * DIM_X_N(BITMAP_MULTIPIER));

				// sqrtf((j - y) * (j - y) + (i - x) * (i - x));
				float dist = hypotf((i - x), (j - y));

				vector3 colorRGB;
				float colorAlpha;

				//float intensity = -dist + M_12_SQRT3 * radius + 1.f;
				float intensity = (3.f / (expf(dist - radius) + 2.f));
				colorRGB = colorMap[colorIdx] * ClipMin<float>(intensity, 1.f);

				//float a = 1.f / (1.f - 1.f / (expf(radius))); // 1/(1 - 1/exp(x))
				//colorAlpha = (a / (expf(dist - radius))) - a + 1.f;
				colorAlpha = (radius / (expf(dist - radius) + radius - 1.f));
				colorAlpha = Clip<float>(colorAlpha, 0.f, 1.f);

				bitmap[4 * offset + 0] = Clip<int>(AlphaBlending(bitmap[4 * offset + 0], colorRGB.x, colorAlpha), 0, 255);
				bitmap[4 * offset + 1] = Clip<int>(AlphaBlending(bitmap[4 * offset + 1], colorRGB.y, colorAlpha), 0, 255);
				bitmap[4 * offset + 2] = Clip<int>(AlphaBlending(bitmap[4 * offset + 2], colorRGB.z, colorAlpha), 0, 255);
			}
		}
	}
}

__global__ void DrawSky(unsigned char* __restrict__ bitmap, const Star_t* __restrict__ stars)
{
	int gpu_offset = threadIdx.x + blockIdx.x * blockDim.x;

	Star_t star = stars[gpu_offset];

	int sx = star.x & 0x0F;
	int sy = star.y & 0x0F;
	int soff = sx + (sy << 4);

	// maybe changing 255 to soff will be better (less iterations)
	/*
	for (int i = 0; i < 255; i++)
	{
		if (soff == i)
		{
			int tempIdx = (int)((star.temperature) * 16.5f); // temperature to color index in colours table from file 'math_helper.h'
			DrawStar(bitmap, star.x, star.y, star.radius, tempIdx);
		}
		__syncthreads();
	}
	*/

	for (int i = 0; i <= soff; i++)
	{
		if (soff == i)
		{
			int tempIdx = (int)((star.temperature) * 16.5f); // temperature to color index in colours table from file 'math_helper.h'
			DrawStar(bitmap, star.x, star.y, star.radius * star.z, tempIdx);
		}
		__syncthreads();
	}
}

void PrepareSkyBackground(unsigned char* dev_bitmap)
{
	Star_t* dev_stars;
	curandStateXORWOW_t* dev_random;

	dim3 gridStar(DIM_X_N(BITMAP_MULTIPIER) * DIM_Y_N(BITMAP_MULTIPIER) / STAR_DENSITY / 32);

	cudaAssert(cudaMalloc((void**)&dev_random, DIM_X_N(BITMAP_MULTIPIER) * DIM_Y_N(BITMAP_MULTIPIER) / STAR_DENSITY * sizeof(curandStateXORWOW_t)));
	cudaAssert(cudaMalloc((void**)&dev_stars, DIM_X_N(BITMAP_MULTIPIER) * DIM_Y_N(BITMAP_MULTIPIER) / STAR_DENSITY * sizeof(Star_t)));

	cudaAssert(cudaMemcpyToSymbol(colorMap, colors, 500 * sizeof(vector3)));

	PrepareRandom << <gridStar, 32 >> > (dev_random);
	GenerateSky << <gridStar, 32 >> > (dev_random, dev_stars);

	// draw stars using CPU on bitmap
	DrawSky << < gridStar, 32 >> > (dev_bitmap, dev_stars);

	cudaAssert(cudaFree(dev_random));
	cudaAssert(cudaFree(dev_stars));
}

