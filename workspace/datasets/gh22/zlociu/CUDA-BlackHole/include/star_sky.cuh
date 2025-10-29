#pragma once
#include "cuda_runtime.h"
#include <curand.h>
#include <curand_kernel.h>

#include "math_helper.cuh"
#include "constants.h"
#include "config.h"
#include "cu_assert.h"
#include "stdio.h"
#include "vector3.h"

/// <summary>
/// Star_t represents a star from 'Main sequence'.
/// </summary>
typedef struct {
	int x, y;			//star position on the screen
	float z;			//distance factor (0.7-1.3)
	float radius;		//radius of star
	float temperature;	//temperature of star (Kelvin), used in selecting color 
} Star_t;

__device__ __constant__ vector3 colorMap[500];

/*
	   -----< Main Sequence stars temperature and colors >-----
			     

	-------------------------------------------------------------
	|  temperature |  colour    |    radius   |  probability    |
	-------------------------------------------------------------
	|	 10-30K	   |    blue    |   1.8-5.6   |      0.13%      |
	-------------------------------------------------------------
	|	7.5-10K    | light blue |   1.4-1.8   |      0.6%       |
	-------------------------------------------------------------
	|	 6-7.5K    |   white    |   1.15-1.4  |        3%       |
	-------------------------------------------------------------
	|	 5.2-6K    |   yellow   |  0.96-1.15  |      7.6%       |
	-------------------------------------------------------------
	|	3.7-5.2K   |   orange   |   0.7-0.96  |      12.1%      |
	-------------------------------------------------------------
	|	2.4-3.7K   |    red     |    <= 0.7   |     76.45%      |
	-------------------------------------------------------------

	Source: https://en.wikipedia.org/wiki/Stellar_classification

*/

/// <summary>
/// Prepare RNG for GPU 
/// </summary>
/// <param name="state"></param>
/// <returns></returns>
__global__ void PrepareRandom(curandStateXORWOW_t* state);

__global__ void PrepareBackground(unsigned char* bitmap);

/// <summary>
/// Blend background color with foreground color and foreground alpha 
/// </summary>
/// <param name="background">background color ()</param>
/// <param name="color">foreground color</param>
/// <param name="alpha">foreground color alpha channel</param>
/// <returns> blended color </returns>
__host__ __device__ int AlphaBlending(int background, int color, float alpha);

/// <summary>
/// Creating all stars; size and temperature is based on 'Main sequence' statistics
/// </summary>
/// <param name="state">rng states</param>
/// <param name="stars">stars collection</param>
/// <returns></returns>
__global__ void GenerateSky(curandStateXORWOW_t* state, Star_t* stars);

/// <summary>
/// Draw star on bitmap. Use GPU
/// </summary>
/// <param name="bitmap">- pixels array </param>
/// <param name="x">- x-value center of star </param>
/// <param name="y">- y-value center of star </param>
/// <param name="radius">- radius of star </param>
/// <param name="colorIdx">- color index, used with 'colors' table.</param>
/// <seealso cref="math_helper.h"/>  
/// <returns></returns>
__device__ void DrawStar(unsigned char* bitmap, int x, int y, float radius, int colorIdx);

/// <summary>
/// Draw all stars from stars collection
/// </summary>
/// <param name="bitmap">main bitmap</param>
/// <param name="stars">stars collection</param>
/// <param name="len">stars collection length</param>
/// <returns></returns>
__global__ void DrawSky(unsigned char* __restrict__ bitmap, const Star_t* __restrict__ stars);

void PrepareSkyBackground(unsigned char* dev_bitmap);