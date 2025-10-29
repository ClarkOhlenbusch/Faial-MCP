#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>
#include "constants.h"
#include "vector3.h"
#include "matrix3.h"
#include "math_helper.cuh"
#include "config.h"

#ifdef GAUSSIAN

/// <summary>
/// Convolve image red channel
/// </summary>
/// <param name="image">- image to be convoluted</param>
/// <param name="gaussImage">- output convoluted image with airy disk kernel</param>
/// <returns></returns>
__global__ void GaussianConvolve(const float* __restrict__ image, unsigned char* __restrict__ gaussImage);

// GenerateGaussKernel can be upgraded 
// gauss kernel has one magic property we can use here
// it has two axis,
// if we calculate one quarter other 3 quaters will be same.
// less calculations, less time we need
// only left upper square 

/// <summary>
/// Generate gauss kernel used in convolution.
/// Sum over all kernel values should be 1.f; value > 0.99f is good enough.
/// </summary>
/// <param name="kernel">- memory</param>
/// <param name="sigmaSqr">- squared value of gauss curve sigma param</param>
/// <returns></returns>
__global__ void GenerateGaussianKernel(float* kernel, float sigmaSqr = 64.f);

#endif // !GAUSSIAN_ROW_COL