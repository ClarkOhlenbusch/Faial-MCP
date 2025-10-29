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
#include "cu_assert.h"

/// <summary>
/// Convolve image red channel
/// </summary>
/// <param name="image">- image to be convoluted</param>
/// <param name="gaussImage">- output convoluted image with airy disk kernel</param>
/// <returns></returns>
__global__ void GaussianConvolveRow(const float* __restrict__ input, float* __restrict__ output);

__global__ void GaussianConvolveCol(const float* __restrict__ input, float* __restrict__ image);

__global__ void GaussianConvolveRowSmall(const float* __restrict__ input, float* __restrict__ output);

__global__ void GaussianConvolveColSmall(const float* __restrict__ input, float* __restrict__ image);

/// <summary>
/// Generate 1D gauss kernel used in convolution.
/// Sum over all kernel values should be 1.f; value > 0.99f is good enough.
/// </summary>
/// <param name="kernel">- memory</param>
/// <param name="sigmaSqr">- squared value of gauss curve sigma param</param>
/// <returns></returns>
__global__ void GenerateGaussianKernel1D(float* kernel, float sigmaSqr = 64.f);

__global__ void GenerateGaussianKernelSmall1D(float* kernel, float sigmaSqr = 64.f);

void CreateGaussKernelRC(void);

void UseGaussianKernelRC(float* __restrict__ dev_imageFloat);