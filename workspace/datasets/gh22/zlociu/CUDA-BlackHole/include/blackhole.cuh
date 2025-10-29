#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdlib.h>
#include "constants.h"
#include "config.h"
#include "vector2.h"
#include "vector3.h"
#include "matrix3.h"
#include "rk4.h"
#include "math_helper.cuh"
#include "gaussRowCol.cuh"
#include "star_sky.cuh"

struct Camera_t
{
	vector3 cameraPosition;		// camera position
    float FoVtangent;           // tangent of FoV / should be 1.5
    vector3 lookAt;             // should be Vector3Zero() initialized
    vector3 upVector;    
    matrix3 viewMatrix;         //final view matrix

};

struct BlackHole_t  
{
    float diskInner;
    float diskOuter;    
    float diskInnerSquared;
    float diskOuterSquared;
};

/// <summary>
/// Convert vector coords to texture offset
/// </summary>
/// <param name="vuv">velocity vector2</param>
/// <returns>offset used to get texture pixel</returns>
__device__ int Lookup(vector2 vuv);

/// <summary>
/// Main raytracing function,
/// calculate every pixel of image, 
/// use GPU
/// </summary>
/// <param name="image">- sRGB pixels values</param>
/// <param name="bitmap">- background image (stars)</param>
/// <param name="camera">- camera object</param>
/// <param name="blackhole">- blackhole object</param>
/// <returns></returns>
__global__ void RayTrace(float* __restrict__ image, cudaTextureObject_t bitmap, Camera_t* camera, BlackHole_t* blackhole);

/// <summary>
/// Finish ray tracing and save to bitmap memory
/// </summary>
/// <param name="imageFloat">input image</param>
/// <param name="image">final bitmap image</param>
/// <returns></returns>
__global__ void FinishRayTracing(const float* __restrict__ imageFloat, unsigned char* __restrict__ image);

/* create camera */
__host__ Camera_t* CreateCamera(void);

/* create black hole */
__host__ BlackHole_t* CreateBlackHole(void);

__host__ void SetCameraParameters(Camera_t* camera, vector3 cameraPosition, float FoVtangent, vector3 lookAt, vector3 upVector);

__host__ void SetBlackHole(BlackHole_t* bh, float diskIn, float diskOut);

void CreateTextureBitmap(unsigned char* dev_image, cudaTextureObject_t* texture_bitmap);

//main
void ProcessRayTracing(unsigned char* dev_image);