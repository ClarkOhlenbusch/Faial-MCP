#pragma once
#include "cuda_runtime.h"
#include "vector3.h"

/// <summary>
/// Clip all values from input array 
/// </summary>
/// <param name="arr">- input array</param>
/// <param name="arr_len">- length of input array</param>
/// <param name="min">- minimum value (inclusive)</param>
/// <param name="max">- maximum value (inclusive)</param>
/// <returns></returns>
__inline__ __host__ __device__ void Clip(float* arr, int arr_len, float min, float max)
{
	for (int i = 0; i < arr_len; i++)
	{
		if (arr[i] < min) arr[i] = min;
		else if (arr[i] > max) arr[i] = max;
		//arr[i] = (arr[i] < min) * min + (arr[i] > max) * max;
	}
}

template<typename T>
__inline__ __host__ __device__ T Clip(T val, T left, T right)
{
	//if (val > max) return max;
	//else if (val < min) return min;
	//return val;
	return max(left, min(right, val));
	//return (val > max) * max + (val < min) * min + ((val <= max) && (val >= min)) * val;
}

/// <summary>
/// Clip value to -INFINITY-right range
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="val">- input value</param>
/// <param name="right">- maximum value (inclusive)</param>
/// <returns></returns>
template<typename T>
__inline__ __host__ __device__ T ClipMax(T val, T right)
{
	//if (val > max) return max;
	//return val;
	return min(val, right);
	//return (val > max) * max + (val <= max) * val;
}

/// <summary>
/// Clip value to left-INFINITY range
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="val">- input value</param>
/// <param name="left">- minimum value (inclusive)</param>
/// <returns>clipped value</returns>
template<typename T>
__inline__ __host__ __device__ T ClipMin(T val, T left)
{
	//if (val < min) return min;
	//return val;
	return max(val, left);
	//return (val >= min) * val + (val < min) * min;
}

/// <summary>
/// Fmaf version using vector3 struct
/// </summary>
/// <returns>_x * _y + _z</returns>
__inline__ __device__ vector3 fma3df(vector3 _x, vector3 _y, vector3 _z)
{
	vector3 res;
	res.x = fmaf(_x.x, _y.x, _z.x);
	res.y = fmaf(_x.y, _y.y, _z.y);
	res.z = fmaf(_x.z, _y.z, _z.z);
	return res;
}

/// <summary>
/// Calculate accretion disk temperature
/// </summary>
/// <param name="sqrR">- squared R of disc </param>
/// <param name="logT0">- log temperature(K) of accretion disk at ISCO </param>
/// <returns></returns>
__inline__ __device__ float DiskTemp(float sqrR, float logT0)
{
	float A = logT0 + M_34_LOG_3;
	return (A - 0.375f * logf(sqrR));
	//float A = logT0 * 2.2795f; // 3^(3/4)
	//return (A / powf(sqrR, 0.375f));
}

/// <summary>
/// 
/// </summary>
/// <param name="T">- temperature of accretion disk</param>
/// <returns>intensity of accretion disk</returns>
__inline__ __device__ float Intensity(float T)
{
	//return 1.f / (expf(29622.4f / ClipMin<float>(T, 1.f)) - 1.f);
	return  1.f / (expf(29622.4f / ClipMin<float>(T, 1.f)) - 1.f);
}

/// <summary>
/// Calc color value from table 'colors'
/// </summary>
/// <param name="T">- temperature</param>
/// <returns>colors from table (TODO: convert to 0.0-1.0)</returns>
__inline__ __device__ int Colour(float T)
{
	// 0.017241f = 1 / 29000 * 500
	float indices = Clip<float>((T - 1000.f) * 0.017241f, 0.f, 498.9f);
	return (int)indices;
}

/// <summary>
/// Blends colours caand cb by placing ca in front of cb
/// </summary>
/// <param name="cb">- disk/horizon color</param>
/// <param name="balpha">- disk/horizon alpha</param>
/// <param name="ca">- current color</param>
/// <param name="aalpha">- current alpha</param>
/// <returns>new color</returns>
__inline__ __device__ vector3 BlendColors(vector3 cb, float balpha, vector3 ca, float aalpha)
{
	return  ca + cb * (balpha * (1.f - aalpha));
}


/// <summary>
/// Is for the final alpha channel after blending
/// </summary>
/// <param name="balpha">- disk/horizon alpha</param>
/// <param name="aalpha">- current alpha</param>
/// <returns></returns>
__inline__ __device__ float BlendAlpha(float balpha, float aalpha)
{
	//return aalpha + balpha * (1.f - aalpha);
	return fmaf(balpha, (1.f - aalpha), aalpha);
}

// https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
__inline__ __device__ float sRGBtoRGB(float color)
{
	if (color > 0.04045f)
		return powf(((color + 0.055f) * 0.947867f), 2.4f);
	else
		return color * 0.077399f;
}

__inline__ __device__ float RGBtosRGB(float color)
{
	if (color > 0.0031308f)
		return fmaf(powf(color, 0.416667f), (1.055f), -0.055f);
	else
		return	12.92f * color;
}

/// <summary>
/// Convert sRGB to linear RGB
/// </summary>
/// <param name="pixel">- pixel to be converted </param>
/// <returns>sRGB pixel</returns>
__inline__ __device__ vector3 sRGBtoRGB(vector3 pixel)
{
	vector3 res;
	res.x = sRGBtoRGB(pixel.x);
	res.y = sRGBtoRGB(pixel.y);
	res.z = sRGBtoRGB(pixel.z);
	return res;
}

/// <summary>
/// Convert linear RGB to sRGB
/// </summary>
/// <param name="pixel">- pixel to be converted </param>
/// <returns>sRGB pixel</returns>
__inline__ __device__ vector3 RGBtosRGB(vector3 pixel)
{
	vector3 res;
	res.x = RGBtosRGB(pixel.x);
	res.y = RGBtosRGB(pixel.y);
	res.z = RGBtosRGB(pixel.z);
	return res;
}

// temperature: 1000K to 30'000K 
const vector3 colors[] = {
vector3{ 255, 50 , 0  },
vector3{ 255, 61 , 0  },
vector3{ 255, 62 , 0  },
vector3{ 255, 80 , 0  },
vector3{ 255, 73 , 0  },
vector3{ 255, 90 , 0  },
vector3{ 255, 90 , 0  },
vector3{ 255, 102, 0  },
vector3{ 255, 110, 0  },
vector3{ 255, 114, 0  },
vector3{ 255, 118, 0  },
vector3{ 255, 123, 0  },
vector3{ 255, 127, 0  },
vector3{ 255, 131, 2  },
vector3{ 255, 135, 2  },
vector3{ 255, 137, 1  },
vector3{ 255, 141, 1  },
vector3{ 255, 144, 3  },
vector3{ 255, 148, 8  },
vector3{ 255, 154, 17 },
vector3{ 255, 159, 26 },
vector3{ 255, 163, 32 },
vector3{ 255, 165, 42 },
vector3{ 255, 169, 47 },
vector3{ 255, 174, 56 },
vector3{ 255, 177, 62 },
vector3{ 255, 180, 69 },
vector3{ 255, 183, 72 },
vector3{ 255, 182, 74 },
vector3{ 255, 185, 76 },
vector3{ 255, 188, 81 },
vector3{ 255, 191, 86 },
vector3{ 255, 197, 97 },
vector3{ 255, 198, 101},
vector3{ 255, 199, 106},
vector3{ 255, 203, 109},
vector3{ 255, 204, 114},
vector3{ 255, 208, 119},
vector3{ 255, 209, 124},
vector3{ 255, 211, 128},
vector3{ 255, 213, 134},
vector3{ 254, 215, 138},
vector3{ 254, 216, 141},
vector3{ 255, 219, 147},
vector3{ 255, 221, 150},
vector3{ 255, 224, 154},
vector3{ 255, 224, 157},
vector3{ 255, 227, 160},
vector3{ 255, 229, 164},
vector3{ 255, 232, 168},
vector3{ 255, 234, 172},
vector3{ 255, 235, 172},
vector3{ 255, 234, 176},
vector3{ 255, 235, 177},
vector3{ 255, 237, 181},
vector3{ 255, 239, 187},
vector3{ 255, 242, 190},
vector3{ 255, 243, 194},
vector3{ 255, 244, 198},
vector3{ 255, 246, 202},
vector3{ 255, 248, 206},
vector3{ 255, 249, 209},
vector3{ 255, 251, 213},
vector3{ 255, 251, 214},
vector3{ 255, 251, 216},
vector3{ 255, 251, 219},
vector3{ 255, 253, 220},
vector3{ 255, 254, 221},
vector3{ 255, 254, 224},
vector3{ 254, 254, 226},
vector3{ 254, 254, 228},
vector3{ 250, 254, 229},
vector3{ 251, 254, 233},
vector3{ 248, 255, 233},
vector3{ 248, 255, 237},
vector3{ 246, 255, 236},
vector3{ 244, 255, 239},
vector3{ 243, 254, 238},
vector3{ 242, 254, 240},
vector3{ 242, 254, 242},
vector3{ 242, 255, 245},
vector3{ 242, 255, 246},
vector3{ 242, 255, 246},
vector3{ 242, 255, 246},
vector3{ 240, 255, 250},
vector3{ 240, 255, 250},
vector3{ 239, 255, 252},
vector3{ 237, 255, 252},
vector3{ 237, 255, 255},
vector3{ 235, 255, 254},
vector3{ 232, 253, 255},
vector3{ 230, 251, 254},
vector3{ 226, 250, 254},
vector3{ 225, 249, 253},
vector3{ 225, 248, 254},
vector3{ 225, 248, 254},
vector3{ 224, 246, 255},
vector3{ 223, 245, 255},
vector3{ 221, 243, 255},
vector3{ 220, 242, 255},
vector3{ 218, 241, 255},
vector3{ 217, 240, 255},
vector3{ 217, 240, 255},
vector3{ 217, 240, 255},
vector3{ 212, 238, 255},
vector3{ 212, 238, 255},
vector3{ 211, 236, 255},
vector3{ 210, 235, 255},
vector3{ 207, 234, 255},
vector3{ 206, 233, 255},
vector3{ 205, 231, 255},
vector3{ 205, 231, 255},
vector3{ 205, 231, 255},
vector3{ 205, 231, 255},
vector3{ 203, 231, 255},
vector3{ 202, 230, 255},
vector3{ 201, 229, 255},
vector3{ 200, 228, 255},
vector3{ 199, 229, 255},
vector3{ 199, 229, 255},
vector3{ 196, 225, 255},
vector3{ 196, 225, 255},
vector3{ 194, 225, 255},
vector3{ 193, 224, 255},
vector3{ 193, 223, 255},
vector3{ 192, 222, 255},
vector3{ 191, 221, 255},
vector3{ 191, 221, 255},
vector3{ 191, 222, 255},
vector3{ 190, 220, 255},
vector3{ 189, 219, 255},
vector3{ 189, 219, 255},
vector3{ 189, 219, 255},
vector3{ 189, 219, 255},
vector3{ 186, 219, 255},
vector3{ 185, 217, 255},
vector3{ 185, 217, 255},
vector3{ 185, 217, 255},
vector3{ 184, 217, 255},
vector3{ 183, 217, 255},
vector3{ 183, 217, 255},
vector3{ 182, 216, 255},
vector3{ 181, 216, 255},
vector3{ 181, 216, 255},
vector3{ 180, 213, 255},
vector3{ 180, 213, 255},
vector3{ 180, 213, 255},
vector3{ 179, 212, 255},
vector3{ 179, 212, 255},
vector3{ 178, 211, 255},
vector3{ 178, 211, 255},
vector3{ 178, 211, 255},
vector3{ 177, 210, 255},
vector3{ 177, 210, 255},
vector3{ 177, 209, 255},
vector3{ 176, 208, 255},
vector3{ 176, 208, 255},
vector3{ 175, 207, 255},
vector3{ 175, 207, 255},
vector3{ 175, 207, 255},
vector3{ 174, 207, 255},
vector3{ 174, 207, 255},
vector3{ 174, 207, 255},
vector3{ 173, 207, 255},
vector3{ 173, 206, 255},
vector3{ 172, 205, 255},
vector3{ 172, 205, 255},
vector3{ 172, 205, 255},
vector3{ 171, 204, 255},
vector3{ 171, 204, 255},
vector3{ 171, 204, 255},
vector3{ 170, 203, 255},
vector3{ 170, 203, 255},
vector3{ 169, 202, 255},
vector3{ 169, 202, 255},
vector3{ 169, 202, 255},
vector3{ 167, 203, 255},
vector3{ 167, 203, 255},
vector3{ 167, 203, 255},
vector3{ 166, 202, 255},
vector3{ 166, 201, 255},
vector3{ 165, 200, 255},
vector3{ 165, 200, 255},
vector3{ 165, 200, 255},
vector3{ 165, 200, 255},
vector3{ 165, 200, 255},
vector3{ 165, 200, 255},
vector3{ 164, 199, 255},
vector3{ 164, 199, 255},
vector3{ 163, 198, 255},
vector3{ 163, 198, 255},
vector3{ 162, 199, 255},
vector3{ 161, 198, 255},
vector3{ 159, 199, 255},
vector3{ 159, 198, 255},
vector3{ 159, 198, 255},
vector3{ 159, 198, 255},
vector3{ 159, 198, 255},
vector3{ 159, 198, 255},
vector3{ 159, 198, 255},
vector3{ 159, 198, 255},
vector3{ 159, 198, 255},
vector3{ 159, 198, 255},
vector3{ 158, 197, 255},
vector3{ 158, 197, 255},
vector3{ 157, 196, 255},
vector3{ 157, 196, 255},
vector3{ 157, 196, 255},
vector3{ 156, 195, 255},
vector3{ 156, 195, 255},
vector3{ 156, 195, 255},
vector3{ 156, 195, 255},
vector3{ 156, 195, 255},
vector3{ 156, 195, 255},
vector3{ 156, 194, 255},
vector3{ 156, 194, 255},
vector3{ 156, 194, 255},
vector3{ 156, 194, 255},
vector3{ 156, 194, 255},
vector3{ 155, 193, 255},
vector3{ 155, 193, 255},
vector3{ 154, 193, 255},
vector3{ 154, 193, 255},
vector3{ 154, 193, 255},
vector3{ 153, 193, 255},
vector3{ 153, 193, 255},
vector3{ 153, 193, 255},
vector3{ 153, 193, 255},
vector3{ 153, 193, 255},
vector3{ 153, 193, 255},
vector3{ 153, 193, 255},
vector3{ 153, 193, 255},
vector3{ 152, 192, 255},
vector3{ 152, 192, 255},
vector3{ 152, 192, 255},
vector3{ 152, 192, 255},
vector3{ 152, 192, 255},
vector3{ 152, 192, 255},
vector3{ 152, 192, 255},
vector3{ 152, 192, 255},
vector3{ 152, 191, 255},
vector3{ 152, 191, 255},
vector3{ 152, 191, 255},
vector3{ 151, 190, 255},
vector3{ 151, 190, 255},
vector3{ 150, 189, 255},
vector3{ 150, 189, 255},
vector3{ 150, 189, 255},
vector3{ 150, 189, 255},
vector3{ 150, 189, 255},
vector3{ 150, 189, 255},
vector3{ 150, 189, 255},
vector3{ 150, 189, 255},
vector3{ 150, 189, 255},
vector3{ 150, 189, 255},
vector3{ 150, 189, 255},
vector3{ 149, 188, 255},
vector3{ 149, 188, 255},
vector3{ 149, 188, 255},
vector3{ 149, 188, 255},
vector3{ 149, 188, 255},
vector3{ 149, 188, 255},
vector3{ 149, 188, 255},
vector3{ 149, 188, 255},
vector3{ 148, 187, 255},
vector3{ 148, 187, 255},
vector3{ 148, 187, 255},
vector3{ 148, 187, 255},
vector3{ 148, 187, 255},
vector3{ 148, 187, 255},
vector3{ 148, 187, 255},
vector3{ 148, 187, 255},
vector3{ 146, 186, 255},
vector3{ 146, 186, 255},
vector3{ 146, 186, 255},
vector3{ 146, 186, 255},
vector3{ 146, 186, 255},
vector3{ 146, 186, 255},
vector3{ 146, 186, 255},
vector3{ 146, 186, 255},
vector3{ 146, 186, 255},
vector3{ 146, 186, 255},
vector3{ 146, 186, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 145, 185, 255},
vector3{ 144, 184, 255},
vector3{ 144, 184, 255},
vector3{ 144, 184, 255},
vector3{ 144, 184, 255},
vector3{ 144, 184, 255},
vector3{ 144, 184, 255},
vector3{ 144, 184, 255},
vector3{ 144, 184, 255},
vector3{ 144, 184, 255},
vector3{ 144, 184, 255},
vector3{ 144, 184, 255},
vector3{ 143, 183, 255},
vector3{ 143, 183, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 143, 183, 255},
vector3{ 143, 183, 255},
vector3{ 143, 183, 255},
vector3{ 143, 183, 255},
vector3{ 143, 183, 255},
vector3{ 143, 183, 255},
vector3{ 143, 183, 255},
vector3{ 143, 183, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 142, 182, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 141, 181, 255},
vector3{ 139, 181, 255},
vector3{ 139, 181, 255},
vector3{ 139, 181, 255},
vector3{ 139, 181, 255},
vector3{ 139, 181, 255},
vector3{ 139, 181, 255},
vector3{ 139, 181, 255},
vector3{ 139, 181, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 138, 180, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 138, 179, 255},
vector3{ 138, 179, 255},
vector3{ 138, 179, 255},
vector3{ 137, 179, 255},
vector3{ 137, 179, 255},
vector3{ 136, 178, 255},
vector3{ 136, 178, 255},
vector3{ 136, 178, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 136, 177, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 135, 176, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 133, 175, 255},
vector3{ 133, 175, 255},
vector3{ 133, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
vector3{ 134, 175, 255},
};