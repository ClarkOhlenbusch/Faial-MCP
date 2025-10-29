
#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"

#include "include/cu_assert.h"
#include "include/config.h"
#include "include/blackhole.cuh"
#include "include/bmp.h"
#include "include/jpeg.h"

void CreateCUDABlackHole(void)
{
	cudaSetDevice(0);

	unsigned char* host_bitmap = null;
	unsigned char* dev_image;

#pragma region Time measurement 
	cudaEvent_t startTime, stopTime;

	cudaAssert(cudaEventCreate(&startTime));
	cudaAssert(cudaEventCreate(&stopTime));
	cudaAssert(cudaEventRecord(startTime));
#pragma endregion

	cudaHostAlloc((void**)&host_bitmap, DIM_X * DIM_Y * 3 * sizeof(unsigned char), cudaHostAllocWriteCombined);
	cudaAssert(cudaMalloc((void**)&dev_image, DIM_X * DIM_Y * 3 * sizeof(unsigned char)));
	ProcessRayTracing(dev_image);
	cudaAssert(cudaMemcpy(host_bitmap, dev_image, DIM_X * DIM_Y * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	cudaAssert(cudaFree(dev_image));

#pragma region Time measurement 
	cudaAssert(cudaEventRecord(stopTime));
	cudaAssert(cudaEventSynchronize(stopTime));

	float time;
	cudaAssert(cudaEventElapsedTime(&time, startTime, stopTime));

	printf("time: %.2f ms\n", time);
#pragma endregion

#if !defined(JPEG)
	BMPfile_t saveImage;
	CreateBMP(&saveImage, DIM_X, DIM_Y, host_bitmap);
	SaveBMPtoFile(&saveImage, FILENAME);
#else
	SaveJPEGtoFile(host_bitmap, FILENAME);
#endif // BMP

	cudaAssert(cudaFreeHost(host_bitmap));
	cudaAssert(cudaDeviceReset());

}

int main()
{
	CreateCUDABlackHole();
	return 0;
}

/*
int main()
{
	cudaSetDevice(0);

	unsigned char* host_bitmap = null;
	cudaHostAlloc((void**)&host_bitmap, DIM_X * DIM_Y * 3 * sizeof(unsigned char), cudaHostAllocWriteCombined);

	//DataBlock data;
	//CPUBitmap cpuBitmap(host_bitmap, DIM_X, DIM_Y, &data);

	unsigned char*	dev_image;
	float*			dev_imageFloat;

	Camera_t*		camera;
	BlackHole_t*	blackHole;

	Camera_t*		dev_camera;
	BlackHole_t*	dev_blackHole;
	Star_t*			dev_stars;

	unsigned char*	dev_bitmap;

	float*			dev_gaussKernel;
	float*			dev_gaussKernelSmall;

	curandStateXORWOW_t* dev_random;

	cudaEvent_t startTime, stopTime;

	camera = CreateCamera();
	blackHole = CreateBlackHole();

	SetBlackHole(blackHole, DISK_IN, DISK_OUT);
	
#ifdef RAYTRACING
	//original settings
	//SetCameraParameters(camera, vector3{0.f, 4.3f, -9.f}, FOV_TANGENT, Vector3Zero(), vector3{0.f, 1.f, 0.f});
	SetCameraParameters(camera, vector3{ 0.0f, -4.3f, -18.f }, FOV_TANGENT, Vector3Zero(), vector3{ 0.f, 1.f, 0.f });
#else
#ifdef JPEG
	SetCameraParameters(camera, vector3{ 0.0f, -1.f, -22.f }, FOV_TANGENT, Vector3Zero(), vector3{ 0.2f, -1.f, 0.f });
#else
	SetCameraParameters(camera, vector3{ 0.0f, -1.f, -20.f }, FOV_TANGENT, Vector3Zero(), vector3{ 0.2f, -1.f, 0.f });
#endif // JPEG
	
#endif // RAYTRACING

#pragma region Time measurement 
	cudaAssert(cudaEventCreate(&startTime));

	cudaAssert(cudaEventCreate(&stopTime));

	cudaAssert(cudaEventRecord(startTime));
	
#pragma endregion

#pragma region Device memory allocation

	cudaAssert(cudaMalloc((void**)&dev_image, DIM_X * DIM_Y * 3 * sizeof(unsigned char)));
	//data.dev_image = dev_image;

	cudaAssert(cudaMalloc((void**)&dev_bitmap, DIM_X_N(BITMAP_MULTIPIER) * DIM_Y_N(BITMAP_MULTIPIER) * 4 * sizeof(unsigned char)));
	
	cudaAssert(cudaMalloc((void**)&dev_imageFloat, DIM_X * DIM_Y * 3 * sizeof(float)));
	
	//data.dev_imageFloat = dev_imageFloat;
	
	cudaAssert(cudaMalloc((void**)&dev_blackHole, sizeof(BlackHole_t)));

	cudaAssert(cudaMalloc((void**)&dev_camera, sizeof(Camera_t)));
	
	cudaAssert(cudaMalloc((void**)&dev_stars, DIM_X_N(BITMAP_MULTIPIER) * DIM_Y_N(BITMAP_MULTIPIER) / STAR_DENSITY * sizeof(Star_t)));
	
	cudaAssert(cudaMalloc((void**)&dev_random, DIM_X_N(BITMAP_MULTIPIER) * DIM_Y_N(BITMAP_MULTIPIER) / STAR_DENSITY * sizeof(curandStateXORWOW_t)));
	

#if defined(GAUSSIAN_ROW_COL)
	cudaAssert(cudaMalloc((void**)&dev_gaussKernel, (2 * GAUSS_SIZE + 1) * sizeof(float)));
	
	cudaAssert(cudaMalloc((void**)&dev_gaussKernelSmall, (2 * GAUSS_SIZE_SMALL + 1) * sizeof(float)));
	
#else
	cudaAssert(cudaMalloc((void**)&dev_gaussKernel, GAUSS_KERNEL_SIZE * sizeof(float)));
	
#endif
	
#pragma endregion
	// blackbody colors 
	cudaAssert(cudaMemcpyToSymbol(colorMap, colors, 500 * sizeof(vector3)));
	
	// small gauss kernel 
	//cudaMemcpyToSymbol(aaKernel, aaGauss, 49 * sizeof(float));

	//Gaussian efect
	//dim3 kerngrid(2 * GAUSS_SIZE + 1, 2 * GAUSS_SIZE + 1);
	
#if defined(GAUSSIAN_ROW_COL)
	//dim3 kerngrid(2 * GAUSS_SIZE + 1);
	//dim3 kerngrid(GAUSS_SIZE + 1);
	GenerateGaussianKernel1D << < GAUSS_SIZE + 1, 1 >> > (dev_gaussKernel, (float)((GAUSS_SIZE / 3) * (GAUSS_SIZE / 3))); //289 = 17^2 // 1156 = 34^2
	cudaAssert(cudaMemcpyToSymbol(convolveKernel1D, dev_gaussKernel, (2 * GAUSS_SIZE + 1) * sizeof(float), 0, cudaMemcpyDeviceToDevice));
	
	GenerateGaussianKernelSmall1D << < GAUSS_SIZE_SMALL + 1, 1 >> > (dev_gaussKernelSmall, (float)((GAUSS_SIZE_SMALL / 3) * (GAUSS_SIZE_SMALL / 3))); //289 = 17^2
	cudaAssert(cudaMemcpyToSymbol(convolveKernelSmall1D, dev_gaussKernelSmall, (2 * GAUSS_SIZE_SMALL + 1) * sizeof(float), 0, cudaMemcpyDeviceToDevice));
#else
	dim3 kerngrid(GAUSS_SIZE + 1, GAUSS_SIZE + 1);
	GenerateGaussianKernel << < kerngrid, 1 >> > (dev_gaussKernel, 289.f); //289 = 17^2
	cudaMemcpyToSymbol(convolveKernel, dev_gaussKernel, GAUSS_KERNEL_SIZE * sizeof(float), 0, cudaMemcpyDeviceToDevice);
#endif
	
	dim3 grid(DIM_X / 16, DIM_Y / 16);
	dim3 threads(16, 16);

	dim3 gridBmp(DIM_X_N(BITMAP_MULTIPIER) / 16, DIM_Y_N(BITMAP_MULTIPIER) / 16);
	// threads identical as 'threads'

	dim3 grid1Axis(DIM_X / 16);
	dim3 threads1Axis(16);

	dim3 gridStar(DIM_X_N(BITMAP_MULTIPIER)* DIM_Y_N(BITMAP_MULTIPIER) / STAR_DENSITY / 32);
	// threads identical as 'threads1Axis'

	cudaAssert(cudaMemcpy(dev_blackHole, blackHole, sizeof(BlackHole_t), cudaMemcpyHostToDevice));
	cudaAssert(cudaMemcpy(dev_camera, camera, sizeof(Camera_t), cudaMemcpyHostToDevice));
	
	// prepare very dark blue sky
	//PrepareBackground<< <gridBmp, threads >> >(dev_bitmap);
	cudaAssert(cudaMemset(dev_bitmap, 0x00, DIM_X_N(BITMAP_MULTIPIER)* DIM_Y_N(BITMAP_MULTIPIER) * sizeof(float)));
	
	// generate some stars
	PrepareRandom << <gridStar, 32 >> > (dev_random);
	GenerateSky << <gridStar, 32 >> > (dev_random, dev_stars);

	// draw stars using CPU on bitmap
	DrawSky << < gridStar, 32 >> >(dev_bitmap, dev_stars);

	// copy bitmap to texture memory
	cudaTextureObject_t textureBitmap = 0;

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = dev_bitmap;
	resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
	resDesc.res.linear.desc.x = 8; // bits per channel
	resDesc.res.linear.desc.y = 8; // bits per channel
	resDesc.res.linear.desc.z = 8; // bits per channel
	resDesc.res.linear.desc.w = 8; // bits per channel
	resDesc.res.linear.sizeInBytes = DIM_X_N(BITMAP_MULTIPIER) * DIM_Y_N(BITMAP_MULTIPIER) * 4 * sizeof(unsigned char);
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.readMode = cudaReadModeElementType;

	cudaAssert(cudaCreateTextureObject(&textureBitmap, &resDesc, &texDesc, null));
	
	// render black hole with accretion disc 
	RayTrace << < grid, threads >> > (dev_imageFloat, textureBitmap, dev_camera, dev_blackHole);

	// add small antialiasing using small gauss kernel
	//Antialiasing << < grid, threads >> > (dev_imageFloat);

	// add gauss effect in postprocessing
#if  defined(GAUSSIAN)
	GaussianConvolve << <grid, threads >> > (dev_imageFloat, dev_image);
#elif defined(GAUSSIAN_ROW_COL)
	float* dev_gaussTmp;
	cudaAssert(cudaMalloc((void**)&dev_gaussTmp, DIM_X * DIM_Y * 3 * sizeof(float)));

	GaussianConvolveRowSmall << < grid, threads >> > (dev_imageFloat, dev_gaussTmp);
	GaussianConvolveColSmall << < grid, threads >> > (dev_gaussTmp, dev_imageFloat);
	//FinishRayTracing << <grid, threads >> > (dev_imageFloat, dev_image);

	GaussianConvolveRow << < grid, threads >> > (dev_imageFloat, dev_gaussTmp);
	GaussianConvolveCol << < grid, threads >> > (dev_gaussTmp, dev_imageFloat);
#endif // GAUSSIAN
	FinishRayTracing << <grid, threads >> > (dev_imageFloat, dev_image);

	cudaAssert(cudaMemcpy(host_bitmap, dev_image, DIM_X * DIM_Y * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	cudaAssert(cudaEventRecord(stopTime));
	
	cudaAssert(cudaEventSynchronize(stopTime));

	float time;
	cudaAssert(cudaEventElapsedTime(&time, startTime, stopTime));
	
	printf("time: %.2f ms\n", time);

#pragma region Cleanup CUDA memory
	cudaAssert(cudaFree(dev_bitmap));

	cudaAssert(cudaFree(dev_blackHole));
	
	cudaAssert(cudaFree(dev_camera));
	
	cudaAssert(cudaFree(dev_gaussKernel));

	cudaAssert(cudaFree(dev_image));
	
	cudaAssert(cudaFree(dev_imageFloat));

	cudaAssert(cudaFree(dev_random));

	cudaAssert(cudaFree(dev_stars));
	
	cudaAssert(cudaDestroyTextureObject(textureBitmap));
	
#pragma endregion

#if !defined(JPEG)
	BMPfile_t saveImage;
	CreateBMP(&saveImage, DIM_X, DIM_Y, host_bitmap);
	SaveBMPtoFile(&saveImage, FILENAME);

	cudaAssert(cudaFreeHost(host_bitmap));
#else
	SaveJPEGtoFile(host_bitmap, FILENAME);
	cudaAssert(cudaFreeHost(host_bitmap));
#endif // BMP

	cudaAssert(cudaDeviceReset());

}
*/