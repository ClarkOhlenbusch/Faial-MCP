#pragma once

#include "../include/jpeg.h"
#include "../include/cu_assert.h"

const char* nvjpegGetErrorString(nvjpegStatus_t status)
{
	switch (status)
	{
	case NVJPEG_STATUS_SUCCESS: return "Nvjpeg Success";
	case NVJPEG_STATUS_NOT_INITIALIZED: return "Nvjpeg Handle not initialized";
	case NVJPEG_STATUS_INVALID_PARAMETER: return "Nvjpeg Invalid parameter";
	case NVJPEG_STATUS_BAD_JPEG: return "Nvjpeg Bad JPEG";
	case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "Nvjpeg JPEG is not supported";
	case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "Nvjpeg Allocator failure";
	case NVJPEG_STATUS_EXECUTION_FAILED: return "Nvjpeg Execution failed"; // specially when memory is not on device
	case NVJPEG_STATUS_ARCH_MISMATCH: return "Nvjpeg Architecture mismatch";
	case NVJPEG_STATUS_INTERNAL_ERROR: return "Nvjpeg Internal error";
	case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED: return "Nvjpeg Implementation is not supported";
	}
	return "unknown error";
}

#define nvjpegAssert(result) if((result) != NVJPEG_STATUS_SUCCESS){printf("%s in line: %d in file: %s\n\n", nvjpegGetErrorString((result)), __LINE__, __FILE__); exit(-1);}

__global__ void InvertBitmap(unsigned char* bitmap)
{
	int gpu_x = threadIdx.x + blockIdx.x * blockDim.x;

	unsigned char b, g, r;

	for (int i = 0; i < DIM_Y / 2; i++)
	{
		int gpu_offset = gpu_x + i * blockDim.x * gridDim.x;
		int gpu_dim_offset = gpu_x + (DIM_Y - 1 - i) * blockDim.x * gridDim.x;

		b = bitmap[3 * gpu_offset + 0];
		g = bitmap[3 * gpu_offset + 1];
		r = bitmap[3 * gpu_offset + 2];

		bitmap[3 * gpu_offset + 0] = bitmap[3 * gpu_dim_offset + 0];
		bitmap[3 * gpu_offset + 1] = bitmap[3 * gpu_dim_offset + 1];
		bitmap[3 * gpu_offset + 2] = bitmap[3 * gpu_dim_offset + 2];

		bitmap[3 * gpu_dim_offset + 0] = b;
		bitmap[3 * gpu_dim_offset + 1] = g;
		bitmap[3 * gpu_dim_offset + 2] = r;
	}
}

void SaveJPEGtoFile(unsigned char* host_bitmap, const char* filename)
{
	nvjpegHandle_t nv_handle;
	nvjpegEncoderState_t nv_enc_state;
	nvjpegEncoderParams_t nv_enc_params;
	cudaStream_t stream = NULL;

	unsigned char* dev_jpeg;
	cudaAssert(cudaMalloc((void**)&dev_jpeg, DIM_X * DIM_Y * 3 * sizeof(unsigned char)));
	cudaAssert(cudaMemcpy(dev_jpeg, host_bitmap, DIM_X * DIM_Y * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

	dim3 gridjpeg(DIM_X / 32);
	dim3 threadsjpeg(32);

	InvertBitmap << <gridjpeg, threadsjpeg >> > (dev_jpeg);
	cudaAssert(cudaDeviceSynchronize());

	// initialize nvjpeg structures
	nvjpegAssert(nvjpegCreateSimple(&nv_handle));
	nvjpegAssert(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
	nvjpegAssert(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));

	nvjpegAssert(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream));
	nvjpegAssert(nvjpegEncoderParamsSetQuality(nv_enc_params, 90, stream));

	nvjpegImage_t nv_image;
	nv_image.channel[0] = dev_jpeg;
	nv_image.pitch[0] = 3 * DIM_X;

	// Compress image
	nvjpegAssert(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
		&nv_image, NVJPEG_INPUT_BGRI, DIM_X, DIM_Y, stream));

	// get compressed stream size
	size_t length = 1;
	nvjpegAssert(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));

	// get stream itself
	cudaStreamSynchronize(stream);
	unsigned char* jpeg = (unsigned char*)malloc(length * sizeof(unsigned char));
	//unsigned char* jpeg;
	//cudaMallocHost((void**)&jpeg, length * sizeof(unsigned char));
	nvjpegAssert(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg, &length, 0));

	// write stream to file
	cudaStreamSynchronize(stream);
	FILE* file;
	if ((file = fopen(filename, "wb")) != 0)
		if (jpeg != 0) fwrite(jpeg, sizeof(unsigned char), length, file);
		else
			printf("Error when trying to open file %s", filename);

	// cleanup
	nvjpegAssert(nvjpegEncoderStateDestroy(nv_enc_state));
	nvjpegAssert(nvjpegEncoderParamsDestroy(nv_enc_params));
	nvjpegDestroy(nv_handle);
	cudaFree(dev_jpeg);

}