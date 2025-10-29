//pass
//--gridDim=[16,32,1] --blockDim=[8,4,2] --warp-sync=32

#include "common2.h"

__global__ void CUDAkernel2DCT(float *dst, float *src, int ImgStride)
{
    __requires(ImgStride == 512);
    __shared__ float block[KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE];

    int OffsThreadInRow = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    int OffsThreadInCol = threadIdx.z * BLOCK_SIZE;
    src += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) + blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
    dst += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) + blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
    float *bl_ptr = block + OffsThreadInCol * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow;

#pragma unroll

    for (unsigned int i = 0;
         #define tid (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)
         __global_invariant(__implies(tid/32 == __other_int(tid)/32
                                      & blockIdx.x == __other_int(blockIdx.x)
                                      & blockIdx.y == __other_int(blockIdx.y)
                                      & blockIdx.z == __other_int(blockIdx.z), !__write(block))),
         #define blockOffset (OffsThreadInCol * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow)
         __global_invariant(__write_implies(block, (__write_offset(block)/sizeof(float) - blockOffset)%KER2_SMEMBLOCK_STRIDE == 0)),
         __global_invariant(__write_implies(block, (__write_offset(block)/sizeof(float) - blockOffset)/KER2_SMEMBLOCK_STRIDE < BLOCK_SIZE)),
         i < BLOCK_SIZE; i++)
        bl_ptr[i * KER2_SMEMBLOCK_STRIDE] = src[i * ImgStride];

    //process rows
    CUDAsubroutineInplaceDCTvector(block + (OffsThreadInCol + threadIdx.x) * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow - threadIdx.x, 1);

    //process columns
    CUDAsubroutineInplaceDCTvector(bl_ptr, KER2_SMEMBLOCK_STRIDE);

    for (unsigned int i = 0; i < BLOCK_SIZE; i++)
        dst[i * ImgStride] = bl_ptr[i * KER2_SMEMBLOCK_STRIDE];
}
