//pass
//--gridDim=[64,64] --blockDim=[16,16]

#include "common.h"

__global__ void transposeDiagonal(float *odata, float *idata, int width, int height, int nreps)
{
    __requires(width == 1024);
    __requires(height == 1024);

    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    // do diagonal reordering
    int bid = blockIdx.x + gridDim.x*blockIdx.y;
    int blockIdx_y = (width == height) ? blockIdx.x : bid%gridDim.y;
    int blockIdx_x = width == height ?
      (blockIdx.x+blockIdx.y)%gridDim.x :
      ((bid/gridDim.y)+blockIdx_y)%gridDim.x;

    // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
    // and similarly for y

    int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    // NATHAN: requires component-like access breaking
    for (int r=0; r < nreps; r++)
    {
        for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
        {
            tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
        }

        __syncthreads();

        for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
        {
            odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
        }

        // IMPERIAL EDIT: add barrier
        __syncthreads();
    }
}
