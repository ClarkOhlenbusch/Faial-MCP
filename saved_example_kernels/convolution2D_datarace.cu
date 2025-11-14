#include <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


#define MASK 5
#define TILE_WIDTH 16
#define RADIUS MASK/2
#define AREA (TILE_WIDTH + MASK - 1)
#define GRID_SIZE(x) (ceil((float)x/TILE_WIDTH))

__device__ inline void setIndexes(unsigned int d,
                                  unsigned int &dX,
                                  unsigned int &dY,
                                  int &sX, int &sY){
  dX = d % AREA;
  dY = d / AREA;
  sX = blockIdx.x * TILE_WIDTH + dX - RADIUS;
  sY = blockIdx.y * TILE_WIDTH + dY - RADIUS;
}

__global__ void convolution(float* I, const float* __restrict__ M, float* P,
                            int channels, int width, int height) {
  __shared__ float tmp[AREA][AREA];

  float acc;
  int sourceY, sourceX;
  unsigned int source, destination;
  unsigned int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
  unsigned int x = blockIdx.x * TILE_WIDTH + threadIdx.x;

  for (unsigned int k = 0; k < channels; k++) {
    unsigned int destinationY, destinationX;
    destination = threadIdx.y * TILE_WIDTH + threadIdx.x;
    setIndexes(destination,
               destinationX,
               destinationY,
               sourceX, sourceY);
    source = (sourceY * width + sourceX) * channels + k;
    if (sourceY >= 0 && sourceY < height && sourceX >= 0 && sourceX < width)
      tmp[destinationY][destinationX] = I[source];
    else
      tmp[destinationY][destinationX] = 0;

    unsigned int destinationY2, destinationX2;
    destination = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    setIndexes(destination,
               destinationX2,
               destinationY2,
               sourceX, sourceY);
    source = (sourceY * width + sourceX) * channels + k;

    if (destinationY2 < AREA)
      if (sourceY >= 0 && sourceY < height && sourceX >= 0 && sourceX < width)
        tmp[destinationY2][destinationX2] = I[source];
      else
        tmp[destinationY2][destinationX2] = 0;

// __syncthreads(); // DATA RACE: Missing syncthreads causes threads to read uninitialized shared memory

    acc = 0;
    #pragma unroll
    for (unsigned int i = 0; i < MASK; i++)
      #pragma unroll
      for (unsigned int j = 0; j < MASK; j++)
        acc += tmp[threadIdx.y + i][threadIdx.x + j] * M[i * MASK + j];

    if (y < height && x < width) P[(y * width + x) * channels + k] = min(max(acc, 0.0), 1.0);

    __syncthreads();
  }
}
