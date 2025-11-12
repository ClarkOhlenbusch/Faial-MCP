// Difficulty: 6
#include <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#ifndef BLOCK_SIZE
# define BLOCK_SIZE 256
#endif

#define HALF_BLOCK_SIZE BLOCK_SIZE << 1

#define MEM_SIZE(size)  \
  size * sizeof(float)

__global__ void post_scan(float* in, float* add, int len) {
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

  if (blockIdx.x) {
     if (start + t < len) in[start + t] += add[blockIdx.x - 1];
     if (start + BLOCK_SIZE + t < len) in[start + BLOCK_SIZE + t] += add[blockIdx.x - 1];
  }
}

__global__ void scan(float* in, float* out, float* post, int len) {
  __shared__ float scan_array[HALF_BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
  int index;

  if (start + t < len) scan_array[t] = in[start + t];
  else scan_array[t] = 0;

  if (start + BLOCK_SIZE + t < len) scan_array[BLOCK_SIZE + t] = in[start + BLOCK_SIZE + t];
  else scan_array[BLOCK_SIZE + t] = 0;

  for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
     index = (t + 1) * stride * 2 - 1;
     if (index < 2 * BLOCK_SIZE) scan_array[index] += scan_array[index - stride];
     __syncthreads();
  }

  for (unsigned int stride = BLOCK_SIZE >> 1; stride; stride >>= 1) {
     index = (t + 1) * stride * 2 - 1;
     if (index + stride < 2 * BLOCK_SIZE) scan_array[index + stride] += scan_array[index];
     __syncthreads();
  }

  if (start + t < len) out[start + t] = scan_array[t];
  if (start + BLOCK_SIZE + t < len) out[start + BLOCK_SIZE + t] = scan_array[BLOCK_SIZE + t];

  if (post && t == 0) post[blockIdx.x] = scan_array[2 * BLOCK_SIZE - 1];
}
