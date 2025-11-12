// Difficulty: 3
//pass
//--blockDim=256 --gridDim=1

// Simple prefix sum (scan) kernel with data race
__global__ void prefixSum(int *input, int *output, int n) {
    __shared__ int temp[256];

    int tid = threadIdx.x;

    // Load input into shared memory
    if (tid < n) {
        temp[tid] = input[tid];
    }

    // __syncthreads();

    // Perform inclusive scan
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }

        __syncthreads();

        if (tid >= stride) {
            temp[tid] += val;
        }

        __syncthreads();
    }

    // Write result
    if (tid < n) {
        output[tid] = temp[tid];
    }
}
