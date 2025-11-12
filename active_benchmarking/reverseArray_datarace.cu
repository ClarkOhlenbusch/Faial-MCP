// Difficulty: 3
//pass
//--blockDim=256 --gridDim=1

// Simple array reversal kernel using shared memory with data race
__global__ void reverseArray(int *input, int *output, int n) {
    __shared__ int temp[256];

    int tid = threadIdx.x;
    int reversedIdx = blockDim.x - 1 - tid;

    // Load input into shared memory
    if (tid < n) {
        temp[tid] = input[tid];
    }


    // Write reversed data to output
    if (reversedIdx < n) {
        output[tid] = temp[reversedIdx];
    }
}
