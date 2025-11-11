// Difficulty: 2
// Example CUDA kernel for testing Faial MCP Server
// This simple kernel demonstrates a basic thread operation

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Example with potential data race (for testing race detection)
__global__ void sharedMemoryExample(int *output) {
    __shared__ int shared_data[256];

    int tid = threadIdx.x;

    // Write to shared memory
    shared_data[tid] = tid;

    // Potential race: missing __syncthreads()
    // Uncomment the next line to fix:
    // __syncthreads();

    // Read from shared memory
    if (tid > 0) {
        output[tid] = shared_data[tid - 1];
    }
}

