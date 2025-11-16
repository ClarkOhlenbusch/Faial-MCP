// Difficulty: 2
// Example CUDA kernel for testing Faial MCP Server
// This kernel demonstrates a data race caused by missing synchronization
__global__ void sharedMemoryExample(int *output) {
    __shared__ int shared_data[256];
    
    int tid = threadIdx.x;
    
    // Write to shared memory
    shared_data[tid] = tid;
    // Read from shared memory
    if (tid > 0) {
        output[tid] = shared_data[tid - 1];
    }
}

