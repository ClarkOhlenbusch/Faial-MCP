#include "class_template_wrapper.cuh"


// allocate memory on device
void allocate_memory_on_device_i(int*& d_ptr, size_t size)
{
    cudaMalloc(&d_ptr, size * sizeof(int));
}

void allocate_memory_on_device_d(double*& d_ptr, size_t size)
{
    cudaMalloc(&d_ptr, size * sizeof(double));
}

// deallocate memory on device
void deallocate_memory_on_device_i(int*& d_ptr)
{
    cudaFree(d_ptr);
}
void deallocate_memory_on_device_d(double*& d_ptr)
{
    cudaFree(d_ptr);
}



// copy memory from host to device
void copy_host_to_device_i(int* d_ptr, int* h_ptr, size_t size)
{
    cudaMemcpy(d_ptr, h_ptr, size * sizeof(int), cudaMemcpyHostToDevice);
}
void copy_host_to_device_d(double* d_ptr, double* h_ptr, size_t size)
{
    cudaMemcpy(d_ptr, h_ptr, size * sizeof(double), cudaMemcpyHostToDevice);
}




// copy memory from device to host
void copy_device_to_host_i(int* h_ptr, int* d_ptr, size_t size)
{
    cudaMemcpy(h_ptr, d_ptr, size * sizeof(int), cudaMemcpyDeviceToHost);
}
void copy_device_to_host_d(double* h_ptr, double* d_ptr, size_t size)
{
    cudaMemcpy(h_ptr, d_ptr, size * sizeof(double), cudaMemcpyDeviceToHost);
}


// kernel function to add 2 to all elements of a vector

// integer
__global__ void add_2_to_all_elements_i_kernel(int* d_ptr, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        d_ptr[i] += 2;
}

void add_2_to_all_elements_i(int* d_ptr, size_t size)
{
    add_2_to_all_elements_i_kernel<<<size / BLOCK_SIZE, BLOCK_SIZE>>>(d_ptr, size);
}

// double
__global__ void add_2_to_all_elements_d_kernel(double* d_ptr, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        d_ptr[i] += 2.0;
}

void add_2_to_all_elements_d(double* d_ptr, size_t size)
{
    add_2_to_all_elements_d_kernel<<<size / BLOCK_SIZE, BLOCK_SIZE>>>(d_ptr, size);
}