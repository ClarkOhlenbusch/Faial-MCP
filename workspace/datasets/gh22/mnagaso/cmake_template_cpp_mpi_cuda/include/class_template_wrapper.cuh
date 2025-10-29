#ifndef CLASS_TEMPLATE_WRAPPER_CUH
#define CLASS_TEMPLATE_WRAPPER_CUH

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#define BLOCK_SIZE 256

// allocate memory on device
void allocate_memory_on_device_i(int *&d_ptr, size_t size);
void allocate_memory_on_device_d(double *&d_ptr, size_t size);

// deallocate memory on device
void deallocate_memory_on_device_i(int *&d_ptr);
void deallocate_memory_on_device_d(double *&d_ptr);


// copy memory from host to device
void copy_host_to_device_i(int *d_ptr, int *h_ptr, size_t size);
void copy_host_to_device_d(double *d_ptr, double *h_ptr, size_t size);

// copy memory from device to host
void copy_device_to_host_i(int *h_ptr, int *d_ptr, size_t size);
void copy_device_to_host_d(double *h_ptr, double *d_ptr, size_t size);

// kernel function to add 2 to all elements of a vector
void add_2_to_all_elements_i(int *d_ptr, size_t size);
void add_2_to_all_elements_d(double *d_ptr, size_t size);

typedef struct gpu_data {
    int* arr_i_device;
    double* arr_d_device;
} gpu_data;

#endif // CLASS_TEMPLATE_WRAPPER_CUH