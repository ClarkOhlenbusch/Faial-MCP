#include <stdio.h>
#include "cuda_runtime.h"

// Device code
// 将数组 A 与 B 中的元素相加存入数组 C 中, N 为数组中元素的数量
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host code
int main()
{
    int N = 1024;
    size_t size = N * sizeof(float);

    // 定义 host 端数组 h_A, h_B, h_C
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    // ...

    // 定义 device 端数组 d_A, d_B, d_C
    // 并在 GPU 中为它们分配对应的显存空间
    // &d_A 的类型为 float **, 此处将其强制转换为 void** 类型.
    float* d_A;
    cudaMalloc((void **)&d_A, size);
    float* d_B;
    cudaMalloc((void **)&d_B, size);
    float* d_C;
    cudaMalloc((void **)&d_C, size);

    // 利用 cudaMemcpy 函数将 host 端 A,B 的值复制到对应的 GPU 内存中
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    // 启动定义好的 GPU kernel，实现数组相加
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // cudaDeviceSynchronize() 会阻塞当前程序的执行, 直到所有线程都执行完 kernel
    // 避免后面的 CPU 代码在 kernel 结束前就执行.
    cudaDeviceSynchronize();

    // 调用 cudaMemcpy 函数将 GPU 端计算结果复制到 CPU 端
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 打印数组相加结果
    for(int i=0; i<N; ++i){
        printf("%d ", h_C[i]);
    }
    printf("\n");

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // reset device before you leave
    cudaDeviceReset();
    return 0;
}
