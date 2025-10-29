//
// Created by alfonso on 26/09/21.
//
#include "../distlocRutaCUDA.h"

static const unsigned int blockSize = 1024;
static const unsigned int gridSize = 24;

/**
 *
 * @param d_presrec
 * @param tam_nodo
 * @param d_aOut
 */
__global__ void kernel_sumaVariaciones(PResRec d_presrec, unsigned int tam_nodo, double *d_aOut) {

    unsigned int tid = threadIdx.x;
    unsigned int gtid = tid + blockIdx.x * blockSize;
    const unsigned int gsize = blockSize * gridDim.x;

    double sum = 0;

    for (unsigned int i = gtid; i < tam_nodo; i += gsize) {
        sum += (d_presrec + i)->delta;
    }

    __shared__ double shArr[blockSize];
    shArr[tid] = sum;
    __syncthreads();

    for (int size = blockSize / 2; size > 0; size /= 2) {
        if (tid < size)
            shArr[tid] += shArr[tid + size];

        __syncthreads();
    }

    if (tid == 0)
        d_aOut[blockIdx.x] = shArr[0];
}

/**
 *
 * @param d_aIn
 * @param tam_nodo
 * @param d_aOut
 */
__global__ void kernel_sumaVariaciones2(double *d_aIn, unsigned int tam_nodo, double *d_aOut) {

    unsigned int tid = threadIdx.x;
    unsigned int gtid = tid + blockIdx.x * blockSize;
    const unsigned int gsize = blockSize * gridDim.x;

    double sum = 0;

    for (unsigned int i = gtid; i < tam_nodo; i += gsize) {
        sum += d_aIn[i];
    }

    __shared__ double shArr[blockSize];
    shArr[tid] = sum;
    __syncthreads();

    for (int size = blockSize / 2; size > 0; size /= 2) {
        if (tid < size)
            shArr[tid] += shArr[tid + size];

        __syncthreads();
    }

    if (tid == 0)
        d_aOut[blockIdx.x] = shArr[0];
}


/**
 *
 * @param d_presrec
 * @param tam_nodo
 * @return
 */
double sumaVariacion(PResRec d_presrec,unsigned int tam_nodo){

    double out;
    double *d_aOut;

    cudaMalloc((void **)&d_aOut,sizeof(double)*gridSize);

    kernel_sumaVariaciones<<<gridSize,blockSize>>>(d_presrec,tam_nodo,d_aOut);

    kernel_sumaVariaciones2<<<1,blockSize>>>(d_aOut,gridSize,d_aOut);

    cudaDeviceSynchronize();

    cudaMemcpy(&out, d_aOut, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_aOut);

    return out;
}
