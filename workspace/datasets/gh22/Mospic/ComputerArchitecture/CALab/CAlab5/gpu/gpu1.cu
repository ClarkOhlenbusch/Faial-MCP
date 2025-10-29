#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

#define N (1 << 10)
#define BLOCKSIZE 8

__global__ void gemm_baseline(float *A, float *B, float *C);
int gemm_verify(float *A, float *B, float *C);
void Initialization(float* M);
void Reverse_Matrix(float *M);


void printboard(float *A)
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf("%f\t",A[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main()
{
    //malloc A,B,C
    
    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));

    srand((unsigned)time(NULL));
    Initialization(A);
    Initialization(B);
    
    memset(C, 0.0, N * N * sizeof(float));

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N*N*sizeof(float));
    cudaMalloc((void **)&d_B, N*N*sizeof(float));
    cudaMalloc((void **)&d_C, N*N*sizeof(float));

    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize(N / BLOCKSIZE, N / BLOCKSIZE);
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE);

    cudaEvent_t gpustart, gpustop;
    float gpuelapsedTime = 0.0;
    cudaEventCreate(&gpustart);
    cudaEventCreate(&gpustop);
    cudaEventRecord(gpustart, 0);
    
    gemm_baseline<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    
    cudaDeviceSynchronize();   
    
    cudaEventRecord(gpustop, 0);
    cudaEventSynchronize(gpustop);
    cudaEventElapsedTime(&gpuelapsedTime, gpustart, gpustop);
    cudaEventDestroy(gpustart);
    cudaEventDestroy(gpustop);
    double gputime = gpuelapsedTime;
    
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);


    printf("GPU time = %lf s\n", gputime);
    if(gemm_verify(A, B, C) == 0)
    {
        printf("Result is Correct!\n");
    }
    else
    {
        printf("Result is Incorrect\n");
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);
    return 0;
}


void Initialization(float* M)
{
    for(int i = 0; i < N * N; i++)
    {
        M[i] = (rand() % 10) / 1.0;
    }
}

int gemm_verify(float *A, float *B, float *C) {
    float* T = (float*)malloc(N * N * sizeof(float));
    memset(T, 0, sizeof(float) * N * N);
    Reverse_Matrix(B);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                T[i * N + j] += A[i * N + k] * B[j * N + k];
            }
        }
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(fabs(C[i * N + j] - T[i * N + j]) > pow(10, -1))
                return 1;
        }
    }
    return 0;
}

__global__ void gemm_baseline(float *A, float *B, float *C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
   

    if ((row < N) && (col < N))
    {
        float t = 0;
        for (int i = 0; i < N; i++)
        {
            t += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = t;
    }
    return ;
}

void Reverse_Matrix(float *M)
{
    float temp;
    for(int i = 1; i < N; i++)
    {
        for(int j = 0; j < i; j++)
        {
            temp = M[i * N + j];
            M[i * N + j] = M[j * N + i];
            M[j * N + i] = temp;
        }
    }
}