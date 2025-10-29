#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

#define N (1 << 10)
#define BLOCKSIZE 64

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
    __shared__ float Ads[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bds[BLOCKSIZE][BLOCKSIZE];

    int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;

    int row = by * BLOCKSIZE + ty;
    int col = bx * BLOCKSIZE + tx;

    float sum = 0;

    for (int i = 0; i < N / BLOCKSIZE; i++)
    {
        Ads[ty][tx] = A[row * N + i * BLOCKSIZE + tx];
        Bds[ty][tx] = B[col + N * (ty + BLOCKSIZE * i)];

        __syncthreads();

        for (int j = 0; j < BLOCKSIZE; ++j)
        {
            sum += Ads[ty][j] * Bds[j][tx];
            __syncthreads();
        }
    }

    C[row * N + col] = sum;
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