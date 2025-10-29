
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define mu 0.01
#define T 40
#define N 1792
#define tempoTotal 15000
#define xInicial 0
#define xFinal 500.0
#define yInicial 0
#define yFinal 500.0
#define	alpha 0.4
#define	gamma 0.8

#define CHECK(call)                                                \
    {                                                              \
        cudaError_t error = call;                                  \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
        }                                                          \
    }

__global__ void waveEquationKernel(double *wave, double *waveFuture, double *wavePast)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    if ((i > 0 && i < N - 1) && (j > 0 && j < N - 1))
        waveFuture[i * blockDim.x + j] = 2 * wave[i * blockDim.x + j] * (1 - alpha * alpha - gamma * gamma) - wavePast[i * blockDim.x + j] + alpha * alpha * wave[(i + 1) * blockDim.x + j] + alpha * alpha * wave[(i - 1) * blockDim.x + j] + gamma * gamma * wave[i * blockDim.x + (j + 1)] + gamma * gamma * wave[i * blockDim.x + (j - 1)];
}

void writeFiles(double *wave, double dx, double dy)
{
    int i, j;
    FILE *fileStaticPlot;

    fileStaticPlot = fopen("WaveStatic2.dat", "w");

    fprintf(fileStaticPlot, "x\ty\tf\n");

    for (i = 1; i < N - 1; i++)
    {
        for (j = 1; j < N - 1; j++)
        {
            fprintf(fileStaticPlot, "%lf\t%lf\t%lf\n", i * dx, j * dy, wave[i * N + j]);
        }
    }

    fclose(fileStaticPlot);
}

void initialCondition(double *wave)
{
    int i, j;
    for (j = 0; j < N; j++)
    {
        for (i = 0; i < N; i++)
        {
            wave[i * N + j] = 4 * sin(M_PI * i / 75.0);
        }
    }
}

void derivativeCondition(double *wave, double *wavePast){

	int i, j;

	for (i = 1; i < N-1; i++)
	{
		for (j = 1; j < N-1; j++)
		{
			wave[i*N+j] = (2*wavePast[i*N + j]*(1 - alpha*alpha - gamma*gamma) + alpha*alpha*wavePast[(i+1)*N + j] + alpha*alpha*wavePast[(i-1)*N + j] + gamma*gamma*wavePast[i*N + (j+1)] +  gamma*gamma*wavePast[i*N + (j-1)])/2;
		}
	}
}

double resultsValidation(double *wave, double dx, double dy){

    int i, j;
    double erro, maiorErro, analitica;

    for(i = 0; i < N; i++){
        for(j = 0; j < N; j ++){

            analitica = sin((j*M_PI*dx)/xFinal)*sin((i*M_PI*dy)/yFinal)*(-(2000/((M_PI)*(M_PI)))*((sin(M_PI*j)*(cos(M_PI*i) - 1))/((j*j - 250000)*i)))*cos(sqrt(mu/T)*(sqrt((j*M_PI/xFinal)*(j*M_PI/xFinal) + (i*M_PI/yFinal)*(i*M_PI/yFinal)))*tempoTotal);

            erro = fabs((wave[i*N + j] - analitica)/analitica);

            if(i == 0 && j == 0){
                maiorErro = erro;
            }
            else{
                if(erro > maiorErro){
                    maiorErro = erro;
                }
            }
        }
    }

    return maiorErro;
}

void deviceCapabilities()
{

    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        printf("\n --- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Device copy overlap: ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Kernel execition timeout : ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("\n --- Memory Information for device %d ---\n", i);
        printf("Total global mem: %ld\n", prop.totalGlobalMem);
        printf("Total constant Mem: %ld\n", prop.totalConstMem);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("Texture Alignment: %ld\n", prop.textureAlignment);
        printf("\n --- MP Information for device %d ---\n", i);
        printf("Multiprocessor count: %d\n",
               prop.multiProcessorCount);
        printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n",
               prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("\n\n");
    }
}

void actionWork(double dx, double dy){

    int i, j, t;

    double *hostWave, *hostWaveFuture, *hostWavePast;       // Host variables
    double *deviceWave, *deviceWaveFuture, *deviceWavePast; // Device Variables
    //double erro;

    //printf("Alocando memoria no host\n");
    hostWave = (double *)calloc((N * N), sizeof(double));
    hostWaveFuture = (double *)calloc((N * N), sizeof(double));
    hostWavePast = (double *)calloc((N * N), sizeof(double));

    //printf("Colocando condição inicial.\n");
    initialCondition(hostWavePast);
    //printf("Colocando condição da derivada.\n"); 
    derivativeCondition(hostWave, hostWavePast);

    //printf("Alocando memoria no Device\n");
    CHECK(cudaMalloc(&deviceWave, (N * N) * sizeof(double)));
    CHECK(cudaMalloc(&deviceWaveFuture, (N * N) * sizeof(double)));
    CHECK(cudaMalloc(&deviceWavePast, (N * N) * sizeof(double))); 
    
    //printf("Iniciando calculo da função de onda.\n");
    for (t = 0; t < tempoTotal; t++)
    {
        CHECK(cudaMemcpy(deviceWave, hostWave, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(deviceWaveFuture, hostWaveFuture, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(deviceWavePast, hostWavePast, (N * N) * sizeof(double), cudaMemcpyHostToDevice));

        waveEquationKernel<<<N, N>>>(deviceWave, deviceWaveFuture, deviceWavePast);

        CHECK(cudaMemcpy(hostWave, deviceWave, (N * N) * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hostWaveFuture, deviceWaveFuture, (N * N) * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hostWavePast, deviceWavePast, (N * N) * sizeof(double), cudaMemcpyDeviceToHost));

        for (i = 1; i < N - 1; i++)
        {
            for (j = 1; j < N - 1; j++)
            {
                hostWavePast[i * N + j] = hostWave[i * N + j];
                hostWave[i * N + j] = hostWaveFuture[i * N + j];
            }
        }
    }

    //printf("Escrevendo no arquivo o resultado do cálculo\n");
    //writeFiles(hostWave, dx, dy);

    //erro = resultsValidation(hostWave, dx, dy);

    //printf("Erro da solução numérica: %lf\n", erro);

    //printf("Liberando memoria no host e device ... \n");

    free(hostWave);
    free(hostWaveFuture);
    free(hostWavePast);

    cudaFree(deviceWave);
    cudaFree(deviceWaveFuture);
    cudaFree(deviceWavePast);
}

int main()
{
    double dx, dy;

    dx = (xFinal - xInicial) / N;
    dy = (yFinal - yInicial) / N;

    //deviceCapabilities();

    //printf("dx: %lf\tdy: %lf\n", dx, dy);
    printf("Malha: %d x %d\n", N, N);
    //printf("Tempo total: %d\n", tempoTotal);

    clock_t beginTime = clock();

    actionWork(dx, dy);

    clock_t endTime = clock();

    printf("Time: %10.2f seconds \n", (endTime - beginTime) / (1.0 * CLOCKS_PER_SEC));
    return 0;
}
