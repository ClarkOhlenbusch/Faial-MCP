#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>


const int  iNumberOfArrayElements = 5;
float *hostInput1=NULL;
float *hostInput2=NULL;
float *hostOutput3=NULL; 

float *deviceInput1=NULL;
float *deviceInput2=NULL;
float *deviceOutput3=NULL; 

//cuda kernel

__global__ void vecAddGPU(float* in1,float* in2,float* out,int len)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;

    if(i < len)
    {
        out[i] = in1[i] + in2[i];
    }  
}

int main(void)          //entry-point function
{
    void cleanup(void);     //function declaration

    int size = iNumberOfArrayElements * sizeof(float);


    hostInput1 = (float*)malloc(size);
    if(hostInput1 == NULL)
    {
        printf("Host allocation is failed for hostInput1 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float*)malloc(size);
    if(hostInput2==NULL)
    {
        printf("host Memory allocation is failed for hostInput2 array. \n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutput3 = (float*)malloc(size);
    if(hostOutput3 == NULL)
    {
        printf("Host memory allocation is failed for hostOutput3 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput1[0] = 101.0; //input values into host array.
    hostInput1[1] = 102.0;
    hostInput1[2] = 103.0;
    hostInput1[3] = 104.0;
    hostInput1[4] = 105.0; 

    hostInput2[0] = 201.0; //input values into host array.
    hostInput2[1] = 202.0;
    hostInput2[2] = 203.0;
    hostInput2[3] = 204.0;
    hostInput2[4] = 205.0; 

    //Devic memory allocation

    cudaError_t result = cudaSuccess;

    result = cudaMalloc((void**)&deviceInput1, size);
    if(result != cudaSuccess)
    {
       printf("Device memory allocation is failed for deviceInput1 array.\n");
       cleanup();
       exit(EXIT_FAILURE); 
    }

    result = cudaMalloc((void**)&deviceInput2, size);
    if(result != cudaSuccess)
    {
        printf("Device memory allocation is failed for deviceInput2 array \n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMalloc((void**)&deviceOutput3, size);
    if(result != cudaSuccess)
    {
        printf("Device memory allocation is failed for deviceOutput3 array \n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    //copy data from host array to device array.

    result = cudaMemcpy(deviceInput1,hostInput1,size,cudaMemcpyHostToDevice);

    if(result != cudaSuccess)
    {
        printf("Host to Device data copy is failed for deviceInput1 array\n ");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMemcpy(deviceInput2,hostInput2,size,cudaMemcpyHostToDevice);
    if(result != cudaSuccess)
    {
        printf("Host to Device data copy is failed for deviceInput2 array\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    dim3 dimGrid = dim3(iNumberOfArrayElements,1,1);
    dim3 dimBlock = dim3(1,1,1);

    //CUDA kernel for vector addition

    vecAddGPU<<< dimGrid , dimBlock >>> (deviceInput1,deviceInput2,deviceOutput3,iNumberOfArrayElements);

    //Copy data from device array into host array

    result = cudaMemcpy(hostOutput3,deviceOutput3,size,cudaMemcpyDeviceToHost);

    if(result != cudaSuccess)
    {
        printf("Device to host data copy is failed for hostOutput3 array\n");
        cleanup();
        exit(EXIT_FAILURE);
    } 

    //Print Vector addition on host

    for(int i=0 ; i< iNumberOfArrayElements ; i++)
    {
        printf("%f + %f = %f\n",hostInput1[i],hostInput2[i],hostOutput3[i]);

    }

    cleanup();
    return 0;
}

void cleanup(void)
{
    if(deviceOutput3)
    {
        cudaFree(deviceOutput3);
        deviceOutput3=NULL;
    }

    if(deviceInput2)
    {
        cudaFree(deviceInput2);
        deviceInput2=NULL;
    }

    if(deviceInput1)
    {
        cudaFree(deviceInput1);
        deviceInput1=NULL;
    }

    if(hostOutput3)
    {
        free(hostOutput3);
        hostOutput3=NULL;
    }

    if(hostInput2)
    {
        free(hostInput2);
        hostInput2=NULL;
    }

    if(hostInput1)
    {
        free(hostInput1);
        hostInput1=NULL;
    }
}