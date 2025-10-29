//header files

#include<stdio.h>
#include<cuda.h>        //cuda header files
#include"helper_timer.h"


//const int iNumberOfArrayElements = 5;

const int iNumberOfArrayElements = 1144777;

float* hostInput1=NULL;
float* hostInput2=NULL;
float* hostOutput=NULL;
float* gold = NULL;

float* deviceInput1=NULL;
float* deviceInput2=NULL;
float* deviceOutput=NULL;

float timeOnCPU=0.0f;
float timeOnGPU=0.0f;

//CUDA Kernel

__global__ void vecAddGPU(float* in1,float* in2,float* out,int len)
{
    
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<len)
    {
        out[i]=in1[i]+in2[i];
    }
}

//Entry-point function

int main(void)
{
    //function declaration
    void fillFloatArrayWithRandomNumbers(float*,int);
    void vecAddCPU(const float*,const float*,float*,int);
    void cleanup(void);

    //variable declaration

    int size=iNumberOfArrayElements*sizeof(float);
    cudaError_t result=cudaSuccess;

    //host memory allocation

    hostInput1 = (float*)malloc(size);
    if(hostInput1 == NULL)
    {
        printf("Host Memory allocation is failed for hostInput1 array\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float*)malloc(size);
    if(hostInput2 == NULL)
    {
        printf("Host memory allocation is failed for hostInput2 array \n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutput = (float*)malloc(size);
    if(hostOutput == NULL)
    {
        printf("Host memory allocation is failed for hostOutput array\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    gold = (float*)malloc(size);
    if(gold == NULL)
    {
        printf("Host memory allocation is failed for gold array\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    //filling values into host arrays

    fillFloatArrayWithRandomNumbers(hostInput1,iNumberOfArrayElements);
    fillFloatArrayWithRandomNumbers(hostInput2,iNumberOfArrayElements);

    //device memory allocation

    result = cudaMalloc((void**)&deviceInput1,size);

    if(result != cudaSuccess)
    {
        printf("Device memory allocation failed for deviceInput1 array\n");
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    result = cudaMalloc((void**)&deviceInput2,size);
    if(result != cudaSuccess)
    {
        printf("Device memory allocation failed for deviceInput2 array \n");
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    result = cudaMalloc((void**)&deviceOutput,size);
    if(result != cudaSuccess)
    {
        printf("Device Memory allocation failed for deviceOutput array \n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    //copy data from host array to device array

    result = cudaMemcpy(deviceInput1,hostInput1,size,cudaMemcpyHostToDevice);
    if(result != cudaSuccess)
    {
        printf("Host to device data copy is failed for deviceInput1\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMemcpy(deviceInput2,hostInput2,size,cudaMemcpyHostToDevice);
    if(result != cudaSuccess)
    {
        printf("Host to device data copy is failed for deviceInput2\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    //cuda kernel configuration

    dim3 dimGrid = dim3((int)ceil((float)iNumberOfArrayElements/256.0f),1,1);
    dim3 dimBlock = dim3(256,1,1);

    //cuda kernel for vector addition

    StopWatchInterface* timer = NULL;
    
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    vecAddGPU<<<dimGrid,dimBlock>>>(deviceInput1,deviceInput2,deviceOutput,iNumberOfArrayElements); //Calling Cuda kernel

    sdkStopTimer(&timer);
    timeOnGPU=sdkGetTimerValue(&timer);
    timer=NULL;

    //copy data from device array to host array

    result = cudaMemcpy(hostOutput,deviceOutput,size,cudaMemcpyDeviceToHost);

    if(result != cudaSuccess)
    {
        printf("Device to Host data copy is failed for Hostoutput array\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    //Vector addition on host

    vecAddCPU(hostInput1,hostInput2,gold,iNumberOfArrayElements);
    


    

    //Comparision

    const float epsilon=0.000001f;
    int breakvalue=-1;
    bool bAccuracy=true;
    
    for(int i=0;i<iNumberOfArrayElements;i++)
    {
        const float val1=gold[i];
        const float val2=hostOutput[i];

        if(fabs(val1-val2) > epsilon)
        {
            bAccuracy=false;
            breakvalue=i;
            break;
        }
    }

    char str[128];

    if(bAccuracy==false)
    {
        sprintf(str,"Comparision of CPU & GPU for Vector addition is not within Accuracy of 0.000001 at array index %d\n",breakvalue);
    }
    else
    {
        sprintf(str,"Comparision of CPU & GPU for Vector addition is within Accuracy of 0.000001\n");
    }

    //output

    printf("Array1: Array1[0]=%.6f to Array1[%d]=%.6f\n",hostInput1[0],iNumberOfArrayElements-1,hostInput1[iNumberOfArrayElements-1]);

    printf("Array2: Array2[0]=%.6f to Array2[%d]=%.6f\n",hostInput2[0],iNumberOfArrayElements-1,hostInput2[iNumberOfArrayElements-1]);

    printf("CUDA kernel Dimensions\n Grid Dimensions: %d %d %d \n Block Dimesions: %d %d %d\n",dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z);

    printf("ArrOutput: ArrOutput[0]=%.6f to ArrOutput[%d]=%.6f\n",hostOutput[0],iNumberOfArrayElements-1,hostOutput[iNumberOfArrayElements-1]);

    //time required display

    printf("Time Required for GPU:  %.6f\n ",timeOnGPU);
    printf("Time Required for CPU:  %.6f\n ",timeOnCPU);
    printf("%s",str);

    //cleanup function call

    cleanup();
    
    return 0;
}

//function for storing random numbers in array

void fillFloatArrayWithRandomNumbers(float* arr,int len)
{
    int i;
    const float fscale=1.0f/(float)RAND_MAX;
    for(i=0; i < len ; i++)
    {
        arr[i]= fscale*rand();
    }
}

//Defination of function vecAddCPU

void vecAddCPU(const float* arr1,const float* arr2,float* out, int len)
{
    StopWatchInterface* timer=NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for(int i=0; i<len ; i++)
    {
        out[i]= arr1[i]+arr2[i];
    }
    sdkStopTimer(&timer);
    timeOnCPU=sdkGetTimerValue(&timer);
    timer=NULL;
} 

//cleanup function defination
void cleanup()
{
    if(deviceOutput)
    {
        cudaFree(deviceOutput);
        deviceOutput=NULL;
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

    if(gold)
    {
        free(gold);
        gold=NULL;
    }

    if(hostOutput)
    {
        free(hostOutput);
        hostOutput=NULL;
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





