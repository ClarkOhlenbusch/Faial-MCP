#include <cuda_runtime.h>
#include <iostream>
#include <thrust/extrema.h>

__global__ void conv(float *img, float *kernel, float *dst,
                     int width, int height,
                     int kernelsize = 3, int stride = 1, int padd=1)
{
    int w = (width - kernelsize + padd * 2) / stride + 1;
    // int h = (height - kernelsize + 1) / stride + 1;

    int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y + threadIdx.y;
    int row = tid / w;
    int col = tid % w;

    int r = row * stride - padd;
    int c = col * stride - padd;

    float tmp = 0.0;
    // (r>=0?r:0)
    for (int i = max(r, 0); i < min(r + kernelsize, height); i++)
    {
        // (c>=0?c:0)
        for (int j = max(c, 0); j < min(c + kernelsize, width); j++)
        {
            tmp += kernel[(i - r) * kernelsize + (j - c)] * img[i * width + j];
        }
    }
    dst[tid] = tmp;
    // dst[tid] = 1;
}

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n",
               cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

int getThreadNum()
{
    cudaDeviceProp prop;
    int count;

    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("gpu num %d\n", count);
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    printf("max grid dimensions: %d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    return prop.maxThreadsPerBlock;
}

int main()
{
    int height = 1080, width = 1920; // 1080p
    float *img = new float[height * width];

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            img[row * width + col] = (col + row) % 256; // 这样横着竖着看图片都是像素值为0123456...
        }
    }

    int kernelsize = 3; // 卷积核大小
    int stride = 1;     // 跳跃为1
    int padd = (kernelsize-1)/2;
    float *kernel = new float[kernelsize * kernelsize];

    for (int i = 0; i < kernelsize * kernelsize; ++i)
    {
        kernel[i] = i % kernelsize - 1;
    }

    for (int row = 0; row < 10; row++)
    {
        for (int col = 0; col < 10; col++)
        {
            // std::cout << img[row * width + col] << " ";
            printf("%2.0f ", img[row*width + col]);
        }
        std::cout << std::endl;
    }

    int target_h = (height - kernelsize + padd * 2) / stride + 1;
    int target_w = (width - kernelsize + padd * 2) / stride + 1;

    // gpu
    float *img_d, *kernel_d, *target_d;
    HANDLE_ERROR(cudaMalloc((void **)&img_d, width * height * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&kernel_d, kernelsize * kernelsize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&target_d, target_w * target_h * sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(img_d, img, width * height * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(kernel_d, kernel, kernelsize * kernelsize * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blocksize(32, 32);
    dim3 gridsize(target_h * target_w / 1024); // 不要自己作死非要定二维，索引难受死了

    // std::cout<<blockDim.x<<blockDim.y<<std::endl;
    // std::cout<<gridDim.x<<gridDim.y<<std::endl;


    conv<<<gridsize, blocksize>>>(img_d, kernel_d,
                                  target_d, width, height,
                                  kernelsize=kernelsize, stride=stride, padd=padd);
    std::cout << "compute success!" << std::endl;

    float *target_host = new float[target_h * target_w];
    cudaMemcpy(target_host, target_d, target_h * target_w * sizeof(float), cudaMemcpyDeviceToHost);

    for(int row = 0; row < 10; ++row)
    {
        for(int col = 0; col < 10; ++col)
        {
            printf("%2.0f ", target_host[col + row * width]);
        }
        printf("\n");
    }
    return 0;
}
