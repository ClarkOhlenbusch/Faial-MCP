#include "../lib/support_lib/myRandom/myRandom_gnr/combined.cuh"
#include "../lib/support_lib/myRandom/myRandom.cuh"
//#include "../lib/support_lib/myRandom/random_numbers.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include <cmath>


//genero numeri casuali, li sommo e vedo se media è consistente
//genero numeri casuali a partire da seed noti e vedo se non cambiano

  

#define NBLOCKS 4
#define TPB 4
#define PPT 2

__global__ void kernel (uint*, double*, double*);
__device__ void rnd_test_dev(uint*, double*, double*);
__host__ void rnd_test_hst(uint*, double*, double*);
__host__ __device__ void rnd_test_generic(uint*, double*, double*, size_t);


__global__ void kernel(uint* seeds, double* dev_sum, double* dev_sq_sum)
{
    rnd_test_dev(seeds, dev_sum, dev_sq_sum);
}

__device__ void rnd_test_dev(uint* seeds, double* dev_sum, double* dev_sq_sum)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < NBLOCKS*TPB)
    {
        rnd_test_generic(seeds, dev_sum, dev_sq_sum, index);
    }
}
__host__ void rnd_test_hst(uint* seeds, double* sum, double* sq_sum)
{
    for(size_t index = 0; index < NBLOCKS*TPB; index++)
    rnd_test_generic(seeds, sum, sq_sum, index);
}
__host__ __device__ void rnd_test_generic(uint* seeds, double* sum, double* sq_sum, size_t index)
{
    uint seed0 = seeds[4 * index];
    uint seed1 = seeds[4 * index + 1];
    uint seed2 = seeds[4 * index + 2];
    uint seed3 = seeds[4 * index + 3];
    rnd::GenCombined* gnr = new rnd::GenCombined(seed0, seed1, seed2, seed3);
    double number;
    for (size_t i = 0; i < PPT; i++)
    {
        number = gnr->genGaussian();
        sum[index] += number;
        sq_sum[index] += number * number;
    }
    

}


int main(int argc, char** argv)
{

    prcr::Device dev;
    dev.CPU = false;
    dev.GPU = false;

    if (prcr::cmdOptionExists(argv, argv + argc, "-gpu"))
        dev.GPU = true;
    if (prcr::cmdOptionExists(argv, argv + argc, "-cpu"))
        dev.CPU = true;

    double host_sum[NBLOCKS * TPB];
    double host_sq_sum[NBLOCKS * TPB];
    uint seeds[4*NBLOCKS* TPB];

    srand(time(NULL));
    for (size_t i = 0; i < 4 * NBLOCKS * TPB; i++)
    {
        seeds[i] = rnd::genSeed(true);
    }
    for(size_t i = 0; i < NBLOCKS*TPB; i++)
    {
	host_sum[i] = 0;
	host_sq_sum[i] = 0;
    }




    if(dev.CPU)
    { 
        rnd_test_hst(seeds, host_sum, host_sq_sum);
    }



    if (dev.GPU)
    {
	    cudaError_t cudaStatus;
        uint* dev_seeds = new uint[4*NBLOCKS*TPB];
        double* dev_sum = new double[NBLOCKS * TPB];
        double* dev_sq_sum = new double[NBLOCKS * TPB];


        cudaStatus = cudaMalloc((void**)&dev_seeds, NBLOCKS *4* TPB * sizeof(uint));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n"); }

        cudaStatus = cudaMalloc((void**)&dev_sum,  NBLOCKS*TPB*sizeof(double));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

        cudaStatus = cudaMalloc((void**)&dev_sq_sum, NBLOCKS * TPB * sizeof(double));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc3 failed!\n"); }

        cudaStatus = cudaMemcpy(dev_seeds, seeds, NBLOCKS * 4 *TPB* sizeof(uint), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy1 failed!\n"); }

        cudaStatus = cudaMemcpy(dev_sum, host_sum, NBLOCKS  *TPB* sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); }

        cudaStatus = cudaMemcpy(dev_sq_sum, host_sq_sum, NBLOCKS *TPB* sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy3 failed!\n"); }

       kernel << <NBLOCKS,TPB >> > (dev_seeds, dev_sum, dev_sq_sum);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(cudaStatus)); }

        cudaStatus = cudaMemcpy(host_sum, dev_sum, NBLOCKS*TPB*sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy back1 failed!\n"); }

        cudaStatus = cudaMemcpy(host_sq_sum, dev_sq_sum, NBLOCKS * TPB * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy back2 failed!\n"); }


       
    }

    double std_dev = 1. / sqrt(NBLOCKS * TPB * PPT);
    std::cout << "std dev teorica (della media): " << std_dev << std::endl;

    double meas_mean = 0;
    double meas_std_dev = 0;

    for (size_t i = 0; i < NBLOCKS * TPB; i++)
    {
        meas_mean += host_sum[i];
        meas_std_dev += host_sq_sum[i];
    }

    meas_mean /= (NBLOCKS * TPB * PPT);
    meas_std_dev =sqrt((meas_std_dev/(NBLOCKS * TPB * PPT)-meas_mean*meas_mean)/(NBLOCKS*TPB*PPT));

    std::cout << " measured std dev:" << meas_std_dev<< std::endl;
    if (abs(meas_mean) < 3 * std_dev) 
    {
        printf("ok, ");
        printf("la media dei numeri generati e': %f\n", meas_mean);
        return 0;
    }
    else 
    {
        printf("La media non e' entro 3 standard deviation: %f\n", meas_mean);
        return 1;
    }
}



