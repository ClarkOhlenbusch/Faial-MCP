#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../lib/path_gen_lib/path/path.cuh"
#include "../lib/support_lib/myRandom/myRandom.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/combined.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/tausworth.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/linCongruential.cuh"
#include "../lib/path_gen_lib/process_eq_imp/process_eq_lognormal_multivariante.cuh"
#include "../lib/path_gen_lib/process_eq_imp/process_eq_lognormal.cuh"
#include "../lib/equity_lib/schedule_lib/schedule.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve_flat.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"


#define NPATH 1  //number of paths             //1 solo cammino
#define STEPS 5  // number of steps
#define NEQ 5  //number of equities


__global__ void kernel_pt2(double*, uint*, double*, double*);
D void createPath_device_pt2(Equity_prices*, Schedule*, double*, size_t, uint*);
H void createPath_host_pt2(Equity_prices*, Schedule*, double*, size_t, uint*);
HD void createPath_generic_pt2(Process_eq*, Equity_prices*, Schedule*, double*, size_t);




__global__ void kernel_pt2(double* path_out, uint* seeds, double* prezzi, double* tempi)
{
    pricer::udb start_prices[NEQ];                              //definiamo oggetti dentro a kernel
    for (int i = 0; i < NEQ; i++)
    {
        start_prices[i] = prezzi[i];
    }
    Equity_description** descr = new Equity_description * [NEQ];

    double start_time = 0.15;
    Volatility_surface* vol = new Volatility_surface(0);
    Yield_curve_flat* yc = new Yield_curve_flat("euro", 0.5);

    for (int i = 0; i < NEQ; i++)
    {
        descr[i] = new Equity_description;
        descr[i]->Set_isin_code("isin codein");
        descr[i]->Set_name("namein ");
        descr[i]->Set_currency("currencyin");
        descr[i]->Set_dividend_yield(0);
        descr[i]->Set_yc(yc);
        descr[i]->Set_vol_surface(vol);
    }

    Equity_prices* starting_point_in = new Equity_prices(start_time, start_prices, NEQ, descr);
    Schedule* calen = new Schedule(tempi, NEQ);

    for (size_t i = 0; i < NEQ; i++)
    {
        path_out[i] = 0;
    }
    createPath_device_pt2(starting_point_in, calen, path_out, NPATH, seeds);

}

D void createPath_device_pt2(Equity_prices* starting_point,
    Schedule* calendar,
    double* path_out,
    size_t totpaths,
    uint* seeds)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;     //in questo test è sempre 1 (vedere sotto: kernel <<<1,1>>>)
    uint seed0;
    uint seed1;
    uint seed2;
    uint seed3;
    //blocco valido in generale, anche se index non è 1
    seed0 = seeds[0 + index * 4];
    seed1 = seeds[1 + index * 4];
    seed2 = seeds[2 + index * 4];
    seed3 = seeds[3 + index * 4];

    rnd::GenCombined* gnr_in = new rnd::GenCombined(seed0, seed1, seed2, seed3);
    Process_eq_lognormal_multivariante* process = new Process_eq_lognormal_multivariante(gnr_in, NEQ);

    createPath_generic_pt2(process, starting_point, calendar, path_out, index);

}

H void createPath_host_pt2(Equity_prices* starting_point,
    Schedule* calendar,
    double* path_out,
    size_t totpaths,
    uint* seeds)
{
    uint seed0;
    uint seed1;
    uint seed2;
    uint seed3;
    for (size_t index = 0; index < totpaths; index++)
    {
        seed0 = seeds[0 + index * 4];
        seed1 = seeds[1 + index * 4];
        seed2 = seeds[2 + index * 4];
        seed3 = seeds[3 + index * 4];
        rnd::GenCombined* gnr_in = new rnd::GenCombined(seed0, seed1, seed2, seed3);
        Process_eq_lognormal_multivariante* process = new Process_eq_lognormal_multivariante(gnr_in, NEQ);

        createPath_generic_pt2(process, starting_point, calendar, path_out, index);
    }
}


HD void createPath_generic_pt2(Process_eq* process,
    Equity_prices* starting_point,
    Schedule* calendar,
    double* path_out,
    size_t index)
{
    Path cammino = Path(starting_point, calendar, process);
    for (size_t i = 0; i < NEQ; i++)
    {
        path_out[i] = cammino.Get_equity_prices(STEPS - 1)->Get_eq_price(i).get_number(); //path out contiente i prezzi delle equities allo step finale
    }
}



int main(int argc, char** argv)
{
    cudaError_t cudaStatus;
    size_t* npath = new size_t(NPATH);
    size_t* neq = new size_t(NEQ);



    srand(time(NULL));
    uint* seeds = new uint[4 * NPATH];
    for (size_t inc = 0; inc < 4 * NPATH; inc++)
    {
        seeds[inc] = rnd::genSeed(true);       
    }
    
    double dt = 0.2;
    double tempi[STEPS];
    double start_time = 0.15;
    for (size_t k = 0; k < STEPS; k++)
    {
        tempi[k] = k * dt;
    }

    double rfr = 0.5;
    double* path = new double[NEQ];

    double* expected_path = new double[NEQ];
    pricer::udb start_prices[NEQ];
    double start_prices_gpu[NEQ];
    for (int i = 0; i < NEQ; i++)
    {
        start_prices[i] = 100 * i;
        start_prices_gpu[i] = start_prices[i].get_number();
        expected_path[i] = start_prices[i].get_number() * exp(rfr * (tempi[STEPS-1] - start_time));
    }

    prcr::Device dev;
    dev.CPU = false;
    dev.GPU = false;

    if (prcr::cmdOptionExists(argv, argv + argc, "-gpu"))
        dev.GPU = true;
    if (prcr::cmdOptionExists(argv, argv + argc, "-cpu"))
        dev.CPU = true;


    if (dev.CPU == true)
    {
        //assegnazione dei campi negli oggetti
        Equity_description** descr = new Equity_description * [NEQ];

        Volatility_surface* vol = new Volatility_surface(0);
        Yield_curve_flat* yc = new Yield_curve_flat("euro", rfr);

        for (int i = 0; i < NEQ; i++)
        {
            descr[i] = new Equity_description;
            descr[i]->Set_isin_code("isin codein");
            descr[i]->Set_name("namein ");
            descr[i]->Set_currency("currencyin");
            descr[i]->Set_dividend_yield(0);
            descr[i]->Set_yc(yc);
            descr[i]->Set_vol_surface(vol);
        }

        Equity_prices* starting_point_in = new Equity_prices(start_time, start_prices, NEQ, descr);

        Schedule* calen = new Schedule(tempi, STEPS);

        createPath_host_pt2(starting_point_in, calen, path, NPATH, seeds);
    }




    if (dev.GPU == true)           //se -gpu setted
    {

        //CudaSetDevice(0);
        double* dev_path;
        

        uint* dev_seeds;
        double* dev_start_prices;
        double* dev_tempi;

        cudaStatus = cudaMalloc((void**)&dev_path, NEQ * sizeof(double));             //malloc vettore che contiene prezzo equities allo step finale
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n"); }

        cudaStatus = cudaMalloc((void**)&dev_seeds, NPATH * 4 * sizeof(uint));       //malloc dei seeds
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }

        cudaStatus = cudaMalloc((void**)&dev_start_prices, NEQ * sizeof(double));             
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc1 failed!\n"); }

        cudaStatus = cudaMalloc((void**)&dev_tempi, STEPS  * sizeof(double));       
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc2 failed!\n"); }



        cudaStatus = cudaMemcpy(dev_seeds, seeds, NPATH * 4 * sizeof(uint), cudaMemcpyHostToDevice);   //memcpy seeds su device
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); }
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

        cudaStatus = cudaMemcpy(dev_start_prices, start_prices_gpu, NEQ * sizeof(double), cudaMemcpyHostToDevice);   //memcpy seeds su device
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy3 failed!\n"); }
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));

        cudaStatus = cudaMemcpy(dev_tempi, tempi, STEPS * sizeof(double), cudaMemcpyHostToDevice);   //memcpy seeds su device
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy4 failed!\n"); }
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));



        kernel_pt2 << <1, 1 >> > (dev_path, dev_seeds, dev_start_prices, dev_tempi);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(cudaStatus)); }



        cudaStatus = cudaMemcpy(path, dev_path, NEQ * sizeof(double), cudaMemcpyDeviceToHost);  //memcpy path_out da device
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); }
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));


        






        
    }
    //stampa 
    for (int k = 0; k < NEQ; k++)
    {
        std::cout << "\nequity " << k << ":";
        std::cout << "\t" << path[k] << "\n";
    }
    std::cout << std::endl;
	
    //confronto con attese
    int err_msg = 0;
    for (int k = 0; k < NEQ; k++)
    {
        //minimo tra prezzo della stessa equity al final step tra gpu e cpu
       if (abs(path[k] - expected_path[k]) >  0.000001 * expected_path[k])
            err_msg += pow(2, k );  //se i valori si discostano troppo tra di loro ("troppo definito da quale tra i due è min)
    } 
    if(err_msg == 0) {std::cout << "test_path2 ok!\n";}                             //ritorniamo errore (così costruito, è unico per ogni caso)
    return err_msg;
}
