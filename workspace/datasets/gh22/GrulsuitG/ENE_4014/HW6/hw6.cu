/*
 * Find BLANK and replace your own code.
 * And submit report why do you replace the blank that way.
 */

#include<stdlib.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<string>

#define TILE_WIDTH_TIMES 1 // fix this to evalute computation speed
#define INPUT_FILENAME "input.txt"

using namespace std;

__global__ void maxpool(float *input, float *output, const int input_size, const int filter_size) {
    // input : input_matrix address
    // output : output buffer address
    // input_size : width, height of input matrix
    // filter_size : filter_size of maxpolling
    // all input, output matrices are vectorized

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // out of bound

    

    int bound = input_size / filter_size ;
    int max = 0;
    
    int global_x = 0 ;
    int global_y = 0;

    for(int i = 0; i < filter_size; i++){
        if( i * bound <= col && col < (i+1) * bound)
            global_x = i;
        if( i * bound <= row && row < (i+1) * bound)
            global_y = i;
    }
    cout << "(" << row << "," << row << ")" << " = " << "(" << global_x << "," << global_y << ")" << "\n";
    
}

__global__ void avgpool(float *input, float *output, const int input_size, const int filter_size) {
    // input : input_matrix address
    // output : output buffer address
    // input_size : width, height of input matrix
    // filter_size : filter_size of maxpolling
    // all input, output matrices are vectorized

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // allocate 2D tiles in __shared__ memory
    // TILE WIDTH = TILE_WIDTH_TIMES*filter_size
    int tile_width = TILE_WIDTH_TIMES * filter_size;

    __shared__ float sum_tile[tile_width][tile_width];

    float result = 0;

    // make sure you handle the case when the matrix sizes are not
    // multiple of tile_width!
    // loop over the tiles of the input in phases
    for(int p = 0; p < input_size/tile_width; ++p){
        // CHANGE

        // You need to use __syncthreads() a few times
        // to synchronize the threads in a thread block.
    }

    // write out the result to output[row*input_size + col] 
    // CHANGE
}

int main(int argc, char **argv) {
    if(argc < 2) {
        cout << "usage : " << argv[0] << " input_size filter_size\n" << "example : " << argv[0] << " 100 2\n";
        return 1;
    }
    const int input_size = stoi(argv[1]);
    const int filter_size = stoi(argv[2]); // used for maxpooling
    
    const int maxpool_output_size = input_size/filter_size;
    const int avgpool_output_size = input_size/filter_size;

    // check input_size is power of 2
    if(input_size == 0 && (input_size & (input_size-1))){
        cout << "input_size must be power of 2\n";
        return 1;
    }

    if(filter_size == 0){
        cout << "filter_size cannot be 0\n";
        return 1;
    }

    float pool_input[input_size*input_size];
    
    // read input matrices 
    ifstream input_in(INPUT_FILENAME);

    for (int i = 0; i < input_size*input_size; ++i) {
        input_in >> pool_input[i];
    }
    
    // prints inputs for debugging.
    cout<<"filter size : "<<filter_size;
    cout<<"\n========== POOL_INPUT ==========\n";
    for (int i = 0; i < input_size * input_size; ++i) {
        if(i%input_size==0) cout<<"\n";
        cout<<pool_input[i]<<" ";
    }

    cout<<'\n';
       
    // set thread, block dimensions
    const dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    const dim3 num_of_maxpool_blocks(maxpool_output_size/block_size.x+1, maxpool_output_size/block_size.y+1);
    const dim3 num_of_avgpool_blocks(avgpool_output_size/block_size.x+1, avgpool_output_size/block_size.y+1);
    const dim3 num_of_blocks(input_size/block_size.x+1, input_size/block_size.y+1);

    // memory allocation for the device
    float *maxpool_output;
    cudaMalloc(&maxpool_output, sizeof(float) * maxpool_output_size * maxpool_output_size);
    
    float *avgpool_output;
    cudaMalloc(&avgpool_output, sizeof(float) * avgpool_output_size * avgpool_output_size);
    

    // copy variable to device memory
    cudaMemcpy(dev_mem_maxpool_input, pool_input, sizeof(float) * input_size * input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mem_avgpool_input, pool_input, sizeof(float) * input_size * input_size, cudaMemcpyHostToDevice);

    // launch CUDA kernels

    // First launch maxpooling kernel
    maxpool<<<num_of_maxpool_blocks, block_size>>>(dev_mem_maxpool_input, maxpool_output, input_size, filter_size);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr, "ERROR %s\n", cudaGetErrorString(error));
        return 1;
    }

    // Then run average pooling
    avgpool<<<num_of_avgpool_blocks, block_size>>>(dev_mem_avgpool_input, avgpool_output, input_size, filter_size);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr, "ERROR %s\n", cudaGetErrorString(error));
        return 1;
    }
 
    // allocate output buf in main memory
    float *maxpool_output_buf = (float*) malloc (sizeof(float)*maxpool_output_size*maxpool_output_size);
    float *avgpool_output_buf = (float*) malloc (sizeof(float)*maxpool_output_size*maxpool_output_size);
    
    // copy results from device to host
    cudaMemcpy(maxpool_output_buf, maxpool_output, sizeof(float)*maxpool_output_size*maxpool_output_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(avgpool_output_buf, avgpool_output, sizeof(float)*maxpool_output_size*maxpool_output_size, cudaMemcpyDeviceToHost);
    
    // prints the results
    cout<<"\n========== MAXPOOL OUTPUT ==========\n";
    for (int i = 0; i < maxpool_output_size * maxpool_output_size; ++i) {
        if(i%maxpool_output_size==0) cout<<"\n";
        cout<<maxpool_output_buf[i]<<" ";
    }
    cout<<'\n';

    cout<<"\n========== AVGPOOL OUTPUT ==========\n";
    for (int i = 0; i < avgpool_output_size * avgpool_output_size; ++i) {
        if(i%avgpool_output_size==0) cout<<"\n";
        cout<<avgpool_output_buf[i]<<" ";
    }
    cout<<'\n';

    cudaFree(dev_mem_input);
    cudaFree(maxpool_output);
    cudaFree(avgpool_output);
    free(maxpool_output_buf);
    free(avgpool_output_buf);
    return 0;
}
