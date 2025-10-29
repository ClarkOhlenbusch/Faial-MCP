#include <iostream>
#include <math.h>
#include <fstream>
#include <random>
#include <ctime>
#include <unordered_map>
#include <curand_kernel.h>
#include <unordered_map>
#include <utility>
#define compare_swap(i,j) if(i > j){ temp = i; i = j; j = temp; }
#define imin(i, j) (i > j ? j : i)

struct pair_hash
{
	template <class T1, class T2>
	std::size_t operator() (const std::pair<T1, T2> &pair) const
	{
		std::size_t other = std::hash<T2>()(pair.second);
		return std::hash<T1>()(pair.first) + 0x9e3779b9 + (other<<6) + (other>>2);
	}
};


inline __device__ int find_num_of_threads(int n, int device_max_thread_per_block){
    if(n > device_max_thread_per_block) return device_max_thread_per_block;
    int i = (n + 31) / 32;
    return i * 32;
}

struct move_struct{
    uint32_t src_start;
    uint32_t src_end;
    uint32_t dst_start;
};

struct gain_struct_t{
    float gain;
    uint32_t i;
    uint32_t j;
};

__device__ float distance_ij(uint32_t city_i, uint32_t city_j, int32_t* coords, uint32_t n){
    float dx = coords[city_i] - coords[city_j];
    float dy = coords[city_i + n] - coords[city_j + n];
    return sqrtf((dx * dx) + (dy * dy));
}


__global__ void apply_2opt_move(int32_t *tour, gain_struct_t *buf, uint32_t n){
    __shared__ gain_struct_t best_gain;
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(threadIdx.x == 0){
        best_gain = buf[0];
    }
    __syncthreads();

    //copy to local
    uint32_t i = best_gain.i;
    uint32_t j = best_gain.j;
    float gain = best_gain.gain;

    if(gain > 0){
        int left = i + id + 1;
        int right = j - id;
        int left_end = (i + j) / 2;
        while(left <= left_end){
            int32_t temp = tour[left];
            int32_t temp2 = tour[left + n];
            tour[left] = tour[right];
            tour[right] = temp;
            tour[left + n] = tour[right + n];
            tour[right + n] = temp2;
            left += 32;
            right -= 32;
        }
    }
}



__global__ void two_opt(int32_t *tour, gain_struct_t *d_inversions, uint32_t n){
    extern __shared__ int32_t s_tour[];
    
    for(int i = threadIdx.x; i < 2 * n; i += blockDim.x){
        s_tour[i] = tour[i];
    }
    __syncthreads();

    int total = ((n - 1) * (n - 2)) / 2;
    
    //int taskPerThread = (total + blockDim.x - 1) / blockDim.x;
    //old this assumes only execute one block,

    int taskPerThread = (total + (gridDim.x * blockDim.x) - 1) / (gridDim.x * blockDim.x); 
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int start = tid * taskPerThread; //including
    int stop = start + taskPerThread; //not including
    if(stop > total - 1) stop = total - 1; 

    int at = start;

    int city_j, city_i, city_i1, city_j1;  
    {
        int b = 3 -  (2 * (int)n);
        int c = 2 * at;
        float delta = (b * b)  - (4 * c);
        float root = (-b - sqrtf(delta)) / 2;
        city_i = (int)(truncf(root));
        float res = (float)city_i * ( ( ((float)city_i) / 2 ) + 1.5 - n) + at;
        city_j = (int)(truncf(res)) + city_i + 2;
    }
    
    gain_struct_t local_best;
    local_best.i = 0;
    local_best.j = 0;
    local_best.gain = 0;

    while(at < stop){
        city_i1 = city_i + 1;
        city_j1 = city_j + 1;
        if(city_j1 == n) city_j1 = 0;        
        float distance_f = 0 - distance_ij(city_i, city_j, s_tour, n) - distance_ij(city_i1, city_j1, s_tour, n) +
        distance_ij(city_i, city_i1, s_tour, n) + distance_ij(city_j, city_j1, s_tour, n);
        if(distance_f > local_best.gain){
            local_best.gain = distance_f;
            local_best.i = city_i;
            local_best.j = city_j;        
        }
        ++at;
        city_j++;
        if(city_j == n){
            city_i++;
            city_j = city_i + 2;
        }
    }
    //write values to global memory
    d_inversions[tid].gain = local_best.gain;
    d_inversions[tid].i = local_best.i;
    d_inversions[tid].j = local_best.j;
}


__global__ void setup_kernel(curandState *state){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    /* Each thread gets same seed, a different sequence number, no offset */    
    curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, uint32_t n, uint32_t *perturbation_indices, uint32_t iteration_count){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    int oldid = id;
    int pos = 4 * id;
    /* Copy state to local memory for efficiency */    
    curandState localState = state[id];
    while(id < iteration_count){
   	 int indices[4];
   	 do{
       	 	indices[0] = curand_uniform(&localState) * n; //likelihood of seeing n is very low, we can say it is btween 0 - n-1
        	indices[1] = curand_uniform(&localState) * n; 
        	indices[2] = curand_uniform(&localState) * n; 
        	indices[3] = curand_uniform(&localState) * n; 
        	int temp;
		compare_swap(indices[0], indices[1]); //network sort
        	compare_swap(indices[2], indices[3]);
        	compare_swap(indices[0], indices[3]);
        	compare_swap(indices[1], indices[2]);
        	compare_swap(indices[0], indices[1]);
        	compare_swap(indices[2], indices[3]);
    	}while(indices[1] <= indices[0] + 1 || indices[2] <= indices[1] + 1 || indices[3] <= indices[2] + 1 || indices[3] + 1 >= n || 
        	(indices[3] == n - 1 && indices[0] == 0));
    	perturbation_indices[pos + 0] = indices[0];
   	perturbation_indices[pos + 1] = indices[1];
    	perturbation_indices[pos + 2] = indices[2];
    	perturbation_indices[pos + 3] = indices[3];
	id += blockDim.x;
	pos = 4*id;
    }
    /* Copy state back to global memory */
    state[oldid] = localState;
}

__global__ void perturbate(int32_t *current_tour, int32_t *other_tour, uint32_t g_i, uint32_t g_j,
    uint32_t g_k, uint32_t g_l, uint32_t g_n){
    uint32_t p_i, p_j, p_k, p_l, n;
   
    p_i = g_i;
    p_j = g_j;
    p_k = g_k;
    p_l = g_l;
    n = g_n;

    move_struct ms[5];
    ms[0].src_start = 0;
    ms[0].src_end = p_i;
    ms[0].dst_start = 0;

    ms[1].src_start = p_k + 1;
    ms[1].src_end = p_l;
    ms[1].dst_start = p_i + 1;
    
    ms[2].src_start = p_j + 1;
    ms[2].src_end = p_k;
    ms[2].dst_start = p_i + p_l - p_k + 1;
    
    ms[3].src_start = p_i + 1;
    ms[3].src_end = p_j;
    ms[3].dst_start = p_i + p_l - p_j + 1;
    
    ms[4].src_start = p_l + 1;
    ms[4].src_end = n - 1;
    ms[4].dst_start = p_l + 1;


    const uint32_t dst_start = ms[blockIdx.x].dst_start;
    const uint32_t src_start = ms[blockIdx.x].src_start;
    const uint32_t src_end = ms[blockIdx.x].src_end;
  
    uint32_t step = threadIdx.x;

    while(src_start + step <= src_end){
        other_tour[dst_start + step] = current_tour[src_start + step];
        other_tour[dst_start + step + n] = current_tour[src_start + step + n];
        step += warpSize;
    }
}
__device__ float calculate_perturbate_cost(uint32_t n, uint32_t g_i, uint32_t g_j, uint32_t g_k, uint32_t g_l,
    int32_t *tour){
    float cost = 0;
    int indices[16];    
    indices[0] = g_i;
    indices[1] = g_i + 1;
    indices[2] = g_j;
    indices[3] = g_j + 1;
    indices[4] = g_k;
    indices[5] = g_k + 1;
    indices[6] = g_l;
    indices[7] = g_l + 1;
    indices[8] = g_i;
    indices[9] = g_k + 1;
    indices[10] = g_l;
    indices[11] = g_j + 1;
    indices[12] = g_k;
    indices[13] = g_i + 1;
    indices[14] = g_j;
    indices[15] = g_l + 1;

    for(int i = 0; i < 8; ++i){
        float distance = distance_ij(indices[2 * i], indices[2 * i + 1], tour, n);
        if(i > 3) cost += distance;
        else cost -= distance;
    }

    return cost;

}

__global__ void reduce(gain_struct_t *idata, gain_struct_t *odata, int len){

    extern __shared__ gain_struct_t sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + (blockIdx.x * (blockDim.x * 2)); //1 thread will load max of 2 values
    
    if(i < len){
        if(i + blockDim.x < len){
            float val1 = idata[i].gain;
            float val2 = idata[i + blockDim.x].gain;
            if(val1 > val2){
                sdata[tid].i = idata[i].i;
                sdata[tid].j = idata[i].j;
                sdata[tid].gain = val1;
            }else{
                sdata[tid].i = idata[i + blockDim.x].i;
                sdata[tid].j = idata[i + blockDim.x].j;
                sdata[tid].gain = val2;
            }
        }else{
            sdata[tid].i = idata[i].i;
            sdata[tid].j = idata[i].j;
            sdata[tid].gain = idata[i].gain;
        }
    }else{
        sdata[tid].i = 0;
        sdata[tid].j = 0;
        sdata[tid].gain = 0;
    }
    __syncthreads();


    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid + s < blockDim.x){
            if(sdata[tid].gain < sdata[tid + s].gain){
                sdata[tid].i = sdata[tid + s].i;
                sdata[tid].j = sdata[tid + s].j;
                sdata[tid].gain = sdata[tid + s].gain;
            }
        }
        __syncthreads();
    }

    if(tid == 0){
        odata[blockIdx.x] = sdata[0];
    }
}


__global__ void ILS(uint32_t *indices,
                    int32_t *tour_1,
                    int32_t *tour_2,
                    uint32_t n,
                    curandState *states,
                    uint32_t iteration_count,
                    gain_struct_t *buf_1,
                    gain_struct_t *buf_2,
                    uint32_t buffer_size        
                ){
    setup_kernel<<<1,256>>>(states);
    generate_kernel<<<1,256>>>(states, n, indices, iteration_count);
    
    int job = ((n - 1) * (n - 2)) / 2;
    int thread;
    if(job > 1024) thread = 1024;
    else if(job > 512) { thread = 1024; }
    else if(job > 256) { thread = 512;  }
    else if(job > 128) { thread = 256;  }
    else if(job > 64)  { thread = 128;  }
    else if(job > 32)  { thread = 64;   }
    else               { thread = 32;   }

    int block = (job + thread - 1) / thread;
    while(thread * block > buffer_size) --block;

    for(int iter = 0; iter < iteration_count; ++iter){
        perturbate<<<5, 32>>>(tour_1, tour_2, indices[4 * iter], indices[4 * iter + 1], indices[4 * iter + 2], indices[4 * iter + 3], n);
        float perturbation_cost = calculate_perturbate_cost(n, indices[4 * iter], indices[4 * iter + 1], indices[4 * iter + 2], indices[4 * iter + 3], tour_1);
        float gain = 0;
        bool positiveMoveExists;
        do{
            two_opt<<<block, thread, 2 * n * sizeof(int32_t)>>>(tour_2, buf_1, n);
            int order = 0;
            int left = buffer_size;
            while(left > 1){
                int reduce_thread_size = find_num_of_threads((left + 1) / 2, 1024); //1 thread can handle 2 values
                int reduce_block_size = (((left + 1) / 2) + reduce_thread_size - 1) / reduce_thread_size;    
                
                reduce<<<reduce_block_size, reduce_thread_size, sizeof(gain_struct_t) * reduce_thread_size>>>
                ((order == 0 ? buf_1 : buf_2), 
                (order == 0 ? buf_2 : buf_1), left);
                left = reduce_block_size;
                order = 1 - order;
            }
            cudaDeviceSynchronize();
            order = 1 - order;
            uint32_t best_i, best_j;
            float best_gain;
            if(order == 0){
                best_i = buf_2[0].i;
                best_j = buf_2[0].j;
                best_gain = buf_2[0].gain;
            }else{
                best_i = buf_1[0].i;
                best_j = buf_1[0].j;
                best_gain = buf_1[0].gain;
            }
            positiveMoveExists = (best_i != 0 || best_j != n - 1) && best_gain > 0.1;
            if(positiveMoveExists) {
                gain += best_gain;
                apply_2opt_move<<<1, 32>>>(tour_2, (order == 0) ? buf_2 : buf_1, n); 
            }
        }while(positiveMoveExists);

        if(gain > perturbation_cost){
            int32_t *temp = tour_1;
            tour_1 = tour_2;
            tour_2 = temp;
        }

    }
}

int main(){
    auto coordinate_city_map = new std::unordered_map<std::pair<int32_t, int32_t>, uint32_t, pair_hash>();
    uint32_t n;
    uint32_t iteration_count = 10000;
    std::ifstream fstream("coordinates.txt");
    if(!fstream.is_open()){
        std::cerr << "coordinates.txt cannot be opened" << std::endl;
        return 1;
    }
    fstream >> n;
    int32_t *h_tour = new int32_t[2 * n];
    uint32_t buffer_size = 4096;
    uint32_t *h_tour_raw = new uint32_t[n];
    gain_struct_t *h_buffer = new gain_struct_t[buffer_size];

    for(int i = 0; i < n; ++i){
        uint32_t city_index;
        int32_t x_coord, y_coord;
        fstream >> city_index; //cities are one based
        fstream >> x_coord;
        fstream >> y_coord;
	h_tour[city_index - 1] = x_coord;
        h_tour[city_index - 1 + n] = y_coord;
	auto pair1 = std::make_pair(x_coord, y_coord);
        if(!coordinate_city_map->insert(std::make_pair(pair1, city_index)).second){
         	std::cout << "i:" << city_index << " x:" << x_coord << " y:" << y_coord  << std::endl;
  	}
	
    }
    fstream.close();

    uint32_t *d_perturbation_indices;
    int32_t *d_tour_1;
    int32_t *d_tour_2;
    gain_struct_t *d_buf_1;
    gain_struct_t *d_buf_2;
    curandState *d_curand_states;

    cudaMalloc(&d_perturbation_indices, sizeof(uint32_t) * 4 * iteration_count);
    cudaMalloc(&d_curand_states, sizeof(curandState) * 256);
    cudaMalloc(&d_tour_1, sizeof(int32_t) * n * 2);
    cudaMalloc(&d_tour_2, sizeof(int32_t) * n * 2);
    cudaMalloc(&d_buf_1, sizeof(gain_struct_t) * buffer_size);
    cudaMalloc(&d_buf_2, sizeof(gain_struct_t) * buffer_size);
    cudaMemcpy(d_buf_1, h_buffer, sizeof(gain_struct_t) * buffer_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf_2, h_buffer, sizeof(gain_struct_t) * buffer_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_tour_1, h_tour, sizeof(int32_t) * n * 2, cudaMemcpyHostToDevice);
    //int *sanity = new int[n];
    //for(int i = 0; i < n; ++i) sanity[i] = 1;
        
    //for(int i = 0; i < n; ++i){
    //    int32_t cx = h_tour[i];
    //    int32_t cy = h_tour[i + n];
    //    uint32_t val = coordinate_city_map->at(std::make_pair(cx, cy));
    //    sanity[val - 1] -= 1;
    //}
    
    //for(int i = 0; i < n; ++i) if(sanity[i] != 0) {std::cout << "bad " << i << std::endl; break; }
   

    ILS<<<1,1>>>(d_perturbation_indices,
                 d_tour_1,
                 d_tour_2,
                 n,
                 d_curand_states,
                 iteration_count,
                 d_buf_1,
                 d_buf_2,
                 buffer_size);

    std::ofstream par("par.txt");
    cudaMemcpy(h_tour, d_tour_1, sizeof(int32_t) * 2 * n, cudaMemcpyDeviceToHost);
    for(int i = 0; i < n; ++i){
        int32_t cx = h_tour[i];
        int32_t cy = h_tour[i + n];
        h_tour_raw[i] = coordinate_city_map->at(std::make_pair(cx, cy));
    	par << h_tour_raw[i] << " ";
    }
    par << std::endl;
    par.close();
                 
    cudaFree(d_perturbation_indices);
    cudaFree(d_curand_states);
    cudaFree(d_tour_1);
    cudaFree(d_tour_2);
    cudaFree(d_buf_1);
    cudaFree(d_buf_2);
    delete[] h_tour;
    delete[] h_buffer;
    delete[] h_tour_raw;
}
