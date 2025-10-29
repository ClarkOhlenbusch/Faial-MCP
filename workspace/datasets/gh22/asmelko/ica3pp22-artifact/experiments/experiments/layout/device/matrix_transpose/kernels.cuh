#pragma once
#include <cuda_runtime.h>

template <char X, char Y, typename T, typename structure_in_t, typename structure_out_t>
__global__ void transpose_basic(const T* __restrict__ input_data, T* __restrict__ output_data,
								const structure_in_t in_structure,
								const structure_out_t out_structure)
{
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	const size_t leading_dim = in_structure.template get_length<X>();

	if (idx >= leading_dim * leading_dim)
		return;

	const size_t i = idx % leading_dim;
	const size_t j = idx / leading_dim;

	out_structure.get_at<X, Y>(output_data, j, i) = in_structure.get_at<X, Y>(input_data, i, j);
}

template <char X, char Y, typename bag_in_t, typename bag_out_t>
void run_transpose_basic(const bag_in_t& input_bag, bag_out_t& output_bag, cudaStream_t stream)
{
	const size_t x_size = input_bag.structure().template get_length<X>();
	const size_t y_size = input_bag.structure().template get_length<Y>();
	const size_t total_size = x_size * y_size;

	const size_t block_size = 256;
	const size_t block_count = (total_size + block_size - 1) / block_size;

	transpose_basic<X, Y><<<(unsigned)block_count, (unsigned)block_size, 0, stream>>>(
		input_bag.data(), output_bag.data(), input_bag.structure(), output_bag.structure());
}

template <char X, char Y, typename T, typename structure_in_t, typename structure_out_t>
__global__ void transpose_coalesced(const T* __restrict__ input_data, T* __restrict__ output_data,
									const structure_in_t in_structure,
								   const structure_out_t out_structure)
{
#define TILE_DIM 16
	__shared__ double tile[TILE_DIM][TILE_DIM + 1];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;

	tile[threadIdx.y][threadIdx.x] = in_structure.get_at<Y, X>(input_data, x, y);

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	out_structure.get_at<Y, X>(output_data, x, y) = tile[threadIdx.x][threadIdx.y];
}

template <char X, char Y, typename bag_in_t, typename bag_out_t>
void run_transpose_coalesced(const bag_in_t& input_bag, bag_out_t& output_bag, cudaStream_t stream)
{
	const size_t x_size = input_bag.structure().template get_length<X>();

	const size_t block_size = TILE_DIM;
	const size_t block_count = x_size / block_size;

	transpose_coalesced<X, Y>
		<<<dim3((unsigned)block_count, (unsigned)block_count), dim3(TILE_DIM, TILE_DIM), 0, stream>>>(
		input_bag.data(), output_bag.data(), input_bag.structure(), output_bag.structure());
}
