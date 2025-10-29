#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <noarr/structures.hpp>

constexpr uint32_t tile_size = 16;

template <char I, char J, typename structure_lhs_t, typename structure_rhs_t, typename structure_out_t>
__global__ void matmul_basic(const char* __restrict__ lhs_data, const char* __restrict__ rhs_data,
							 char* __restrict__ output_data, const structure_lhs_t lhs_structure,
							 const structure_rhs_t rhs_structure, const structure_out_t out_structure)
{
	const uint32_t k = lhs_structure.template get_length<J>();

	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= rhs_structure.template get_length<J>() || y >= lhs_structure.template get_length<I>())
		return;

	noarr::scalar_t<decltype(out_structure.unwrap())> acc = 0.0;
	for (uint32_t i = 0; i < k; ++i)
	{
		acc +=
			lhs_structure.template get_at<I, J>(lhs_data, y, i) * rhs_structure.template get_at<I, J>(rhs_data, i, x);
	}

	out_structure.template get_at<I, J>(output_data, y, x) = acc;
}

template <char I, char J, typename structure_lhs_t, typename structure_rhs_t, typename structure_out_t>
__global__ void matmul_shared(const char* __restrict__ lhs_data, const char* __restrict__ rhs_data,
							  char* __restrict__ output_data, const structure_lhs_t lhs_structure,
							  const structure_rhs_t rhs_structure, const structure_out_t out_structure)
{
	using F = noarr::scalar_t<decltype(out_structure.unwrap())>;

	constexpr auto tile_structure = noarr::wrapper(noarr::array<I, tile_size, noarr::array<J, tile_size, noarr::scalar<F>>>());
	__shared__ F l_tile[tile_size * tile_size];
	__shared__ F r_tile[tile_size * tile_size];

	const uint32_t k = lhs_structure.template get_length<J>();

	const uint32_t x = blockIdx.x * tile_size + threadIdx.x;
	const uint32_t y = blockIdx.y * tile_size + threadIdx.y;
	F acc = (F)0;

	for (uint32_t i = 0; i < k; i += tile_size)
	{
		tile_structure.get_at<I, J>(l_tile, threadIdx.y, threadIdx.x) =
			lhs_structure.template get_at<I, J>(lhs_data, y, threadIdx.x + i);
		tile_structure.get_at<I, J>(r_tile, threadIdx.y, threadIdx.x) =
			rhs_structure.template get_at<I, J>(rhs_data, threadIdx.y + i, x);

		__syncthreads();

		for (uint32_t j = 0; j < tile_size; j++)
			acc += tile_structure.get_at<I, J>(l_tile, threadIdx.y, j)
				   * tile_structure.get_at<I, J>(r_tile, j, threadIdx.x);

		__syncthreads();
	}

	out_structure.template get_at<I, J>(output_data, y, x) = acc;
}

#define DIV_UP(X, Y) (X + Y - 1) / Y

template <char I, char J, typename bag_lhs_t, typename bag_rhs_t, typename bag_out_t>
void run_matmul_basic(const bag_lhs_t& lhs_bag, const bag_rhs_t& rhs_bag, bag_out_t& output_bag, cudaStream_t stream)
{
	const size_t I_size = lhs_bag.structure().template get_length<I>();
	const size_t J_size = rhs_bag.structure().template get_length<J>();

	const size_t block_size = tile_size;
	const size_t block_count_x = DIV_UP(J_size, block_size);
	const size_t block_count_y = DIV_UP(I_size, block_size);

	matmul_shared<I, J>
		<<<dim3((unsigned)block_count_x, (unsigned)block_count_y), dim3((unsigned)block_size, (unsigned)block_size), 0,
		   stream>>>(lhs_bag.data(), rhs_bag.data(), output_bag.data(), lhs_bag.structure(), rhs_bag.structure(),
					 output_bag.structure());
}
