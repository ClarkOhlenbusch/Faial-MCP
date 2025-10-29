#pragma once
#include <cuda_runtime.h>

#include <noarr/structures.hpp>

using dim_t = char;

template <dim_t X, dim_t Y, typename T, typename structure_in_t, typename structure_out_t>
__global__ void transform_internal(const T* input_data, T* output_data, const structure_in_t in_structure,
								   const structure_out_t out_structure, const size_t leading_dim,
								   const size_t total_size)
{
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_size)
		return;

	const size_t i = idx % leading_dim;
	const size_t j = idx / leading_dim;

	out_structure.get_at<X, Y>(output_data, i, j) = in_structure.get_at<X, Y>(input_data, i, j);
}

struct base_cuda_transformer_2d
{
public:
	template <dim_t X, dim_t Y, typename bag_in_t, typename bag_out_t>
	static void transform(const bag_in_t& input_bag, bag_out_t& output_bag, cudaStream_t stream)
	{
		const size_t x_size = input_bag.structure().template get_length<X>();
		const size_t y_size = input_bag.structure().template get_length<Y>();
		const size_t total_size = x_size * y_size;

		const size_t block_size = 128;
		const size_t block_count = (total_size + block_size - 1) / block_size;

		transform_internal<X, Y><<<(unsigned)block_count, (unsigned)block_size, 0, stream>>>(
			input_bag.data(), output_bag.data(), input_bag.structure(), output_bag.structure(), x_size, total_size);
	}
};
