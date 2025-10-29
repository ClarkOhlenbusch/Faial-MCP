#include <gtest/gtest.h>
#include <noarr/structures.hpp>
#include <noarr/structures_extended.hpp>

#include "memory-traces/device/cuda_model.h"
#include "memory-traces/device/cuda_trace_bag.cuh"
#include "memory-traces/device/cuda_trace_policy.cuh"

template <typename Structure>
__global__ void test_kernel_bag(Structure s, char* data, cuda_trace_policy policy)
{
	auto bag = make_cuda_trace_bag(s, data, policy);
	bag.template at<'a'>(threadIdx.x, access_type_t::WRITE, threadIdx.x) = 1;
}

TEST(cuda_trace, bag)
{
	cudaSetDevice(0);

	auto s = noarr::sized_vector<'a', noarr::scalar<int>>(noarr::scalar<int>(), 10);

	int* data;
	cudaMalloc(&data, 10 * sizeof(int));

	cuda_trace_policy p;
	p.initialize();

	test_kernel_bag<<<1, 6>>>(s, (char*)data, p);
	cudaDeviceSynchronize();

	auto map = p.get_traces();

	p.release();

	ASSERT_EQ(map.size(), 6);

	for (size_t i = 0; i < 6; i++)
	{
		auto&& traces = map.at(i);
		ASSERT_EQ(traces.size(), 1);
		ASSERT_EQ(traces.front(),
				  trace_t(memory_type_t::CUDA_GLOBAL, access_type_t::WRITE, i, (uintptr_t)(data + i), sizeof(int)));
	}

	cudaFree(data);
}

class cuda_trace_model : public testing::TestWithParam<std::tuple<std::string, double>>
{};

using matrix_rows = noarr::vector<'m', noarr::vector<'n', noarr::scalar<int>>>;
using matrix_cols = noarr::vector<'n', noarr::vector<'m', noarr::scalar<int>>>;

template <typename Structure>
__global__ void test_kernel_model(Structure s, char* data, cuda_trace_policy policy)
{
	auto bag = make_cuda_trace_bag(s, data, policy);
	bag.template at<'m', 'n'>(threadIdx.x + blockDim.x * threadIdx.y, access_type_t::WRITE, threadIdx.y, threadIdx.x) =
		1;
}

TEST_P(cuda_trace_model, global)
{
	auto structure = std::get<0>(GetParam());
	auto result = std::get<1>(GetParam());

	cudaSetDevice(0);


	int* data;
	cudaMalloc(&data, 1024 * sizeof(int));

	cuda_trace_policy p;
	p.initialize();

	if (structure == "row")
	{
		auto rows_s = matrix_rows() | noarr::set_length<'m'>(32) | noarr::set_length<'n'>(32);

		test_kernel_model<<<1, dim3(32, 32)>>>(rows_s, (char*)data, p);
	}
	else if (structure == "col")
	{
		auto cols_s = matrix_cols() | noarr::set_length<'n'>(32) | noarr::set_length<'m'>(32);

		test_kernel_model<<<1, dim3(32, 32)>>>(cols_s, (char*)data, p);
	}

	cudaDeviceSynchronize();

	auto map = p.get_traces();

	cuda_memory_model m;

	auto scores = m.score(map);

	p.release();

	ASSERT_EQ(scores.size(), 32);
	for (size_t i = 0; i < 32; i++)
	{
		ASSERT_EQ(scores[i], result);
	}
}

INSTANTIATE_TEST_SUITE_P(rows_cols, cuda_trace_model,
						 ::testing::Values(std::make_pair("row", 0.0), std::make_pair("col", 1.0)),
						 [](auto&& info) { return std::get<0>(info.param); });
