#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>
#include <noarr/structures.hpp>

#include "layouts/layouts.h"
#include "structure-transformers/device/transformer_pipeline.cuh"
#include "structure-transformers/host/transformer_pipeline.h"
#include "utils/matrix.h"
#include "utils/pipelines.h"

using namespace noarr::pipelines;

template <typename T, char X, char Y>
class matrix_kit : public matrix_generator<T, X, Y>
{
	bool validation_result_;

public:
	matrix_kit(size_t height, size_t width) : matrix_generator<T, X, Y>(height, width), validation_result_(true) {}

	template <typename bag_t>
	void validate(const bag_t& bag)
	{
		for (size_t i = 0; i < this->height; i++)
		{
			for (size_t j = 0; j < this->width; j++)
			{
				if (bag.template at<X, Y>(i, j) != this->data_[i * this->width + j])
				{
					std::cout << "Error at [" << i << ", " << j << "] " << bag.template at<X, Y>(i, j)
							  << " != " << this->data_[i * this->width + j] << std::endl;
					validation_result_ = false;
					return;
				}
			}
		}
		validation_result_ = true;
		return;
	}

	bool validation_successful() const { return validation_result_ == true; }
};


template <typename T>
using matrix_rows = noarr::vector<'m', noarr::vector<'n', noarr::scalar<T>>>;
template <typename T>
using matrix_columns = noarr::vector<'n', noarr::vector<'m', noarr::scalar<T>>>;

class transform : public testing::TestWithParam<std::tuple<bool, std::pair<int, int>>>
{};

TEST_P(transform, base)
{
	CudaPipelines::register_extension();

	bool device = std::get<0>(GetParam());
	size_t size_m = std::get<1>(GetParam()).first;
	size_t size_n = std::get<1>(GetParam()).second;

	size_t buffer_size = size_n * size_m * sizeof(double);

	auto rows_struct = matrix_rows<double>() | noarr::set_length<'n'>(size_n) | noarr::set_length<'m'>(size_m);
	auto test_struct =
		resize<'m', 'n'>(rc_curve<'n', 'm', noarr::sized_vector<'a', noarr::scalar<double>>, 16>(), size_m, size_n);

	matrix_kit<double, 'm', 'n'> kit(size_m, size_n);

	auto [in_hub, in_node] = construct_generator(kit, rows_struct, buffer_size);
	auto [out_hub, out_node] = construct_validator(kit, test_struct, buffer_size);

	std::unique_ptr<ComputeNode> transformer;
	if (device)
	{
		transformer = construct_cuda_transformer_2d_node<'m', 'n', decltype(test_struct)>(
			in_hub->to_consume(Device::DEVICE_INDEX), out_hub->to_produce(Device::DEVICE_INDEX));
	}
	else
		transformer = construct_host_transformer_2d_node<'m', 'n', decltype(test_struct)>(
			in_hub->to_consume(Device::HOST_INDEX), out_hub->to_produce(Device::HOST_INDEX));

	SimpleScheduler scheduler;
	scheduler << *in_hub << *in_node << *transformer << *out_hub << *out_node;
	scheduler.run();

	ASSERT_TRUE(kit.validation_successful());
}

INSTANTIATE_TEST_SUITE_P(rectangles, transform,
						 ::testing::Combine(::testing::Values(true, false),
											::testing::Values(std::make_pair(16, 16), std::make_pair(16, 32),
															  std::make_pair(32, 16))),
						 [](auto&& info) {
							 const std::string what = std::get<0>(info.param) == true ? "device" : "host";
							 return what + "_" + std::to_string(std::get<1>(info.param).first) + "_"
									+ std::to_string(std::get<1>(info.param).second);
						 });
