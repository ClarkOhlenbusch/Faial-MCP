#pragma once
#include <memory>

#include <noarr/pipelines.hpp>
#include <noarr/cuda-pipelines.hpp>
#include <noarr/structures-pipelines.hpp>

#include "layouts/layouts.h"
#include "transformer.cuh"
#include "utils/pipelines.h"

template <dim_t X, dim_t Y, typename transform_struct_t, typename in_link_t, typename out_link_t>
auto construct_cuda_transformer_2d_node(in_link_t& consume_link, out_link_t& produce_link)
{
	auto transformer = std::make_unique<timer_cuda_compute_node>("cuda-transformer");
	auto& input_link = transformer->link(consume_link);
	auto& output_link = transformer->link(produce_link);

	transformer->advance_cuda([&, &transformer = *transformer](cudaStream_t stream) {
		auto input_bag = noarr::pipelines::bag_from_link(input_link);
		auto x_length = input_bag.structure().template get_length<X>();
		auto y_length = input_bag.structure().template get_length<Y>();

		output_link.envelope->structure = resize<X, Y>(transform_struct_t(), x_length, y_length);

		auto output_bag = noarr::pipelines::bag_from_link(output_link);

		base_cuda_transformer_2d::transform<X, Y>(input_bag, output_bag, stream);
	});

	return transformer;
}

template <dim_t X, dim_t Y, typename transform_struct_t, typename in_hub_t>
auto construct_cuda_transformer(in_hub_t& input_hub, size_t buffer_size)
{
	auto transformer_hub = std::make_unique<noarr::pipelines::Hub<transform_struct_t>>(buffer_size);
	transformer_hub->allocate_envelopes(noarr::pipelines::Device::DEVICE_INDEX, 1);
	transformer_hub->allocate_envelopes(noarr::pipelines::Device::HOST_INDEX, 1);

	auto transformer_node = construct_cuda_transformer_2d_node<X, Y, transform_struct_t>(
		input_hub.to_consume(noarr::pipelines::Device::DEVICE_INDEX),
		transformer_hub->to_produce(noarr::pipelines::Device::DEVICE_INDEX));

	return std::make_pair(std::move(transformer_hub), std::move(transformer_node));
}
