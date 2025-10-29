#pragma once

#include <noarr/structures.hpp>

#include "device_experiment.cuh"
#include "kernels.cuh"
#include "kit.h"

class matrix_transpose_experiment : public device_experiment<matrix_transpose_experiment>
{
	size_t height_ = 2048;
	
	using rows_structure_t = decltype(matrix_rows() | noarr::set_length<'m'>(0) | noarr::set_length<'n'>(0));
	using cols_structure_t = decltype(matrix_cols() | noarr::set_length<'m'>(0) | noarr::set_length<'n'>(0));

public:
	transpose_kit<data_t, 'm', 'n'> kit { height_, height_ };

	std::tuple<std::pair<std::string, rows_structure_t>, std::pair<std::string, cols_structure_t>> types {
		{ "rows ", rows_structure_t() }, { "cols ", cols_structure_t() }
	};

	constexpr static size_t number_args = 1;

	matrix_transpose_experiment() { noarr::pipelines::CudaPipelines::register_extension(); }

	template <typename transture_t, typename scheduler_t>
	void construct(scheduler_t& scheduler, bool optimized)
	{
		auto in_hub = construct_in_tr_generic<transture_t>(scheduler, std::make_pair(height_, height_), kit);

		auto out_hub = construct_out_tr_generic<rows_structure_t>(scheduler, std::make_pair(height_, height_), kit);

		construct_transpose_node(scheduler, *in_hub, *out_hub, optimized);
	}

protected:
	template <typename in_hub_t, typename out_hub_t, typename scheduler_t>
	auto construct_transpose_node(scheduler_t& scheduler, in_hub_t& input_hub, out_hub_t& output_hub, bool optimized)
	{
		auto transpose_node = std::make_unique<timer_cuda_compute_node>("transpose");
		auto& input_link = transpose_node->link(input_hub.to_consume(noarr::pipelines::Device::DEVICE_INDEX));
		auto& output_link = transpose_node->link(output_hub.to_produce(noarr::pipelines::Device::DEVICE_INDEX));

		transpose_node->advance_cuda([&, optimized](cudaStream_t stream) {
			auto input_bag = noarr::pipelines::bag_from_link(input_link);
			auto height = input_bag.structure().template get_length<'m'>();

			output_link.envelope->structure =
				output_link.envelope->structure | noarr::set_length<'m'>(height) | noarr::set_length<'n'>(height);

			auto output_bag = noarr::pipelines::bag_from_link(output_link);
			if (optimized)
				run_transpose_coalesced<'m', 'n'>(input_bag, output_bag, stream);
			else
				run_transpose_basic<'m', 'n'>(input_bag, output_bag, stream);
		});

		action_duration.add(transpose_node->advance_duration);
		scheduler << std::move(transpose_node);
	}

	virtual result run_base() override
	{
		return result("rows", run_single<rows_structure_t>(false));
	}

	virtual result run_optimized() override { return result("rows", run_single<rows_structure_t>(true)); }
};
