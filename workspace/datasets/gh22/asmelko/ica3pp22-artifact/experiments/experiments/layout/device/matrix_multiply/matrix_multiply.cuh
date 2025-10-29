#pragma once

#include <string>

#include <noarr/structures.hpp>

#include "device_experiment.cuh"
#include "kernels.cuh"
#include "kit.h"

class matrix_multiply_experiment : public device_experiment<matrix_multiply_experiment>
{
	size_t lhs_height_;
	size_t lhs_width_;
	size_t rhs_height_;
	size_t rhs_width_;

	using rows_structure_t = decltype(matrix_rows() | noarr::set_length<'m'>(0) | noarr::set_length<'n'>(0));
	using cols_structure_t = decltype(matrix_cols() | noarr::set_length<'m'>(0) | noarr::set_length<'n'>(0));
	using rc_structure_t = rc_curve<'m', 'n', noarr::sized_vector<'a', noarr::scalar<data_t>>, tile_size>;
	using cr_structure_t = rc_curve<'n', 'm', noarr::sized_vector<'a', noarr::scalar<data_t>>, tile_size>;
	using rr_structure_t = rr_curve<'m', 'n', noarr::sized_vector<'a', noarr::scalar<data_t>>, tile_size>;
	using cc_structure_t = rr_curve<'n', 'm', noarr::sized_vector<'a', noarr::scalar<data_t>>, tile_size>;

public:
	multipty_kit<data_t, 'm', 'n'> kit { lhs_height_, lhs_width_, rhs_width_ };

	template <typename struct_t>
	using struct_meta = std::pair<std::string, struct_t>;

	std::tuple<struct_meta<rows_structure_t>, struct_meta<cols_structure_t>, struct_meta<rc_structure_t>,
			   struct_meta<cr_structure_t>, struct_meta<rr_structure_t>, struct_meta<cc_structure_t>>
		types { { "R", rows_structure_t() }, { "C", cols_structure_t() }, { "RC", rc_structure_t() },
				{ "CR", cr_structure_t() },	 { "RR", rr_structure_t() },  { "CC", cc_structure_t() } };

	constexpr static size_t number_args = 3;

	matrix_multiply_experiment(const std::vector<std::string>& args)
		: lhs_height_(args.size() == 3 ? std ::stoi(args[0]) : 8192),
		  lhs_width_(args.size() == 3 ? std ::stoi(args[1]) : 4096),
		  rhs_height_(lhs_width_),
		  rhs_width_(args.size() == 3 ? std ::stoi(args[2]) : 2048)
	{
		noarr::pipelines::CudaPipelines::register_extension();
	}

	~matrix_multiply_experiment() {}

	template <typename transture_lhs_t, typename transture_rhs_t, typename transture_out_t, typename scheduler_t>
	void construct(scheduler_t& scheduler, bool optimized)
	{
		auto l_hub = construct_in_tr_generic<transture_lhs_t>(scheduler, std::make_pair(lhs_height_, lhs_width_),
															  kit.lhs_generator);

		auto r_hub = construct_in_tr_generic<transture_rhs_t>(scheduler, std::make_pair(rhs_height_, rhs_width_),
															  kit.rhs_generator);

		auto out_hub =
			construct_out_tr_generic<transture_out_t>(scheduler, std::make_pair(lhs_height_, rhs_width_), kit);

		construct_multiply_node(scheduler, *l_hub, *r_hub, *out_hub,
								resize<'m', 'n'>(transture_out_t(), lhs_height_, rhs_width_), optimized);
	}

protected:
	template <typename in_lhs_hub_t, typename in_rhs_hub_t, typename out_hub_t, typename scheduler_t,
			  typename out_struct_t>
	auto construct_multiply_node(scheduler_t& scheduler, in_lhs_hub_t& input_lhs_hub, in_rhs_hub_t& input_rhs_hub,
								 out_hub_t& output_hub, out_struct_t out_structure, bool optimized)
	{
		auto multiply_node = std::make_unique<timer_cuda_compute_node>("matmul");
		auto& lhs_link = multiply_node->link(input_lhs_hub.to_consume(noarr::pipelines::Device::DEVICE_INDEX));
		auto& rhs_link = multiply_node->link(input_rhs_hub.to_consume(noarr::pipelines::Device::DEVICE_INDEX));
		auto& output_link = multiply_node->link(output_hub.to_produce(noarr::pipelines::Device::DEVICE_INDEX));

		multiply_node->advance_cuda([&, optimized, out_structure](cudaStream_t stream) {
			auto lhs_bag = noarr::pipelines::bag_from_link(lhs_link);
			auto rhs_bag = noarr::pipelines::bag_from_link(rhs_link);

			output_link.envelope->structure = out_structure;

			auto output_bag = noarr::pipelines::bag_from_link(output_link);

			run_matmul_basic<'m', 'n'>(lhs_bag, rhs_bag, output_bag, stream);
		});

		action_duration.add(multiply_node->advance_duration);
		scheduler << std::move(multiply_node);
	}

	virtual result run_base() override
	{
		return result("rows rows rows", run_single<rows_structure_t, rows_structure_t, rows_structure_t>(false));
	}

	virtual result run_optimized() override
	{
		return result("cols cols cols", run_single<cols_structure_t, cols_structure_t, cols_structure_t>(true));
	}

public:
	void run_one()
	{
		smart_ptr_scheduler s;
		construct<cr_structure_t, rc_structure_t, rows_structure_t>(s, false);
		s.run();
	}
};
