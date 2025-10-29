#pragma once

#include "memory-traces/trace.h"
#include "noarr/structures/struct_traits.hpp"
#include "noarr/structures/structs.hpp"

template <typename Structure, typename TracePolicy>
class cuda_trace_bag
{
	memory_type_t memory_type_ = memory_type_t::CUDA_GLOBAL;

	noarr::wrapper<Structure> structure_;
	char* data_;

	TracePolicy trace_policy_;

	template <typename T>
	__device__ void add_trace(access_type_t access_type, size_t worker_id, T* address) const
	{
		trace_policy_.add(
			trace_t(memory_type_, access_type, worker_id, (uintptr_t)address, sizeof(noarr::scalar_t<Structure>)));
	}

public:
	decltype(auto) get_traces() const { return trace_policy_.get_traces(); }

	constexpr cuda_trace_bag(Structure structure, char* data, TracePolicy trace_policy, memory_type_t memory_type)
		: memory_type_(memory_type), structure_(noarr::wrap(structure)), data_(data), trace_policy_(trace_policy)
	{}

	template <char... Dims, typename... Ts>
	__device__ decltype(auto) at(size_t worker_id, access_type_t access_type, Ts... ts)
	{
		auto&& address = structure_.template get_at<Dims...>(data_, ts...);
		add_trace(access_type, worker_id, &address);
		return address;
	}
};

template <typename Structure, typename TracePolicy>
constexpr auto make_cuda_trace_bag(Structure s, char* data, TracePolicy trace_policy,
								   memory_type_t memory_type = memory_type_t::CUDA_GLOBAL)
{
	return cuda_trace_bag<Structure, TracePolicy>(s, data, trace_policy, memory_type);
}
