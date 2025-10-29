#pragma once

#include "../trace.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void load_and_free(size_t* sizes, trace_t** traces, trace_t* linearized_traces)
{
	size_t acc = 0;
	for (size_t i = 0; i < 1024; i++)
	{
		auto count = sizes[i];
		for (size_t j = 0; j < count; j++)
		{
			linearized_traces[acc + j] = traces[i][j];
		}
		acc += count;
		free(traces[i]);
	}
}

struct cuda_trace_policy
{
	trace_t** traces;
	size_t* sizes;

	__host__ void initialize()
	{
		cudaMallocManaged(&sizes, 1024 * sizeof(size_t));
		cudaMemset(sizes, 0, 1024 * sizeof(size_t));

		cudaMallocManaged(&traces, 1024 * sizeof(trace_t*));
		cudaMemset(traces, 0, 1024 * sizeof(size_t*));
	}

	__device__ void add(trace_t&& trace) const
	{
		const size_t count = sizes[trace.worker_id];
		auto old_arr = traces[trace.worker_id];
		const auto is_full = (count & (count - 1)) == 0;

		if (is_full)
		{
			const auto new_size = count == 0 ? 1 : count * 2;
			trace_t* new_arr = (trace_t*)malloc(new_size * sizeof(trace_t));
			memcpy(new_arr, old_arr, count * sizeof(trace_t));

			if (old_arr != nullptr)
				free(old_arr);

			traces[trace.worker_id] = new_arr;
		}

		traces[trace.worker_id][count] = trace;

		sizes[trace.worker_id] = count + 1;
	}

	__host__ worker_trace_map_t get_traces()
	{
		size_t total_size = 0;
		for (size_t i = 0; i < 1024; i++)
		{
			total_size += sizes[i];
		}

		trace_t* linearized_traces;
		cudaMallocManaged(&linearized_traces, total_size * sizeof(trace_t));

		load_and_free<<<1, 1>>>(sizes, traces, linearized_traces);
		cudaDeviceSynchronize();

		worker_trace_map_t map;
		size_t acc = 0;

		for (size_t i = 0; i < 1024; i++)
		{
			if (sizes[i] == 0)
				continue;

			map[i].resize(sizes[i]);
			memcpy(map[i].data(), linearized_traces + acc, sizes[i] * sizeof(trace_t));
			acc += sizes[i];
		}

		cudaFree(linearized_traces);

		return map;
	}

	__host__ void release()
	{
		cudaFree(sizes);
		cudaFree(traces);
	}
};
