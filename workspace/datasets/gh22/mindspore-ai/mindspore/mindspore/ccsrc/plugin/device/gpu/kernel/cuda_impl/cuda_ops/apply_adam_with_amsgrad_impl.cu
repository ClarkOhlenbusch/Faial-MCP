/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_adam_with_amsgrad_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T sqrtFunc(T x) {
  return sqrt(x);
}

template <>
__device__ __forceinline__ half sqrtFunc(half x) {
  return sqrt(__half2float(x));
}

template <typename T>
__device__ __forceinline__ T maxFunc(T x, T y) {
  return x > y? x : y;
}

template <>
__device__ __forceinline__ half maxFunc(half x, half y) {
  return __half2float(x) > __half2float(y)? __half2float(x) : __half2float(y);
}

template <typename T>
__global__ void CalApplyAdamWithAmsgradKernel(const size_t input_elements, const int64_t batch_size, T *var, T *m,
                                              T *v, T *vhat, T *beta1_power, T *beta2_power, const T *lr,
                                              const T *grad, const float beta1, const float beta2,
                                              const float epsilon) {
  auto all_elements = input_elements * batch_size;
  const T one = static_cast<T>(1.0);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < all_elements; pos += gridDim.x * blockDim.x) {
    auto batch = pos / input_elements;
    auto new_learning_rate = lr[batch] * sqrtFunc(one - beta2_power[batch]) / (one - beta1_power[batch]);
    m[pos] += (grad[pos] - m[pos]) * (one - static_cast<T>(beta1));
    v[pos] += (grad[pos] * grad[pos] - v[pos]) * (one - static_cast<T>(beta2));
    vhat[pos] = maxFunc(vhat[pos], v[pos]);
    var[pos] -= new_learning_rate * m[pos] / (sqrtFunc(vhat[pos]) + static_cast<T>(epsilon));
  }
}

template <typename T>
void CalApplyAdamWithAmsgrad(const size_t input_elements, const int64_t batch_size, T *var, T *m, T *v, T *vhat,
                             T *beta1_power, T *beta2_power, const T *lr, const T *grad, const float beta1,
                             const float beta2, const float epsilon, const uint32_t &device_id,
                             cudaStream_t stream_ptr) {
  CalApplyAdamWithAmsgradKernel<<<CUDA_BLOCKS(device_id, input_elements * batch_size), CUDA_THREADS(device_id), 0,
                                   stream_ptr>>>(input_elements, batch_size, var, m, v, vhat, beta1_power, beta2_power,
                                   lr, grad, beta1, beta2, epsilon);
}

template CUDA_LIB_EXPORT void CalApplyAdamWithAmsgrad<double>(const size_t size, const int64_t batch_size, double *var,
                                                              double *m, double *v, double *vhat, double *beta1_power,
                                                              double *beta2_power, const double *lr,
                                                              const double *grad, const float beta1, const float beta2,
                                                              const float epsilon, const uint32_t &device_id,
                                                              cudaStream_t stream_ptr);

template CUDA_LIB_EXPORT void CalApplyAdamWithAmsgrad<float>(const size_t size, const int64_t batch_size, float *var,
                                                             float *m, float *v, float *vhat, float *beta1_power,
                                                             float *beta2_power, const float *lr, const float *grad,
                                                             const float beta1, const float beta2, const float epsilon,
                                                             const uint32_t &device_id, cudaStream_t stream_ptr);

template CUDA_LIB_EXPORT void CalApplyAdamWithAmsgrad<half>(const size_t size, const int64_t batch_size, half *var,
                                                            half *m, half *v, half *vhat, half *beta1_power,
                                                            half *beta2_power, const half *lr, const half *grad,
                                                            const float beta1, const float beta2, const float epsilon,
                                                            const uint32_t &device_id, cudaStream_t stream_ptr);
