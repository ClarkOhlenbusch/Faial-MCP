#include<vector>
#include<cuda_runtime.h>
#include<torch/extension.h>
#include<cuda.h>
#include <cublas_v2.h>

/*一些可以参考的cuda上的激活函数实现方式*/
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}
template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}

template <typename scalar_t>
__global__ void mm_kernel(scalar_t& A,scalar_t& B, scalar_t& dst){



}

std::vector<torch::Tensor> linear_cuda_forward(torch::Tensor W,
                                               torch::Tensor x)
{
    auto width = x.sizes()[1];
    auto height = W.sizes()[1];
    auto dst = torch::zeros({width, height});

    // 定义kernel的执行配置
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);
    // 执行kernel

  AT_DISPATCH_FLOATING_TYPES(W.type(), "lltm_forward_cuda", ([&] {
    mm_kernel<scalar_t><<<gridSize, blockSize>>>(x.transpose(0, 1).data<scalar_t>(), W.data<scalar_t>(), dst.data<scalar_t>());
  }));
}
std::vector<torch::Tensor> linear_cuda_backward(torch::Tensor dy, 
                        torch::Tensor x, torch::Tensor w);






