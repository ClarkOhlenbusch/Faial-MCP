//pass
//--gridDim=[1024] --blockDim=[1024]
/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

template <typename T>
__global__ void MatrixTransposeKernel(const T *input, int elements, int row, int col, T *output) {
  // if (col <= 0 || row <= 0 || row != col || elements != blockDim.x * gridDim.x) {
  //   return;
  // }
  __requires (col > 0);
  __requires (row > 0);
  __requires (row == col);
  //__requires (elements > blockDim.x * gridDim.x);
  const int matrix_size = row * col;
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < elements; pos += blockDim.x * gridDim.x) {
    const int b = pos / matrix_size;
    const int b_stride = b * matrix_size;
    const int r = (pos - b_stride) / col;
    const int c = (pos - b_stride) % col;
    // For output,  new position is  b_stride + c * col + r.
    output[b_stride + c * col + r] = input[pos];
  }
}
