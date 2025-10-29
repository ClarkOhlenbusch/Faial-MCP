## Qian-Molloy/PaddlePaddle/paddle/fluid/operators/math/pooling.cu
#### Analysis:

Contains multiple kernels. Unknowns in *imp* correctly identified.
- `KernelPool2D` contains triple-nested for loop.

- `KernelPool2DGrad` contains triple-nested for-loop.

- `KernelMaxPool2DGrad` contains triple-nested for-loop. ~~Read is missing~~ (this is fine, CLang could not recognize it).
  ```
  if (maxIndex != -1) {
    // atomic add
    platform::CudaAtomicAdd(input_grad + maxIndex, output_grad[index]);
  }
  ```
  ```
  if ((maxIndex) (!=.bool) ((0) (-.int) (1))) {
  {}
  } else {
  {}
  }
  ```  

- `KernelPool3D` contains quadruple-nested for-loop.
   Inconsistency in `@func` keyword in *imp*. Same for `@parm` keyword.
   ```
   int wend = min(wstart + ksize_width, input_width);
   dstart = max(dstart, 0);
   ```
   ```
   decl {
        wend = @func min(@parm input_width, (wstart) + (@parm ksize_width))
        }
   (dstart) = (max(0, dstart))
   ```
   Additional appearance of statement `ele` after a write.
   ```
   output_data[index] = ele;
   ```
   ```
   {
   rw output_data[index] = ele;
   ele
   }
   ```

- `KernelPool3DGrad` contains quadruple-nested for-loop.

- `KernelMaxPool3DGrad` contains quadruple-nested for-loop with an if in its body.

- `KernelMaxPool2dWithIdx` contains triple-nested for-loop with an if in its body.

- `KernelMaxPool2DWithIdxGrad` contains triple-nested for-loop with an if in its body.

- `KernelMaxPool2DWithIdxGrad` contains quadruple-nested for-loop with an if in its body.

- `KernelMaxPool3DWithIdxGrad` contains quadruple-nested for-loop with an if in its body.
