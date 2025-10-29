# Assigned 07/23/2022

## mlcommons/training_results_v2.0/CASIA/benchmarks/maskrcnn/implementations/pytorch/maskrcnn_benchmark/csrc/cuda/ROIPool_cuda.cu

### Kernels:

1. RoIPoolFForward
2. RoIPoolFBackward

There were two kernels in the CUDA file, and both of them were analyzed by `c-ast`.

### Warnings:

None

### 1. RoIPoolFForward

<hr>

### Reads/Writes

There are 7 reads and 2 writes in this kernel:

```cuda
// Reads
int roi_batch_ind = offset_bottom_rois[0];
int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);
if (offset_bottom_data[bottom_index] > maxval) {
  maxval = offset_bottom_data[bottom_index];

// Writes
top_data[index] = maxval;
argmax_data[index] = maxidx;
```

All of them were captured by `c-ast`.

### Loops/Conditionals

There are 3 loops and 1 conditional in this kernel:

```cuda
// Loops
CUDA_1D_KERNEL_LOOP(index, nthreads) {
for (int h = hstart; h < hend; ++h) {
for (int w = wstart; w < wend; ++w) {

// Conditionals
if (offset_bottom_data[bottom_index] > maxval) {
```

All of them were captured by `c-ast`.

### 2. RoIPoolFBackward

<hr>

There are 3 reads in this kernel:

```cuda
int roi_batch_ind = offset_bottom_rois[0];
int argmax = offset_argmax_data[ph * pooled_width + pw];
static_cast<T>(offset_top_diff[ph * pooled_width + pw]));
```

All of them were captured by `c-ast`.

### Loops/Conditionals

There is 1 conditional in this kernel:

```cuda
if (argmax != -1) {
```

It was captured by `c-ast`, albeit with a unary negation being converted to binary subtraction:

```
if (argmax != 0 - 1) {
```
