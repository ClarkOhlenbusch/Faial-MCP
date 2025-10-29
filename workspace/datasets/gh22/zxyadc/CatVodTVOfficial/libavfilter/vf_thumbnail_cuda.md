# Assigned 07/15/2022

## zxyadc/CatVodTVOfficial/libavfilter/vf_thumbnail_cuda.cu

### Kernels:

1. Thumbnail_uchar
2. Thumbnail_uchar2
3. Thumbnail_ushort
4. Thumbnail_ushort2

There were four kernels in the CUDA file, and all of them were analyzed by `c-ast`.

### Warnings:

None

### 1. Thumbnail_uchar

<hr>

### Reads/Writes

There is 1 read in this kernel:

```cuda
atomicAdd(&histogram[pixel], 1);
```

It was captured by `c-ast`.

### Loops/Conditionals

There is 1 conditional in this kernel:

```cuda
if (y < src_height && x < src_width)
```

It was captured by `c-ast`.

### 2. Thumbnail_uchar2

<hr>

### Reads/Writes

There are 2 reads in this kernel:

```cuda
atomicAdd(&histogram[pixel.x], 1);
atomicAdd(&histogram[256 + pixel.y], 1);
```

Both of them were captured by `c-ast`.

### Loops/Conditionals

There is 1 conditional in this kernel:

```cuda
if (y < src_height && x < src_width)
```

It was captured by `c-ast`.

### 3. Thumbnail_ushort

<hr>

### Reads/Writes

There is 1 read in this kernel:

```cuda
atomicAdd(&histogram[pixel], 1);
```

It was captured by `c-ast`.

### Loops/Conditionals

There is 1 conditional in this kernel:

```cuda
if (y < src_height && x < src_width)
```

It was captured by `c-ast`.

### 4. Thumbnail_ushort2

<hr>

### Reads/Writes

There are 2 reads in this kernel:

```cuda
atomicAdd(&histogram[(pixel.x + 128) >> 8], 1);
atomicAdd(&histogram[256 + ((pixel.y + 128) >> 8)], 1);
```

Both of them were captured by `c-ast`.

### Loops/Conditionals

There is 1 conditional in this kernel:

```cuda
if (y < src_height && x < src_width)
```

It was captured by `c-ast`.
