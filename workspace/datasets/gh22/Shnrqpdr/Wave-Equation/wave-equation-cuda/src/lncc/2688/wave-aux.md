# Assigned 07/23/2022

## Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu

### Kernels:

1. waveEquationKernel

There was one kernel in the CUDA file, and it was analyzed by `c-ast`.

### Warnings:

```
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:100:27: error: definition of type 'char1' conflicts with typedef of the same name
struct __device_builtin__ char1
                          ^
/usr/local/share/c-to-json/include/cuda.h:204:1: note: 'char1' declared here
__MAKE_VECTOR_OPERATIONS(signed char,char)
^
/usr/local/share/c-to-json/include/cuda.h:165:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##1;         \
    ^
<scratch space>:2:1: note: expanded from here
char1
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:105:27: error: definition of type 'uchar1' conflicts with typedef of the same name
struct __device_builtin__ uchar1
                          ^
/usr/local/share/c-to-json/include/cuda.h:205:1: note: 'uchar1' declared here
__MAKE_VECTOR_OPERATIONS(unsigned char,uchar)
^
/usr/local/share/c-to-json/include/cuda.h:165:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##1;         \
    ^
<scratch space>:33:1: note: expanded from here
uchar1
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:111:40: error: definition of type 'char2' conflicts with typedef of the same name
struct __device_builtin__ __align__(2) char2
                                       ^
/usr/local/share/c-to-json/include/cuda.h:204:1: note: 'char2' declared here
__MAKE_VECTOR_OPERATIONS(signed char,char)
^
/usr/local/share/c-to-json/include/cuda.h:168:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##2;         \
    ^
<scratch space>:3:1: note: expanded from here
char2
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:116:40: error: definition of type 'uchar2' conflicts with typedef of the same name
struct __device_builtin__ __align__(2) uchar2
                                       ^
/usr/local/share/c-to-json/include/cuda.h:205:1: note: 'uchar2' declared here
__MAKE_VECTOR_OPERATIONS(unsigned char,uchar)
^
/usr/local/share/c-to-json/include/cuda.h:168:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##2;         \
    ^
<scratch space>:34:1: note: expanded from here
uchar2
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:121:27: error: definition of type 'char3' conflicts with typedef of the same name
struct __device_builtin__ char3
                          ^
/usr/local/share/c-to-json/include/cuda.h:204:1: note: 'char3' declared here
__MAKE_VECTOR_OPERATIONS(signed char,char)
^
/usr/local/share/c-to-json/include/cuda.h:171:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##3;         \
    ^
<scratch space>:4:1: note: expanded from here
char3
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:126:27: error: definition of type 'uchar3' conflicts with typedef of the same name
struct __device_builtin__ uchar3
                          ^
/usr/local/share/c-to-json/include/cuda.h:205:1: note: 'uchar3' declared here
__MAKE_VECTOR_OPERATIONS(unsigned char,uchar)
^
/usr/local/share/c-to-json/include/cuda.h:171:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##3;         \
    ^
<scratch space>:35:1: note: expanded from here
uchar3
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:131:40: error: definition of type 'char4' conflicts with typedef of the same name
struct __device_builtin__ __align__(4) char4
                                       ^
/usr/local/share/c-to-json/include/cuda.h:204:1: note: 'char4' declared here
__MAKE_VECTOR_OPERATIONS(signed char,char)
^
/usr/local/share/c-to-json/include/cuda.h:174:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##4;         \
    ^
<scratch space>:5:1: note: expanded from here
char4
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:136:40: error: definition of type 'uchar4' conflicts with typedef of the same name
struct __device_builtin__ __align__(4) uchar4
                                       ^
/usr/local/share/c-to-json/include/cuda.h:205:1: note: 'uchar4' declared here
__MAKE_VECTOR_OPERATIONS(unsigned char,uchar)
^
/usr/local/share/c-to-json/include/cuda.h:174:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##4;         \
    ^
<scratch space>:36:1: note: expanded from here
uchar4
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:141:27: error: definition of type 'short1' conflicts with typedef of the same name
struct __device_builtin__ short1
                          ^
/usr/local/share/c-to-json/include/cuda.h:206:1: note: 'short1' declared here
__MAKE_VECTOR_OPERATIONS(short, short)
^
/usr/local/share/c-to-json/include/cuda.h:165:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##1;         \
    ^
<scratch space>:64:1: note: expanded from here
short1
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:146:27: error: definition of type 'ushort1' conflicts with typedef of the same name
struct __device_builtin__ ushort1
                          ^
/usr/local/share/c-to-json/include/cuda.h:207:1: note: 'ushort1' declared here
__MAKE_VECTOR_OPERATIONS(unsigned short,ushort)
^
/usr/local/share/c-to-json/include/cuda.h:165:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##1;         \
    ^
<scratch space>:95:1: note: expanded from here
ushort1
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:151:40: error: definition of type 'short2' conflicts with typedef of the same name
struct __device_builtin__ __align__(4) short2
                                       ^
/usr/local/share/c-to-json/include/cuda.h:206:1: note: 'short2' declared here
__MAKE_VECTOR_OPERATIONS(short, short)
^
/usr/local/share/c-to-json/include/cuda.h:168:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##2;         \
    ^
<scratch space>:65:1: note: expanded from here
short2
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:156:40: error: definition of type 'ushort2' conflicts with typedef of the same name
struct __device_builtin__ __align__(4) ushort2
                                       ^
/usr/local/share/c-to-json/include/cuda.h:207:1: note: 'ushort2' declared here
__MAKE_VECTOR_OPERATIONS(unsigned short,ushort)
^
/usr/local/share/c-to-json/include/cuda.h:168:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##2;         \
    ^
<scratch space>:96:1: note: expanded from here
ushort2
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:161:27: error: definition of type 'short3' conflicts with typedef of the same name
struct __device_builtin__ short3
                          ^
/usr/local/share/c-to-json/include/cuda.h:206:1: note: 'short3' declared here
__MAKE_VECTOR_OPERATIONS(short, short)
^
/usr/local/share/c-to-json/include/cuda.h:171:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##3;         \
    ^
<scratch space>:66:1: note: expanded from here
short3
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:166:27: error: definition of type 'ushort3' conflicts with typedef of the same name
struct __device_builtin__ ushort3
                          ^
/usr/local/share/c-to-json/include/cuda.h:207:1: note: 'ushort3' declared here
__MAKE_VECTOR_OPERATIONS(unsigned short,ushort)
^
/usr/local/share/c-to-json/include/cuda.h:171:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##3;         \
    ^
<scratch space>:97:1: note: expanded from here
ushort3
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:171:30: error: definition of type 'short4' conflicts with typedef of the same name
__cuda_builtin_vector_align8(short4, short x; short y; short z; short w;);
                             ^
/usr/local/share/c-to-json/include/cuda.h:206:1: note: 'short4' declared here
__MAKE_VECTOR_OPERATIONS(short, short)
^
/usr/local/share/c-to-json/include/cuda.h:174:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##4;         \
    ^
<scratch space>:67:1: note: expanded from here
short4
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:172:30: error: definition of type 'ushort4' conflicts with typedef of the same name
__cuda_builtin_vector_align8(ushort4, unsigned short x; unsigned short y; unsigned short z; unsigned short w;);
                             ^
/usr/local/share/c-to-json/include/cuda.h:207:1: note: 'ushort4' declared here
__MAKE_VECTOR_OPERATIONS(unsigned short,ushort)
^
/usr/local/share/c-to-json/include/cuda.h:174:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##4;         \
    ^
<scratch space>:98:1: note: expanded from here
ushort4
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:174:27: error: definition of type 'int1' conflicts with typedef of the same name
struct __device_builtin__ int1
                          ^
/usr/local/share/c-to-json/include/cuda.h:208:1: note: 'int1' declared here
__MAKE_VECTOR_OPERATIONS(int,int)
^
/usr/local/share/c-to-json/include/cuda.h:165:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##1;         \
    ^
<scratch space>:126:1: note: expanded from here
int1
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:179:27: error: definition of type 'uint1' conflicts with typedef of the same name
struct __device_builtin__ uint1
                          ^
/usr/local/share/c-to-json/include/cuda.h:209:1: note: 'uint1' declared here
__MAKE_VECTOR_OPERATIONS(unsigned int,uint)
^
/usr/local/share/c-to-json/include/cuda.h:165:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##1;         \
    ^
<scratch space>:157:1: note: expanded from here
uint1
^
In file included from /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu:3:
In file included from /usr/include/device_launch_parameters.h:53:
/usr/include/vector_types.h:184:30: error: definition of type 'int2' conflicts with typedef of the same name
__cuda_builtin_vector_align8(int2, int x; int y;);
                             ^
/usr/local/share/c-to-json/include/cuda.h:208:1: note: 'int2' declared here
__MAKE_VECTOR_OPERATIONS(int,int)
^
/usr/local/share/c-to-json/include/cuda.h:168:5: note: expanded from macro '__MAKE_VECTOR_OPERATIONS'
  } NAME##2;         \
    ^
<scratch space>:127:1: note: expanded from here
int2
^
fatal error: too many errors emitted, stopping now [-ferror-limit=]
20 errors generated when compiling for host.
Error while processing /home/udaya/gpu-oss-dataset/Shnrqpdr/Wave-Equation/wave-equation-cuda/src/lncc/2688/wave-aux.cu.
```

### Reads/Writes

There were 6 reads and 1 write in this kernel:

```cuda
// Reads
wave[i * blockDim.x + j]
wavePast[i * blockDim.x + j]
wave[(i + 1) * blockDim.x + j]
wave[(i - 1) * blockDim.x + j]
wave[i * blockDim.x + (j + 1)]
wave[i * blockDim.x + (j - 1)];

// Writes
waveFuture[i * blockDim.x + j]
```

All of them were captured by `c-ast`.

### Loops/Conditionals

There was 1 conditional in this kernel:

```cuda
if ((i > 0 && i < N - 1) && (j > 0 && j < N - 1))
```

Which was converted to the following in `c-ast`:

```
decl local __unk0;
if (__unk0 != 0) {
```

This is likely due to the use of `N`, which is declared here at the top of the file:

```cuda
#define N XYXY
```
