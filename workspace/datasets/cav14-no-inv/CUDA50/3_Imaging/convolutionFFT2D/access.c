#include <cstdio>
#include <cstddef>
#include <cstdlib>

//#define POWER_OF_TWO 1

typedef unsigned int uint;
typedef struct {
  uint x;
  uint y;
  uint z;
} uint3;

int __ffs(int x) {
  if (x == 0) return 0;
  return __builtin_ffs(x) - 1;
}

void udivmod(uint &dividend, uint divisor, uint &rem) {
#if(!POWER_OF_TWO)
  rem = dividend % divisor;
  dividend /= divisor;
#else
  rem = dividend & (divisor - 1);
  dividend >>= (__ffs(divisor) - 1);
#endif
}

int main(int argc, char **argv) {
  uint3 threadIdx, blockIdx;
  if (argc != 3) {
    printf("%s <threadIdx.x> <blockIdx.x>\n", argv[0]);
    return 1;
  }
  threadIdx.x = atoi(argv[1]); threadIdx.y = 0; threadIdx.z = 0;
  blockIdx.x = atoi(argv[2]); blockIdx.y = 0; blockIdx.z = 0;
  printf("threadIdx.x = %d\n", threadIdx.x);
  printf("blockIdx.x = %d\n", blockIdx.x);

  uint3 blockDim;
  blockDim.x = 256; blockDim.y = 1; blockDim.z = 1;
  uint3 gridDim;
  gridDim.x = 4096; gridDim.y = 1; gridDim.z = 1;

  uint DX = 1024;
  uint DY = 2048;
  uint threadCount = 1048576;

  const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
  printf("threadId = %d\n", threadId);
  if (threadId >= threadCount) {
    printf("threadId >= threadCount\n");
    return 0;
  }

  uint x, y, i = threadId;
  udivmod(i, DX, x);
  udivmod(i, DY / 2, y);

  printf("x = %d\n", x);
  printf("y = %d\n", y);
  printf("i = %d\n", i);

  uint offset = i * DY * DX;
  printf("offset = %d\n", offset);
  if ((y == 0) && (x > DX / 2)) {
    printf("y == 0 && x > DX/2");
    return 1;
  }
  uint pos1 = offset +          y * DX +          x;
  printf("pos1 = %d\n", pos1);

  return 0;
}
