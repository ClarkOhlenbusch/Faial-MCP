//pass
//--gridDim=128              --blockDim=[128,3,1]

#define MUL(a, b) __umul24(a, b)
#define QRNG_DIMENSIONS 3
#define QRNG_RESOLUTION 31
#define INT_SCALE (1.0f / (float)0x80000001U)

static __constant__ unsigned int c_Table[QRNG_DIMENSIONS][QRNG_RESOLUTION];

__global__ void quasirandomGeneratorKernel(
    float *d_Output,
    unsigned int seed,
    unsigned int N
)
{
    __requires(N == 1048576);
    unsigned int *dimBase = &c_Table[threadIdx.y][0];
    unsigned int      tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int  threadN = MUL(blockDim.x, gridDim.x);

    for (unsigned int pos = tid;
         __invariant(tid <= pos),
         __invariant(       pos < N + threadN),
         __invariant(pos % threadN == tid),
         __global_invariant(__write_implies(d_Output, (threadIdx.y * N) <= (__write_offset(d_Output)/sizeof(float)))),
         __global_invariant(__write_implies(d_Output, (__write_offset(d_Output)/sizeof(float) - (threadIdx.y * N)) % threadN == tid)),
         __global_invariant(__write_implies(d_Output, (__write_offset(d_Output)/sizeof(float)/N) == threadIdx.y)),
         pos < N; pos += threadN)
    {
        unsigned int result = 0;
        unsigned int data = seed + pos;

        for (int bit = 0; bit < QRNG_RESOLUTION; bit++, data >>= 1)
            if (data & 1)
            {
                result ^= dimBase[bit];
            }

        d_Output[MUL(threadIdx.y, N) + pos] = (float)(result + 1) * INT_SCALE;
    }
}
