#include <stdio.h>
#include <stdlib.h>
#include <util/check_cuda_errors.h>

void check_cuda(cudaError_t result, char const* const func, char const* const file, int const line)
{
    if (result) {
        fprintf(stderr, "CUDA error %d at %s : %d in %s\n", (unsigned)result, file, line, func);
        fprintf(stderr, "Message: '%s'\n", cudaGetErrorString(result));
        cudaDeviceReset();
        exit(99);
    }
}
