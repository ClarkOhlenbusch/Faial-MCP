//pass
//--gridDim=[5,8,1] --blockDim=[64,4,1]

#include "common.h"
  
__global__ void
diffuseProject_k(cData *vx, cData *vy, int dx, int dy, float dt,
                 float visc, int lb)
{
    __requires(dx == 512);
    __requires(dy == 512);
    __requires(lb == 16);

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    cData xterm, yterm;

    // gtidx is the domain location in x for this thread
    if (gtidx < dx)
    {
        for (p = 0;
            #define __vx_r_offset __read_offset(vx)/sizeof(cData)
            #define __vx_w_offset __write_offset(vx)/sizeof(cData)
            #define __vy_r_offset __read_offset(vy)/sizeof(cData)
            #define __vy_w_offset __write_offset(vy)/sizeof(cData)
             __global_invariant(__write_implies(vx, threadIdx.y == (__vx_w_offset/dx/lb)            - (threadIdx.x/dx/lb)                                                   - (blockIdx.x*blockDim.x/dx/lb)            - (blockIdx.y      *blockDim.y      ))),
             __global_invariant(__write_implies(vx, blockIdx.y  == (__vx_w_offset/dx/lb/blockDim.y) - (threadIdx.x/dx/lb/blockDim.y) - (threadIdx.y            /blockDim.y) - (blockIdx.x*blockDim.x/dx/lb/blockDim.y))),
             __global_invariant(__read_implies( vx, threadIdx.y == (__vx_r_offset/dx/lb)            - (threadIdx.x/dx/lb)                                                   - (blockIdx.x*blockDim.x/dx/lb)            - (blockIdx.y      *blockDim.y      ))),
             __global_invariant(__read_implies( vx, blockIdx.y  == (__vx_r_offset/dx/lb/blockDim.y) - (threadIdx.x/dx/lb/blockDim.y) - (threadIdx.y            /blockDim.y) - (blockIdx.x*blockDim.x/dx/lb/blockDim.y))),
             __global_invariant(__write_implies(vy, threadIdx.y == (__vy_w_offset/dx/lb)            - (threadIdx.x/dx/lb)                                                   - (blockIdx.x*blockDim.x/dx/lb)            - (blockIdx.y      *blockDim.y      ))),
             __global_invariant(__write_implies(vy, blockIdx.y  == (__vy_w_offset/dx/lb/blockDim.y) - (threadIdx.x/dx/lb/blockDim.y) - (threadIdx.y            /blockDim.y) - (blockIdx.x*blockDim.x/dx/lb/blockDim.y))),
             __global_invariant(__read_implies( vy, threadIdx.y == (__vy_r_offset/dx/lb)            - (threadIdx.x/dx/lb)                                                   - (blockIdx.x*blockDim.x/dx/lb)            - (blockIdx.y      *blockDim.y      ))),
             __global_invariant(__read_implies( vy, blockIdx.y  == (__vy_r_offset/dx/lb/blockDim.y) - (threadIdx.x/dx/lb/blockDim.y) - (threadIdx.y            /blockDim.y) - (blockIdx.x*blockDim.x/dx/lb/blockDim.y))),
             p < lb; p++)
        {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;

            if (fi < dy)
            {
                int fj = fi * dx + gtidx;
                xterm = vx[fj];
                yterm = vy[fj];

                // Compute the index of the wavenumber based on the
                // data order produced by a standard NN FFT.
                int iix = gtidx;
                int iiy = (fi>dy/2)?(fi-(dy)):fi;

                // Velocity diffusion
                float kk = (float)(iix * iix + iiy * iiy); // k^2
                float diff = 1.f / (1.f + visc * dt * kk);
                xterm.x *= diff;
                xterm.y *= diff;
                yterm.x *= diff;
                yterm.y *= diff;

                // Velocity projection
                if (kk > 0.f)
                {
                    float rkk = 1.f / kk;
                    // Real portion of velocity projection
                    float rkp = (iix * xterm.x + iiy * yterm.x);
                    // Imaginary portion of velocity projection
                    float ikp = (iix * xterm.y + iiy * yterm.y);
                    xterm.x -= rkk * rkp * iix;
                    xterm.y -= rkk * ikp * iix;
                    yterm.x -= rkk * rkp * iiy;
                    yterm.y -= rkk * ikp * iiy;
                }

                vx[fj] = xterm;
                vy[fj] = yterm;
            }
        }
    }
}
