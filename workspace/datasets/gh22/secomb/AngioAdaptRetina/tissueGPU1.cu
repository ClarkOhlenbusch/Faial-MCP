/***********************************************************
tissueGPU1.cu
GPU kernel to accumulate contributions of tissue source
strengths qt to tissue solute levels pt.
Each vessel point is assigned a number (stepGPU) of threads
TWS December 2011
Modified to do blocks of tissue points separately for improved convergence
TWS May 2016
Cuda 10.1 Version, August 2019
************************************************************/
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void tissueGPU1Kernel(int* d_tisspoints, float* d_dtt000, float* d_pt000, float* d_qt000,
	int nnt, int stepGPU, int blockstep, int iblockstep)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int itp = i / stepGPU * blockstep + iblockstep - 1;
	int itp1 = i % stepGPU;
	int jtp, ixyz, ix, iy, iz, jx, jy, jz, nnt2 = 2 * nnt, istep;
	float p = 0.;
	if (itp < nnt) {
		ix = d_tisspoints[itp];
		iy = d_tisspoints[itp + nnt];
		iz = d_tisspoints[itp + nnt2];
		for (jtp = itp1; jtp < nnt; jtp += stepGPU) {
			jx = d_tisspoints[jtp];
			jy = d_tisspoints[jtp + nnt];
			jz = d_tisspoints[jtp + nnt2];
			ixyz = abs(jx - ix) + abs(jy - iy) + abs(jz - iz);
			p += d_qt000[jtp] * d_dtt000[ixyz];
		}
		if (itp1 == 0) d_pt000[itp] = p;
		__syncthreads();
		//The following is apparently needed to assure that d_pt000 is incremented in sequence from the needed threads
		for (istep = 1; istep < stepGPU; istep++) if (itp1 == istep) d_pt000[itp] += p;
	}
}

extern "C" void tissueGPU1(int* d_tisspoints, float* d_dtt000, float* d_pt000, float* d_qt000, int nnt, int stepGPU, int blockstep, int iblockstep)
{
	int threadsPerBlock = 256;
	int nnt1 = (nnt + blockstep - 1) / blockstep;
	//int stepGPU = 4;//has to be a power of two apparently
	int blocksPerGrid = (stepGPU * nnt1 + threadsPerBlock - 1) / threadsPerBlock;
	tissueGPU1Kernel <<<blocksPerGrid, threadsPerBlock >>> (d_tisspoints, d_dtt000, d_pt000, d_qt000, nnt, stepGPU, blockstep, iblockstep);
}
