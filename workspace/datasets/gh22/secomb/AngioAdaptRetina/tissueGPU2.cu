/***********************************************************
tissueGPU2.cu
GPU kernel to accumulate contributions of tissue source
strengths qt to vessel solute levels pv.
Each vessel point is assigned a number (stepGPU) of threads
TWS January 2012
Cuda 10.1 Version, August 2019
************************************************************/
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
/*
__device__ void eval(float dist2, float r2d2, float req2, float req, int is2d, float * gvt){
	if (dist2 < req2) {
		if (is2d) *gvt += log(r2d2 / req2) + 1. - dist2 / req2;
		else *gvt += (1.5 - 0.5 * dist2 / req2) / req;
	}
	else {
		if (is2d) *gvt += log(r2d2 / dist2);
		else *gvt += 1. / sqrt(dist2);
	}
	return;
}
*/

__global__ void tissueGPU2Kernel(float* d_tissxyz, float* d_vessxyz, float* d_pv000, float* d_qt000, float* d_hex_norm, float aphexp,
	int nnt, int nnv, int is2d, float req, float r2d, int stepGPU, int hexperiodic)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int ivp = i / stepGPU;
	int ivp1 = i % stepGPU;
	int jtp, nnt2 = 2 * nnt;
	int istep, idomain, ndomain;
	float p = 0., xv, yv, zv, x, y, z, dist2, gvt, req2 = req * req, r2d2 = r2d * r2d, x1, y1;

	if (ivp < nnv) {
		xv = d_vessxyz[ivp];
		yv = d_vessxyz[ivp + nnv];
		zv = d_vessxyz[ivp + nnv * 2];
		for (jtp = ivp1; jtp < nnt; jtp += stepGPU) {
			x = d_tissxyz[jtp] - xv;
			y = d_tissxyz[jtp + nnt] - yv;
			z = d_tissxyz[jtp + nnt2] - zv;
			if (hexperiodic) ndomain = 7;
			else ndomain = 1;
			gvt = 0.;
			for (idomain = 0; idomain < ndomain; idomain++) {
				x1 = x;
				y1 = y;
				if (idomain > 0) {
					x1 += 2. * aphexp * d_hex_norm[idomain - 1];
					y1 += 2. * aphexp * d_hex_norm[idomain + 5];
				}
				dist2 = x1 * x1 + y1 * y1 + z * z;
				//eval(dist2, r2d2, req2, req, is2d, &gvt);	//for testing __device__ usage
				if (dist2 < req2) {
					if (is2d) gvt += log(r2d2 / req2) + 1. - dist2 / req2;
					else gvt += (1.5 - 0.5 * dist2 / req2) / req;
				}
				else {
					if (is2d) gvt += log(r2d2 / dist2);
					else gvt += 1. / sqrt(dist2);
				}
			}
			p += d_qt000[jtp] * gvt;
		}
		if (ivp1 == 0) d_pv000[ivp] = p;
		__syncthreads();
		//The following is apparently needed to assure that d_pt000 is incremented in sequence from the needed threads
		for (istep = 1; istep < stepGPU; istep++) if (ivp1 == istep) d_pv000[ivp] += p;
	}
}

extern "C" void tissueGPU2(float* d_tissxyz, float* d_vessxyz, float* d_pv000, float* d_qt000, float* d_hex_norm, float aphexp,
	int nnt, int nnv, int is2d, float req, float r2d, int stepGPU, int hexperiodic)
{
	int threadsPerBlock = 256;
	//int stepGPU = 4;
	int blocksPerGrid = (stepGPU * nnv + threadsPerBlock - 1) / threadsPerBlock;
	tissueGPU2Kernel <<<blocksPerGrid, threadsPerBlock >>> (d_tissxyz, d_vessxyz, d_pv000, d_qt000, d_hex_norm, aphexp,
		nnt, nnv, is2d, req, r2d, stepGPU, hexperiodic);
}
