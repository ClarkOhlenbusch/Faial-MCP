/***********************************************************
tissueGPU3.cu
GPU kernel to accumulate contributions of vessel source
strengths qv to tissue solute levels pt.
Each vessel point is assigned a number (stepGPU) of threads
TWS January 2012
Cuda 10.1 Version, August 2019
************************************************************/
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void tissueGPU3Kernel(float *d_tissxyz, float *d_vessxyz, float *d_pt000, float *d_qv000, float *d_hex_norm, float aphexp,
	int nnt, int nnv, int is2d, float req, float r2d, int stepGPU, int hexperiodic)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int itp = i/stepGPU;
    int itp1 = i%stepGPU;
	int jvp,nnv2=2*nnv;
	int istep,idomain,ndomain;
	float p = 0., xt,yt,zt,x,y,z,dist2,gtv,req2=req*req,r2d2=r2d*r2d,x1,y1;

    if(itp < nnt){
		xt = d_tissxyz[itp];
		yt = d_tissxyz[itp+nnt];
		zt = d_tissxyz[itp+nnt*2];
		for(jvp=itp1; jvp<nnv; jvp+=stepGPU){
			x = d_vessxyz[jvp] - xt;
			y = d_vessxyz[jvp+nnv] - yt;
			z = d_vessxyz[jvp+nnv2] - zt;
			if(hexperiodic) ndomain = 7;
			else ndomain = 1;
			gtv = 0.;
			for(idomain=0; idomain<ndomain; idomain++){
				x1 = x;
				y1 = y;
				if(idomain > 0){
					x1 += 2.*aphexp*d_hex_norm[idomain-1];
					y1 += 2.*aphexp*d_hex_norm[idomain+5];
				}
				dist2 = x1*x1 + y1*y1 + z*z;
				if(dist2 < req2){
					if(is2d) gtv += log(r2d2/req2) + 1. - dist2/req2;
					else gtv += (1.5 - 0.5*dist2/req2)/req;
				}
				else{
					if(is2d) gtv += log(r2d2/dist2);
					else gtv += 1./sqrt(dist2);
				}
			}
			p += d_qv000[jvp]*gtv;
		}
		if (itp1 == 0) d_pt000[itp] = p;
		__syncthreads();
//The following is apparently needed to assure that d_pt000 is incremented in sequence from the needed threads
		for(istep=1; istep<stepGPU; istep++) if(itp1 == istep) d_pt000[itp] += p;
	}
}

extern "C" void tissueGPU3(float *d_tissxyz, float *d_vessxyz, float *d_pt000, float *d_qv000, float *d_hex_norm, float aphexp,
		int nnt, int nnv, int is2d, float req, float r2d, int stepGPU, int hexperiodic)
{
	int threadsPerBlock = 256;
	//int stepGPU = 4;
	int blocksPerGrid = (stepGPU*nnt + threadsPerBlock - 1) / threadsPerBlock;
	tissueGPU3Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tissxyz, d_vessxyz, d_pt000, d_qv000, d_hex_norm, aphexp,
		nnt, nnv, is2d, req, r2d, stepGPU, hexperiodic);
}
