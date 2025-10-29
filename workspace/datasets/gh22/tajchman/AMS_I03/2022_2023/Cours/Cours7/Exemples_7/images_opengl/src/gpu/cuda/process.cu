#include <stdio.h>
#include "operation.h"

#include "io_png.hxx"
#include <iostream>
#include <cmath>
#include "timer.hxx"
#include "cuda_check.cuh"
#include "cImageGPU.h"
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

int  blockSize = 512;


__global__ void setGreyGPU(double *g_odata,
			   double *g_ired,
			   double *g_igreen,
			   double *g_iblue,
			   size_t n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (id < n) {
    g_odata[id] = 0.21*g_ired[id] + 0.72*g_igreen[id] + 0.07*g_iblue[id];
  }
}


void setGrey(cImageGPU &imageOut, const cImageGPU &imageIn)
{
  Timer T;
  T.start();

  int n = imageIn.width * imageOut.height;
  int gridSize = (int)ceil((double)n/blockSize);
  setGreyGPU<<<gridSize, blockSize>>>(imageOut.d_coef[0],
				      imageIn.d_coef[0],
				      imageIn.d_coef[1],
				      imageIn.d_coef[2],
				      imageIn.imageSize);
  CUDA_CHECK_KERNEL();

  T.stop();
  std::cerr << "\t\tTime Gray image generation "
	    << T.elapsed() << " s" << std::endl;

}

#define nGauss 3
#define nGauss2 (2*nGauss+1)

void filterCreation(double GKernel[][nGauss2]) 
{ 
    double sigma = 1.0; 
    double g, r, s = 2.0 * sigma * sigma; 
  
    double sum = 0.0; 
  
    // generating nGauss2xnGauss2 kernel 
    for (int x = -nGauss; x <= nGauss; x++) { 
        for (int y = -nGauss; y <= nGauss; y++) { 
            r = sqrt(x * x + y * y);
            g = (exp(-(r * r) / s)) / (M_PI * s);
            GKernel[x + nGauss][y + nGauss] = g; 
            sum += g; 
        } 
    } 
  
    for (int i = 0; i < nGauss2; ++i) 
        for (int j = 0; j < nGauss2; ++j) 
            GKernel[i][j] /= sum; 
}

__constant__ double d_GKernel[nGauss2][nGauss2];

__global__ void printKernel()
{
    int i,j;
    for (j=0; j<nGauss2; j++) {
      for (i=0; i<nGauss2; i++)
	      printf("%10.4e ", d_GKernel[i][j]);
      printf("\n");
    } 
}

__global__ void copyImageGPU(double *g_odata,
			     double *g_idata,
			     int width, int height)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
 
  if ((idx < width) && (idy < height)) {
    g_odata[idx + idy*width] = g_idata[idx + idy*width];
  }
}

void copyImage(cImageGPU &imageOut, const cImageGPU &imageIn)
{
  Timer T;
  T.start();

  dim3 blockSize (8, 8);
  dim3 gridSize  ((imageIn.width + blockSize.x)/blockSize.x,
		 (imageIn.height + blockSize.y)/blockSize.y);
  
  
  copyImageGPU<<<gridSize, blockSize>>>(imageOut.d_coef[0],
					imageIn.d_coef[0],
					imageIn.width, imageIn.height);
  CUDA_CHECK_KERNEL();

  T.stop();
  std::cerr << "\t\tTime image copy            "
	    << T.elapsed() << " s" << std::endl;  
 }

__global__ void smoothGPU(double *g_odata,
			  double *g_idata,
			  int width, int height)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
 
  if ((idx < width) && (idy < height)) {

    if ((idx >= nGauss) && (idx < width-nGauss) &&
	      (idy >= nGauss) && (idy < height-nGauss)) {
      double sum = 0;
      for (int x = -nGauss; x <= nGauss; x++) { 
	      for (int y = -nGauss; y <= nGauss; y++) { 
	        sum += d_GKernel[x+nGauss][y+nGauss]*g_idata[(idx+x)+(idy+y)*width];
      	}
      }
      g_odata[idx + idy*width] = sum;
    }
    else
      g_odata[idx + idy*width] = g_idata[idx + idy*width];
  }
}

void smooth(cImageGPU &imageOut, const cImageGPU &imageIn)
{
  Timer T;
  T.start();

  double h_GKernel[nGauss2][nGauss2];
  filterCreation(h_GKernel);
  cudaMemcpyToSymbol(d_GKernel, h_GKernel, nGauss2*nGauss2*sizeof(double));
  //  printKernel<<<1,1>>>();

  dim3 blockSize (8, 8);
  dim3 gridSize  (
		  (imageIn.width + blockSize.x)/blockSize.x,
		  (imageIn.height + blockSize.y)/blockSize.y);
  
  smoothGPU<<<gridSize, blockSize>>>(imageOut.d_coef[0],
					imageIn.d_coef[0],
					imageIn.width, imageIn.height);
  CUDA_CHECK_KERNEL();

  T.stop();
  std::cerr << "\t\tTime Gauss smoothing       "
	    << T.elapsed() << " s" << std::endl;  
 }

__constant__ int d_dx[3][3] = {{ 1, 0,-1},{ 2, 0,-2},{ 1, 0,-1}};
__constant__ int d_dy[3][3] = {{ 1, 2, 1},{ 0, 0, 0},{-1,-2,-1}};

__global__ void sobelGPU(double * gOut, double * gIn,
			 int width, int height)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
 
  if ((idx < width) && (idy < height)) {

    if ((idx > 0) && (idx < width-1) &&
	      (idy > 0) && (idy < height-1)) {
      int x,y;
      double s;
      double sum, sumx = 0, sumy = 0;
//      printf("%d %d : %f\n", idx, idy, gIn[idx + idy*width] );

      for(x=-1; x<=1; x++)
	      for(y=-1; y<=1; y++) {
	        s = gIn[(idx+x)+(idy+y)*width];
	        sumx+=s*d_dx[x+1][y+1];
	        sumy+=s*d_dy[x+1][y+1];
        }
        
      sum=fabs(sumx)+fabs(sumy);
      gOut[idx + idy*width] = (sum>255.0) ? 255.0 : sum;
//      printf("%d %d : %f\n", idx, idy, sum);
    }
    else
      gOut[idx + idy*width] = gIn[idx + idy*width];
  }
}

void sobel(cImageGPU &imageOut, const cImageGPU &imageIn)
{
  Timer T;
  T.start();

  dim3 blockSize (8, 8);
  dim3 gridSize  ((imageIn.width + blockSize.x)/blockSize.x,
		          (imageIn.height + blockSize.y)/blockSize.y);
  
  sobelGPU<<<gridSize, blockSize>>>(imageOut.d_coef[0],
				    imageIn.d_coef[0],
				    imageIn.width, imageIn.height);
  CUDA_CHECK_KERNEL();
 
  T.stop();
  std::cerr << "\t\tTime Sobel filter          "
	    << T.elapsed() << " s" << std::endl;  
}

void process(cImage &imageOut, const cImage &imageIn)
{
  Timer T;
  T.start();
  
  int w = imageIn.width,
    h = imageIn.height;
  
  cImageGPU
    imageTemp0(imageIn),
    imageTemp1(w, h, 1),
    imageTemp2(w, h, 1),
    imageTemp3(w, h, 1);
  
  T.stop();
  std::cerr << "\n\tTime send to GPU           "
	    << T.elapsed() << " s" << std::endl;

  Timer T_compute;
  T_compute.start();
  
  if (imageTemp0.ncolors == 3)
    setGrey(imageTemp1, imageTemp0);
  else
    copyImage(imageTemp1, imageTemp0);
  

  smooth (imageTemp2, imageTemp1);
  sobel  (imageTemp3, imageTemp2);

  T_compute.stop();
  std::cerr << "\n\tTime compute on GPU        "
	    << T_compute.elapsed() << " s" << std::endl;


  T.restart();
  imageOut = cImage(imageTemp3);
  
  T.stop();
  std::cerr << "\tTime get from GPU          "
	    << T.elapsed() << " s" << std::endl;
}
