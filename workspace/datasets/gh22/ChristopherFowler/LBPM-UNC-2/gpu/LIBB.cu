/*
  Copyright 2013--2018 James E. McClure, Virginia Polytechnic & State University

  This file is part of the Open Porous Media project (OPM).
  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <math.h>
#include <stdio.h>
#include <cuda_profiler_api.h>

#define NBLOCKS 1024
#define NTHREADS 256


#define WBC 1

/*
     int S = Np/NBLOCKS/NTHREADS + 1;
       for (int s=0; s<S; s++) {
           //........Get 1-D index for this thread....................
           n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
           if (n<finish) {
*/


__global__ void dvc_InitExtrapolateScalarField(int *Map, char * id, double * phi, double *phi2, int start, int finish, int Ni, int strideY, int strideZ) {
    int ijk,nn;
    
    int n;
    
    double m1 = 0;
    
    
    
    double sum_weight = 0;
    
       int S = Ni/NBLOCKS/NTHREADS + 1;
       for (int s=0; s<S; s++) {
           //........Get 1-D index for this thread....................
           n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
           if (n<finish) {
       
               ijk = Map[n];
        
        m1 = 0.0;
        sum_weight = 0;
        
        nn = ijk-1;
        if (id[nn]==3) { // Neighbor is at least partially fluid
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;

        }
        nn = ijk+1;
        if (id[nn]==3) { // Neighbor is at least partially fluid
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;

        }
        nn = ijk-strideY;
        if (id[nn]==3) { // Neighbor is at least partially fluid
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;

        }
        nn = ijk+strideY;
        if (id[nn]==3) { // Neighbor is at least partially fluid
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;

        }
        nn = ijk-strideZ;
        if (id[nn]==3) { // Neighbor is at least partially fluid
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;

        }
        nn = ijk+strideZ;
        if (id[nn]==3) { // Neighbor is at least partially fluid
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;

        }
        
        nn = ijk-strideY-1;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;

        }
        nn = ijk+strideY+1;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;

        }
        nn = ijk+strideY-1;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;

        }
        nn = ijk-strideY+1;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;

        }
        nn = ijk-strideZ-1;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;

        }
        nn = ijk+strideZ+1;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;

        }
        nn = ijk+strideZ-1;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;

        }
        nn = ijk-strideZ+1;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;

        }
        nn = ijk-strideZ-strideY;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;

        }
        nn = ijk+strideZ+strideY;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;

        }
        nn = ijk+strideZ-strideY;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;

        }
        nn = ijk-strideZ+strideY;
        if (id[nn]==3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
        }


        if (sum_weight == 0.0) { sum_weight = 1.0; m1 = 0.0; }
        m1 /= sum_weight;

//        phi2[ijk] = 3;
        if (m1 > 0) phi2[ijk] = 1.0;
        if (m1 < 0) phi2[ijk] = -1.0;
        if (m1 == 0) phi2[ijk] = -1.0; // just choose one of the fluids for the edge case
       
       
       
        }
    }
}



__global__ void dvc_InitExtrapolatePhaseFieldInactive(int *Map, char * id, double *phi, double *phi2, int start, int finish, int strideY, int strideZ, int Np) {
    int ijk,nn;
    int n;
    double m1 = 0;
    double sum_weight = 0;
    
        int S = Np/NBLOCKS/NTHREADS + 1;
       for (int s=0; s<S; s++) {
           //........Get 1-D index for this thread....................
           n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
           if (n<finish) {
       
                ijk = Map[n];
        m1 = 0.0;
        sum_weight = 0;

        nn = ijk-1;
        if (id[nn]<=3) { // neighbor is fully fluid
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
        }
        //........................................................................
        nn = ijk+1;
        if (id[nn]<=3) {
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
            
        }
        //........................................................................
        nn = ijk-strideY;
        if (id[nn]<=3) {
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
            
        }
        //........................................................................
        nn = ijk+strideY;
        if (id[nn]<=3) {
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
            
        }
        //........................................................................
        nn = ijk-strideZ;
        if (id[nn]<=3) {
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
            
        }
        //........................................................................
        nn = ijk+strideZ;
        if (id[nn]<=3) {
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
            
        }
        // ........................................................................
        
        
        // IF Parallel plates, don't use the diagonal extrapolated values
        
        nn = ijk-strideY-1;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideY+1;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideY-1;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk-strideY+1;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk-strideZ-1;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideZ+1;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideZ-1;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk-strideZ+1;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk-strideZ-strideY;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideZ+strideY;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideZ-strideY;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk-strideZ+strideY;
        if (id[nn]<=3) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        
       // phi2[ijk] = 4;
        
        if (sum_weight == 0.0) { sum_weight = 1.0; m1 = 0.0; }
        m1 /= sum_weight;
        if (m1 > 0) phi2[ijk] = 1.0;
        if (m1 < 0) phi2[ijk] = -1.0;
        if (m1 == 0) phi2[ijk] = -1.0;
       
       
        
        }
    }
}

__global__ void dvc_ExtrapolateScalarField(int *Map, int * neighborList, double * phi, double *phi2, int start, int finish, int Nsb, int strideY, int strideZ) {
    int ijk,nn;
    
    int n;
    double m1 = 0;
    double sum_weight = 0;
    
        int S = Nsb/NBLOCKS/NTHREADS + 1;
       for (int s=0; s<S; s++) {
           //........Get 1-D index for this thread....................
           n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
           if (n<finish) {


        ijk = Map[n];
        m1 = 0;
        sum_weight = 0;

        nn = neighborList[n];
        m1 += 0.05555555555555556*phi[nn];
        sum_weight += 0.05555555555555556;


        nn = neighborList[n+Nsb];
        m1 += 0.05555555555555556*phi[nn];
        sum_weight += 0.05555555555555556;


        nn = neighborList[n+2*Nsb];
        m1 += 0.05555555555555556*phi[nn];
        sum_weight += 0.05555555555555556;


        nn = neighborList[n+3*Nsb];
        m1 += 0.05555555555555556*phi[nn];
        sum_weight += 0.05555555555555556;


        nn = neighborList[n+4*Nsb];
        m1 += 0.05555555555555556*phi[nn];
        sum_weight += 0.05555555555555556;


        nn = neighborList[n+5*Nsb];
        m1 += 0.05555555555555556*phi[nn];
        sum_weight += 0.05555555555555556;
       
        
        // IF Parallel plates
        nn = neighborList[n+6*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;

        nn = neighborList[n+7*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;
        
        nn = neighborList[n+8*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;
        
        nn = neighborList[n+9*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;

        nn = neighborList[n+10*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;

        nn = neighborList[n+11*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;

        nn = neighborList[n+12*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;

        nn = neighborList[n+13*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;

        nn = neighborList[n+14*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;

        nn = neighborList[n+15*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;

        nn = neighborList[n+16*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;

        nn = neighborList[n+17*Nsb];
        m1 += 0.02777777777777778*phi[nn];
        sum_weight += 0.02777777777777778;
    
        if (sum_weight == 0.0) { sum_weight = 1.0; m1 = 0.0; }
        phi2[ijk] = m1/sum_weight;
        
        
        }
    }
}

__global__ void dvc_ComputeGradPhi(double input_angle, int *Map, double * Phi,
                               double * GradPhiX, double * GradPhiY, double * GradPhiZ, double * CField,
                               double * GradSDsX, double * GradSDsY, double * GradSDsZ,
                               int strideY, int strideZ, int start, int finish, int Np, int WBC_Flag) {
    
    double npluscoefA,npluscoefB,nminuscoefA,nminuscoefB;
    double tempnx, tempny, tempnz;
    int ijk,nn,n;
    double nplus_x,nplus_y,nplus_z;
    double nminus_x,nminus_y,nminus_z;
    double denom;
    double SolC;
    double Euclidean_distance_plus;
    double Euclidean_distance_minus;
    double snx, sny, snz;
    double prescribed_angle_radians;
    double theta_prime;

    double temp_nx, temp_ny, temp_nz;

    double TODEGREES = 180.0/3.14159265359;
    double TORADIANS = 3.14159265359/180.0;
    double m1,m2,m4,m6,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m3,m5,m7;
    double nx,ny,nz,C;
    int count = 0;
    int flag = 0;

        int S = Np/NBLOCKS/NTHREADS + 1;
       for (int s=0; s<S; s++) {
           //........Get 1-D index for this thread....................
           n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
           if (n<finish) {
               flag = 0;
               ijk = Map[n];
        nn=ijk+1; m1=Phi[nn];
        nn=ijk-1; m2=Phi[nn];
        nn=ijk+strideY; m3=Phi[nn];
        nn=ijk-strideY; m4=Phi[nn];
        nn=ijk+strideZ; m5=Phi[nn];
        nn=ijk-strideZ; m6=Phi[nn];
        nn=ijk+1+strideY; m7=Phi[nn];
        nn=ijk-1-strideY; m8=Phi[nn];
        nn=ijk+1-strideY; m9=Phi[nn];
        nn=ijk-1+strideY; m10=Phi[nn];
        nn=ijk+1+strideZ; m11=Phi[nn];
        nn=ijk-1-strideZ; m12=Phi[nn];
        nn=ijk+1-strideZ; m13=Phi[nn];
        nn=ijk-1+strideZ; m14=Phi[nn];
        nn=ijk+strideY+strideZ; m15=Phi[nn];
        nn=ijk-strideY-strideZ; m16=Phi[nn];
        nn=ijk+strideY-strideZ; m17=Phi[nn];
        nn=ijk-strideY+strideZ; m18=Phi[nn];
    
        nx = 0.16666666666666666*m1 - 0.08333333333333333*m10 + 0.08333333333333333*m11 -
        0.08333333333333333*m12 + 0.08333333333333333*m13 - 0.08333333333333333*m14 -
        0.16666666666666666*m2 + 0.08333333333333333*m7 - 0.08333333333333333*m8 +
        0.08333333333333333*m9;
        ny = 0.08333333333333333*m10 + 0.08333333333333333*m15 - 0.08333333333333333*m16 +
        0.08333333333333333*m17 - 0.08333333333333333*m18 + 0.16666666666666666*m3 -
        0.16666666666666666*m4 + 0.08333333333333333*m7 - 0.08333333333333333*m8 -
        0.08333333333333333*m9;
        nz = 0.08333333333333333*m11 - 0.08333333333333333*m12 - 0.08333333333333333*m13 +
        0.08333333333333333*m14 + 0.08333333333333333*m15 - 0.08333333333333333*m16 -
        0.08333333333333333*m17 + 0.08333333333333333*m18 + 0.16666666666666666*m5 -
        0.16666666666666666*m6;

       C = sqrt(nx*nx + ny*ny + nz*nz);
       if (C==0) {C=1; flag=1;}
       nx /= C;
       ny /= C;
       nz /= C;

       temp_nx = nx; temp_ny=ny; temp_nz=nz;
     
#ifdef WBC
        snx = GradSDsX[ijk];
        sny = GradSDsY[ijk];
        snz = GradSDsZ[ijk];

        SolC = sqrt(snx*snx + sny*sny + snz*snz);
        if (SolC==0.0) {SolC=1.0; flag=1;}
        snx /= SolC;
        sny /= SolC;
        snz /= SolC;

        prescribed_angle_radians = input_angle*TORADIANS;
        theta_prime = acos(snx*nx + sny*ny + snz*nz);
        denom = sin(theta_prime);
        npluscoefA = cos(prescribed_angle_radians) - sin(prescribed_angle_radians)*cos(theta_prime)/denom;
        npluscoefB = sin(prescribed_angle_radians)/denom;

        nplus_x = npluscoefA * snx + npluscoefB * nx;
        nplus_y = npluscoefA * sny + npluscoefB * ny;
        nplus_z = npluscoefA * snz + npluscoefB * nz;

        nminuscoefA = cos(-prescribed_angle_radians) - sin(-prescribed_angle_radians)*cos(theta_prime)/denom;
        nminuscoefB = sin(-prescribed_angle_radians)/denom;

        nminus_x = nminuscoefA * snx + nminuscoefB * nx;
        nminus_y = nminuscoefA * sny + nminuscoefB * ny;
        nminus_z = nminuscoefA * snz + nminuscoefB * nz;

        Euclidean_distance_plus  = sqrt( (nx - nplus_x) *(nx - nplus_x)
                                        + (ny - nplus_y) *(ny - nplus_y)
                                        + (nz - nplus_z) *(nz - nplus_z)  );

        Euclidean_distance_minus = sqrt( (nx - nminus_x)*(nx - nminus_x)
                                        + (ny - nminus_y)*(ny - nminus_y)
                                        + (nz - nminus_z)*(nz - nminus_z) );



        if (Euclidean_distance_minus > Euclidean_distance_plus) {
            tempnx = nplus_x;
            tempny = nplus_y;
            tempnz = nplus_z;
        }
        if (Euclidean_distance_minus < Euclidean_distance_plus) {
            tempnx = nminus_x;
            tempny = nminus_y;
            tempnz = nminus_z;
        }
        if (Euclidean_distance_plus == Euclidean_distance_minus) {
            tempnx = snx;
            tempny = sny;
            tempnz = snz;
        }

        nx = tempnx;
        ny = tempny;
        nz = tempnz;
#endif
        C = sqrt(nx*nx + ny*ny + nz*nz);
        if (C==0) {C=1; flag=1;}
        // if (abs(nx) < 0.001 && abs(ny) < 0.001 && abs(nz) < 0.001){
        //     C = 1;
        // } else{
        //     C = sqrt(nx*nx + ny*ny + nz*nz);
        // }
        nx /= C;
        ny /= C;
        nz /= C;

        theta_prime *= TODEGREES;
        double theta_prime_new = acos(snx*nx + sny*ny + snz*nz)*TODEGREES;
        double delta_theta = (theta_prime - theta_prime_new);
        
        GradPhiX[ijk] = nx;
        GradPhiY[ijk] = ny;
        GradPhiZ[ijk] = nz;
        CField[ijk] = C;

        //printf("diff: %f | ",delta_theta);
        // if (delta_theta > 10.0 || delta_theta < -10.0){
        //     nx = temp_nx;
        //     ny = temp_ny;
        //     nz = temp_nz;
        //     C = 1;
        //     // if (abs(nx) < 0.001 && abs(ny) < 0.001 && abs(nz) < 0.001){
        //     //     C = 1;
        //     // } else{
        //     //     C = sqrt(nx*nx + ny*ny + nz*nz);
        //     // }
        //     // nx /= C;
        //     // ny /= C;
        //     // nz /= C;
        //     theta_prime_new = acos(snx*nx + sny*ny + snz*nz)*TODEGREES;
        //     delta_theta = (theta_prime - theta_prime_new);
        //     if (delta_theta != 0.0 && count == 0.0){
        //         count = count + 1.0;
        //         printf("Old angle: %f New angle: %f diff: %f | \n",theta_prime,theta_prime_new,delta_theta);
        //         printf("orig nx ny nz: %f %f %f \n",temp_nx,temp_ny,temp_nz);
        //         printf("C: %f \n",C);
        //         printf("new nx ny nz: %f %f %f \n",nx, ny, nz);
        //     }
        //     //printf("#");
        // }
        
        // if (flag == 0){
        //     GradPhiX[ijk] = nx;
        //     GradPhiY[ijk] = ny;
        //     GradPhiZ[ijk] = nz;
        //     CField[ijk] = C;
        // } else {
        //     GradPhiX[ijk] = temp_nx;
        //     GradPhiY[ijk] = temp_ny;
        //     GradPhiZ[ijk] = temp_nz;
        //     CField[ijk] = 1;
        // }

        // if (snx == sny == snz == 0.0){
        //     GradPhiX[ijk] = temp_nx;
        //     GradPhiY[ijk] = temp_ny;
        //     GradPhiZ[ijk] = temp_nz;
        //     CField[ijk] = 1;
        // }

             
       
       
        }
     }
     
}

__global__ void dvc_InitExtrapolatePhaseFieldActive(int *Map, double * VFmask, double *phi, double *phi2, int start, int finish, int strideY, int strideZ, int Np) {
    int ijk,nn;
    int n;
    double m1 = 0;
    double sum_weight = 0;
    
         int S = Np/NBLOCKS/NTHREADS + 1;
       for (int s=0; s<S; s++) {
           //........Get 1-D index for this thread....................
           n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
           if (n<finish) {
        ijk = Map[n];
        m1 = 0.0;
        sum_weight = 0;

        nn = ijk-1;
        if (VFmask[nn] < 0.5) { // neighbor is fully fluid
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
        }
        //........................................................................
        nn = ijk+1;
        if (VFmask[nn] < 0.5) {
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
            
        }
        //........................................................................
        nn = ijk-strideY;
        if (VFmask[nn] < 0.5) {
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
            
        }
        //........................................................................
        nn = ijk+strideY;
        if (VFmask[nn] < 0.5) {
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
            
        }
        //........................................................................
        nn = ijk-strideZ;
        if (VFmask[nn] < 0.5) {
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
            
        }
        //........................................................................
        nn = ijk+strideZ;
        if (VFmask[nn] < 0.5) {
            m1 += 0.05555555555555556*phi[nn];
            sum_weight += 0.05555555555555556;
            
        }
        // ........................................................................
        
        
        // IF Parallel plates, don't use the diagonal extrapolated values
        
        nn = ijk-strideY-1;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideY+1;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideY-1;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk-strideY+1;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk-strideZ-1;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideZ+1;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideZ-1;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk-strideZ+1;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk-strideZ-strideY;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideZ+strideY;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk+strideZ-strideY;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        //........................................................................
        nn = ijk-strideZ+strideY;
        if (VFmask[nn] < 0.5) {
            m1 += 0.02777777777777778*phi[nn];
            sum_weight += 0.02777777777777778;
            
        }
        
        
        if (sum_weight == 0.0) { sum_weight = 1.0; m1 = 0.0; }
        m1 /= sum_weight;
        if (m1 > 0) phi2[ijk] = 1.0;
        if (m1 < 0) phi2[ijk] = -1.0;
        if (m1 == 0) phi2[ijk] = -1.0;
        }
    }
}



__global__ void dvc_Inactive_Color_LIBB(int * scalarList, int *Map, double *DenA, double *DenB, double * DenA2, double * DenB2, double *Phi, double *Velx2, double * Vely2, double * Velz2, double beta,  int strideY, int strideZ, int start, int finish, int Np, int N, double*  GradPhiX, double*GradPhiY, double* GradPhiZ, double*  CField) {
    
    int nn,ijk;
    
    double nAB, nA, nB, delta;
    double ux,uy,uz;
    
    double C,nx,ny,nz;
    int n;

           
    double s0,s1,s2,s4,s6,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s3,s5,s7;
    double t0,t1,t2,t4,t6,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t3,t5,t7;
    double b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18;
    int sl1,sl2,sl3,sl4,sl5,sl6,sl7,sl8,sl9,sl10,sl11,sl12,sl13,sl14,sl15,sl16,sl17,sl18;
    int S = Np/NBLOCKS/NTHREADS + 1;
    for (int s=0; s<S; s++) {
        //........Get 1-D index for this thread....................
        n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
        if (n<finish) {
        
                    ijk = Map[n];
        sl1 = scalarList[n];
        sl2 = scalarList[n+Np];
        sl3 = scalarList[n+2*Np];
        sl4 = scalarList[n+3*Np];
        sl5 = scalarList[n+4*Np];
        sl6 = scalarList[n+5*Np];
        sl7 = scalarList[n+6*Np];
        sl8 = scalarList[n+7*Np];
        sl9  = scalarList[n+8*Np];
        sl10 = scalarList[n+9*Np];
        sl11 = scalarList[n+10*Np];
        sl12 = scalarList[n+11*Np];
        sl13 = scalarList[n+12*Np];
        sl14 = scalarList[n+13*Np];
        sl15 = scalarList[n+14*Np];
        sl16 = scalarList[n+15*Np];
        sl17 = scalarList[n+16*Np];
        sl18 = scalarList[n+17*Np];
         
        
         
        (sl1 == ijk)  ? b1  = -1 : b1 = 1;
        (sl2 == ijk)  ? b2  = -1 : b2 = 1;
        (sl3 == ijk)  ? b3  = -1 : b3 = 1;
        (sl4 == ijk)  ? b4  = -1 : b4 = 1;
        (sl5 == ijk)  ? b5  = -1 : b5 = 1;
        (sl6 == ijk)  ? b6  = -1 : b6 = 1;
        (sl7 == ijk)  ? b7  = -1 : b7 = 1;
        (sl8 == ijk)  ? b8  = -1 : b8 = 1;
        (sl9 == ijk)  ? b9  = -1 : b9 = 1;
        (sl10 == ijk) ? b10 = -1 : b10 = 1;
        (sl11 == ijk) ? b11 = -1 : b11 = 1;
        (sl12 == ijk) ? b12 = -1 : b12 = 1;
        (sl13 == ijk) ? b13 = -1 : b13 = 1;
        (sl14 == ijk) ? b14 = -1 : b14 = 1;
        (sl15 == ijk) ? b15 = -1 : b15 = 1;
        (sl16 == ijk) ? b16 = -1 : b16 = 1;
        (sl17 == ijk) ? b17 = -1 : b17 = 1;
        (sl18 == ijk) ? b18 = -1 : b18 = 1;
      
         
         
         t1 = DenA[sl1];  // i+1
         s1 = DenB[sl1];  // i+1
         
         t2 = DenA[sl2];  // i-1
         s2 = DenB[sl2];  // i-1

         t3 = DenA[sl3];  // j+1
         s3 = DenB[sl3];  // j+1
         
         t4 = DenA[sl4];  // j-1
         s4 = DenB[sl4];  // j-1

         t5 = DenA[sl5];  // k+1
         s5 = DenB[sl5];  // k+1
         
         t6 = DenA[sl6];  // k-1
         s6 = DenB[sl6];  // k-1

         t7 = DenA[sl7];  // i+1, j+1
         s7 = DenB[sl7];  // i+1, j+1

         t8 = DenA[sl8];  // i-1, j-1
         s8 = DenB[sl8];  // i-1, j-1

         t9 = DenA[sl9];  // k-1
         s9 = DenB[sl9];  // k-1

         t10 = DenA[sl10];  // k-1
         s10 = DenB[sl10];  // k-1

         t11 = DenA[sl11];  // k-1
         s11 = DenB[sl11];  // k-1

         t12 = DenA[sl12];  // k-1
         s12 = DenB[sl12];  // k-1

         t13 = DenA[sl13];  // k-1
         s13 = DenB[sl13];  // k-1

         t14 = DenA[sl14];  // k-1
         s14 = DenB[sl14];  // k-1

         t15 = DenA[sl15];  // k-1
         s15 = DenB[sl15];  // k-1

         t16 = DenA[sl16];  // k-1
         s16 = DenB[sl16];  // k-1

         t17 = DenA[sl17];  // k-1
         s17 = DenB[sl17];  // k-1

         t18 = DenA[sl18];  // k-1
         s18 = DenB[sl18];  // k-1
         
         double densityA, densityB;
         delta = 0;
         nA = DenA[ijk];
         nB = DenB[ijk];
        
        
        
        

         densityA = nA/3.;
         densityB = nB/3.;

        nA = t1; nB = s1; ux = Velx2[sl1]; nx = GradPhiX[sl1];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*nx)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b1*delta  + nA*(1  + b1*ux))/18.;
        densityB+= (-b1*delta + nB*(1  + b1*ux))/18.;
        
        nA = t2; nB = s2; ux = Velx2[sl2]; nx = GradPhiX[sl2];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*nx)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b2*delta + nA*(1  - b2*ux))/18.;
        densityB+= (b2*delta  + nB*(1  - b2*ux))/18.;

        nA = t3; nB = s3; uy = Vely2[sl3]; ny = GradPhiY[sl3];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*ny)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b3*delta  + nA*(1  + b3*uy))/18.;
        densityB+= (-b3*delta + nB*(1  + b3*uy))/18.;
        
        nA = t4; nB = s4; uy = Vely2[sl4]; ny = GradPhiY[sl4];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*ny)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b4*delta + nA*(1  - b4*uy))/18.;
        densityB+= (b4*delta  + nB*(1  - b4*uy))/18.;

        nA = t5; nB = s5; uz = Velz2[sl5]; nz = GradPhiZ[sl5];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*nz)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b5*delta  + nA*(1  + b5*uz))/18.;
        densityB+= (-b5*delta + nB*(1  + b5*uz))/18.;
        
        nA = t6; nB = s6; uz = Velz2[sl6]; nz = GradPhiZ[sl6];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*nz)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b6*delta + nA*(1  - b6*uz))/18.;
        densityB+= (b6*delta  + nB*(1  - b6*uz))/18.;

        nA = t7; nB = s7;  ux = Velx2[sl7]; uy = Vely2[sl7];  nx = GradPhiX[sl7]; ny = GradPhiY[sl7];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx + ny))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b7*delta  + nA*(1  + b7*(ux + uy)))/36.;
        densityB+= (-b7*delta + nB*(1  + b7*(ux + uy)))/36.;

        nA = t8; nB = s8;  ux = Velx2[sl8]; uy = Vely2[sl8];  nx = GradPhiX[sl8]; ny = GradPhiY[sl8];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx + ny))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b8*delta + nA*(1  + b8*(-ux - uy)))/36.;
        densityB+= (b8*delta  + nB*(1  + b8*(-ux - uy)))/36.;

        nA = t9; nB = s9;  ux = Velx2[sl9]; uy = Vely2[sl9];  nx = GradPhiX[sl9]; ny = GradPhiY[sl9];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx - ny))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b9*delta  + nA*(1  + b9*(ux - uy)))/36.;
        densityB+= (-b9*delta + nB*(1  + b9*(ux - uy)))/36.;

        nA = t10; nB = s10;  ux = Velx2[sl10]; uy = Vely2[sl10];  nx = GradPhiX[sl10]; ny = GradPhiY[sl10];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx - ny))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b10*delta + nA*(1  + b10*(-ux + uy)))/36.;
        densityB+= (b10*delta  + nB*(1  + b10*(-ux + uy)))/36.;

        nA = t11; nB = s11;  ux = Velx2[sl11]; uz = Velz2[sl11];  nx = GradPhiX[sl11]; nz = GradPhiZ[sl11];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx + nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b11*delta  + nA*(1  + b11*(ux + uz)))/36.;
        densityB+= (-b11*delta + nB*(1  + b11*(ux + uz)))/36.;

        nA = t12; nB = s12;  ux = Velx2[sl12]; uz = Velz2[sl12];  nx = GradPhiX[sl12]; nz = GradPhiZ[sl12];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx + nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b12*delta + nA*(1  + b12*(-ux - uz)))/36.;
        densityB+= (b12*delta  + nB*(1  + b12*(-ux - uz)))/36.;

        nA = t13; nB = s13;  ux = Velx2[sl13]; uz = Velz2[sl13];  nx = GradPhiX[sl13]; nz = GradPhiZ[sl13];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx - nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b13*delta  + nA*(1  + b13*(ux - uz)))/36.;
        densityB+= (-b13*delta + nB*(1  + b13*(ux - uz)))/36.;

        nA = t14; nB = s14;  ux = Velx2[sl14]; uz = Velz2[sl14];  nx = GradPhiX[sl14]; nz = GradPhiZ[sl14];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx - nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b14*delta + nA*(1  + b14*(-ux + uz)))/36.;
        densityB+= (b14*delta  + nB*(1  + b14*(-ux + uz)))/36.;

        nA = t15; nB = s15;  uy = Vely2[sl15]; uz = Velz2[sl15];  ny = GradPhiY[sl15]; nz = GradPhiZ[sl15];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(ny + nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b15*delta  + nA*(1  + b15*(uy + uz)))/36.;
        densityB+= (-b15*delta + nB*(1  + b15*(uy + uz)))/36.;

        nA = t16; nB = s16;  uy = Vely2[sl16]; uz = Velz2[sl16];  ny = GradPhiY[sl16]; nz = GradPhiZ[sl16];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(ny + nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b16*delta + nA*(1  + b16*(-uy - uz)))/36.;
        densityB+= (b16*delta  + nB*(1  + b16*(-uy - uz)))/36.;

        nA = t17; nB = s17;  uy = Vely2[sl17]; uz = Velz2[sl17];  ny = GradPhiY[sl17]; nz = GradPhiZ[sl17];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(ny - nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b17*delta  + nA*(1  + b17*(uy - uz)))/36.;
        densityB+= (-b17*delta + nB*(1  + b17*(uy - uz)))/36.;

        nA = t18; nB = s18;  uy = Vely2[sl18]; uz = Velz2[sl18];  ny = GradPhiY[sl18]; nz = GradPhiZ[sl18];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(ny - nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b18*delta + nA*(1  + b18*(-uy + uz)))/36.;
        densityB+= (b18*delta  + nB*(1  + b18*(-uy + uz)))/36.;
         
         DenA2[ijk] = densityA;
         DenB2[ijk] = densityB;
         
         Phi[ijk] = (densityA-densityB)/(densityA+densityB);
       
        if (densityA == 0 && densityB == 0) Phi[ijk] = 0;
        
  
        }
    }
}

__global__  void dvc_ScaLBL_D3Q7_PhaseField_LIBB(int* interpolationList, int *neighborList,  int *Map, double *Aq, double *Bq, double *savedAq, double *savedBq,  double *Den, double *Phi, int start, int finish, int Np, int N, double * LIBBqA, double * LIBBqBC, double * LIBBqD)
{}



__global__ void dvc_ScaLBL_D3Q19_Color_LIBB(int * scalarList, int * interpolationList, int *neighborList, int *Map, double *dist, double *dist2, double *savedfq, double *Aq, double *Bq, double *DenA, double *DenB, double * DenA2, double * DenB2, double *Phi, double *Velx, double * Vely, double * Velz, double *Velx2, double * Vely2, double * Velz2, double *Press, double rhoA, double rhoB, double tauA, double tauB, double alpha, double beta, double Fx, double Fy, double Fz, int strideY, int strideZ, int start, int finish, int Np, int N, double * LIBBqA, double * LIBBqBC, double * LIBBqD, double* GradPhiX, double*GradPhiY, double* GradPhiZ, double * CField) {
    
    int n,nn,ijk,nread;
    int nr1,nr2,nr3,nr4,nr5,nr6;
    int nr7,nr8,nr9,nr10;
    int nr11,nr12,nr13,nr14;
    int nr15,nr16,nr17,nr18;
    
    int dir_X,dir_x,dir_Y,dir_y,dir_Z,dir_z;
    
    double fq;
    // conserved momemnts
    double rho,jx,jy,jz;
    double jxp,jyp,jzp;
    // non-conserved moments
    double m0,m1,m2,m4,m6,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18;
    double s0,s1,s2,s4,s6,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s3,s5,s7;
    double t0,t1,t2,t4,t6,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t3,t5,t7;
    double m3,m5,m7;
    double nA,nB; // number density
    double a1,a2,nAB,delta;
    double C,nx,ny,nz; //color gradient magnitude and direction
    double ux,uy,uz;
    double phi,tau,rho0,rlx_setA,rlx_setB;
    
        double rho_diff;
            double densityA, densityB;
            
                double b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18;
                
        int sl1;
    int sl2;
    int sl3;
    int sl4;
    int sl5;
    int sl6;
    int sl7;
    int sl8;
    int sl9;
    int sl10;
    int sl11;
    int sl12;
    int sl13;
    int sl14;
    int sl15;
    int sl16;
    int sl17;
    int sl18;
    
    const double mrt_V1=0.05263157894736842;
    const double mrt_V2=0.012531328320802;
    const double mrt_V3=0.04761904761904762;
    const double mrt_V4=0.004594820384294068;
    const double mrt_V5=0.01587301587301587;
    const double mrt_V6=0.0555555555555555555555555;
    const double mrt_V7=0.02777777777777778;
    const double mrt_V8=0.08333333333333333333333;
    const double mrt_V9=0.003341687552213868;
    const double mrt_V10=0.003968253968253968;
    const double mrt_V11=0.01388888888888889;
    const double mrt_V12=0.04166666666666666;
    double f0;
    double f1,f2,f3,f4,f5,f6,f7,f8,f9;
    double f10,f11,f12,f13,f14,f15,f16,f17,f18;
    
    double g1,g2,g3,g4,g5,g6,g7,g8,g9;
    double g10,g11,g12,g13,g14,g15,g16,g17,g18;
    
    double rho_early;
    
    double nplus_x,nplus_y,nplus_z;
    double nminus_x,nminus_y,nminus_z;
    
    double TORADIANS;
    
    double snx, sny, snz;
    
    double q;
    
    double A1,A2,A3,A4,A5,A6;
    double A7,A8,A9,A10,A11,A12;
    double A13,A14,A15,A16,A17,A18;
    
    double BC1,BC2,BC3,BC4,BC5,BC6;
    double BC7,BC8,BC9,BC10,BC11,BC12;
    double BC13,BC14,BC15,BC16,BC17,BC18;
    
    double D1,D2,D3,D4,D5,D6;
    double D7,D8,D9,D10,D11,D12;
    double D13,D14,D15,D16,D17,D18;
    
    double r1,r2,r3,r4,r5,r6,r7,r8,r9;
    
    int lr1,lr2,lr3,lr4,lr5,lr6;
    int lr7,lr8,lr9,lr10;
    int lr11,lr12,lr13,lr14;
    int lr15,lr16,lr17,lr18;
    
    int S = Np/NBLOCKS/NTHREADS + 1;
       for (int s=0; s<S; s++) {
           //........Get 1-D index for this thread....................
           n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
           if (n<finish) {
           
                
        ijk = Map[n];
        /* Accessed in same way as neighborList */
        lr1 = interpolationList[n];
        lr2 = interpolationList[n+Np];
        lr3 = interpolationList[n+2*Np];
        lr4 = interpolationList[n+3*Np];
        lr5 = interpolationList[n+4*Np];
        lr6 = interpolationList[n+5*Np];
        lr7 = interpolationList[n+6*Np];
        lr8 = interpolationList[n+7*Np];
        lr9 = interpolationList[n+8*Np];
        lr10 = interpolationList[n+9*Np];
        lr11 = interpolationList[n+10*Np];
        lr12 = interpolationList[n+11*Np];
        lr13 = interpolationList[n+12*Np];
        lr14 = interpolationList[n+13*Np];
        lr15 = interpolationList[n+14*Np];
        lr16 = interpolationList[n+15*Np];
        lr17 = interpolationList[n+16*Np];
        lr18 = interpolationList[n+17*Np];
        
        nr1 = neighborList[n];
        nr2 = neighborList[n+Np];
        nr3 = neighborList[n+2*Np];
        nr4 = neighborList[n+3*Np];
        nr5 = neighborList[n+4*Np];
        nr6 = neighborList[n+5*Np];
        nr7 = neighborList[n+6*Np];
        nr8 = neighborList[n+7*Np];
        nr9 = neighborList[n+8*Np];
        nr10 = neighborList[n+9*Np];
        nr11 = neighborList[n+10*Np];
        nr12 = neighborList[n+11*Np];
        nr13 = neighborList[n+12*Np];
        nr14 = neighborList[n+13*Np];
        nr15 = neighborList[n+14*Np];
        nr16 = neighborList[n+15*Np];
        nr17 = neighborList[n+16*Np];
        nr18 = neighborList[n+17*Np];
    
        
        A1 = LIBBqA[n + 0*Np];
        BC1 = LIBBqBC[n + 0*Np];
        D1 = LIBBqD[n + 0*Np];
        
        A2 = LIBBqA[n + 1*Np];
        BC2 = LIBBqBC[n + 1*Np];
        D2 = LIBBqD[n + 1*Np];
        
        A3 = LIBBqA[n + 2*Np];
        BC3 = LIBBqBC[n + 2*Np];
        D3 = LIBBqD[n + 2*Np];
        
        A4 = LIBBqA[n + 3*Np];
        BC4 = LIBBqBC[n + 3*Np];
        D4 = LIBBqD[n + 3*Np];
        
        A5 = LIBBqA[n + 4*Np];
        BC5 = LIBBqBC[n + 4*Np];
        D5 = LIBBqD[n + 4*Np];
        
        A6 = LIBBqA[n + 5*Np];
        BC6 = LIBBqBC[n + 5*Np];
        D6 = LIBBqD[n + 5*Np];
        
        A7 = LIBBqA[n + 6*Np];
        BC7 = LIBBqBC[n + 6*Np];
        D7 = LIBBqD[n + 6*Np];
        
        A8 = LIBBqA[n + 7*Np];
        BC8 = LIBBqBC[n + 7*Np];
        D8 = LIBBqD[n + 7*Np];
        
        A9 = LIBBqA[n + 8*Np];
        BC9 = LIBBqBC[n + 8*Np];
        D9 = LIBBqD[n + 8*Np];
        
        A10 = LIBBqA[n + 9*Np];
        BC10 = LIBBqBC[n + 9*Np];
        D10= LIBBqD[n + 9*Np];
        
        A11 = LIBBqA[n + 10*Np];
        BC11 = LIBBqBC[n + 10*Np];
        D11 = LIBBqD[n + 10*Np];
        
        A12 = LIBBqA[n + 11*Np];
        BC12 = LIBBqBC[n + 11*Np];
        D12 = LIBBqD[n + 11*Np];
        
        A13 = LIBBqA[n + 12*Np];
        BC13 = LIBBqBC[n + 12*Np];
        D13 = LIBBqD[n + 12*Np];
        
        A14 = LIBBqA[n + 13*Np];
        BC14 = LIBBqBC[n + 13*Np];
        D14 = LIBBqD[n + 13*Np];
        
        A15 = LIBBqA[n + 14*Np];
        BC15 = LIBBqBC[n + 14*Np];
        D15 = LIBBqD[n + 14*Np];
        
        A16 = LIBBqA[n + 15*Np];
        BC16 = LIBBqBC[n + 15*Np];
        D16 = LIBBqD[n + 15*Np];
        
        A17 = LIBBqA[n + 16*Np];
        BC17 = LIBBqBC[n + 16*Np];
        D17 = LIBBqD[n + 16*Np];
        
        A18 = LIBBqA[n + 17*Np];
        BC18 = LIBBqBC[n + 17*Np];
        D18 = LIBBqD[n + 17*Np];
        
        // q = 1
        fq = dist[n];
        f1 = dist[nr1];
        // Autogenerating LBM block...
        f2 = dist[nr2];
        f3 = dist[nr3];
        f4 = dist[nr4];
        f5 = dist[nr5];
        f6 = dist[nr6];
        f7 = dist[nr7];
        f8 = dist[nr8];
        f9 = dist[nr9];
        f10 = dist[nr10];
        f11 = dist[nr11];
        f12 = dist[nr12];
        f13 = dist[nr13];
        f14 = dist[nr14];
        f15 = dist[nr15];
        f16 = dist[nr16];
        f17 = dist[nr17];
        f18 = dist[nr18];    //done...
        rho_early = fq + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18;
        
        
        g1 = savedfq[lr1];
        // Autogenerating LBM block...
        g2 = savedfq[lr2];
        g3 = savedfq[lr3];
        g4 = savedfq[lr4];
        g5 = savedfq[lr5];
        g6 = savedfq[lr6];
        g7 = savedfq[lr7];
        g8 = savedfq[lr8];
        g9 = savedfq[lr9];
        g10 = savedfq[lr10];
        g11 = savedfq[lr11];
        g12 = savedfq[lr12];
        g13 = savedfq[lr13];
        g14 = savedfq[lr14];
        g15 = savedfq[lr15];
        g16 = savedfq[lr16];
        g17 = savedfq[lr17];
        g18 = savedfq[lr18];    //done...
        
        
        // local density
        rho0=rhoA + 0.5*(1.0-Phi[ijk])*(rhoB-rhoA);  // 1.0 for single phase
        // local relaxation time
        tau=tauA + 0.5*(1.0-Phi[ijk])*(tauB-tauA); // tau = tauA for single phase
        rlx_setA = 1.0/tau;
        rlx_setB = 8.*(2.-rlx_setA)/(8.-rlx_setA); // 8.f*(2.f-rlx_setA)/(8.f-rlx_setA);
#ifdef SP
        nx = ny = nz = C = 0;
#else
        nx = GradPhiX[ijk];
        ny = GradPhiY[ijk];
        nz = GradPhiZ[ijk];
        C = CField[ijk];
        
       // nx = ny = nz = C = 0;
#endif
        
        f1*= BC2; f1 += (D2*dist[nr2] + A2*g1);
        // Autogenerating LBM block...
        f2*= BC1; f2 += (D1*dist[nr1] + A1*g2);
        f3*= BC4; f3 += (D4*dist[nr4] + A4*g3);
        f4*= BC3; f4 += (D3*dist[nr3] + A3*g4);
        f5*= BC6; f5 += (D6*dist[nr6] + A6*g5);
        f6*= BC5; f6 += (D5*dist[nr5] + A5*g6);
        f7*= BC8; f7 += (D8*dist[nr8] + A8*g7);
        f8*= BC7; f8 += (D7*dist[nr7] + A7*g8);
        f9*= BC10; f9 += (D10*dist[nr10] + A10*g9);
        f10*= BC9; f10 += (D9*dist[nr9] + A9*g10);
        f11*= BC12; f11 += (D12*dist[nr12] + A12*g11);
        f12*= BC11; f12 += (D11*dist[nr11] + A11*g12);
        f13*= BC14; f13 += (D14*dist[nr14] + A14*g13);
        f14*= BC13; f14 += (D13*dist[nr13] + A13*g14);
        f15*= BC16; f15 += (D16*dist[nr16] + A16*g15);
        f16*= BC15; f16 += (D15*dist[nr15] + A15*g16);
        f17*= BC18; f17 += (D18*dist[nr18] + A18*g17);
        f18*= BC17; f18 += (D17*dist[nr17] + A17*g18);
        //done...
        
        // q=0
        
        rho = fq;
        m1  = -30.0*fq;
        m2  = 12.0*fq;
        
        // q = 1
        rho += f1;
        m1 -= 11.0*f1;
        m2 -= 4.0*f1;
        jx = f1;
        m4 = -4.0*f1;
        m9 = 2.0*f1;
        m10 = -4.0*f1;
        
        // q = 2
        rho += f2;
        m1 -= 11.0*(f2);
        m2 -= 4.0*(f2);
        jx -= f2;
        m4 += 4.0*(f2);
        m9 += 2.0*(f2);
        m10 -= 4.0*(f2);
        
        // q = 3
        rho += f3;
        m1 -= 11.0*f3;
        m2 -= 4.0*f3;
        jy = f3;
        m6 = -4.0*f3;
        m9 -= f3;
        m10 += 2.0*f3;
        m11 = f3;
        m12 = -2.0*f3;
        
        // q = 4
        rho+= f4;
        m1 -= 11.0*f4;
        m2 -= 4.0*f4;
        jy -= f4;
        m6 += 4.0*f4;
        m9 -= f4;
        m10 += 2.0*f4;
        m11 += f4;
        m12 -= 2.0*f4;
        
        // q=5
        rho += f5;
        m1 -= 11.0*f5;
        m2 -= 4.0*f5;
        jz = f5;
        m8 = -4.0*f5;
        m9 -= f5;
        m10 += 2.0*f5;
        m11 -= f5;
        m12 += 2.0*f5;
        
        // q = 6
        rho+= f6;
        m1 -= 11.0*f6;
        m2 -= 4.0*f6;
        jz -= f6;
        m8 += 4.0*f6;
        m9 -= f6;
        m10 += 2.0*f6;
        m11 -= f6;
        m12 += 2.0*f6;
        
        // q=7
        rho += f7;
        m1 += 8.0*f7;
        m2 += f7;
        jx += f7;
        m4 += f7;
        jy += f7;
        m6 += f7;
        m9  += f7;
        m10 += f7;
        m11 += f7;
        m12 += f7;
        m13 = f7;
        m16 = f7;
        m17 = -f7;
        
        // q = 8
        rho += f8;
        m1 += 8.0*f8;
        m2 += f8;
        jx -= f8;
        m4 -= f8;
        jy -= f8;
        m6 -= f8;
        m9 += f8;
        m10 += f8;
        m11 += f8;
        m12 += f8;
        m13 += f8;
        m16 -= f8;
        m17 += f8;
        
        // q=9
        rho += f9;
        m1 += 8.0*f9;
        m2 += f9;
        jx += f9;
        m4 += f9;
        jy -= f9;
        m6 -= f9;
        m9 += f9;
        m10 += f9;
        m11 += f9;
        m12 += f9;
        m13 -= f9;
        m16 += f9;
        m17 += f9;
        
        // q = 10
        rho += f10;
        m1 += 8.0*f10;
        m2 += f10;
        jx -= f10;
        m4 -= f10;
        jy += f10;
        m6 += f10;
        m9 += f10;
        m10 += f10;
        m11 += f10;
        m12 += f10;
        m13 -= f10;
        m16 -= f10;
        m17 -= f10;
        
        // q=11
        rho += f11;
        m1 += 8.0*f11;
        m2 += f11;
        jx += f11;
        m4 += f11;
        jz += f11;
        m8 += f11;
        m9 += f11;
        m10 += f11;
        m11 -= f11;
        m12 -= f11;
        m15 = f11;
        m16 -= f11;
        m18 = f11;
        
        // q=12
        rho += f12;
        m1 += 8.0*f12;
        m2 += f12;
        jx -= f12;
        m4 -= f12;
        jz -= f12;
        m8 -= f12;
        m9 += f12;
        m10 += f12;
        m11 -= f12;
        m12 -= f12;
        m15 += f12;
        m16 += f12;
        m18 -= f12;
        
        // q=13
        rho += f13;
        m1 += 8.0*f13;
        m2 += f13;
        jx += f13;
        m4 += f13;
        jz -= f13;
        m8 -= f13;
        m9 += f13;
        m10 += f13;
        m11 -= f13;
        m12 -= f13;
        m15 -= f13;
        m16 -= f13;
        m18 -= f13;
        
        // q=14
        rho += f14;
        m1 += 8.0*f14;
        m2 += f14;
        jx -= f14;
        m4 -= f14;
        jz += f14;
        m8 += f14;
        m9 += f14;
        m10 += f14;
        m11 -= f14;
        m12 -= f14;
        m15 -= f14;
        m16 += f14;
        m18 += f14;
        
        // q=15
        rho += f15;
        m1 += 8.0*f15;
        m2 += f15;
        jy += f15;
        m6 += f15;
        jz += f15;
        m8 += f15;
        m9 -= 2.0*f15;
        m10 -= 2.0*f15;
        m14 = f15;
        m17 += f15;
        m18 -= f15;
        
        // q=16
        rho += f16;
        m1 += 8.0*f16;
        m2 += f16;
        jy -= f16;
        m6 -= f16;
        jz -= f16;
        m8 -= f16;
        m9 -= 2.0*f16;
        m10 -= 2.0*f16;
        m14 += f16;
        m17 -= f16;
        m18 += f16;
        
        // q=17
        rho += f17;
        m1 += 8.0*f17;
        m2 += f17;
        jy += f17;
        m6 += f17;
        jz -= f17;
        m8 -= f17;
        m9 -= 2.0*f17;
        m10 -= 2.0*f17;
        m14 -= f17;
        m17 += f17;
        m18 += f17;
        
        // q=18
        rho += f18;
        m1 += 8.0*f18;
        m2 += f18;
        jy -= f18;
        m6 -= f18;
        jz += f18;
        m8 += f18;
        m9 -= 2.0*f18;
        m10 -= 2.0*f18;
        m14 -= f18;
        m17 -= f18;
        m18 -= f18;
        rho_diff = rho-rho_early;
//#ifdef SP
        Press[n] = 0.3333333333333333*rho;
//        Velx[ijk] = jx + 0.5*Fx;
//        Vely[ijk] = jy + 0.5*Fy;
//        Velz[ijk] = jz + 0.5*Fz;
//#else
//        Press[n] = 0.3333333333333333*rho;
        Velx[ijk] = jx/rho0;     // + 0.5*Fx;
        Vely[ijk] = jy/rho0;  // + 0.5*Fy;
        Velz[ijk] = jz/rho0;// + 0.5*Fz;
//
        jxp = jx;// + 0.5*Fx;
        jyp = jy;// + 0.5*Fy;
        jzp = jz;// + 0.5*Fz;
//#endif
    
        // ENTER MOMENTUM INTO THE MOMENTS, NOT VELOCITY
        //..........Toelke, Fruediger et. al. 2006................................
        
#ifdef SP
        /* Stokes flow - single phase */
        m1 = m1 + rlx_setA*(- 11*rho - m1);
        m2 = m2 + rlx_setA*(3*rho - m2);
        m4 = m4 + rlx_setB*((-0.6666666666666666*jx)- m4);
        m6 = m6 + rlx_setB*((-0.6666666666666666*jy)- m6);
        m8 = m8 + rlx_setB*((-0.6666666666666666*jz)- m8);
        // pxx
        m9 = m9 + rlx_setA*( - m9);
        // PIxx
        m10 = m10 + rlx_setA*( - m10);
        // pww
        m11 = m11 + rlx_setA*( - m11);
        // PIww
        m12 = m12 + rlx_setA*( - m12);
        m13 = m13 + rlx_setA*( - m13);
        m14 = m14 + rlx_setA*( - m14);
        m15 = m15 + rlx_setA*( - m15);
        m16 = m16 + rlx_setB*( - m16);
        m17 = m17 + rlx_setB*( - m17);
        m18 = m18 + rlx_setB*( - m18);
#else
        m1 = m1 + rlx_setA*((19*(jxp*jxp+jyp*jyp+jzp*jzp)/rho0 - 11*rho) -alpha*C - m1);
        m2 = m2 + rlx_setA*((3*rho - 5.5*(jxp*jxp+jyp*jyp+jzp*jzp)/rho0)- m2);
        m4 = m4 + rlx_setB*((-0.6666666666666666*jx)- m4);
        m6 = m6 + rlx_setB*((-0.6666666666666666*jy)- m6);
        m8 = m8 + rlx_setB*((-0.6666666666666666*jz)- m8);
        // pxx
        m9 = m9 + rlx_setA*(   ((2*jxp*jxp-jyp*jyp-jzp*jzp)/(3.0*rho0)) + 0.5*alpha*C*(2*nx*nx-ny*ny-nz*nz) - m9);
        // PIxx
        m10 = m10 + rlx_setA*( -0.5*((2*jxp*jxp-jyp*jyp-jzp*jzp)/(3.0*rho0))  - m10);
        // pww
        m11 = m11 + rlx_setA*(   ((jyp*jyp-jzp*jzp)/rho0) + 0.5*alpha*C*(ny*ny-nz*nz)- m11);
        // PIww
        m12 = m12 + rlx_setA*( -0.5*((jyp*jyp-jzp*jzp)/rho0)  - m12);
        m13 = m13 + rlx_setA*( (jxp*jyp/rho0) + 0.5*alpha*C*nx*ny - m13);
        m14 = m14 + rlx_setA*( (jyp*jzp/rho0) + 0.5*alpha*C*ny*nz - m14);
        m15 = m15 + rlx_setA*( (jxp*jzp/rho0) + 0.5*alpha*C*nx*nz - m15);
        m16 = m16 + rlx_setB*( - m16);
        m17 = m17 + rlx_setB*( - m17);
        m18 = m18 + rlx_setB*( - m18);
#endif
       
       
        //.................inverse transformation......................
        fq = 0. - 0.012531328320802004*m1 + 0.047619047619047616*m2 + 0.05263157894736842*rho;
        
//#ifdef SP
//        dist2[n] = fq;
//#else
        dist2[n] = fq-rho_diff;
//#endif
        
        fq = 0. + 0.1*jx - 0.004594820384294068*m1 - 0.05555555555555555*m10 - 0.015873015873015872*m2 -
        0.1*m4 + 0.05555555555555555*m9 + 0.05263157894736842*rho + 0.16666666666666667*Fx;
        dist2[n+Np] = fq;
        
        fq =  0. - 0.1*jx - 0.004594820384294068*m1 - 0.05555555555555555*m10 - 0.015873015873015872*m2 +
        0.1*m4 + 0.05555555555555555*m9 + 0.05263157894736842*rho - 0.16666666666666667*Fx;
        dist2[n+2*Np] = fq;
        
        fq = 0. + 0.1*jy - 0.004594820384294068*m1 + 0.027777777777777776*m10 + 0.0833333333333333333*m11 -
        0.0833333333333333333*m12 - 0.015873015873015872*m2 - 0.1*m6 - 0.027777777777777776*m9 +
        0.05263157894736842*rho + 0.16666666666666667*Fy;
        dist2[n+3*Np] = fq;
        
        fq =  0. - 0.1*jy - 0.004594820384294068*m1 + 0.027777777777777776*m10 +
        0.0833333333333333333*m11 - 0.0833333333333333333*m12 - 0.015873015873015872*m2 + 0.1*m6 -
        0.027777777777777776*m9 + 0.05263157894736842*rho - 0.16666666666666667*Fy;
        dist2[n+4*Np] = fq;
        
        
        fq =  0. + 0.1*jz - 0.004594820384294068*m1 + 0.027777777777777776*m10 - 0.0833333333333333333*m11 +
        0.0833333333333333333*m12 - 0.015873015873015872*m2 - 0.1*m8 - 0.027777777777777776*m9 +
        0.05263157894736842*rho + 0.16666666666666667*Fz;
        dist2[n+5*Np] = fq;
        
        
        fq = 0. - 0.1*jz - 0.004594820384294068*m1 + 0.027777777777777776*m10 -
        0.0833333333333333333*m11 + 0.0833333333333333333*m12 - 0.015873015873015872*m2 + 0.1*m8 -
        0.027777777777777776*m9 + 0.05263157894736842*rho - 0.16666666666666667*Fz;
        dist2[n+6*Np] = fq;
        
        fq =  0. + 0.1*jx + 0.1*jy + 0.003341687552213868*m1 + 0.013888888888888888*m10 +
        0.0833333333333333333*m11 + 0.041666666666666664*m12 + 0.25*m13 + 0.125*m16 - 0.125*m17 +
        0.003968253968253968*m2 + 0.025*m4 + 0.025*m6 + 0.027777777777777776*m9 +
        0.05263157894736842*rho + 0.08333333333333333*(Fx+Fy);
        dist2[n+7*Np] = fq;
        
        fq = 0. - 0.1*jx - 0.1*jy + 0.003341687552213868*m1 +
        0.013888888888888888*m10 + 0.0833333333333333333*m11 + 0.041666666666666664*m12 + 0.25*m13 -
        0.125*m16 + 0.125*m17 + 0.003968253968253968*m2 - 0.025*m4 - 0.025*m6 +
        0.027777777777777776*m9 + 0.05263157894736842*rho - 0.08333333333333333*(Fx+Fy);
        dist2[n+8*Np] = fq;
        
        fq = 0. + 0.1*jx - 0.1*jy + 0.003341687552213868*m1 + 0.013888888888888888*m10 +
        0.0833333333333333333*m11 + 0.041666666666666664*m12 - 0.25*m13 + 0.125*m16 + 0.125*m17 +
        0.003968253968253968*m2 + 0.025*m4 - 0.025*m6 + 0.027777777777777776*m9 +
        0.05263157894736842*rho + 0.08333333333333333*(Fx-Fy);
        dist2[n+9*Np] = fq;
        
        fq = 0. - 0.1*jx + 0.1*jy + 0.003341687552213868*m1 +
        0.013888888888888888*m10 + 0.0833333333333333333*m11 + 0.041666666666666664*m12 - 0.25*m13 -
        0.125*m16 - 0.125*m17 + 0.003968253968253968*m2 - 0.025*m4 + 0.025*m6 +
        0.027777777777777776*m9 + 0.05263157894736842*rho - 0.08333333333333333*(Fx-Fy);
        dist2[n+10*Np] = fq;
        
        fq =   0. + 0.1*jx + 0.1*jz + 0.003341687552213868*m1 + 0.013888888888888888*m10 -
        0.0833333333333333333*m11 - 0.041666666666666664*m12 + 0.25*m15 - 0.125*m16 + 0.125*m18 +
        0.003968253968253968*m2 + 0.025*m4 + 0.025*m8 + 0.027777777777777776*m9 +
        0.05263157894736842*rho + 0.08333333333333333*(Fx+Fz);
        dist2[n+11*Np] = fq;
        
        fq = 0. - 0.1*jx - 0.1*jz + 0.003341687552213868*m1 +
        0.013888888888888888*m10 - 0.0833333333333333333*m11 - 0.041666666666666664*m12 + 0.25*m15 +
        0.125*m16 - 0.125*m18 + 0.003968253968253968*m2 - 0.025*m4 - 0.025*m8 +
        0.027777777777777776*m9 + 0.05263157894736842*rho - 0.08333333333333333*(Fx+Fz);
        dist2[n+12*Np] = fq;
        
        fq =   0. + 0.1*jx - 0.1*jz + 0.003341687552213868*m1 + 0.013888888888888888*m10 -
        0.0833333333333333333*m11 - 0.041666666666666664*m12 - 0.25*m15 - 0.125*m16 - 0.125*m18 +
        0.003968253968253968*m2 + 0.025*m4 - 0.025*m8 + 0.027777777777777776*m9 +
        0.05263157894736842*rho + 0.08333333333333333*(Fx-Fz);
        dist2[n+13*Np] = fq;
        
        fq = 0. - 0.1*jx + 0.1*jz + 0.003341687552213868*m1 +
        0.013888888888888888*m10 - 0.0833333333333333333*m11 - 0.041666666666666664*m12 - 0.25*m15 +
        0.125*m16 + 0.125*m18 + 0.003968253968253968*m2 - 0.025*m4 + 0.025*m8 +
        0.027777777777777776*m9 + 0.05263157894736842*rho - 0.08333333333333333*(Fx-Fz);
        dist2[n+14*Np] = fq;
        
        fq =  0. + 0.1*jy + 0.1*jz + 0.003341687552213868*m1 - 0.027777777777777776*m10 + 0.25*m14 +
        0.125*m17 - 0.125*m18 + 0.003968253968253968*m2 + 0.025*m6 + 0.025*m8 -
        0.05555555555555555*m9 + 0.05263157894736842*rho + 0.08333333333333333*(Fy+Fz);
        dist2[n+15*Np] = fq;
        
        fq =  0. - 0.1*jy - 0.1*jz + 0.003341687552213868*m1 - 0.027777777777777776*m10 + 0.25*m14 -
        0.125*m17 + 0.125*m18 + 0.003968253968253968*m2 - 0.025*m6 - 0.025*m8 -
        0.05555555555555555*m9 + 0.05263157894736842*rho - 0.08333333333333333*(Fy+Fz);
        dist2[n+16*Np] = fq;
        
        fq =   0. + 0.1*jy - 0.1*jz + 0.003341687552213868*m1 - 0.027777777777777776*m10 - 0.25*m14 +
        0.125*m17 + 0.125*m18 + 0.003968253968253968*m2 + 0.025*m6 - 0.025*m8 -
        0.05555555555555555*m9 + 0.05263157894736842*rho + 0.08333333333333333*(Fy-Fz);
        dist2[n+17*Np] = fq;
        
        fq =   0. - 0.1*jy + 0.1*jz + 0.003341687552213868*m1 - 0.027777777777777776*m10 - 0.25*m14 -
        0.125*m17 - 0.125*m18 + 0.003968253968253968*m2 - 0.025*m6 + 0.025*m8 -
        0.05555555555555555*m9 + 0.05263157894736842*rho - 0.08333333333333333*(Fy-Fz);
        dist2[n+18*Np] = fq;
        
#ifdef SP
        
#else
        sl1 = scalarList[n+0*Np];
        sl2 = scalarList[n+1*Np];
        sl3 = scalarList[n+2*Np];
        sl4 = scalarList[n+3*Np];
        sl5 = scalarList[n+4*Np];
        sl6 = scalarList[n+5*Np];
        sl7 = scalarList[n+6*Np];
        sl8 = scalarList[n+7*Np];
        sl9  = scalarList[n+8*Np];
        sl10 = scalarList[n+9*Np];
        sl11 = scalarList[n+10*Np];
        sl12 = scalarList[n+11*Np];
        sl13 = scalarList[n+12*Np];
        sl14 = scalarList[n+13*Np];
        sl15 = scalarList[n+14*Np];
        sl16 = scalarList[n+15*Np];
        sl17 = scalarList[n+16*Np];
        sl18 = scalarList[n+17*Np];
        
    
        
        (sl1 == ijk)  ? b1  = -1 : b1 = 1;
        (sl2 == ijk)  ? b2  = -1 : b2 = 1;
        (sl3 == ijk)  ? b3  = -1 : b3 = 1;
        (sl4 == ijk)  ? b4  = -1 : b4 = 1;
        (sl5 == ijk)  ? b5  = -1 : b5 = 1;
        (sl6 == ijk)  ? b6  = -1 : b6 = 1;
        (sl7 == ijk)  ? b7  = -1 : b7 = 1;
        (sl8 == ijk)  ? b8  = -1 : b8 = 1;
        (sl9 == ijk)  ? b9  = -1 : b9 = 1;
        (sl10 == ijk) ? b10 = -1 : b10 = 1;
        (sl11 == ijk) ? b11 = -1 : b11 = 1;
        (sl12 == ijk) ? b12 = -1 : b12 = 1;
        (sl13 == ijk) ? b13 = -1 : b13 = 1;
        (sl14 == ijk) ? b14 = -1 : b14 = 1;
        (sl15 == ijk) ? b15 = -1 : b15 = 1;
        (sl16 == ijk) ? b16 = -1 : b16 = 1;
        (sl17 == ijk) ? b17 = -1 : b17 = 1;
        (sl18 == ijk) ? b18 = -1 : b18 = 1;
     
        t1 = DenA[sl1];  // i+1
        s1 = DenB[sl1];  // i+1
        
        t2 = DenA[sl2];  // i-1
        s2 = DenB[sl2];  // i-1

        t3 = DenA[sl3];  // j+1
        s3 = DenB[sl3];  // j+1
        
        t4 = DenA[sl4];  // j-1
        s4 = DenB[sl4];  // j-1

        t5 = DenA[sl5];  // k+1
        s5 = DenB[sl5];  // k+1
        
        t6 = DenA[sl6];  // k-1
        s6 = DenB[sl6];  // k-1
        
        t7 = DenA[sl7];  // i+1, j+1
        s7 = DenB[sl7];  // i+1, j+1

        t8 = DenA[sl8];  // i-1, j-1
        s8 = DenB[sl8];  // i-1, j-1

        t9 = DenA[sl9];  // k-1
        s9 = DenB[sl9];  // k-1

        t10 = DenA[sl10];  // k-1
        s10 = DenB[sl10];  // k-1

        t11 = DenA[sl11];  // k-1
        s11 = DenB[sl11];  // k-1

        t12 = DenA[sl12];  // k-1
        s12 = DenB[sl12];  // k-1

        t13 = DenA[sl13];  // k-1
        s13 = DenB[sl13];  // k-1

        t14 = DenA[sl14];  // k-1
        s14 = DenB[sl14];  // k-1

        t15 = DenA[sl15];  // k-1
        s15 = DenB[sl15];  // k-1

        t16 = DenA[sl16];  // k-1
        s16 = DenB[sl16];  // k-1

        t17 = DenA[sl17];  // k-1
        s17 = DenB[sl17];  // k-1

        t18 = DenA[sl18];  // k-1
        s18 = DenB[sl18];  // k-1
        
        delta = 0;
        nA = DenA[ijk];
        nB = DenB[ijk];
       

        densityA = nA/3.;
        densityB = nB/3.;

        nA = t1; nB = s1; ux = Velx2[sl1]; nx = GradPhiX[sl1];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*nx)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b1*delta  + nA*(1  + b1*ux))/18.;
        densityB+= (-b1*delta + nB*(1  + b1*ux))/18.;
        
        nA = t2; nB = s2; ux = Velx2[sl2]; nx = GradPhiX[sl2];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*nx)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b2*delta + nA*(1  - b2*ux))/18.;
        densityB+= (b2*delta  + nB*(1  - b2*ux))/18.;

        nA = t3; nB = s3; uy = Vely2[sl3]; ny = GradPhiY[sl3];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*ny)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b3*delta  + nA*(1  + b3*uy))/18.;
        densityB+= (-b3*delta + nB*(1  + b3*uy))/18.;
        
        nA = t4; nB = s4; uy = Vely2[sl4]; ny = GradPhiY[sl4];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*ny)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b4*delta + nA*(1  - b4*uy))/18.;
        densityB+= (b4*delta  + nB*(1  - b4*uy))/18.;

        nA = t5; nB = s5; uz = Velz2[sl5]; nz = GradPhiZ[sl5];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*nz)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b5*delta  + nA*(1  + b5*uz))/18.;
        densityB+= (-b5*delta + nB*(1  + b5*uz))/18.;
        
        nA = t6; nB = s6; uz = Velz2[sl6]; nz = GradPhiZ[sl6];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*nz)*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b6*delta + nA*(1  - b6*uz))/18.;
        densityB+= (b6*delta  + nB*(1  - b6*uz))/18.;

        nA = t7; nB = s7;  ux = Velx2[sl7]; uy = Vely2[sl7];  nx = GradPhiX[sl7]; ny = GradPhiY[sl7];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx + ny))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b7*delta  + nA*(1  + b7*(ux + uy)))/36.;
        densityB+= (-b7*delta + nB*(1  + b7*(ux + uy)))/36.;

        nA = t8; nB = s8;  ux = Velx2[sl8]; uy = Vely2[sl8];  nx = GradPhiX[sl8]; ny = GradPhiY[sl8];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx + ny))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b8*delta + nA*(1  + b8*(-ux - uy)))/36.;
        densityB+= (b8*delta  + nB*(1  + b8*(-ux - uy)))/36.;

        nA = t9; nB = s9;  ux = Velx2[sl9]; uy = Vely2[sl9];  nx = GradPhiX[sl9]; ny = GradPhiY[sl9];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx - ny))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b9*delta  + nA*(1  + b9*(ux - uy)))/36.;
        densityB+= (-b9*delta + nB*(1  + b9*(ux - uy)))/36.;

        nA = t10; nB = s10;  ux = Velx2[sl10]; uy = Vely2[sl10];  nx = GradPhiX[sl10]; ny = GradPhiY[sl10];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx - ny))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b10*delta + nA*(1  + b10*(-ux + uy)))/36.;
        densityB+= (b10*delta  + nB*(1  + b10*(-ux + uy)))/36.;

        nA = t11; nB = s11;  ux = Velx2[sl11]; uz = Velz2[sl11];  nx = GradPhiX[sl11]; nz = GradPhiZ[sl11];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx + nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b11*delta  + nA*(1  + b11*(ux + uz)))/36.;
        densityB+= (-b11*delta + nB*(1  + b11*(ux + uz)))/36.;

        nA = t12; nB = s12;  ux = Velx2[sl12]; uz = Velz2[sl12];  nx = GradPhiX[sl12]; nz = GradPhiZ[sl12];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx + nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b12*delta + nA*(1  + b12*(-ux - uz)))/36.;
        densityB+= (b12*delta  + nB*(1  + b12*(-ux - uz)))/36.;

        nA = t13; nB = s13;  ux = Velx2[sl13]; uz = Velz2[sl13];  nx = GradPhiX[sl13]; nz = GradPhiZ[sl13];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx - nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b13*delta  + nA*(1  + b13*(ux - uz)))/36.;
        densityB+= (-b13*delta + nB*(1  + b13*(ux - uz)))/36.;

        nA = t14; nB = s14;  ux = Velx2[sl14]; uz = Velz2[sl14];  nx = GradPhiX[sl14]; nz = GradPhiZ[sl14];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(nx - nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b14*delta + nA*(1  + b14*(-ux + uz)))/36.;
        densityB+= (b14*delta  + nB*(1  + b14*(-ux + uz)))/36.;

        nA = t15; nB = s15;  uy = Vely2[sl15]; uz = Velz2[sl15];  ny = GradPhiY[sl15]; nz = GradPhiZ[sl15];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(ny + nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b15*delta  + nA*(1  + b15*(uy + uz)))/36.;
        densityB+= (-b15*delta + nB*(1  + b15*(uy + uz)))/36.;

        nA = t16; nB = s16;  uy = Vely2[sl16]; uz = Velz2[sl16];  ny = GradPhiY[sl16]; nz = GradPhiZ[sl16];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(ny + nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b16*delta + nA*(1  + b16*(-uy - uz)))/36.;
        densityB+= (b16*delta  + nB*(1  + b16*(-uy - uz)))/36.;

        nA = t17; nB = s17;  uy = Vely2[sl17]; uz = Velz2[sl17];  ny = GradPhiY[sl17]; nz = GradPhiZ[sl17];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(ny - nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (b17*delta  + nA*(1  + b17*(uy - uz)))/36.;
        densityB+= (-b17*delta + nB*(1  + b17*(uy - uz)))/36.;

        nA = t18; nB = s18;  uy = Vely2[sl18]; uz = Velz2[sl18];  ny = GradPhiY[sl18]; nz = GradPhiZ[sl18];
        nAB = 1./(nA+nB);
        delta = (beta*nA*nB*(ny - nz))*nAB;
        if (!(nA*nB*nAB>0)) delta=0;
        densityA+= (-b18*delta + nA*(1  + b18*(-uy + uz)))/36.;
        densityB+= (b18*delta  + nB*(1  + b18*(-uy + uz)))/36.;
        
        DenA2[ijk] = densityA;
        DenB2[ijk] = densityB;
        
        
        Phi[ijk] = (densityA-densityB)/(densityA+densityB);
      
        if (densityA == 0 && densityB == 0) Phi[ijk] = 0;
        
     
#endif
        
        
           
           
           
        }
    }
}




__global__ void dvc_ScaLBL_PhaseField_Init_LIBB(int *Map, double *Phi, double *Den, double *Aq, double *Bq, int start, int finish, int Np){
    int idx,n;
    double phi,nA,nB;

     int S = Np/NBLOCKS/NTHREADS + 1;
       for (int s=0; s<S; s++) {
           //........Get 1-D index for this thread....................
           idx =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
           if (idx<finish) {



        n = Map[idx];
        phi = Phi[n];
        if (phi > 0.0){
            nA = 1.0; nB = 0.0;
        }
        else if (phi < 0.0){
            nB = 1.0; nA = 0.0;
        }
        else{
            nA=0.5*(phi+1.0);
            nB=0.5*(1.0-phi);
        }
        Den[idx] = nA;
        Den[Np+idx] = nB;


        Aq[idx] = 0.3333333333333333*nA;
        Aq[Np+idx] = 0.055555555555555555*nA;
        Aq[2*Np+idx] = 0.055555555555555555*nA;
        Aq[3*Np+idx] = 0.055555555555555555*nA;
        Aq[4*Np+idx] = 0.055555555555555555*nA;
        Aq[5*Np+idx] = 0.055555555555555555*nA;
        Aq[6*Np+idx] = 0.055555555555555555*nA;
        Aq[7*Np+idx] = 0.0277777777777778*nA;
        Aq[8*Np+idx] = 0.0277777777777778*nA;
        Aq[9*Np+idx] = 0.0277777777777778*nA;
        Aq[10*Np+idx] = 0.0277777777777778*nA;
        Aq[11*Np+idx] = 0.0277777777777778*nA;
        Aq[12*Np+idx] = 0.0277777777777778*nA;
        Aq[13*Np+idx] = 0.0277777777777778*nA;
        Aq[14*Np+idx] = 0.0277777777777778*nA;
        Aq[15*Np+idx] = 0.0277777777777778*nA;
        Aq[16*Np+idx] = 0.0277777777777778*nA;
        Aq[17*Np+idx] = 0.0277777777777778*nA;
        Aq[18*Np+idx] = 0.0277777777777778*nA;

        Bq[idx] = 0.3333333333333333*nB;
        Bq[Np+idx] = 0.055555555555555555*nB;
        Bq[2*Np+idx] = 0.055555555555555555*nB;
        Bq[3*Np+idx] = 0.055555555555555555*nB;
        Bq[4*Np+idx] = 0.055555555555555555*nB;
        Bq[5*Np+idx] = 0.055555555555555555*nB;
        Bq[6*Np+idx] = 0.055555555555555555*nB;
        Bq[7*Np+idx] = 0.0277777777777778*nB;
        Bq[8*Np+idx] = 0.0277777777777778*nB;
        Bq[9*Np+idx] = 0.0277777777777778*nB;
        Bq[10*Np+idx] = 0.0277777777777778*nB;
        Bq[11*Np+idx] = 0.0277777777777778*nB;
        Bq[12*Np+idx] = 0.0277777777777778*nB;
        Bq[13*Np+idx] = 0.0277777777777778*nB;
        Bq[14*Np+idx] = 0.0277777777777778*nB;
        Bq[15*Np+idx] = 0.0277777777777778*nB;
        Bq[16*Np+idx] = 0.0277777777777778*nB;
        Bq[17*Np+idx] = 0.0277777777777778*nB;
        Bq[18*Np+idx] = 0.0277777777777778*nB;
    }
}
}



extern "C" void ScaLBL_PhaseField_Init_LIBB(int *Map, double *Phi, double *Den, double *Aq, double *Bq, int start, int finish, int Np) {

        dvc_ScaLBL_PhaseField_Init_LIBB<<<NBLOCKS,NTHREADS >>>(Map,Phi,Den,Aq,Bq,start,finish,Np);
}


extern "C" void ScaLBL_D3Q7_PhaseField_LIBB(int* interpolationList, int *neighborList,  int *Map, double *Aq, double *Bq, double *savedAq, double *savedBq,  double *Den, double *Phi, int start, int finish, int Np, int N, double * LIBBqA, double * LIBBqBC, double * LIBBqD) {

    dvc_ScaLBL_D3Q7_PhaseField_LIBB<<<NBLOCKS,NTHREADS >>>(interpolationList, neighborList,
    Map, Aq, Bq, savedAq, savedBq, Den, Phi, start, finish, Np, N, LIBBqA, LIBBqBC, LIBBqD);
}


extern "C" void ScaLBL_D3Q19_Color_LIBB(int * scalarList, int * interpolationList, int *neighborList, int *Map, double *dist, double *dist2, double *savedfq, double *Aq, double *Bq, double *DenA, double *DenB, double * DenA2, double * DenB2, double *Phi, double *Velx, double * Vely, double * Velz, double *Velx2, double * Vely2, double * Velz2, double *Press, double rhoA, double rhoB, double tauA, double tauB, double alpha, double beta, double Fx, double Fy, double Fz, int strideY, int strideZ, int start, int finish, int Np, int N, double * LIBBqA, double * LIBBqBC, double * LIBBqD, double* GradPhiX, double*GradPhiY, double* GradPhiZ, double * CField) {

    dvc_ScaLBL_D3Q19_Color_LIBB<<<NBLOCKS,NTHREADS >>>(scalarList, interpolationList, neighborList, Map, dist, dist2, savedfq, Aq, Bq, DenA, DenB, DenA2, DenB2,Phi, Velx, Vely, Velz, Velx2, Vely2, Velz2, Press, rhoA,  rhoB, tauA, tauB, alpha, beta, Fx, Fy, Fz, strideY, strideZ, start, finish, Np, N, LIBBqA, LIBBqBC, LIBBqD,  GradPhiX, GradPhiY, GradPhiZ, CField);
}

extern "C" void Inactive_Color_LIBB(int * scalarList, int *Map, double *DenA, double *DenB, double * DenA2, double * DenB2, double *Phi, double *Velx2, double * Vely2, double * Velz2, double beta,  int strideY, int strideZ, int start, int finish, int Np, int N, double*  GradPhiX, double*GradPhiY, double* GradPhiZ, double*  CField) {

    dvc_Inactive_Color_LIBB<<<NBLOCKS,NTHREADS >>>(scalarList, Map, DenA, DenB, DenA2, DenB2, Phi, Velx2, Vely2, Velz2, beta, strideY, strideZ, start, finish, Np, N, GradPhiX, GradPhiY, GradPhiZ, CField);



}



extern "C" void InitExtrapolatePhaseFieldActive(int *Map, double * VFmask, double *phi, double *phi2, int start, int finish, int strideY, int strideZ, int Np) {


    dvc_InitExtrapolatePhaseFieldActive<<<NBLOCKS,NTHREADS >>>(Map, VFmask, phi, phi2,
    start, finish, strideY, strideZ, Np);


}




extern "C" void ComputeGradPhi(double input_angle, int *Map, double * Phi,
                               double * GradPhiX, double * GradPhiY, double * GradPhiZ, double * CField,
                               double * GradSDsX, double * GradSDsY, double * GradSDsZ,
                               int strideY, int strideZ, int start, int finish, int Np, int WBCFlag) {
                               
    dvc_ComputeGradPhi<<<NBLOCKS,NTHREADS >>>(input_angle, Map, Phi,
                               GradPhiX,  GradPhiY,  GradPhiZ,  CField,
                               GradSDsX,  GradSDsY,  GradSDsZ,
                            strideY, strideZ, start, finish, Np, WBCFlag);
                               
                               
}


extern "C" void ExtrapolateScalarField(int *Map, int * neighborList, double * phi, double *phi2, int start, int finish, int Nsb, int strideY, int strideZ) {

    dvc_ExtrapolateScalarField<<<NBLOCKS,NTHREADS >>>(Map,  neighborList,  phi,  phi2, start, finish, Nsb, strideY, strideZ);
}


extern "C" void InitExtrapolatePhaseFieldInactive(int *Map, char * id, double *phi, double *phi2, int start, int finish, int strideY, int strideZ,int Np) {

    dvc_InitExtrapolatePhaseFieldInactive<<<NBLOCKS,NTHREADS >>>(Map, id,  phi,  phi2, start, finish, strideY, strideZ,Np);

}


extern "C" void InitExtrapolateScalarField(int *Map, char * id, double * phi, double *phi2, int start, int finish, int Ni, int strideY, int strideZ) {

    dvc_InitExtrapolateScalarField<<<NBLOCKS,NTHREADS >>>(Map,id, phi,  phi2, start, finish, Ni, strideY, strideZ);

}
