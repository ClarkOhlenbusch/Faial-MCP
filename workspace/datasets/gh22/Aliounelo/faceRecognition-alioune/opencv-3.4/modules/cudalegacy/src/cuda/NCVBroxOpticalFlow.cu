/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

////////////////////////////////////////////////////////////////////////////////
//
// NVIDIA CUDA implementation of Brox et al Optical Flow algorithm
//
// Algorithm is explained in the original paper:
//      T. Brox, A. Bruhn, N. Papenberg, J. Weickert:
//      High accuracy optical flow estimation based on a theory for warping.
//      ECCV 2004.
//
// Implementation by Mikhail Smirnov
// email: msmirnov@nvidia.com, devsupport@nvidia.com
//
// Credits for help with the code to:
// Alexey Mendelenko, Anton Obukhov, and Alexander Kharlamov.
//
////////////////////////////////////////////////////////////////////////////////

typedef float FloatVector;

/////////////////////////////////////////////////////////////////////////////////////////
// Implementation specific constants
/////////////////////////////////////////////////////////////////////////////////////////
__device__ const float eps2 = 1e-6f;

/////////////////////////////////////////////////////////////////////////////////////////
// Additional defines
/////////////////////////////////////////////////////////////////////////////////////////

// rounded up division
inline int iDivUp(int a, int b)
{
    return (a + b - 1)/b;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Texture references
/////////////////////////////////////////////////////////////////////////////////////////

texture<float, 2, cudaReadModeElementType> tex_coarse;
texture<float, 2, cudaReadModeElementType> tex_fine;

texture<float, 2, cudaReadModeElementType> tex_I1;
texture<float, 2, cudaReadModeElementType> tex_I0;

texture<float, 2, cudaReadModeElementType> tex_Ix;
texture<float, 2, cudaReadModeElementType> tex_Ixx;
texture<float, 2, cudaReadModeElementType> tex_Ix0;

texture<float, 2, cudaReadModeElementType> tex_Iy;
texture<float, 2, cudaReadModeElementType> tex_Iyy;
texture<float, 2, cudaReadModeElementType> tex_Iy0;

texture<float, 2, cudaReadModeElementType> tex_Ixy;

texture<float, 1, cudaReadModeElementType> tex_u;
texture<float, 1, cudaReadModeElementType> tex_v;
texture<float, 1, cudaReadModeElementType> tex_du;
texture<float, 1, cudaReadModeElementType> tex_dv;
texture<float, 1, cudaReadModeElementType> tex_numerator_dudv;
texture<float, 1, cudaReadModeElementType> tex_numerator_u;
texture<float, 1, cudaReadModeElementType> tex_numerator_v;
texture<float, 1, cudaReadModeElementType> tex_inv_denominator_u;
texture<float, 1, cudaReadModeElementType> tex_inv_denominator_v;
texture<float, 1, cudaReadModeElementType> tex_diffusivity_x;
texture<float, 1, cudaReadModeElementType> tex_diffusivity_y;


/////////////////////////////////////////////////////////////////////////////////////////
// SUPPLEMENTARY FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// \brief performs pointwise summation of two vectors stored in device memory
/// \param d_res    - pointer to resulting vector (device memory)
/// \param d_op1    - term #1 (device memory)
/// \param d_op2    - term #2 (device memory)
/// \param len    - vector size
///////////////////////////////////////////////////////////////////////////////
__global__ void pointwise_add(float *d_res, const float *d_op1, const float *d_op2, const int len)
{
    const int pos = blockIdx.x*blockDim.x + threadIdx.x;

    if(pos >= len) return;

    d_res[pos] = d_op1[pos] + d_op2[pos];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief wrapper for summation kernel.
///  Computes \b op1 + \b op2 and stores result to \b res
/// \param res   array, containing op1 + op2 (device memory)
/// \param op1   term #1 (device memory)
/// \param op2   term #2 (device memory)
/// \param count vector size
///////////////////////////////////////////////////////////////////////////////
static void add(float *res, const float *op1, const float *op2, const int count, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks(iDivUp(count, threads.x));

    pointwise_add<<<blocks, threads, 0, stream>>>(res, op1, op2, count);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief wrapper for summation kernel.
/// Increments \b res by \b rhs
/// \param res   initial vector, will be replaced with result (device memory)
/// \param rhs   increment (device memory)
/// \param count vector size
///////////////////////////////////////////////////////////////////////////////
static void add(float *res, const float *rhs, const int count, cudaStream_t stream)
{
    add(res, res, rhs, count, stream);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief kernel for scaling vector by scalar
/// \param d_res  scaled vector (device memory)
/// \param d_src  source vector (device memory)
/// \param scale  scalar to scale by
/// \param len    vector size (number of elements)
///////////////////////////////////////////////////////////////////////////////
__global__ void scaleVector(float *d_res, const float *d_src, float scale, const int len)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= len) return;

    d_res[pos] = d_src[pos] * scale;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief scale vector by scalar
///
/// kernel wrapper
/// \param d_res  scaled vector (device memory)
/// \param d_src  source vector (device memory)
/// \param scale  scalar to scale by
/// \param len    vector size (number of elements)
/// \param stream CUDA stream
///////////////////////////////////////////////////////////////////////////////
static void ScaleVector(float *d_res, const float *d_src, float scale, const int len, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks(iDivUp(len, threads.x));

    scaleVector<<<blocks, threads, 0, stream>>>(d_res, d_src, scale, len);
}

const int SOR_TILE_WIDTH = 32;
const int SOR_TILE_HEIGHT = 6;
const int PSOR_TILE_WIDTH = 32;
const int PSOR_TILE_HEIGHT = 6;
const int PSOR_PITCH = PSOR_TILE_WIDTH + 4;
const int PSOR_HEIGHT = PSOR_TILE_HEIGHT + 4;

///////////////////////////////////////////////////////////////////////////////
///\brief Utility function. Compute smooth term diffusivity along x axis
///\param s (out) pointer to memory location for result (diffusivity)
///\param pos (in) position within shared memory array containing \b u
///\param u (in) shared memory array containing \b u
///\param v (in) shared memory array containing \b v
///\param du (in) shared memory array containing \b du
///\param dv (in) shared memory array containing \b dv
///////////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ void diffusivity_along_x(float *s, int pos, const float *u, const float *v, const float *du, const float *dv)
{
    //x derivative between pixels (i,j) and (i-1,j)
    const int left = pos-1;
    float u_x = u[pos] + du[pos] - u[left] - du[left];
    float v_x = v[pos] + dv[pos] - v[left] - dv[left];
    const int up        = pos + PSOR_PITCH;
    const int down      = pos - PSOR_PITCH;
    const int up_left   = up - 1;
    const int down_left = down-1;
    //y derivative between pixels (i,j) and (i-1,j)
    float u_y = 0.25f*(u[up] + du[up] + u[up_left] + du[up_left] - u[down] - du[down] - u[down_left] - du[down_left]);
    float v_y = 0.25f*(v[up] + dv[up] + v[up_left] + dv[up_left] - v[down] - dv[down] - v[down_left] - dv[down_left]);
    *s = 0.5f / sqrtf(u_x*u_x + v_x*v_x + u_y*u_y + v_y*v_y + eps2);
}

///////////////////////////////////////////////////////////////////////////////
///\brief Utility function. Compute smooth term diffusivity along y axis
///\param s (out) pointer to memory location for result (diffusivity)
///\param pos (in) position within shared memory array containing \b u
///\param u (in) shared memory array containing \b u
///\param v (in) shared memory array containing \b v
///\param du (in) shared memory array containing \b du
///\param dv (in) shared memory array containing \b dv
///////////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ void diffusivity_along_y(float *s, int pos, const float *u, const float *v, const float *du, const float *dv)
{
    //y derivative between pixels (i,j) and (i,j-1)
    const int down = pos-PSOR_PITCH;
    float u_y = u[pos] + du[pos] - u[down] - du[down];
    float v_y = v[pos] + dv[pos] - v[down] - dv[down];
    const int right      = pos + 1;
    const int left       = pos - 1;
    const int down_right = down + 1;
    const int down_left  = down - 1;
    //x derivative between pixels (i,j) and (i,j-1);
    float u_x = 0.25f*(u[right] + u[down_right] + du[right] + du[down_right] - u[left] - u[down_left] - du[left] - du[down_left]);
    float v_x = 0.25f*(v[right] + v[down_right] + dv[right] + dv[down_right] - v[left] - v[down_left] - dv[left] - dv[down_left]);
    *s = 0.5f/sqrtf(u_x*u_x + v_x*v_x + u_y*u_y + v_y*v_y + eps2);
}

///////////////////////////////////////////////////////////////////////////////
///\brief Utility function. Load element of 2D global memory to shared memory
///\param smem pointer to shared memory array
///\param is shared memory array column
///\param js shared memory array row
///\param w number of columns in global memory array
///\param h number of rows in global memory array
///\param p global memory array pitch in floats
///////////////////////////////////////////////////////////////////////////////
template<int tex_id>
__forceinline__ __device__ void load_array_element(float *smem, int is, int js, int i, int j, int w, int h, int p)
{
    //position within shared memory array
    const int ijs = js * PSOR_PITCH + is;
    //mirror reflection across borders
    i = max(i, -i-1);
    i = min(i, w-i+w-1);
    j = max(j, -j-1);
    j = min(j, h-j+h-1);
    const int pos = j * p + i;
    switch(tex_id){
        case 0:
            smem[ijs] = tex1Dfetch(tex_u, pos);
            break;
        case 1:
            smem[ijs] = tex1Dfetch(tex_v, pos);
            break;
        case 2:
            smem[ijs] = tex1Dfetch(tex_du, pos);
            break;
        case 3:
            smem[ijs] = tex1Dfetch(tex_dv, pos);
            break;
    }
}

///////////////////////////////////////////////////////////////////////////////
///\brief Utility function. Load part (tile) of 2D global memory to shared memory
///\param smem pointer to target shared memory array
///\param ig column number within source
///\param jg row number within source
///\param w number of columns in global memory array
///\param h number of rows in global memory array
///\param p global memory array pitch in floats
///////////////////////////////////////////////////////////////////////////////
template<int tex>
__forceinline__ __device__ void load_array(float *smem, int ig, int jg, int w, int h, int p)
{
    const int i = threadIdx.x + 2;
    const int j = threadIdx.y + 2;
    load_array_element<tex>(smem, i, j, ig, jg, w, h, p);//load current pixel
    __syncthreads();
    if(threadIdx.y < 2)
    {
        //load bottom shadow elements
        load_array_element<tex>(smem, i, j-2, ig, jg-2, w, h, p);
        if(threadIdx.x < 2)
        {
            //load bottom right shadow elements
            load_array_element<tex>(smem, i+PSOR_TILE_WIDTH, j-2, ig+PSOR_TILE_WIDTH, jg-2, w, h, p);
            //load middle right shadow elements
            load_array_element<tex>(smem, i+PSOR_TILE_WIDTH, j, ig+PSOR_TILE_WIDTH, jg, w, h, p);
        }
        else if(threadIdx.x >= PSOR_TILE_WIDTH-2)
        {
            //load bottom left shadow elements
            load_array_element<tex>(smem, i-PSOR_TILE_WIDTH, j-2, ig-PSOR_TILE_WIDTH, jg-2, w, h, p);
            //load middle left shadow elements
            load_array_element<tex>(smem, i-PSOR_TILE_WIDTH, j, ig-PSOR_TILE_WIDTH, jg, w, h, p);
        }
    }
    else if(threadIdx.y >= PSOR_TILE_HEIGHT-2)
    {
        //load upper shadow elements
        load_array_element<tex>(smem, i, j+2, ig, jg+2, w, h, p);
        if(threadIdx.x < 2)
        {
            //load upper right shadow elements
            load_array_element<tex>(smem, i+PSOR_TILE_WIDTH, j+2, ig+PSOR_TILE_WIDTH, jg+2, w, h, p);
            //load middle right shadow elements
            load_array_element<tex>(smem, i+PSOR_TILE_WIDTH, j, ig+PSOR_TILE_WIDTH, jg, w, h, p);
        }
        else if(threadIdx.x >= PSOR_TILE_WIDTH-2)
        {
            //load upper left shadow elements
            load_array_element<tex>(smem, i-PSOR_TILE_WIDTH, j+2, ig-PSOR_TILE_WIDTH, jg+2, w, h, p);
            //load middle left shadow elements
            load_array_element<tex>(smem, i-PSOR_TILE_WIDTH, j, ig-PSOR_TILE_WIDTH, jg, w, h, p);
        }
    }
    else
    {
        //load middle shadow elements
        if(threadIdx.x < 2)
        {
            //load middle right shadow elements
            load_array_element<tex>(smem, i+PSOR_TILE_WIDTH, j, ig+PSOR_TILE_WIDTH, jg, w, h, p);
        }
        else if(threadIdx.x >= PSOR_TILE_WIDTH-2)
        {
            //load middle left shadow elements
            load_array_element<tex>(smem, i-PSOR_TILE_WIDTH, j, ig-PSOR_TILE_WIDTH, jg, w, h, p);
        }
    }
    __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
/// \brief computes matrix of linearised system for \c du, \c dv
/// Computed values reside in GPU memory. \n
/// Matrix computation is divided into two steps. This kernel performs first step\n
/// - compute smoothness term diffusivity between pixels - psi dash smooth
/// - compute robustness factor in the data term - psi dash data
/// \param diffusivity_x (in/out) diffusivity between pixels along x axis in smoothness term
/// \param diffusivity_y (in/out) diffusivity between pixels along y axis in smoothness term
/// \param denominator_u (in/out) precomputed part of expression for new du value in SOR iteration
/// \param denominator_v (in/out) precomputed part of expression for new dv value in SOR iteration
/// \param numerator_dudv (in/out) precomputed part of expression for new du and dv value in SOR iteration
/// \param numerator_u (in/out) precomputed part of expression for new du value in SOR iteration
/// \param numerator_v (in/out) precomputed part of expression for new dv value in SOR iteration
/// \param w (in) frame width
/// \param h (in) frame height
/// \param pitch (in) pitch in floats
/// \param alpha (in) alpha in Brox model (flow smoothness)
/// \param gamma (in) gamma in Brox model (edge importance)
///////////////////////////////////////////////////////////////////////////////

__global__ void prepare_sor_stage_1_tex(float *diffusivity_x, float *diffusivity_y,
                                                        float *denominator_u, float *denominator_v,
                                                        float *numerator_dudv,
                                                        float *numerator_u, float *numerator_v,
                                                        int w, int h, int s,
                                                        float alpha, float gamma)
{
    __shared__ float u[PSOR_PITCH * PSOR_HEIGHT];
    __shared__ float v[PSOR_PITCH * PSOR_HEIGHT];
    __shared__ float du[PSOR_PITCH * PSOR_HEIGHT];
    __shared__ float dv[PSOR_PITCH * PSOR_HEIGHT];

    //position within tile
    const int i = threadIdx.x;
    const int j = threadIdx.y;
    //position within smem arrays
    const int ijs = (j+2) * PSOR_PITCH + i + 2;
    //position within global memory
    const int ig  = blockIdx.x * blockDim.x + threadIdx.x;
    const int jg  = blockIdx.y * blockDim.y + threadIdx.y;
    const int ijg = jg * s + ig;
    //position within texture
    float x = (float)ig + 0.5f;
    float y = (float)jg + 0.5f;
    //load u  and v to smem
    load_array<0>(u, ig, jg, w, h, s);
    load_array<1>(v, ig, jg, w, h, s);
    load_array<2>(du, ig, jg, w, h, s);
    load_array<3>(dv, ig, jg, w, h, s);
    //warped position
    float wx = (x + u[ijs])/(float)w;
    float wy = (y + v[ijs])/(float)h;
    x /= (float)w;
    y /= (float)h;
    //compute image derivatives
    const float Iz  = tex2D(tex_I1, wx, wy) - tex2D(tex_I0, x, y);
    const float Ix  = tex2D(tex_Ix, wx, wy);
    const float Ixz = Ix - tex2D(tex_Ix0, x, y);
    const float Ixy = tex2D(tex_Ixy, wx, wy);
    const float Ixx = tex2D(tex_Ixx, wx, wy);
    const float Iy  = tex2D(tex_Iy, wx, wy);
    const float Iyz = Iy - tex2D(tex_Iy0, x, y);
    const float Iyy = tex2D(tex_Iyy, wx, wy);
    //compute data term
    float q0, q1, q2;
    q0 = Iz  + Ix  * du[ijs] + Iy  * dv[ijs];
    q1 = Ixz + Ixx * du[ijs] + Ixy * dv[ijs];
    q2 = Iyz + Ixy * du[ijs] + Iyy * dv[ijs];
    float data_term = 0.5f * rsqrtf(q0*q0 + gamma*(q1*q1 + q2*q2) + eps2);
    //scale data term by 1/alpha
    data_term /= alpha;
    //compute smoothness term (diffusivity)
    float sx, sy;

    if(ig >= w || jg >= h) return;

    diffusivity_along_x(&sx, ijs, u, v, du, dv);
    diffusivity_along_y(&sy, ijs, u, v, du, dv);

    if(ig == 0) sx = 0.0f;
    if(jg == 0) sy = 0.0f;

    numerator_dudv[ijg] = data_term * (Ix*Iy + gamma * Ixy*(Ixx + Iyy));
    numerator_u[ijg]    = data_term * (Ix*Iz + gamma * (Ixx*Ixz + Ixy*Iyz));
    numerator_v[ijg]    = data_term * (Iy*Iz + gamma * (Iyy*Iyz + Ixy*Ixz));
    denominator_u[ijg]  = data_term * (Ix*Ix + gamma * (Ixy*Ixy + Ixx*Ixx));
    denominator_v[ijg]  = data_term * (Iy*Iy + gamma * (Ixy*Ixy + Iyy*Iyy));
    diffusivity_x[ijg]  = sx;
    diffusivity_y[ijg]  = sy;
}

///////////////////////////////////////////////////////////////////////////////
///\brief computes matrix of linearised system for \c du, \c dv
///\param inv_denominator_u
///\param inv_denominator_v
///\param w
///\param h
///\param s
///////////////////////////////////////////////////////////////////////////////
__global__ void prepare_sor_stage_2(float *inv_denominator_u, float *inv_denominator_v,
                                    int w, int h, int s)
{
    __shared__ float sx[(PSOR_TILE_WIDTH+1) * (PSOR_TILE_HEIGHT+1)];
    __shared__ float sy[(PSOR_TILE_WIDTH+1) * (PSOR_TILE_HEIGHT+1)];
    //position within tile
    const int i = threadIdx.x;
    const int j = threadIdx.y;
    //position within smem arrays
    const int ijs = j*(PSOR_TILE_WIDTH+1) + i;
    //position within global memory
    const int ig  = blockIdx.x * blockDim.x + threadIdx.x;
    const int jg  = blockIdx.y * blockDim.y + threadIdx.y;
    const int ijg = jg*s + ig;
    int inside = ig < w && jg < h;
    float denom_u;
    float denom_v;
    if(inside)
    {
        denom_u = inv_denominator_u[ijg];
        denom_v = inv_denominator_v[ijg];
    }
    if(inside)
    {
        sx[ijs] = tex1Dfetch(tex_diffusivity_x, ijg);
        sy[ijs] = tex1Dfetch(tex_diffusivity_y, ijg);
    }
    else
    {
        sx[ijs] = 0.0f;
        sy[ijs] = 0.0f;
    }
    int up = ijs+PSOR_TILE_WIDTH+1;
    if(j == PSOR_TILE_HEIGHT-1)
    {
        if(jg < h-1 && inside)
        {
            sy[up] = tex1Dfetch(tex_diffusivity_y, ijg + s);
        }
        else
        {
            sy[up] = 0.0f;
        }
    }
    int right = ijs + 1;
    if(threadIdx.x == PSOR_TILE_WIDTH-1)
    {
        if(ig < w-1 && inside)
        {
            sx[right] = tex1Dfetch(tex_diffusivity_x, ijg + 1);
        }
        else
        {
            sx[right] = 0.0f;
        }
    }
    __syncthreads();
    float diffusivity_sum;
    diffusivity_sum = sx[ijs] + sx[ijs+1] + sy[ijs] + sy[ijs+PSOR_TILE_WIDTH+1];
    if(inside)
    {
        denom_u += diffusivity_sum;
        denom_v += diffusivity_sum;
        inv_denominator_u[ijg] = 1.0f/denom_u;
        inv_denominator_v[ijg] = 1.0f/denom_v;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
// Red-Black SOR
/////////////////////////////////////////////////////////////////////////////////////////

template<int isBlack> __global__ void sor_pass(float *new_du,
                                               float *new_dv,
                                               const float *g_inv_denominator_u,
                                               const float *g_inv_denominator_v,
                                               const float *g_numerator_u,
                                               const float *g_numerator_v,
                                               const float *g_numerator_dudv,
                                               float omega,
                                               int width,
                                               int height,
                                               int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= width || j >= height)
        return;

    const int pos = j * stride + i;
    const int pos_r = i < width - 1 ? pos + 1 : pos;
    const int pos_u = j < height - 1 ? pos + stride : pos;
    const int pos_d = j > 0 ? pos - stride : pos;
    const int pos_l = i > 0 ? pos - 1 : pos;

    //load smooth term
    float s_up, s_left, s_right, s_down;
    s_left = tex1Dfetch(tex_diffusivity_x, pos);
    s_down = tex1Dfetch(tex_diffusivity_y, pos);
    if(i < width-1)
        s_right = tex1Dfetch(tex_diffusivity_x, pos_r);
    else
        s_right = 0.0f; //Neumann BC
    if(j < height-1)
        s_up = tex1Dfetch(tex_diffusivity_y, pos_u);
    else
        s_up = 0.0f; //Neumann BC

    //load u, v and du, dv
    float u_up, u_left, u_right, u_down, u;
    float v_up, v_left, v_right, v_down, v;
    float du_up, du_left, du_right, du_down, du;
    float dv_up, dv_left, dv_right, dv_down, dv;

    u_left  = tex1Dfetch(tex_u, pos_l);
    u_right = tex1Dfetch(tex_u, pos_r);
    u_down  = tex1Dfetch(tex_u, pos_d);
    u_up    = tex1Dfetch(tex_u, pos_u);
    u       = tex1Dfetch(tex_u, pos);

    v_left  = tex1Dfetch(tex_v, pos_l);
    v_right = tex1Dfetch(tex_v, pos_r);
    v_down  = tex1Dfetch(tex_v, pos_d);
    v       = tex1Dfetch(tex_v, pos);
    v_up    = tex1Dfetch(tex_v, pos_u);

    du       = tex1Dfetch(tex_du, pos);
    du_left  = tex1Dfetch(tex_du, pos_l);
    du_right = tex1Dfetch(tex_du, pos_r);
    du_down  = tex1Dfetch(tex_du, pos_d);
    du_up    = tex1Dfetch(tex_du, pos_u);

    dv       = tex1Dfetch(tex_dv, pos);
    dv_left  = tex1Dfetch(tex_dv, pos_l);
    dv_right = tex1Dfetch(tex_dv, pos_r);
    dv_down  = tex1Dfetch(tex_dv, pos_d);
    dv_up    = tex1Dfetch(tex_dv, pos_u);

    float numerator_dudv    = g_numerator_dudv[pos];

    if((i+j)%2 == isBlack)
    {
        // update du
        float numerator_u = (s_left*(u_left + du_left) + s_up*(u_up + du_up) + s_right*(u_right + du_right) + s_down*(u_down + du_down) -
                             u * (s_left + s_right + s_up + s_down) - g_numerator_u[pos] - numerator_dudv*dv);

        du = (1.0f - omega) * du + omega * g_inv_denominator_u[pos] * numerator_u;

        // update dv
        float numerator_v = (s_left*(v_left + dv_left) + s_up*(v_up + dv_up) + s_right*(v_right + dv_right) + s_down*(v_down + dv_down) -
                             v * (s_left + s_right + s_up + s_down) - g_numerator_v[pos] - numerator_dudv*du);

        dv = (1.0f - omega) * dv + omega * g_inv_denominator_v[pos] * numerator_v;
    }
    new_du[pos] = du;
    new_dv[pos] = dv;
}
