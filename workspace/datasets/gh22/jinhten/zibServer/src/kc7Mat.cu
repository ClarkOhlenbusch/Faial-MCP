// base header
#include "../inc/kc7Mat.h"

#include <mma.h>

#ifdef __CUDA_IMMA__

using namespace nvcuda;

#endif

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// standard defines

////////////////////////////////////////////////////////////
// standard function definition for reduction

#define KC_RDC_WHOLE_FRONT(V0) \
	extern __shared__ float sm[];\
	KC_RDC_INDEX_EXDM0(b); \
	T val = V0; \
	for(int ix2 = 0; ix2 < nx2; ++ix2) { \
		const int i1 = ix1 + nx1*ix2; if(i1 >= b.n1) continue;

#define KC_RDC_WHOLE_BACK(A) \
	} \
	sm[ix1] = val; \
	for(int size_s = nx1>>1; size_s > 0; size_s >>= 1) { \
		__syncthreads(); \
		if(ix1 < size_s) A(sm[ix1], sm[ix1 + size_s]); } \
	if(ix1 == 0) *a = sm[0]; \

#define KC_RDC_WHOLE_BACK_SUM() \
	} \
	sm[ix1] = val; \
	for(int size_s = nx1>>1; size_s > 0; size_s >>= 1) { \
		__syncthreads(); \
		if(ix1 < size_s) sm[ix1] += sm[ix1 + size_s]; } \
	if(ix1 == 0) *a = sm[0]; \

#define KC1_RDC_CALL_DFUN(A,V0) \
	KC_RDC_WHOLE_FRONT(V0) \
	A(val, b(i1)); \
	KC_RDC_WHOLE_BACK(A) \

#define KC2_RDC_CALL_DFUN(A,V0) \
	KC_RDC_WHOLE_FRONT(V0) \
	for(int i2 = 0; i2 < b.n2; ++i2) { A(val, b(i1,i2));} \
	KC_RDC_WHOLE_BACK(A) \

#define KC3_RDC_CALL_DFUN(A,V0) \
	KC_RDC_WHOLE_FRONT(V0) \
	for(int i3 = 0; i3 < b.n3; ++i3) \
	for(int i2 = 0; i2 < b.n2; ++i2) { A(val, b(i1,i2,i3));} \
	KC_RDC_WHOLE_BACK(A) \

#define KC4_RDC_CALL_DFUN(A, V0) \
	KC_RDC_WHOLE_FRONT(V0) \
	for(int i4 = 0; i4 < b.n4; ++i4) \
	for(int i3 = 0; i3 < b.n3; ++i3) \
	for(int i2 = 0; i2 < b.n2; ++i2) { A(val, b(i1,i2,i3,i4)); } \
	KC_RDC_WHOLE_BACK(A) \

#define KC1_RDC_CALL_DFUN_SUM(A,V0) \
	KC_RDC_WHOLE_FRONT(V0) \
	A(val, b(i1)); \
	KC_RDC_WHOLE_BACK_SUM() \

#define KC2_RDC_CALL_DFUN_SUM(A,V0) \
	KC_RDC_WHOLE_FRONT(V0) \
	for(int i2 = 0; i2 < b.n2; ++i2) { A(val, b(i1,i2));} \
	KC_RDC_WHOLE_BACK_SUM() \

#define KC3_RDC_CALL_DFUN_SUM(A,V0) \
	KC_RDC_WHOLE_FRONT(V0) \
	for(int i3 = 0; i3 < b.n3; ++i3) \
	for(int i2 = 0; i2 < b.n2; ++i2) { A(val, b(i1,i2,i3));} \
	KC_RDC_WHOLE_BACK_SUM() \

#define KC4_RDC_CALL_DFUN_SUM(A, V0) \
	KC_RDC_WHOLE_FRONT(V0) \
	for(int i4 = 0; i4 < b.n4; ++i4) \
	for(int i3 = 0; i3 < b.n3; ++i3) \
	for(int i2 = 0; i2 < b.n2; ++i2) { A(val, b(i1,i2,i3,i4)); } \
	KC_RDC_WHOLE_BACK_SUM() \

#define KC_RDC_CALL_KERNEL(A,DIM) \
	kcMat1<T> a(1); \
	dim3 bk, gd; int sm; b.GetBkGdRdc(bk, gd, sm); \
	A<<<gd,bk,sm,s>>>(a.P(), kckMat##DIM<T>(b)); \
	return kmMat1<T>(a)(0)

#define KC_RDC_WHOLE_DEFINITION(NAME, DEVFUN, V0) \
\
template<typename T> __global__ void kck##NAME(T* a, const kckMat1<T> b) { KC1_RDC_CALL_DFUN(DEVFUN, V0); } \
template<typename T> __global__ void kck##NAME(T* a, const kckMat2<T> b) { KC2_RDC_CALL_DFUN(DEVFUN, V0); } \
template<typename T> __global__ void kck##NAME(T* a, const kckMat3<T> b) { KC3_RDC_CALL_DFUN(DEVFUN, V0); } \
template<typename T> __global__ void kck##NAME(T* a, const kckMat4<T> b) { KC4_RDC_CALL_DFUN(DEVFUN, V0); } \
\
template<typename T> T NAME(const kcMat1<T>& b, cudaStream_t s) { KC_RDC_CALL_KERNEL(kck##NAME, 1);}; \
template<typename T> T NAME(const kcMat2<T>& b, cudaStream_t s) { KC_RDC_CALL_KERNEL(kck##NAME, 2);}; \
template<typename T> T NAME(const kcMat3<T>& b, cudaStream_t s) { KC_RDC_CALL_KERNEL(kck##NAME, 3);}; \
template<typename T> T NAME(const kcMat4<T>& b, cudaStream_t s) { KC_RDC_CALL_KERNEL(kck##NAME, 4);};

#define KC_RDC_WHOLE_DEFINITION_SUM(NAME, DEVFUN, V0) \
\
template<typename T> __global__ void kck##NAME(T* a, const kckMat1<T> b) { KC1_RDC_CALL_DFUN_SUM(DEVFUN, V0); } \
template<typename T> __global__ void kck##NAME(T* a, const kckMat2<T> b) { KC2_RDC_CALL_DFUN_SUM(DEVFUN, V0); } \
template<typename T> __global__ void kck##NAME(T* a, const kckMat3<T> b) { KC3_RDC_CALL_DFUN_SUM(DEVFUN, V0); } \
template<typename T> __global__ void kck##NAME(T* a, const kckMat4<T> b) { KC4_RDC_CALL_DFUN_SUM(DEVFUN, V0); } \
\
template<typename T> T NAME(const kcMat1<T>& b, cudaStream_t s) { KC_RDC_CALL_KERNEL(kck##NAME, 1);}; \
template<typename T> T NAME(const kcMat2<T>& b, cudaStream_t s) { KC_RDC_CALL_KERNEL(kck##NAME, 2);}; \
template<typename T> T NAME(const kcMat3<T>& b, cudaStream_t s) { KC_RDC_CALL_KERNEL(kck##NAME, 3);}; \
template<typename T> T NAME(const kcMat4<T>& b, cudaStream_t s) { KC_RDC_CALL_KERNEL(kck##NAME, 4);};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// math functions for kcMat

template<typename T> inline
__device__ void kcdAddEq(T& a, const T& b)
{
	a = __fadd_rn(a, b);
}

////////////////////////////////////////////////////////
// dot product and add... a = b*c

template<typename T>
__global__ void kckDot(kckMat2<T> a, const kckMat2<T> b, const kckMat2<T> c)
{
	// init idx
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);

	if(i1 >= a.n1 || i2 >= a.n2) return;
		
	// calc multification
	T val = 0;

	for(int j = 0; j < c.n1; ++j) val += b(i1,j)*c(j,i2);

	// set the result
	a(i1,i2) = val;
}

template<typename T>
kcMat2<T>& _Dot(kcMat2<T>& a, const kcMat2<T>& b, const kcMat2<T>& c, cudaStream_t s)
{
	// check size
	ASSERTFA(b.N2() == c.N1(), "_Dot in 111 of kc7Mat.cu");
	
	if(a.N1() != b.N1() || a.N2() != c.N2()) a.Recreate(b.N1(), c.N2());

	// call kernel
	dim3 bk, gd; a.GetBkGd(bk, gd);

	kckDot<<<gd,bk,0,s>>>(kckMat2<T>(a), kckMat2<T>(b), kckMat2<T>(c));

	return a;
}

kcMat2f32& Dot(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s) { return _Dot(a, b, c, s);}

template<typename T>
__global__ void kckDotS1(kckMat2<T> a, const kckMat2<T> b, const kckMat2<T> c)
{
	// declare shared memory
	extern __shared__ T sm[];

	// init global idx
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);

	// init local idx
	int j1 = threadIdx.x;
	int j2 = threadIdx.y;

	// init size of k-axis
	int nk  = c.n1;
	int nk1 = blockDim.x;
	int nk2 = (nk - 1)/nk1 + 1;
	
	// init shared memory block
	kckMat2<T> sb(sm          , nk1, nk1);
	kckMat2<T> sc(sm + nk1*nk1, nk1, nk1);

	// calc multification
	T val = 0;
		
	for(int k2 = 0; k2 < nk2; ++k2)
	{
		// copy to shared memory
		const int kstp = nk1*k2;
		const int bi2  = j2 + kstp;
		const int ci1  = j1 + kstp;

		__syncthreads();

		if( i1 < b.n1 && bi2 < b.n2) sb(j1,j2) = b( i1, bi2);
		if(ci1 < c.n1 &&  i2 < c.n2) sc(j1,j2) = c(ci1,  i2);

		__syncthreads();
	
		// calc mma
		if(i1 < a.n1 && i2 < a.n2)
		for(int k1 = 0; k1 < nk1; ++k1)		
		{
			if(k1 + nk1*k2 < nk) val += sb(j1,k1)*sc(k1,j2);
		}
	}

	// set the result
	if(i1 < a.n1 && i2 < a.n2) a(i1,i2) = val;
}

template<typename T>
kcMat2<T>& _DotS1(kcMat2<T>& a, const kcMat2<T>& b, const kcMat2<T>& c, cudaStream_t s)
{
	// check size
	ASSERTFA(b.N2() == c.N1(), "_DotS1 in 147 of kc7Mat.cu");
	
	if(a.N1() != b.N1() || a.N2() != c.N2()) a.Recreate(b.N1(), c.N2());

	// init parameters.. a(m,n) = b(m,k)*c(k,n)
	int m = a.N1(), n = a.N2(), k = b.N2();

	// call kernel	
	dim3 bk = {32, 32, 1u};
	dim3 gd = {(m - 1)/bk.x + 1u, (n - 1)/bk.y + 1u, 1u};

	uint sm_byte = bk.x*bk.y*sizeof(T)*2;

	kckDotS1<<<gd,bk,sm_byte,s>>>(kckMat2<T>(a), kckMat2<T>(b), kckMat2<T>(c));

	return a;
}

kcMat2f32& DotS1(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s) { return _DotS1(a, b, c, s);}

// shared memory version with the cutluss architecture
template<typename T, int mtile, int ntile, int ktile, int mwrp, int nwrp, int mthd, int nthd>
__global__ void kckDotS2(kckMat2<T> a, const kckMat2<T> b, const kckMat2<T> c)
{
	// declare shared memory
	extern __shared__ T sm[];

	// init parameters	
	const int m = a.n1, n = a.n2, k = b.n2;	
	const int mwp2 = (mtile - 1)/mwrp + 1; // mtile = 128, mwp2 = 2	
	const int nreg = 4;

	// init idx	
	const int imb = blockIdx.x*mtile;
	const int inb = blockIdx.y*ntile;
	const int ith = threadIdx.x, nth = blockDim.x; // idx of thread, nth = 128

	const int iwp  = ith/32;       // 0~3
	const int ith1 = ith - 32*iwp; // 0~31

	const int iwp2 = iwp/mwp2;
	const int iwp1 = iwp - mwp2*iwp2;	

	const int imw = iwp1*mwrp;
	const int inw = iwp2*nwrp;
		
	const int mtrg = mthd/nreg;
	const int inth = (ith1/mtrg)*nreg;
	const int imth = (ith1 - mtrg*(ith1/mtrg))*nreg;

	// init shared memory block
	kckMat2<T> sb(sm          , mtile    , ktile);
	kckMat2<T> sc(sb.End() + 1, ktile + 1, ntile); // ktile + 1 for decreasing bank conflict

	// init regisiter file
	T sum[8][8] = {0,}, bfrg[8], cfrg[8];

	// main loop... kblock	
	for(int ikb = 0; ikb < k; ikb += ktile)
	{
		__syncthreads();
		// load B, C tiles to shared memory
#pragma unroll
		for(int ikt = 0; ikt < ktile; ++ikt)
		{			
			int im = imb + ith, ik = ikb + ikt;
			
			if(im < m && ik < k) sb(ith, ikt) = b(im, ik); // nth = mtile
		
			int ic1 = ith + nth*ikt;
			int ic2 = ic1/ktile; ic1 -= ic2*ktile;

			int in = inb + ic2; ik = ikb + ic1;
		
			if(ic2 < ntile) if(ik < k && in < n) sc(ic1,ic2) = c(ik, in);
		}
		__syncthreads();

		// main loop... ktile
#pragma unroll
		for(int ikt = 0; ikt < ktile && ikt + ikb < k; ++ikt)
		{
			// load a tile from shared memory to registers
			int imwt = imw + imth;
			//bfrg[0] = sb(imwt    , ikt);
			//bfrg[1] = sb(imwt + 1, ikt);
			//bfrg[2] = sb(imwt + 2, ikt);
			//bfrg[3] = sb(imwt + 3, ikt);
			*((int4*) bfrg) = *((int4*)&sb(imwt,ikt));

			imwt += mthd;
			//bfrg[4] = sb(imwt    , ikt);
			//bfrg[5] = sb(imwt + 1, ikt);
			//bfrg[6] = sb(imwt + 2, ikt);
			//bfrg[7] = sb(imwt + 3, ikt);
			*((int4*)&bfrg[4]) = *((int4*)&sb(imwt,ikt));

			int inwt = inw + inth;
			cfrg[0] = sc(ikt, inwt    );
			cfrg[1] = sc(ikt, inwt + 1);
			cfrg[2] = sc(ikt, inwt + 2);
			cfrg[3] = sc(ikt, inwt + 3);

			inwt += nthd;
			cfrg[4] = sc(ikt, inwt    );
			cfrg[5] = sc(ikt, inwt + 1);
			cfrg[6] = sc(ikt, inwt + 2);
			cfrg[7] = sc(ikt, inwt + 3);

			// main loop
#pragma unroll
			for(int i = 0; i < 8; ++i)
#pragma unroll
			for(int j = 0; j < 8; ++j)
			{
				sum[j][i] += bfrg[i]*cfrg[j];
			}
		}
	}

	// write result	
	const int im = imb + imw + imth;
	const int in = inb + inw + inth;

#pragma unroll		
	for(int i = 0; i < 8; ++i)
	{
		const int ima = im + ((i < 4) ? i:(i + mthd - 4));

		if(ima >= m) break;
#pragma unroll		
		for(int j = 0; j < 8; ++j)
		{
			const int ina = in + ((j < 4) ? j:(j + nthd - 4));

			if(ina >= n) break;
	
			a(ima, ina) = sum[j][i];
		}
	}
}

template<typename T>
kcMat2<T>& _DotS2(kcMat2<T>& a, const kcMat2<T>& b, const kcMat2<T>& c, cudaStream_t s)
{
	// check size
	ASSERTFA(b.N2() == c.N1(), "_DotS2 in 255 of kc7Mat.cu");
	
	if(a.N1() != b.N1() || a.N2() != c.N2()) a.Recreate(b.N1(), c.N2());

	// init parameters.. a(m,n) = b(m,k)*c(k,n)
	int m = a.N1(), n = a.N2(), k = b.N2();

	const int mtile = 128, ntile = 64, ktile = 16;
	const int mwrp  =  64,  nwrp = 32;
	const int mthd  =  32,  nthd = 16; // mthd*nthd = nidx*nreg

	// call kernel... a = b*c
	dim3 bk = {128u, 1u, 1u};
	dim3 gd = {(m - 1)/mtile + 1u, (n - 1)/ntile + 1u, 1u};

	uint sm_byte = (mtile*ktile + (ktile+1)*ntile)*sizeof(T);

	kckDotS2<float,mtile,ntile,ktile,mwrp,nwrp,mthd,nthd><<<gd,bk,sm_byte,s>>>
		    (kckMat2<T>(a), kckMat2<T>(b), kckMat2<T>(c));

	return a;
}

kcMat2f32& DotS2(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s) { return _DotS2(a, b, c, s);}

#ifdef __CUDA_IMMA__

// tensor core version
template<int mtile, int ntile, int ktile, int mwrp, int nwrp, int mwm, int nwm>
__global__ void kckDotS3(kckMat2f32 a, const kckMat2f32 b, const kckMat2f32 c)
{
	// declare shared memory
	extern __shared__ half sm0[];

	// init parameters	
	const int m = a.n1, n = a.n2, k = b.n2;	
	const int mwp2 = (mtile - 1)/mwrp + 1; // mtile = 128, mwp2 = 2		

	// init idx	
	const int imb = blockIdx.x*mtile;
	const int inb = blockIdx.y*ntile;
	const int ith = threadIdx.x, nth = blockDim.x; // idx of thread, nth = 128

	const int iwp  = ith/32;       // 0~3
	const int iwp2 = iwp/mwp2;
	const int iwp1 = iwp - mwp2*iwp2;

	const int imw = iwp1*mwrp;
	const int inw = iwp2*nwrp;

	const int loop_mk = (mtile*ktile - 1)/nth + 1;
	const int loop_kn = (ktile*ntile - 1)/nth + 1;
	
	// init shared memory block
	kckMat2f16 sb (sm0       , ktile  , mtile);
	kckMat2f16 sb0(sb.End()+1, mtile+2, ktile);
	kckMat2f16 sc (sb.End()+1, ktile  , ntile);

	wmma::fragment<wmma::matrix_a   , 16,16,16, half, wmma::row_major> bmat[mwm];
	wmma::fragment<wmma::matrix_b   , 16,16,16, half, wmma::col_major> cmat;
	wmma::fragment<wmma::accumulator, 16,16,16, float>                 amat[nwm][mwm];

#pragma unroll
	for(int j = 0; j < nwm; ++j) 
#pragma unroll
	for(int i = 0; i < mwm; ++i)
	{	
		wmma::fill_fragment(amat[j][i], 0.f);
	}
	
	// main loop... kblock	
	for(int ikb = 0; ikb < k; ikb += ktile)
	{
		__syncthreads();
		// load B tiles to shared memory... with avoiding bank conflict
#pragma unroll
		for(int loop = 0, idx = ith; loop < loop_mk; ++loop, idx += nth)
		{
			const int i2 = idx/mtile, i1 = idx - mtile*i2;

			if(i2 >= ktile) break;

			const int im = imb + i1 , ik = ikb + i2; 
				
			sb0(i1,i2) = (im < m && ik < k) ? __float2half(b(im, ik)) : (half)0;
		}
		__syncthreads();
#pragma unroll
		for(int loop = 0, idx = ith; loop < loop_mk; ++loop, idx += nth)
		{
			const int i2 = idx/ktile, i1 = idx - ktile*i2;

			if(i2 >= mtile) break;
		
			sb(i1,i2) = sb0(i2,i1);
		}

		// load C tiles to shared memory
#pragma unroll
		for(int loop = 0, idx = ith; loop < loop_kn; ++loop, idx += nth)
		{
			const int i2 = idx/ktile, i1 = idx - ktile*i2;

			if(i2 >= ntile) break;

			const int ik = ikb + i1 , in = inb + i2; 

			sc(i1,i2) = (ik < k && in < n) ? __float2half(c(ik, in)) : (half)0;
		}
		__syncthreads();

		// load fragment
#pragma unroll
		for(int i = 0; i < mwm; ++i) wmma::load_matrix_sync(bmat[i], &sb(0, imw + i*16), 16);

		// main loop
#pragma unroll
		for(int j = 0; j < nwm; ++j)
		{
			// load fragment
			wmma::load_matrix_sync(cmat, &sc(0, inw + j*16), 16);
#pragma unroll
			for(int i = 0; i < mwm; ++i)
			{	
				// calc mma
				wmma::mma_sync(amat[j][i], bmat[i], cmat, amat[j][i]);
			}
		}
	}

	// store result to direct global mem
	//for(int j = 0; j < 2; ++j)
	//{
	//	float* pa = &a(imb + imw, inb + inw + j*16);
	//
	//	for(int i = 0; i < 4; ++i, pa += 16)
	//	{	
	//		wmma::store_matrix_sync(pa, amat[j][i], a.p1, wmma::mem_col_major);
	//	}
	//}

	// store result	via smem
	const kckMat2f32 sa((float*)sm0, mwrp, nwrp);
	
	const int loop_mn = (mwrp*nwrp - 1)/nth + 1;
	const int n_wrp   = nth/32;
	const int n_wrpm  = mtile/mwrp;
	
	for(int iwrp = 0; iwrp < n_wrp; ++iwrp)
	{
		// copyt to smem
		__syncthreads();
	
		if(iwp == iwrp)
		for(int j = 0; j < nwm; ++j)
		{
			float* pa = &sa(0,j*16);
		
			for(int i = 0; i < mwm; ++i, pa += 16)
			{	
				wmma::store_matrix_sync(pa, amat[j][i], sa.p1, wmma::mem_col_major);
			}
		}
		__syncthreads();
	
		// copy to global
		
		const int iwrp2 = iwrp/n_wrpm;
		const int iwrp1 = iwrp - n_wrpm*iwrp2;
	
		int im0 = imb + mwrp*iwrp1, in0 = inb + nwrp*iwrp2;
	
		for(int loop = 0, idx = ith; loop < loop_mn; ++loop, idx += nth)
		{
			const int i2 = idx/mwrp, i1 = idx - mwrp*i2;
	
			const int im = im0 + i1 , in = in0 + i2;
	
			if(i2 < nwrp) if(im < m && in < n) a(im,in) = sa(i1,i2);
		}
	}
}	

template<typename T>
kcMat2<T>& _DotS3(kcMat2<T>& a, const kcMat2<T>& b, const kcMat2<T>& c, cudaStream_t s)
{
	// check size
	ASSERTFA(b.N2() == c.N1(), "_DotS2 in 255 of kc7Mat.cu");
	
	if(a.N1() != b.N1() || a.N2() != c.N2()) a.Recreate(b.N1(), c.N2());

	// init parameters.. a(m,n) = b(m,k)*c(k,n)
	int m = a.N1(), n = a.N2(), k = b.N2();

	const int mtile = 128, ntile = 128, ktile = 16;
	const int mwrp  = 32,  nwrp  = 32;

	const int mwm =  mwrp/16, nwm = nwrp/16;

	const int nth = (mtile/mwrp)*(ntile/nwrp)*32;

	// call kernel... a = b*c
	dim3 bk = {(uint)nth, 1u, 1u};
	dim3 gd = {(m - 1)/mtile + 1u, (n - 1)/ntile + 1u, 1u};

	PRINTF_DIM(bk,gd);
	PRINTFA("\n");

	uint sm0_byte = (mtile*ktile + ktile*ntile    )*sizeof(half);
	uint sm1_byte = (mtile*ktile + ktile*(mtile+2))*sizeof(half);
	uint sm2_byte = (mwrp+16)*nwrp*sizeof(float);
	uint sm_byte = MAX(MAX(sm0_byte,sm1_byte),sm2_byte);

	kckDotS3<mtile,ntile,ktile,mwrp,nwrp,mwm,nwm><<<gd,bk,sm_byte,s>>>
		    (kckMat2<T>(a), kckMat2<T>(b), kckMat2<T>(c));

	return a;
}

kcMat2f32& DotS3(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s) { return _DotS3(a, b, c, s);}
#endif

////////////////////////////////////////////////////////
// dot product and add... a = b*c + d

template<typename T>
__global__ void kckDotAdd(kckMat2<T> a, const kckMat2<T> b, 
	                const kckMat2<T> c, const kckMat2<T> d)
{
	// init idx
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);

	if(i1 >= a.n1 || i2 >= a.n2) return;
	
	// calc multification
	T val = 0;

	for(int j = 0; j < c.n1; ++j) val += b(i1, j)*c(j, i2);

	// calc addition
	val += d(i1,i2);

	// set the result
	a(i1,i2) = val;
}

template<typename T>
__global__ void kckDotAdd(kckMat3<T> a, const kckMat3<T> b, 
	                const kckMat3<T> c, const kckMat3<T> d)
{
	// init idx
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);
	int i3 = GetStdIdx3(blockIdx);

	if(i1 >= a.n1 || i2 >= a.n2) return;
	
	// calc multification
	T val = 0;

	for(int j = 0; j < c.n1; ++j) val += b(i1, j, i3)*c(j, i2, i3);

	// calc addition
	val += d(i1,i2,i3);

	// set the result
	a(i1,i2,i3) = val;
}

template<typename T>
__global__ void kckDotAdd(kckMat4<T> a, const kckMat4<T> b, 
	                const kckMat4<T> c, const kckMat4<T> d)
{
	// init idx
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);
	int i3 = GetStdIdx3(blockIdx);
	int i4 = GetStdIdx4(i3, a.n3);

	if(i1 >= a.n1 || i2 >= a.n2) return;
	
	// calc multification
	T val = 0;

	for(int j = 0; j < c.n1; ++j) val += b(i1, j, i3, i4)*c(j, i2, i3, i4);

	// calc addition
	val += d(i1,i2,i3,i4);

	// set the result
	a(i1,i2,i3,i4) = val;
}

template<typename T>
kcMat2<T>& _DotAdd(kcMat2<T>& a, const kcMat2<T>& b, 
             const kcMat2<T>& c, const kcMat2<T>& d, cudaStream_t s)
{
	// check size
	ASSERTFA(b.N2() == c.N1(), "_MultiAdd in 185 of kc7Mat.cu");
	ASSERTFA(b.N1() == d.N1(), "_MultiAdd in 186 of kc7Mat.cu");
	ASSERTFA(c.N2() == d.N2(), "_MultiAdd in 187 of kc7Mat.cu");

	if(a.N1() != b.N1() || a.N2() != c.N2()) a.Recreate(b.N1(), c.N2());

	// call kernel
	dim3 bk, gd; a.GetBkGd(bk, gd);

	kckDotAdd<<<gd,bk,0,s>>>(kckMat2<T>(a), kckMat2<T>(b), kckMat2<T>(c), kckMat2<T>(d));

	return a;
}

template<typename T>
kcMat3<T>& _DotAdd(kcMat3<T>& a, const kcMat3<T>& b, 
             const kcMat3<T>& c, const kcMat3<T>& d, cudaStream_t s)
{
	// check size
	ASSERTFA(b.N2() == c.N1(), "_MultiAdd in 218 of kc7Mat.cu");
	ASSERTFA(b.N1() == d.N1(), "_MultiAdd in 219 of kc7Mat.cu");
	ASSERTFA(c.N2() == d.N2(), "_MultiAdd in 220 of kc7Mat.cu");
	ASSERTFA(b.N3() == c.N3(), "_MultiAdd in 221 of kc7Mat.cu");
	ASSERTFA(b.N3() == d.N3(), "_MultiAdd in 222 of kc7Mat.cu");

	if(a.N1() != b.N1() || a.N2() != c.N2() || a.N3() != b.N3())
	{
		a.Recreate(b.N1(), c.N2(), b.N3());
	}

	// call kernel
	dim3 bk, gd; a.GetBkGd(bk, gd);

	kckDotAdd<<<gd,bk,0,s>>>(kckMat3<T>(a), kckMat3<T>(b), kckMat3<T>(c), kckMat3<T>(d));

	return a;
}

template<typename T>
kcMat4<T>& _DotAdd(kcMat4<T>& a, const kcMat4<T>& b, 
             const kcMat4<T>& c, const kcMat4<T>& d, cudaStream_t s)
{
	// check size
	ASSERTFA(b.N2() == c.N1(), "_MultiAdd in 257 of kc7Mat.cu");
	ASSERTFA(b.N1() == d.N1(), "_MultiAdd in 258 of kc7Mat.cu");
	ASSERTFA(c.N2() == d.N2(), "_MultiAdd in 259 of kc7Mat.cu");
	ASSERTFA(b.N3() == c.N3(), "_MultiAdd in 260 of kc7Mat.cu");
	ASSERTFA(b.N3() == d.N3(), "_MultiAdd in 261 of kc7Mat.cu");
	ASSERTFA(b.N4() == c.N4(), "_MultiAdd in 262 of kc7Mat.cu");
	ASSERTFA(b.N4() == d.N4(), "_MultiAdd in 263 of kc7Mat.cu");

	if(a.N1() != b.N1() || a.N2() != c.N2() || a.N3() != b.N3() || a.N4() != b.N4())
	{
		a.Recreate(b.N1(), c.N2(), b.N3(), b.N4());
	}

	// call kernel
	dim3 bk, gd; a.GetBkGd(bk, gd);

	kckDotAdd<<<gd,bk,0,s>>>(kckMat4<T>(a), kckMat4<T>(b), kckMat4<T>(c), kckMat4<T>(d));

	return a;
}

kcMat2f32& DotAdd(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, const kcMat2f32& d, cudaStream_t s) { return _DotAdd(a, b, c, d, s);}
kcMat3f32& DotAdd(kcMat3f32& a, const kcMat3f32& b, const kcMat3f32& c, const kcMat3f32& d, cudaStream_t s) { return _DotAdd(a, b, c, d, s);}
kcMat4f32& DotAdd(kcMat4f32& a, const kcMat4f32& b, const kcMat4f32& c, const kcMat4f32& d, cudaStream_t s) { return _DotAdd(a, b, c, d, s);}

kcMat2f32 DotAdd(const kcMat2f32& b, const kcMat2f32& c, const kcMat2f32& d, cudaStream_t s) { kcMat2f32 a; _DotAdd(a, b, c, d, s); return a;}
kcMat3f32 DotAdd(const kcMat3f32& b, const kcMat3f32& c, const kcMat3f32& d, cudaStream_t s) { kcMat3f32 a; _DotAdd(a, b, c, d, s); return a;}
kcMat4f32 DotAdd(const kcMat4f32& b, const kcMat4f32& c, const kcMat4f32& d, cudaStream_t s) { kcMat4f32 a; _DotAdd(a, b, c, d, s); return a;}

////////////////////////////////////////////////////////
// Sum... a = sum(b, dim1, dim2) using reduction

template<typename T>
__global__ void kckSum(kckMat1<T> a, const kckMat3<T> b)
{
	// declare shared memory
	extern __shared__ float sm[];

	// init index
	KC_RDC_INDEX_EXDM1(b);

	const int i3 = ie1;
	
	// pre-reduction
	T val = 0;	

	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2; if(i1 >= b.n1) continue;

		for(int i2 = 0; i2 < b.n2; ++i2) val += b(i1,i2,i3);
	}
	
	// reduction
	sm[ix1] = val;

	for(int size_s = nx1>>1; size_s > 0; size_s >>= 1)
	{
		__syncthreads();

		if(ix1 < size_s) sm[ix1] += sm[ix1 + size_s];
	}

	// update
	if(ix1 == 0) a(i3) = sm[0];
}

template<typename T>
kcMat1<T>& _Sum(kcMat1<T>& a, const kcMat3<T>& b, cudaStream_t s)
{
	// check size
	if(a.N1() != b.N3()) a.Recreate(b.N3()); 
	
	// call kernel
	dim3 bk, gd; int sm; b.GetBkGdRdc(bk, gd, sm, 3);

	kckSum<<<gd,bk,sm,s>>>(kckMat1<T>(a), kckMat3<T>(b));	

	return a;
}

kcMat1f32& Sum(kcMat1f32& a, const kcMat3f32& b, cudaStream_t s) { return _Sum(a, b, s);}

////////////////////////////////////////////////////////
// Sum and Mean ... T a = sum(b) using reduction

KC_RDC_WHOLE_DEFINITION(_Sum, kcdAddEq, 0)

float Sum(const kcMat1f32& b, cudaStream_t s) { return _Sum(b,s);}
float Sum(const kcMat2f32& b, cudaStream_t s) { return _Sum(b,s);}
float Sum(const kcMat3f32& b, cudaStream_t s) { return _Sum(b,s);}
float Sum(const kcMat4f32& b, cudaStream_t s) { return _Sum(b,s);}

float Mean(const kcMat1f32& b, cudaStream_t s) { return _Sum(b,s)/b.N();}
float Mean(const kcMat2f32& b, cudaStream_t s) { return _Sum(b,s)/b.N();}
float Mean(const kcMat3f32& b, cudaStream_t s) { return _Sum(b,s)/b.N();}
float Mean(const kcMat4f32& b, cudaStream_t s) { return _Sum(b,s)/b.N();}

////////////////////////////////////////////////////////
// variance and standard deviation ... T a = var(b) using reduction

template<typename T> inline
__device__ void kcdAddSq(T& a, const T& b) { a += b*b; }

KC_RDC_WHOLE_DEFINITION_SUM(_SumSq, kcdAddSq, 0)

float SumSq(const kcMat1f32& b, cudaStream_t s) { return _SumSq(b,s);}
float SumSq(const kcMat2f32& b, cudaStream_t s) { return _SumSq(b,s);}
float SumSq(const kcMat3f32& b, cudaStream_t s) { return _SumSq(b,s);}
float SumSq(const kcMat4f32& b, cudaStream_t s) { return _SumSq(b,s);}

float Var(const kcMat1f32& b, cudaStream_t s) { return _SumSq(b,s)/b.N() - powf(Mean(b,s),2.f);}
float Var(const kcMat2f32& b, cudaStream_t s) { return _SumSq(b,s)/b.N() - powf(Mean(b,s),2.f);}
float Var(const kcMat3f32& b, cudaStream_t s) { return _SumSq(b,s)/b.N() - powf(Mean(b,s),2.f);}
float Var(const kcMat4f32& b, cudaStream_t s) { return _SumSq(b,s)/b.N() - powf(Mean(b,s),2.f);}

float Std(const kcMat1f32& b, cudaStream_t s) { return sqrtf(Var(b,s));}
float Std(const kcMat2f32& b, cudaStream_t s) { return sqrtf(Var(b,s));}
float Std(const kcMat3f32& b, cudaStream_t s) { return sqrtf(Var(b,s));}
float Std(const kcMat4f32& b, cudaStream_t s) { return sqrtf(Var(b,s));}

////////////////////////////////////////////////////////
// Max... T a = Max(a, b) using reduction

template<typename T> inline
__device__ void kcdMax(T& a, const T& b) { if(a < b) a = b; }

KC_RDC_WHOLE_DEFINITION(_Max, kcdMax, FLOAT_MIN)

float Max(const kcMat1f32& b, cudaStream_t s) { return _Max(b,s);}
float Max(const kcMat2f32& b, cudaStream_t s) { return _Max(b,s);}
float Max(const kcMat3f32& b, cudaStream_t s) { return _Max(b,s);}
float Max(const kcMat4f32& b, cudaStream_t s) { return _Max(b,s);}

////////////////////////////////////////////////////////
// Max... T a = Min(a, b) using reduction

template<typename T> inline
__device__ void kcdMin(T& a, const T& b) { if(a > b) a = b; }

KC_RDC_WHOLE_DEFINITION(_Min, kcdMin, FLOAT_MAX)

float Min(const kcMat1f32& b, cudaStream_t s) { return _Min(b,s);}
float Min(const kcMat2f32& b, cudaStream_t s) { return _Min(b,s);}
float Min(const kcMat3f32& b, cudaStream_t s) { return _Min(b,s);}
float Min(const kcMat4f32& b, cudaStream_t s) { return _Min(b,s);}

////////////////////////////////////////////////////////
// Hilbert transform... a = hilbert(b)
// : b is real signal
// : a is complex or imaginary

// a = hilbert(b) : a is imaginary of b
// n_tap_h is fixed with 32
// boundary method : KFO_BD_NO_CALC
// < (nx-1)/nx1+1, ny, nz), (nx1, 1)>
template<typename T, typename Y>
__global__ void kckHilbert32(T* a, const Y* b, int nx, int a_nxp, int b_nxp)
{
	const int nkh = 32;

	// init index
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, iy = blockIdx.y, iz = blockIdx.z;
	const int nx1 = blockDim .x;
	const int ny  = gridDim  .y;	
	const int ix  = ix1 + nx1*ix2 + nkh;

	// check dimension size
	if(ix > nx - 65) return;

	// init variables
	a += ix + a_nxp*(iy + ny*iz);
	b += ix + b_nxp*(iy + ny*iz);
	
	// hilbert transform
	int i = 1;	float sum = 0;

	sum += HTC_00*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_01*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_02*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_03*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_04*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_05*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_06*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_07*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_08*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_09*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_10*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_11*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_12*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_13*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_14*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_15*(*(b-i) - *(b+i));
		
	// set output
	*(a) = (T) sum;		// imag part
}

// a = hilbert(b) : a is "complex".
// * This is the speicalized template function of the above function with float2
// < (nx-1)/nx1+1, ny, nz), (nx1, 1)>
template<typename T>
__global__ void kckHilbert32(float2* a, const T* b, int nx, int a_nxp, int b_nxp)
{
	const int nkh = 32;

	// init index
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, iy = blockIdx.y, iz  = blockIdx.z;
	const int nx1 = blockDim .x;
	const int ny  = gridDim  .y;	
	const int ix  = ix1 + nx1*ix2 + nkh;

	// check dimension size
	if(ix > nx - 65) return;

	// init variables
	a += ix + a_nxp*(iy + ny*iz);
	b += ix + b_nxp*(iy + ny*iz);
	
	// hilbert transform
	int i   = 1;	float   sum = 0;

	sum += HTC_00*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_01*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_02*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_03*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_04*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_05*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_06*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_07*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_08*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_09*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_10*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_11*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_12*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_13*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_14*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_15*(*(b-i) - *(b+i));
		
	// set output
	a->x = (float) *(b);	// real part	
	a->y =          sum;	// imag part
}

template<typename T, typename Y>
void _Hilbert(kcMat3<T>& a, const kcMat3<Y>& b, cudaStream_t s = 0)
{
	// clear output
	a.SetZero(s);

	// check matrix sizes
	ASSERTFA(a.IsEqualSize(b), "Hilbert in 1184");
			
	// init grid, block size
	const int nx1 = 128;		
	dim3      bk(nx1,1,1);
	dim3      gd((a.N1()-1)/nx1+1, a.N2(), a.N3());
	
	// call cuda kernel
	kckHilbert32<<<gd,bk,0,s>>>(a.P(), b.P(), a.N1(), a.P1(), b.P1());
};

void Hilbert(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s) { _Hilbert(a, b, s); }
void Hilbert(kcMat3c32& a, const kcMat3f32& b, cudaStream_t s) { _Hilbert(a, b, s); }

////////////////////////////////////////////////////////
// Hilbert transform... a = abs(hilbert(b))
// : b is real signal
// : a is amplitude of b

// a = hilbert(b) : a is amplitude of b
// n_tap_h is fixed with 32
// boundary method : KFO_BD_NO_CALC
// < (nx-1)/nx1+1, ny, nz), (nx1, 1)>
template<typename T, typename Y>
__global__ void kckHilbertAbs32(T* a, const Y* b, int nx, int a_nxp, int b_nxp)
{
	const int nkh = 32;

	// init index
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, iy = blockIdx.y, iz  = blockIdx.z;
	const int nx1 = blockDim .x;
	const int ny  = gridDim  .y;	
	const int ix  = ix1 + nx1*ix2 + nkh;

	// check dimension size
	if(ix > nx - 65) return;

	// init variables
	a += ix + a_nxp*(iy + ny*iz);
	b += ix + b_nxp*(iy + ny*iz);
	
	// hilbert transform
	int i = 1; float sum = 0;

	sum += HTC_00*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_01*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_02*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_03*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_04*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_05*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_06*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_07*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_08*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_09*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_10*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_11*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_12*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_13*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_14*(*(b-i) - *(b+i)); i+=2;
	sum += HTC_15*(*(b-i) - *(b+i));
		
	// set output
	*(a) = (T) hypotf(sum, (float) (*b)); // amplitude
}

template<typename T>
void _HilbertAbs(kcMat3<T>& a, const kcMat3<T>& b, cudaStream_t s = 0)
{
	// clear output
	a.SetZero(s);

	// check matrix sizes
	ASSERTFA(a.IsEqualSize(b), "Hilbert in 1184");
			
	// init grid, block size
	const int nx1 = 128;		
	dim3      bk(nx1,1,1);
	dim3      gd((a.N1()-1)/nx1+1, a.N2(), a.N3());
	
	// call cuda kernel
	kckHilbertAbs32<<<gd,bk,0,s>>>(a.P(), b.P(), a.N1(), a.P1(), b.P1());
};

void HilbertAbs(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s) { _HilbertAbs(a, b, s); }

////////////////////////////////////////////////////////
// IQ Demodulation 

// a = IqDemodulate(b) : a is "complex".
// * This is the speicalized template function of the above function with float2
// < (nx-1)/nx1+1, ny, nz), (nx1, 1)>
template<typename T>
__global__ void kckIqDemodulate(float2* a, const T* b, int nx, int a_nxp, int b_nxp, float w_rad)
{
	// init index
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, iy = blockIdx.y, iz  = blockIdx.z;
	const int nx1 = blockDim .x;
	const int ny  = gridDim  .y;	
	const int ix  = ix1 + nx1*ix2;

	// check dimension size
	if(ix >= nx) return;

	// init variables
	a += ix + a_nxp*(iy + ny*iz);
	b += ix + b_nxp*(iy + ny*iz);
	
	// calc cos and sin	
	float cos, sin;

	__sincosf(w_rad*ix, &sin, &cos);
		
	// set output
	const float bv = *(b);

	a->x =  bv*cos;	// real part
	a->y = -bv*sin;	// imag part
}

// a = IqDemodulate(b) : a is complex if T is float2.
//                  otherwise, a is imaginary of b	
// fs_MHz : sampling rate 
// fd_MHz : demodulating freq	
template<typename T>
void _IqDemodulate(kcMat3c32& a, const kcMat3<T>& b, float fs_MHz, float fd_MHz, cudaStream_t s = 0)
{
	// clear output
	a.SetZero(s);

	// check matrix sizes
	ASSERTFA(a.IsEqualSize(b), "IqDemodulate in 1411");

	// init parameters
	const float w_rad = 2.f*PIf*fd_MHz/fs_MHz;
			
	// init grid, block size
	const int nx1 = 128;		
	dim3      bk(nx1,1,1);
	dim3      gd((a.N1()-1)/nx1+1, a.N2(), a.N3());
	
	// call cuda kernel
	kckIqDemodulate<<<gd,bk,0,s>>>
					  (a.Begin(), b.Begin(), a.N1(), a.P1(), b.P1(), w_rad);
}

void IqDemodulate(kcMat3c32& a, const kcMat3f32& b, float fs_MHz, float fd_MHz, cudaStream_t s)
{
	_IqDemodulate(a, b, fs_MHz, fd_MHz, s);
}

////////////////////////////////////////////////////////
// Filtering

// a(x,y,z) = filter(b(x,y,z), k(nk,1))
// a_nxp : the pitch of a
// b_nxp : the pitch of b
// boundary method : KFO_BD_NO_CALC
// < ((nx-1)/nx1+1, ny, nz), (nx1,1) > : loop(nk)
template <typename U>
__global__ void kckFilter1DX1_N(float2* a, const float2* b, const U* k, int nx, int nk,
	                          int nkh, int a_nxp, int b_nxp)
{
	// init index
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, iy = blockIdx.y, iz  = blockIdx.z;
	const int nx1 = blockDim .x;
	const int ny  = gridDim  .y;

	const int ix  = ix1 + nx1*ix2;
	
	// check dimension size
	if(ix < nkh || ix >= nx - nkh) return;

	// init parameters
	const int iyz    = iy + ny*iz;

	a += ix + a_nxp*iyz;
	b += ix + b_nxp*iyz - nkh;
	
	// init variables			
	float2 sum = {0, 0};
	for(int ik = 0; ik < nk; ++ik)
	{
		// sum of data
		sum.x += *(k  ) * (b  )->x;
		sum.y += *(k++) * (b++)->y;
	}
	*(a) = (sum);	
}

// a(x,y,z) = filter(b(x,y,z), k(nk,1))
// a_nxp : the pitch of a
// b_nxp : the pitch of b
// boundary method : KFO_BD_NO_CALC
// < ((nx-1)/nx1+1, ny, nz), (nx1,1) > : loop(nk)
template <typename T, typename Y, typename U>
__global__ void kckFilter1DX1_N(T* a, const Y* b, const U* k, int nx, int nk, 
	                          int nkh, int a_nxp, int b_nxp)
{
	// init index
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, iy = blockIdx.y, iz  = blockIdx.z;
	const int nx1 = blockDim .x;
	const int ny  = gridDim  .y;

	const int ix  = ix1 + nx1*ix2;
	
	// check dimension size
	if(ix < nkh || ix >= nx - nkh) return;

	// init parameters
	const int iyz = iy + ny*iz;

	a += ix + a_nxp*iyz;
	b += ix + b_nxp*iyz - nkh;
	
	// init variables			
	T sum = 0;	
	for(int ik = 0; ik < nk; ++ik)
	{
		// sum of data
		sum += *(k++) * *(b++);
	}
	*(a) = (sum);	
}

// a = filtering b with 1D kernel k(nk, 1) in x1 axis
// boundary method : KFO_BD_NO_CALC
template<typename T, typename Y, typename U>
void _Filter1DX1_N(kcMat2<T>& a, const kcMat2<Y>& b, const kcMat1<U>& k, cudaStream_t s = 0)
{
	// check matrix sizes
	ASSERTFA(a.IsEqualSize(b),"Filter1DX1");

	// init parameters
	const int nk  = k.N1();
	const int nkh = (nk-1)/2;

	// init grid, block size
	const int nx1 = 128;
	dim3      bk(nx1,1,1);
	dim3      gd( (a.N1()-1)/nx1+1, a.N2(), 1);

	// call cuda kernel
	kckFilter1DX1_N<<<gd,bk,0,s>>>(a.Begin(), b.Begin(), k.Begin(), a.N1(), nk, nkh, a.P1(), b.P1());
}

void Filter1DX1_N(kcMat2f32& a, const kcMat2f32& b, const kcMat1f32& k, cudaStream_t s) { _Filter1DX1_N(a, b, k, s);}
void Filter1DX1_N(kcMat2c32& a, const kcMat2c32& b, const kcMat1f32& k, cudaStream_t s) { _Filter1DX1_N(a, b, k, s);}

////////////////////////////////////////////////////////
// down sampling

// a = downsample(b)
// < ((nx-1)/nx1+1, (ny-1)/ny1+1, nz), (nx1,ny1) >
template<typename T>
__global__ void kckDownsample(T* a, int nx, int ny, int nxp, const T* b, int b_nxp, int downrate)
{
	// init index
	const int ix1 = threadIdx.x, iy1 = threadIdx.y;
	const int ix2 = blockIdx .x, iy2 = blockIdx .y;
	const int iz  = blockIdx .z;
	const int nx1 = blockDim .x, ny1 = blockDim .y;
	const int ix  = ix1 + nx1*ix2;
	const int iy  = iy1 + ny1*iy2;
	
	// check dimension size
	if(ix >= nx || iy >= ny) return;
	
	// init parameters		
	const int iyz   = (iy + ny*iz);
	const int a_idx = ix +   nxp*iyz;
	const int b_idx = ix*downrate + b_nxp*iyz;
	
	// downsampling
	*(a + a_idx) = *(b + b_idx);
}

// Downsample(a, b, downrate)
template<typename T>
void _Downsample(kcMat3<T>& a, const kcMat3<T>& b, const int downrate, cudaStream_t s = 0)
{
	// check size
	ASSERTFA(downrate > 0,"Downsample in 3807");
	ASSERTFA(a.N1() == b.N1()/downrate && a.N2() == b.N2() && a.N3() == b.N3(),"Downsample in 3808");

	// init grid, block size
	const int nx1 = 128;
	const int ny1 = 1;
	dim3      bk(nx1, ny1,1);
	dim3      gd( (a.N1()-1)/nx1+1, (a.N2()-1)/ny1+1, a.N3());

	// call cuda kernel		
	kckDownsample<<<gd,bk,0,s>>>(a.Begin(), a.N1(), a.N2(), a.P1(), b.Begin(), b.P1(), downrate);
}

void Downsample(kcMat3f32& a, const kcMat3f32& b, const int downrate, cudaStream_t s) { _Downsample(a, b, downrate, s); };
void Downsample(kcMat3c32& a, const kcMat3c32& b, const int downrate, cudaStream_t s) { _Downsample(a, b, downrate, s); };

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// loop functions for lambda function

//////////////////////////
// for one argument

template<typename T, class L>
__global__ void kckForeach(kckArr<T> a, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);	

	if(i1 >= a.size) return;

	lambdafun(a(i1));
}

template<typename T, class L>
__global__ void kckForeach(kckMat1<T> a, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);	

	if(i1 >= a.n1) return;

	lambdafun(a(i1));
}

template<typename T, class L>
__global__ void kckForeach(kckMat2<T> a, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2));
}

template<typename T, class L>
__global__ void kckForeach(kckMat3<T> a, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);
	int i3 = GetStdIdx3(blockIdx);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2,i3));
}

template<typename T, class L>
__global__ void kckForeach(kckMat4<T> a, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);
	int i3 = GetStdIdx3(blockIdx);
	int i4 = GetStdIdx4(i3, a.n3);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2,i3,i4));
}

template<template<typename> class M, typename T, class L>
void Foreach(M<T>& a, const L lambdafun, cudaStream_t s = 0)
{
	// check size
	if(a.N() == 0) return;

	// check noblank
	if(a.IsNoBlank())
	{
		// call kernel with down-dimesion
		kcMat1<T> _a(a.P(), a.N());

		dim3 bk, gd; _a.GetBkGd(bk, gd);

		kckForeach<<<gd,bk,0,s>>>(kckMat(_a), lambdafun);
	}
	else
	{
		// call kernel
		dim3 bk, gd; a.GetBkGd(bk, gd);

		kckForeach<<<gd,bk,0,s>>>(kckMat(a), lambdafun);
	}
}

//////////////////////////
// for two argument

template<typename T, typename Y, class L>
__global__ void kckForeach(kckArr<T> a, kckArr<Y> b, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);	

	if(i1 >= a.size) return;

	lambdafun(a(i1), b(i1));
}

template<typename T, typename Y, class L>
__global__ void kckForeach(kckMat1<T> a, kckMat1<Y> b, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);	

	if(i1 >= a.n1) return;

	lambdafun(a(i1), b(i1));
}

template<typename T, typename Y, class L>
__global__ void kckForeach(kckMat2<T> a, kckMat2<Y> b, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2), b(i1,i2));
}

template<typename T, typename Y, class L>
__global__ void kckForeach(kckMat3<T> a, kckMat3<Y> b, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);
	int i3 = GetStdIdx3(blockIdx);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2,i3), b(i1,i2,i3));
}

template<typename T, typename Y, class L>
__global__ void kckForeach(kckMat4<T> a, kckMat4<Y> b, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);
	int i3 = GetStdIdx3(blockIdx);
	int i4 = GetStdIdx4(i3, a.n3);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2,i3,i4), b(i1,i2,i3,i4));
}

template<template<typename> class M, typename T, typename Y, class L>
void Foreach(M<T>& a, const M<Y>& b, const L lambdafun, cudaStream_t s = 0)
{
	// check size
	if(!a.IsEqualSizeDim(b)) a.Recreate(b);

	// check noblank
	if(a.IsNoBlank() && b.IsNoBlank())
	{
		// call kernel with down-dimension
		kcMat1<T> _a(a.P(), a.N());
		kcMat1<Y> _b(b.P(), b.N());
	
		dim3 bk, gd; _a.GetBkGd(bk, gd);
	
		kckForeach<<<gd,bk,0,s>>>(kckMat(_a), kckMat(_b), lambdafun);
	}
	else
	{
		// call kernel
		dim3 bk, gd; a.GetBkGd(bk, gd);

		kckForeach<<<gd,bk,0,s>>>(kckMat(a), kckMat(b), lambdafun);
	}
}

//////////////////////////
// for three argument

template<typename T, typename Y, typename U, class L>
__global__ void kckForeach(kckArr<T> a, kckArr<Y> b, kckArr<U> c, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);

	if(i1 >= a.size) return;

	lambdafun(a(i1), b(i1), c(i1));
}

template<typename T, typename Y, typename U, class L>
__global__ void kckForeach(kckMat1<T> a, kckMat1<Y> b, kckMat1<U> c, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);	

	if(i1 >= a.n1) return;

	lambdafun(a(i1), b(i1), c(i1));
}

template<typename T, typename Y, typename U, class L>
__global__ void kckForeach(kckMat2<T> a, kckMat2<Y> b, kckMat2<U> c, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2), b(i1,i2), c(i1,i2));
}

template<typename T, typename Y, typename U, class L>
__global__ void kckForeach(kckMat3<T> a, kckMat3<Y> b, kckMat3<U> c, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);
	int i3 = GetStdIdx3(blockIdx);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2,i3), b(i1,i2,i3), c(i1,i2,i3));
}

template<typename T, typename Y, typename U, class L>
__global__ void kckForeach(kckMat4<T> a, kckMat4<Y> b, kckMat4<U> c, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);
	int i3 = GetStdIdx3(blockIdx);
	int i4 = GetStdIdx4(i3, a.n3);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2,i3,i4), b(i1,i2,i3,i4), c(i1,i2,i3,i4));
}

template<template<typename> class M, typename T, typename Y, typename U, class L>
void Foreach(M<T>& a, const M<Y>& b, const M<U>& c, const L lambdafun, cudaStream_t s = 0)
{
	// check size
	ASSERTFA(b.IsEqualSizeDim(c), "foreach in 1831");
	if(!a.IsEqualSizeDim(b)) a.Recreate(b);	

	// check noblank
	if(a.IsNoBlank() && b.IsNoBlank() && c.IsNoBlank())
	{
		// call kernel with down-dimension
		kcMat1<T> _a(a.P(), a.N());
		kcMat1<Y> _b(b.P(), b.N());
		kcMat1<U> _c(c.P(), c.N());

		dim3 bk, gd; _a.GetBkGd(bk, gd);

		kckForeach<<<gd,bk,0,s>>>(kckMat(_a), kckMat(_b), kckMat(_c), lambdafun);
	}
	else
	{
		// call kernel
		dim3 bk, gd; a.GetBkGd(bk, gd);

		kckForeach<<<gd,bk,0,s>>>(kckMat(a), kckMat(b), kckMat(c), lambdafun);
	}
}

//////////////////////////
// for four argument

template<typename T, typename Y, typename U, typename P, class L>
__global__ void kckForeach(kckArr<T> a, kckArr<Y> b, kckArr<U> c, kckArr<P> d, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);	

	if(i1 >= a.size) return;

	lambdafun(a(i1), b(i1), c(i1), d(i1));
}

template<typename T, typename Y, typename U, typename P, class L>
__global__ void kckForeach(kckMat1<T> a, kckMat1<Y> b, kckMat1<U> c, kckMat1<P> d, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);	

	if(i1 >= a.n1) return;

	lambdafun(a(i1), b(i1), c(i1), d(i1));
}

template<typename T, typename Y, typename U, typename P, class L>
__global__ void kckForeach(kckMat2<T> a, kckMat2<Y> b, kckMat2<U> c, kckMat2<P> d, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2), b(i1,i2), c(i1,i2), d(i1,i2));
}

template<typename T, typename Y, typename U, typename P, class L>
__global__ void kckForeach(kckMat3<T> a, kckMat3<Y> b, kckMat3<U> c, kckMat3<P> d, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);
	int i3 = GetStdIdx3(blockIdx);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2,i3), b(i1,i2,i3), c(i1,i2,i3), d(i1,i2,i3));
}

template<typename T, typename Y, typename U, typename P, class L>
__global__ void kckForeach(kckMat4<T> a, kckMat4<Y> b, kckMat4<U> c, kckMat4<P> d, const L lambdafun)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);
	int i3 = GetStdIdx3(blockIdx);
	int i4 = GetStdIdx4(i3, a.n3);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	lambdafun(a(i1,i2,i3,i4), b(i1,i2,i3,i4), c(i1,i2,i3,i4), d(i1,i2,i3,i4));
}

template<template<typename> class M, typename T, typename Y, typename U, typename P, class L>
void Foreach(M<T>& a, const M<Y>& b, const M<U>& c, const M<P>& d, const L lambdafun, cudaStream_t s = 0)
{
	// check size
	ASSERTFA(b.IsEqualSizeDim(c), "foreach in 1831");
	if(!a.IsEqualSizeDim(b)) a.Recreate(b);	
	
	// check noblank
	if(a.IsNoBlank() && b.IsNoBlank() && c.IsNoBlank() && d.IsNoBlank())
	{
		// call kernel with down-dimension
		kcMat1<T> _a(a.P(), a.N());
		kcMat1<Y> _b(b.P(), b.N());
		kcMat1<U> _c(c.P(), c.N());
		kcMat1<P> _d(d.P(), d.N());

		dim3 bk, gd; _a.GetBkGd(bk, gd);

		kckForeach<<<gd,bk,0,s>>>(kckMat(_a), kckMat(_b), kckMat(_c), kckMat(_d), lambdafun);
	}
	else
	{
		// call kernel
		dim3 bk, gd; a.GetBkGd(bk, gd);

		kckForeach<<<gd,bk,0,s>>>(kckMat(a), kckMat(b), kckMat(c), kckMat(d), lambdafun);
	}
}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// casting functions a = (T) b (b is (Y))

template<template<typename> class M, typename T, typename Y>
M<T>& _TCast(M<T>& a, const M<Y>& b, cudaStream_t s)
{
	Foreach(a, b, []__device__(T& a, Y b){ a = (T)b; },s);
	return a;
};

kcArr0f32& TCast(kcArr0f32& a, const kcArr0i8 & b, cudaStream_t s) { return _TCast(a, b, s);}
kcArr0f32& TCast(kcArr0f32& a, const kcArr0i16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcArr0f32& TCast(kcArr0f32& a, const kcArr0i32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcArr0f32& TCast(kcArr0f32& a, const kcArr0i64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcArr0f32& TCast(kcArr0f32& a, const kcArr0f16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcArr0f32& TCast(kcArr0f32& a, const kcArr0f64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcArr0f16& TCast(kcArr0f16& a, const kcArr0i8 & b, cudaStream_t s) { return _TCast(a, b, s);}
kcArr0f16& TCast(kcArr0f16& a, const kcArr0i16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcArr0f16& TCast(kcArr0f16& a, const kcArr0i32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcArr0f16& TCast(kcArr0f16& a, const kcArr0i64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcArr0f16& TCast(kcArr0f16& a, const kcArr0f32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcArr0f16& TCast(kcArr0f16& a, const kcArr0f64& b, cudaStream_t s) { return _TCast(a, b, s);}

kcMat1f32& TCast(kcMat1f32& a, const kcMat1i8 & b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat1f32& TCast(kcMat1f32& a, const kcMat1i16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat1f32& TCast(kcMat1f32& a, const kcMat1i32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat1f32& TCast(kcMat1f32& a, const kcMat1i64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat1f32& TCast(kcMat1f32& a, const kcMat1f16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat1f32& TCast(kcMat1f32& a, const kcMat1f64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat1f16& TCast(kcMat1f16& a, const kcMat1i8 & b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat1f16& TCast(kcMat1f16& a, const kcMat1i16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat1f16& TCast(kcMat1f16& a, const kcMat1i32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat1f16& TCast(kcMat1f16& a, const kcMat1i64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat1f16& TCast(kcMat1f16& a, const kcMat1f32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat1f16& TCast(kcMat1f16& a, const kcMat1f64& b, cudaStream_t s) { return _TCast(a, b, s);}

kcMat2f32& TCast(kcMat2f32& a, const kcMat2i8 & b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat2f32& TCast(kcMat2f32& a, const kcMat2i16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat2f32& TCast(kcMat2f32& a, const kcMat2i32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat2f32& TCast(kcMat2f32& a, const kcMat2i64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat2f32& TCast(kcMat2f32& a, const kcMat2f16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat2f32& TCast(kcMat2f32& a, const kcMat2f64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat2f16& TCast(kcMat2f16& a, const kcMat2i8 & b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat2f16& TCast(kcMat2f16& a, const kcMat2i16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat2f16& TCast(kcMat2f16& a, const kcMat2i32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat2f16& TCast(kcMat2f16& a, const kcMat2i64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat2f16& TCast(kcMat2f16& a, const kcMat2f32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat2f16& TCast(kcMat2f16& a, const kcMat2f64& b, cudaStream_t s) { return _TCast(a, b, s);}

kcMat3f32& TCast(kcMat3f32& a, const kcMat3i8 & b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat3f32& TCast(kcMat3f32& a, const kcMat3i16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat3f32& TCast(kcMat3f32& a, const kcMat3i32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat3f32& TCast(kcMat3f32& a, const kcMat3i64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat3f32& TCast(kcMat3f32& a, const kcMat3f16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat3f32& TCast(kcMat3f32& a, const kcMat3f64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat3f16& TCast(kcMat3f16& a, const kcMat3i8 & b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat3f16& TCast(kcMat3f16& a, const kcMat3i16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat3f16& TCast(kcMat3f16& a, const kcMat3i32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat3f16& TCast(kcMat3f16& a, const kcMat3i64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat3f16& TCast(kcMat3f16& a, const kcMat3f32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat3f16& TCast(kcMat3f16& a, const kcMat3f64& b, cudaStream_t s) { return _TCast(a, b, s);}

kcMat4f32& TCast(kcMat4f32& a, const kcMat4i8 & b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat4f32& TCast(kcMat4f32& a, const kcMat4i16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat4f32& TCast(kcMat4f32& a, const kcMat4i32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat4f32& TCast(kcMat4f32& a, const kcMat4i64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat4f32& TCast(kcMat4f32& a, const kcMat4f16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat4f32& TCast(kcMat4f32& a, const kcMat4f64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat4f16& TCast(kcMat4f16& a, const kcMat4i8 & b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat4f16& TCast(kcMat4f16& a, const kcMat4i16& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat4f16& TCast(kcMat4f16& a, const kcMat4i32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat4f16& TCast(kcMat4f16& a, const kcMat4i64& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat4f16& TCast(kcMat4f16& a, const kcMat4f32& b, cudaStream_t s) { return _TCast(a, b, s);}
kcMat4f16& TCast(kcMat4f16& a, const kcMat4f64& b, cudaStream_t s) { return _TCast(a, b, s);}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// math functions

////////////////////////////////////////////////////////
// log-compression

template<template<typename> class M, typename T, typename Y>
void _LogCompression(M<T>& a, const M<Y>& b, cudaStream_t s)
{
	Foreach(a, b, []__device__(T& a, Y& b){ a = log10(hypotf(b.x, b.y) + T(1)); }, s); 
};

void LogCompression(kcMat2f32& a, const kcMat2c32& b, cudaStream_t s)	{ _LogCompression(a, b, s); };
void LogCompression(kcMat3f32& a, const kcMat3c32& b, cudaStream_t s)	{ _LogCompression(a, b, s); };
void LogCompression(kcMat3f32& a, const kcMat3q16& b, cudaStream_t s)	{ _LogCompression(a, b, s); };

kcMat2f32 LogCompression(const kcMat2c32& b, cudaStream_t s) { kcMat2f32 a; LogCompression(a, b, s); return a; };
kcMat3f32 LogCompression(const kcMat3c32& b, cudaStream_t s) { kcMat3f32 a; LogCompression(a, b, s); return a; };
kcMat3f32 LogCompression(const kcMat3q16& b, cudaStream_t s) { kcMat3f32 a; LogCompression(a, b, s); return a; };

////////////////////////////////////////////////////////
// abs for complex

template<template<typename> class M, typename T, typename Y>
void _Abs(M<T>& a, const M<Y>& b, cudaStream_t s)
{
	Foreach(a, b, []__device__(T& a, Y& b){ a = hypotf(b.x, b.y); }, s); 
};

void Abs(kcMat2f32& a, const kcMat2c32& b, cudaStream_t s)	{ _Abs(a, b, s); };
void Abs(kcMat3f32& a, const kcMat3c32& b, cudaStream_t s)	{ _Abs(a, b, s); };
void Abs(kcMat3f32& a, const kcMat3q16& b, cudaStream_t s)	{ _Abs(a, b, s); };

kcMat2f32 Abs(const kcMat2c32& b, cudaStream_t s) { kcMat2f32 a; Abs(a, b, s); return a; };
kcMat3f32 Abs(const kcMat3c32& b, cudaStream_t s) { kcMat3f32 a; Abs(a, b, s); return a; };
kcMat3f32 Abs(const kcMat3q16& b, cudaStream_t s) { kcMat3f32 a; Abs(a, b, s); return a; };

////////////////////////////////////////////////////////
// sigmoid... a = sigmoid(b)

template<template<typename> class M, typename T>
M<T>& _Sigmoid(M<T>& a, const M<T>& b, cudaStream_t s)
{
	Foreach(a, b, []__device__(T& a, T b){ a = fdividef(1.f, 1.f + expf(-b)); },s);
	return a;
};

kcMat1f32& Sigmoid(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s) { return _Sigmoid(a, b, s);}
kcMat2f32& Sigmoid(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s) { return _Sigmoid(a, b, s);}
kcMat3f32& Sigmoid(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s) { return _Sigmoid(a, b, s);}
kcMat4f32& Sigmoid(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s) { return _Sigmoid(a, b, s);}
kcMat1f16& Sigmoid(kcMat1f16& a, const kcMat1f16& b, cudaStream_t s) { return _Sigmoid(a, b, s);}
kcMat2f16& Sigmoid(kcMat2f16& a, const kcMat2f16& b, cudaStream_t s) { return _Sigmoid(a, b, s);}
kcMat3f16& Sigmoid(kcMat3f16& a, const kcMat3f16& b, cudaStream_t s) { return _Sigmoid(a, b, s);}
kcMat4f16& Sigmoid(kcMat4f16& a, const kcMat4f16& b, cudaStream_t s) { return _Sigmoid(a, b, s);}

kcMat1f32 Sigmoid(const kcMat1f32& b, cudaStream_t s) { kcMat1f32 a; _Sigmoid(a, b, s); return a;}
kcMat2f32 Sigmoid(const kcMat2f32& b, cudaStream_t s) { kcMat2f32 a; _Sigmoid(a, b, s); return a;}
kcMat3f32 Sigmoid(const kcMat3f32& b, cudaStream_t s) { kcMat3f32 a; _Sigmoid(a, b, s); return a;}
kcMat4f32 Sigmoid(const kcMat4f32& b, cudaStream_t s) { kcMat4f32 a; _Sigmoid(a, b, s); return a;}
kcMat1f16 Sigmoid(const kcMat1f16& b, cudaStream_t s) { kcMat1f16 a; _Sigmoid(a, b, s); return a;}
kcMat2f16 Sigmoid(const kcMat2f16& b, cudaStream_t s) { kcMat2f16 a; _Sigmoid(a, b, s); return a;}
kcMat3f16 Sigmoid(const kcMat3f16& b, cudaStream_t s) { kcMat3f16 a; _Sigmoid(a, b, s); return a;}
kcMat4f16 Sigmoid(const kcMat4f16& b, cudaStream_t s) { kcMat4f16 a; _Sigmoid(a, b, s); return a;}

////////////////////////////////////////////////////////
// ReLU... a = ReLU(b)

template<template<typename> class M, typename T>
M<T>& _ReLU(M<T>& a, const M<T>& b, cudaStream_t s)
{
	Foreach(a, b, []__device__(T& a, T b){ a = MAX(0, float(b)); },s);
	return a;
};

kcMat1f32& ReLU(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s) { return _ReLU(a, b, s);}
kcMat2f32& ReLU(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s) { return _ReLU(a, b, s);}
kcMat3f32& ReLU(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s) { return _ReLU(a, b, s);}
kcMat4f32& ReLU(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s) { return _ReLU(a, b, s);}
kcMat1f16& ReLU(kcMat1f16& a, const kcMat1f16& b, cudaStream_t s) { return _ReLU(a, b, s);}
kcMat2f16& ReLU(kcMat2f16& a, const kcMat2f16& b, cudaStream_t s) { return _ReLU(a, b, s);}
kcMat3f16& ReLU(kcMat3f16& a, const kcMat3f16& b, cudaStream_t s) { return _ReLU(a, b, s);}
kcMat4f16& ReLU(kcMat4f16& a, const kcMat4f16& b, cudaStream_t s) { return _ReLU(a, b, s);}

kcMat1f32 ReLU(const kcMat1f32& b, cudaStream_t s) { kcMat1f32 a; _ReLU(a, b, s); return a;}
kcMat2f32 ReLU(const kcMat2f32& b, cudaStream_t s) { kcMat2f32 a; _ReLU(a, b, s); return a;}
kcMat3f32 ReLU(const kcMat3f32& b, cudaStream_t s) { kcMat3f32 a; _ReLU(a, b, s); return a;}
kcMat4f32 ReLU(const kcMat4f32& b, cudaStream_t s) { kcMat4f32 a; _ReLU(a, b, s); return a;}
kcMat1f16 ReLU(const kcMat1f16& b, cudaStream_t s) { kcMat1f16 a; _ReLU(a, b, s); return a;}
kcMat2f16 ReLU(const kcMat2f16& b, cudaStream_t s) { kcMat2f16 a; _ReLU(a, b, s); return a;}
kcMat3f16 ReLU(const kcMat3f16& b, cudaStream_t s) { kcMat3f16 a; _ReLU(a, b, s); return a;}
kcMat4f16 ReLU(const kcMat4f16& b, cudaStream_t s) { kcMat4f16 a; _ReLU(a, b, s); return a;}

////////////////////////////////////////////////////////
// exponential... a = e^(b)

template<template<typename> class M, typename T>
M<T>& _Exp(M<T>& a, const M<T>& b, cudaStream_t s)
{
	Foreach(a, b, []__device__(T& a, T b){ a = exp(b); },s);
	return a;
};

kcMat1f32& Exp(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s) { return _Exp(a, b, s);}
kcMat2f32& Exp(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s) { return _Exp(a, b, s);}
kcMat3f32& Exp(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s) { return _Exp(a, b, s);}
kcMat4f32& Exp(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s) { return _Exp(a, b, s);}

kcMat1f32 Exp(const kcMat1f32& b, cudaStream_t s) { kcMat1f32 a; _Exp(a, b, s); return a;}
kcMat2f32 Exp(const kcMat2f32& b, cudaStream_t s) { kcMat2f32 a; _Exp(a, b, s); return a;}
kcMat3f32 Exp(const kcMat3f32& b, cudaStream_t s) { kcMat3f32 a; _Exp(a, b, s); return a;}
kcMat4f32 Exp(const kcMat4f32& b, cudaStream_t s) { kcMat4f32 a; _Exp(a, b, s); return a;}

////////////////////////////////////////////////////////
// softmax... a = softmax(b)

template<template<typename> class M, typename T> 
M<T>& _Softmax(M<T>& a, const M<T>& b, cudaStream_t s)
{
	Exp(a, b, s); 
	const T invs = (T)1/(Sum(a, s) + FLOAT_SMALL); 
	MulEq(a, invs, s); 
	return a;
}

kcMat1f32& Softmax(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s) { return _Softmax(a, b, s);}
kcMat2f32& Softmax(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s) { return _Softmax(a, b, s);}
kcMat3f32& Softmax(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s) { return _Softmax(a, b, s);}
kcMat4f32& Softmax(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s) { return _Softmax(a, b, s);}

kcMat1f32 Softmax(const kcMat1f32& b, cudaStream_t s) { kcMat1f32 a; _Softmax(a, b, s); return a;}
kcMat2f32 Softmax(const kcMat2f32& b, cudaStream_t s) { kcMat2f32 a; _Softmax(a, b, s); return a;}
kcMat3f32 Softmax(const kcMat3f32& b, cudaStream_t s) { kcMat3f32 a; _Softmax(a, b, s); return a;}
kcMat4f32 Softmax(const kcMat4f32& b, cudaStream_t s) { kcMat4f32 a; _Softmax(a, b, s); return a;}

////////////////////////////////////////////////////////
// operator+=.. a += b

template<template<typename> class M, typename T>
M<T>& _AddEq(M<T>& a, const M<T>& b, cudaStream_t s)
{
	Foreach(a, b, []__device__(T& a, T b){ a += b; },s);
	return a;
};

template<template<typename> class M, typename T>
M<T>& _AddEq(M<T>& a, const T b, cudaStream_t s)
{
	Foreach(a, [b]__device__(T& a){ a += b; },s);
	return a;
};

kcMat1f32& AddEq(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s) { return _AddEq(a, b, s);}
kcMat2f32& AddEq(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s) { return _AddEq(a, b, s);}
kcMat3f32& AddEq(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s) { return _AddEq(a, b, s);}
kcMat4f32& AddEq(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s) { return _AddEq(a, b, s);}
kcMat1f16& AddEq(kcMat1f16& a, const kcMat1f16& b, cudaStream_t s) { return _AddEq(a, b, s);}
kcMat2f16& AddEq(kcMat2f16& a, const kcMat2f16& b, cudaStream_t s) { return _AddEq(a, b, s);}
kcMat3f16& AddEq(kcMat3f16& a, const kcMat3f16& b, cudaStream_t s) { return _AddEq(a, b, s);}
kcMat4f16& AddEq(kcMat4f16& a, const kcMat4f16& b, cudaStream_t s) { return _AddEq(a, b, s);}
kcMat1f32& AddEq(kcMat1f32& a, const float      b, cudaStream_t s) { return _AddEq(a, b, s);}
kcMat2f32& AddEq(kcMat2f32& a, const float      b, cudaStream_t s) { return _AddEq(a, b, s);}
kcMat3f32& AddEq(kcMat3f32& a, const float      b, cudaStream_t s) { return _AddEq(a, b, s);}
kcMat4f32& AddEq(kcMat4f32& a, const float      b, cudaStream_t s) { return _AddEq(a, b, s);}


kcMat1f32& operator+=(kcMat1f32& a, const kcMat1f32& b) { return _AddEq(a, b, 0);}
kcMat2f32& operator+=(kcMat2f32& a, const kcMat2f32& b) { return _AddEq(a, b, 0);}
kcMat3f32& operator+=(kcMat3f32& a, const kcMat3f32& b) { return _AddEq(a, b, 0);}
kcMat4f32& operator+=(kcMat4f32& a, const kcMat4f32& b) { return _AddEq(a, b, 0);}
kcMat1f16& operator+=(kcMat1f16& a, const kcMat1f16& b) { return _AddEq(a, b, 0);}
kcMat2f16& operator+=(kcMat2f16& a, const kcMat2f16& b) { return _AddEq(a, b, 0);}
kcMat3f16& operator+=(kcMat3f16& a, const kcMat3f16& b) { return _AddEq(a, b, 0);}
kcMat4f16& operator+=(kcMat4f16& a, const kcMat4f16& b) { return _AddEq(a, b, 0);}
kcMat1f32& operator+=(kcMat1f32& a, const float      b) { return _AddEq(a, b, 0);}
kcMat2f32& operator+=(kcMat2f32& a, const float      b) { return _AddEq(a, b, 0);}
kcMat3f32& operator+=(kcMat3f32& a, const float      b) { return _AddEq(a, b, 0);}
kcMat4f32& operator+=(kcMat4f32& a, const float      b) { return _AddEq(a, b, 0);}

////////////////////////////////////////////////////////
// operator-=.. a -= b

template<template<typename> class M, typename T>
M<T>& _SubEq(M<T>& a, const M<T>& b, cudaStream_t s)
{
	Foreach(a, b, []__device__(T& a, T b){ a -= b; },s);
	return a;
};

kcMat1f32& SubEq(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s) { return _SubEq(a, b, s);}
kcMat2f32& SubEq(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s) { return _SubEq(a, b, s);}
kcMat3f32& SubEq(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s) { return _SubEq(a, b, s);}
kcMat4f32& SubEq(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s) { return _SubEq(a, b, s);}

kcMat1f32& operator-=(kcMat1f32& a, const kcMat1f32& b) { return _SubEq(a, b, 0);}
kcMat2f32& operator-=(kcMat2f32& a, const kcMat2f32& b) { return _SubEq(a, b, 0);}
kcMat3f32& operator-=(kcMat3f32& a, const kcMat3f32& b) { return _SubEq(a, b, 0);}
kcMat4f32& operator-=(kcMat4f32& a, const kcMat4f32& b) { return _SubEq(a, b, 0);}

////////////////////////////////////////////////////////
// operator*=.. a *= b

template<template<typename> class M, typename T>
M<T>& _MulEq(M<T>& a, const M<T>& b, cudaStream_t s)
{
	Foreach(a, b, []__device__(T& a, T b){ a *= b; },s);
	return a;
};

template<template<typename> class M, typename T>
M<T>& _MulEq(M<T>& a, const T b, cudaStream_t s)
{
	Foreach(a, [b]__device__(T& a){ a *= b; },s);
	return a;
};

kcMat1f32& MulEq(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s) { return _MulEq(a, b, s);}
kcMat2f32& MulEq(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s) { return _MulEq(a, b, s);}
kcMat3f32& MulEq(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s) { return _MulEq(a, b, s);}
kcMat4f32& MulEq(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s) { return _MulEq(a, b, s);}
kcMat1f32& MulEq(kcMat1f32& a, const float      b, cudaStream_t s) { return _MulEq(a, b, s);}
kcMat2f32& MulEq(kcMat2f32& a, const float      b, cudaStream_t s) { return _MulEq(a, b, s);}
kcMat3f32& MulEq(kcMat3f32& a, const float      b, cudaStream_t s) { return _MulEq(a, b, s);}
kcMat4f32& MulEq(kcMat4f32& a, const float      b, cudaStream_t s) { return _MulEq(a, b, s);}

kcMat1f32& operator*=(kcMat1f32& a, const kcMat1f32& b) { return _MulEq(a, b, 0);}
kcMat2f32& operator*=(kcMat2f32& a, const kcMat2f32& b) { return _MulEq(a, b, 0);}
kcMat3f32& operator*=(kcMat3f32& a, const kcMat3f32& b) { return _MulEq(a, b, 0);}
kcMat4f32& operator*=(kcMat4f32& a, const kcMat4f32& b) { return _MulEq(a, b, 0);}
kcMat1f32& operator*=(kcMat1f32& a, const float      b) { return _MulEq(a, b, 0);}
kcMat2f32& operator*=(kcMat2f32& a, const float      b) { return _MulEq(a, b, 0);}
kcMat3f32& operator*=(kcMat3f32& a, const float      b) { return _MulEq(a, b, 0);}
kcMat4f32& operator*=(kcMat4f32& a, const float      b) { return _MulEq(a, b, 0);}

////////////////////////////////////////////////////////
// Add... a = b + c

template<template<typename> class M, typename T>
M<T>& _Add(M<T>& a, const M<T>& b, const M<T>& c, cudaStream_t s)
{
	Foreach(a, b, c, []__device__(T& a, const T b, const T c){ a = b + c; },s);
	return a;
};

kcMat1f32& Add(kcMat1f32& a, const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s) { return _Add(a, b, c, s);}
kcMat2f32& Add(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s) { return _Add(a, b, c, s);}
kcMat3f32& Add(kcMat3f32& a, const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s) { return _Add(a, b, c, s);}
kcMat4f32& Add(kcMat4f32& a, const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s) { return _Add(a, b, c, s);}
kcMat1f16& Add(kcMat1f16& a, const kcMat1f16& b, const kcMat1f16& c, cudaStream_t s) { return _Add(a, b, c, s);}
kcMat2f16& Add(kcMat2f16& a, const kcMat2f16& b, const kcMat2f16& c, cudaStream_t s) { return _Add(a, b, c, s);}
kcMat3f16& Add(kcMat3f16& a, const kcMat3f16& b, const kcMat3f16& c, cudaStream_t s) { return _Add(a, b, c, s);}
kcMat4f16& Add(kcMat4f16& a, const kcMat4f16& b, const kcMat4f16& c, cudaStream_t s) { return _Add(a, b, c, s);}

////////////////////////////////////////////////////////
// SubEqMul.. a -= b*c

template<template<typename> class M, typename T>
M<T>& _SubEqMul(M<T>& a, const M<T>& b, const T c, cudaStream_t s)
{
	Foreach(a, b, [c]__device__(T& a, T b){ a -= b*c; },s);
	return a;
};

kcMat1f32& SubEqMul(kcMat1f32& a, const kcMat1f32& b, const float c, cudaStream_t s) { return _SubEqMul(a, b, c, s);}
kcMat2f32& SubEqMul(kcMat2f32& a, const kcMat2f32& b, const float c, cudaStream_t s) { return _SubEqMul(a, b, c, s);}
kcMat3f32& SubEqMul(kcMat3f32& a, const kcMat3f32& b, const float c, cudaStream_t s) { return _SubEqMul(a, b, c, s);}
kcMat4f32& SubEqMul(kcMat4f32& a, const kcMat4f32& b, const float c, cudaStream_t s) { return _SubEqMul(a, b, c, s);}

////////////////////////////////////////////////////////
// square error... a = sqrt(b - c)

template<template<typename> class M, typename T>
M<T>& _Sqe(M<T>& a, const M<T>& b, const M<T>& c, cudaStream_t s)
{
	Foreach(a, b, c, []__device__(T& a, T b, T c){ a = (b - c)*(b - c); },s);
	return a;
};

kcMat1f32& Sqe(kcMat1f32& a, const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s) { return _Sqe(a, b, c, s);}
kcMat2f32& Sqe(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s) { return _Sqe(a, b, c, s);}
kcMat3f32& Sqe(kcMat3f32& a, const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s) { return _Sqe(a, b, c, s);}
kcMat4f32& Sqe(kcMat4f32& a, const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s) { return _Sqe(a, b, c, s);}

////////////////////////////////////////////////////////
// mean square error... a = mean(sqrt(b - c))

float Mse(const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s) { kcMat1f32 a(b); return Sum(_Sqe(a, b, c, s))/a.N();}
float Mse(const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s) { kcMat2f32 a(b); return Sum(_Sqe(a, b, c, s))/a.N();}
float Mse(const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s) { kcMat3f32 a(b); return Sum(_Sqe(a, b, c, s))/a.N();}
float Mse(const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s) { kcMat4f32 a(b); return Sum(_Sqe(a, b, c, s))/a.N();}

////////////////////////////////////////////////////////
// GradMse... a = 2*(b - c)

template<template<typename> class M, typename T>
M<T>& _GradMse(M<T>& a, const M<T>& b, const M<T>& c, cudaStream_t s)
{
	Foreach(a, b, c, []__device__(T& a, T b, T c){ a = T(2)*(b - c); },s);
	return a;
};

kcMat1f32& GradMse(kcMat1f32& a, const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s) { return _GradMse(a, b, c, s);}
kcMat2f32& GradMse(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s) { return _GradMse(a, b, c, s);}
kcMat3f32& GradMse(kcMat3f32& a, const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s) { return _GradMse(a, b, c, s);}
kcMat4f32& GradMse(kcMat4f32& a, const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s) { return _GradMse(a, b, c, s);}
kcMat1f16& GradMse(kcMat1f16& a, const kcMat1f16& b, const kcMat1f16& c, cudaStream_t s) { return _GradMse(a, b, c, s);}
kcMat2f16& GradMse(kcMat2f16& a, const kcMat2f16& b, const kcMat2f16& c, cudaStream_t s) { return _GradMse(a, b, c, s);}
kcMat3f16& GradMse(kcMat3f16& a, const kcMat3f16& b, const kcMat3f16& c, cudaStream_t s) { return _GradMse(a, b, c, s);}
kcMat4f16& GradMse(kcMat4f16& a, const kcMat4f16& b, const kcMat4f16& c, cudaStream_t s) { return _GradMse(a, b, c, s);}

////////////////////////////////////////////////////////
// cross entropy error... a = -sum(c*log(b))

//template<typename T> inline
//__device__ void kcdMulLog10(T& a, const T& b, const T& c)
//{
//	a = float(c)*log10f(float(b) + 1e-12f);
//}

template<template<typename> class M, typename T>
M<T>& _MulLog10(M<T>& a, const M<T>& b, const M<T>& c, cudaStream_t s)
{
	Foreach(a, b, c, []__device__(T& a, T b, T c){ a = c*log10(b + T(1e-12)); },s);
	return a;
};

float Cee(const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s) { kcMat1f32 a(b); return -Sum(_MulLog10(a, b, c, s));}
float Cee(const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s) { kcMat2f32 a(b); return -Sum(_MulLog10(a, b, c, s));}
float Cee(const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s) { kcMat3f32 a(b); return -Sum(_MulLog10(a, b, c, s));}
float Cee(const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s) { kcMat4f32 a(b); return -Sum(_MulLog10(a, b, c, s));}

////////////////////////////////////////////////////////
// GradCee... a = -c / MAX(b, 1e-10f)

//template<typename T> inline
//__device__ void kcdGradCee(T& a, const T& b, const T& c)
//{
//	a = -float(c)/MAX(float(b), 1e-12f);
//}

template<template<typename> class M, typename T>
M<T>& _GradCee(M<T>& a, const M<T>& b, const M<T>& c, cudaStream_t s)
{
	Foreach(a, b, c, []__device__(T& a, T b, T c){ a = -c/(MAX(b, T(1e-12))); },s);
	return a;
};

kcMat1f32& GradCee(kcMat1f32& a, const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s) { return _GradCee(a, b, c, s);}
kcMat2f32& GradCee(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s) { return _GradCee(a, b, c, s);}
kcMat3f32& GradCee(kcMat3f32& a, const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s) { return _GradCee(a, b, c, s);}
kcMat4f32& GradCee(kcMat4f32& a, const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s) { return _GradCee(a, b, c, s);}
kcMat1f16& GradCee(kcMat1f16& a, const kcMat1f16& b, const kcMat1f16& c, cudaStream_t s) { return _GradCee(a, b, c, s);}
kcMat2f16& GradCee(kcMat2f16& a, const kcMat2f16& b, const kcMat2f16& c, cudaStream_t s) { return _GradCee(a, b, c, s);}
kcMat3f16& GradCee(kcMat3f16& a, const kcMat3f16& b, const kcMat3f16& c, cudaStream_t s) { return _GradCee(a, b, c, s);}
kcMat4f16& GradCee(kcMat4f16& a, const kcMat4f16& b, const kcMat4f16& c, cudaStream_t s) { return _GradCee(a, b, c, s);}

////////////////////////////////////////////////////////
// set daigonal stripes

template<typename T>
__global__ void kckSetDiagStripe(kckMat2<T> a, float vs, float ve, int cnt0)
{	
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);

	if(i1 >= a.n1 || i2 >= a.n2) return;

	const int cnt = (cnt0 + i1 + i2)%32;

	const float v = vs + float(cnt)*(ve - vs)/31.f;

	a(i1,i2).x =  v;
	a(i1,i2).y = -v;
}

// Downsample(a, b, downrate)
template<typename T>
void _SetDiagStripe(kcMat2<T>& a, float vs, float ve, int cnt0 = 0, cudaStream_t s = 0)
{
	dim3 bk, gd; a.GetBkGd(bk, gd);

	kckSetDiagStripe<<<gd,bk,0,s>>>(kckMat2<T>(a), vs, ve, cnt0);
}

void SetDiagStripe(kcMat2c32& a, float vs, float ve, int cnt0, cudaStream_t s) { _SetDiagStripe(a, vs, ve, cnt0, s); };

////////////////////////////////////////////////////////
// color coding with cmap
//
// * Note that this kernel has not been optimized using shared memory

template<typename T>
__global__ void kckConvertBgrTp(kckMat2<kmBgr> img, const kckMat2<T> a, T min_v, T max_v, const kckMat1<kmBgr> cmap)
{
	int i1 = GetStdIdx1(threadIdx, blockIdx, blockDim);
	int i2 = GetStdIdx2(threadIdx, blockIdx, blockDim);

	if(i1 >= img.n1 || i2 >= img.n2) return;

	const float coff = 255.f/float(max_v - min_v);

	T val = MIN(MAX(min_v, a(i2,i1)),max_v);

	const int idx = int(float(val - min_v)*coff);

	img(i1,i2) = cmap(MIN(MAX(0, idx), 255));
}

template<typename T>
void _ConvertBgrTp(kcMat2<kmBgr>& img, const kcMat2<T>& a, T cmin, T cmax, const kcMat1<kmBgr>& cmap, cudaStream_t s = 0)
{
	dim3 bk, gd; img.GetBkGd(bk, gd);

	kckConvertBgrTp<<<gd,bk,0,s>>>(kckMat2<kmBgr>(img), kckMat2<T>(a), cmin, cmax, kckMat1<kmBgr>(cmap));
}

void ConvertBgrTp(kcMat2<kmBgr>& img, const kcMat2f32& a, float cmin, float cmax, const kcMat1<kmBgr>& cmap, cudaStream_t s)
{
	_ConvertBgrTp(img, a, cmin, cmax, cmap, s);
}

////////////////////////////////////////////////////////
// decompose complex

template<template<typename> class M, typename T>
void _DecomposeCmplx(M<T>& a, M<T>& b, M<T>& c, const M<float2>& d, cudaStream_t s)
{
	Foreach(a, b, c, d, []__device__(T& a, T& b, T& c, float2 d){ a = d.x; b = d.y; c = hypotf(d.x, d.y); },s);
};

void DecomposeCmplx(kcMat2f32& re, kcMat2f32& im, kcMat2f32& ma, const kcMat2c32& cmplx, cudaStream_t s)
{
	_DecomposeCmplx(re, im, ma, cmplx, s);
};
void DecomposeCmplx(kcMat3f32& re, kcMat3f32& im, kcMat3f32& ma, const kcMat3c32& cmplx, cudaStream_t s)
{
	_DecomposeCmplx(re, im, ma, cmplx, s);
};

////////////////////////////////////////////////////////
// contrast functions.... T a = Contrast(b) using reduction

// constrast with Sobel
template<typename T>
__global__ void kckSumSqSobel(T* a, const kckMat2<T> b)
{
	extern __shared__ float sm[];

	KC_RDC_INDEX_EXDM0(b);

	T val = 0;
	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2; if(i1 < 1 || i1 >= b.n1 - 1) continue;

		for(int i2 = 1; i2 < b.n2 - 1; ++i2)
		{
			T gx  = b(i1-1, i2-1) + 2.f*b(i1-1,i2) + b(i1-1,i2+1);
			  gx -= b(i1+1, i2-1) + 2.f*b(i1+1,i2) + b(i1+1,i2+1);

			T gy  = b(i1-1, i2-1) + 2.f*b(i1,i2-1) + b(i1+1,i2-1);
			  gy -= b(i1-1, i2+1) + 2.f*b(i1,i2+1) + b(i1+1,i2+1);

			val += gx*gx + gy*gy;
		}
	} 
	sm[ix1] = val;

	for(int size_s = nx1>>1; size_s > 0; size_s >>= 1)
	{
		__syncthreads(); if(ix1 < size_s) sm[ix1] += sm[ix1 + size_s];
	}
	if(ix1 == 0) *a = sm[0];
}

template<typename T>
float _SumSqSobel(const kcMat2<T>& b, cudaStream_t s)
{
	// call kernel
	kcMat1<T> a(1);

	dim3 bk, gd; int sm; b.GetBkGdRdc(bk, gd, sm);	

	kckSumSqSobel<<<gd,bk,sm,s>>>(a.P(), kckMat2<T>(b));

	return kmMat1<T>(a)(0);	
};

float ContrastSobel(const kcMat2f32& b, cudaStream_t s)
{
	const int64 n1 = b.N1(), n2 = b.N2();
	
	if(n1 < 3 || n2 < 3) return 0;

	return _SumSqSobel(b, s)/((n1 - 2)*(n2 - 2));
};

// constrast with Laplacian
template<typename T>
__global__ void kckSumSqLp(T* a, const kckMat2<T> b)
{
	extern __shared__ float sm[];

	KC_RDC_INDEX_EXDM0(b);

	T val = 0;
	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2; if(i1 < 1 || i1 >= b.n1 - 1) continue;

		for(int i2 = 1; i2 < b.n2 - 1; ++i2)
		{
			const T g0 = -4.f*b(i1,i2) + b(i1-1,i2) + b(i1+1,i2) + b(i1,i2-1) + b(i1,i2+1);

			val += g0*g0;
		}
	}
	sm[ix1] = val;

	for(int size_s = nx1>>1; size_s > 0; size_s >>= 1)
	{
		__syncthreads(); if(ix1 < size_s) sm[ix1] += sm[ix1 + size_s];
	}
	if(ix1 == 0) *a = sm[0];
}

template<typename T>
float _SumSqLp(const kcMat2<T>& b, cudaStream_t s)
{
	// call kernel
	kcMat1<T> a(1);

	dim3 bk, gd; int sm; b.GetBkGdRdc(bk, gd, sm);	

	kckSumSqLp<<<gd,bk,sm,s>>>(a.P(), kckMat2<T>(b));

	return kmMat1<T>(a)(0);	
};

float ContrastLp(const kcMat2f32& b, cudaStream_t s)
{
	const int64 n1 = b.N1(), n2 = b.N2();
	
	if(n1 < 3 || n2 < 3) return 0;

	return _SumSqLp(b, s)/((n1 - 2)*(n2 - 2));
};
 

