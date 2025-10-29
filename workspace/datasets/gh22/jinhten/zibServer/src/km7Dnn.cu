// base header
#include "../inc/km7Dnn.h"

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// math functions for kmDnn

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// kernel functions for kmDnn

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// member functions for kmDnn

////////////////////////////////////
// kmDnnLyActf's members

// ReLU - backward
template<typename T>
__global__ void kckGReLU(kckMat4<T> gx0, const kckMat4<T> gx1, const kckMat4<T> x1)
{
	KC4_STD_INDEX(gx0);	// i1, i2, i3, i4
	
	const float x1v = x1(i1,i2,i3,i4);

	if(x1v > 0)
	{
		T& gx0r = gx0(i1,i2,i3,i4);	
		
		gx0r = float(gx0r) + float(gx1(i1,i2,i3,i4));
	}
}

void kmDnnLyActf::cBackward_ReLU(kcMat4f32& gx0, const kcMat4f32& gx1, const kcMat4f32& x1)
{
	dim3 bk, gd; gx0.GetBkGd(bk, gd);

	kckGReLU<float><<<gd,bk>>>(gx0, gx1, x1);
}

void kmDnnLyActf::cBackward_ReLU(kcMat4f16& gx0, const kcMat4f16& gx1, const kcMat4f16& x1)
{
	dim3 bk, gd; gx0.GetBkGd(bk, gd);

	kckGReLU<half><<<gd,bk>>>(gx0, gx1, x1);
}

// sigmoid - backward
template<typename T>
__global__ void kckGSigmoid(kckMat4<T> gx0, const kckMat4<T> gx1, const kckMat4<T> x1)
{
	KC4_STD_INDEX(gx0);	// i1, i2, i3, i4
	
	const float gx1v = gx1(i1,i2,i3,i4);
	const float  x1v =  x1(i1,i2,i3,i4);	
	T&          gx0v = gx0(i1,i2,i3,i4); 

	gx0v = float(gx0v) + x1v*(1.f - x1v)*gx1v;
}

void kmDnnLyActf::cBackward_Sigm(kcMat4f32& gx0, const kcMat4f32& gx1, const kcMat4f32& x1)
{
	dim3 bk, gd; gx0.GetBkGd(bk, gd);

	kckGSigmoid<float><<<gd,bk>>>(gx0, gx1, x1);
}

void kmDnnLyActf::cBackward_Sigm(kcMat4f16& gx0, const kcMat4f16& gx1, const kcMat4f16& x1)
{
	dim3 bk, gd; gx0.GetBkGd(bk, gd);

	kckGSigmoid<half><<<gd,bk>>>(gx0, gx1, x1);
}

// softmax - forward
template<typename T>
__global__ void kckSoftmax(kckMat4<T> x1, const kckMat4<T> x0)
{
	// declare shared memory
	extern __shared__ float sm[];

	KC_RDC_INDEX_EXDM1(x1); // ix1, ix2, ie1

	const int i4 = ie1;

	/////////////////////////////// step1
	// get max ... max_x

	// pre-reduction
	float val = FLOAT_MIN;	

	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2; if(i1 >= x1.n1) continue;

		for(int i2 = 0; i2 < x1.n2; ++i2)
		for(int i3 = 0; i3 < x1.n3; ++i3)
		{
			const float x0v = x0(i1,i2,i3,i4);

			if(val < x0v) val = x0v;
		}
	}
	// reduction
	sm[ix1] = val;

	for(int size_s = nx1>>1; size_s > 0; size_s >>= 1)
	{
		__syncthreads();
		if(ix1 < size_s)
		{
			if(sm[ix1] < sm[ix1 + size_s]) sm[ix1] = sm[ix1 + size_s];
		}
	}
	// update
	const float max_x = sm[0];

	//////////////////////////////////// step2
	// get sum.... sum_ex

	// pre-reduction
	val = 0;

	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2; if(i1 >= x1.n1) continue;

		for(int i2 = 0; i2 < x1.n2; ++i2)
		for(int i3 = 0; i3 < x1.n3; ++i3)
		{
			val += expf(float(x0(i1,i2,i3,i4)) - max_x);
		}
	}
	// reduction
	sm[ix1] = val;

	for(int size_s = nx1>>1; size_s > 0; size_s >>= 1)
	{
		__syncthreads();
		if(ix1 < size_s) sm[ix1] += sm[ix1 + size_s];
	}
	// update
	const float sum_ex = sm[0];

	//////////////////////////////////// step3
	// update x1

	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2; if(i1 >= x1.n1) continue;

		for(int i2 = 0; i2 < x1.n2; ++i2)
		for(int i3 = 0; i3 < x1.n3; ++i3)
		{
			x1(i1,i2,i3,i4) = expf(float(x0(i1,i2,i3,i4)) - max_x)/sum_ex;
		}
	}
}

void kmDnnLyActf::Softmax(kcMat4f32& x1, const kcMat4f32& x0, cudaStream_t s)
{
	dim3 bk, gd; int sm; x1.GetBkGdRdc(bk, gd, sm, 4);
		
	kckSoftmax<float><<<gd,bk,sm,s>>>(x1, x0);
}

void kmDnnLyActf::Softmax(kcMat4f16& x1, const kcMat4f16& x0, cudaStream_t s)
{
	dim3 bk, gd; int sm; x1.GetBkGdRdc(bk, gd, sm, 4);
		
	kckSoftmax<half><<<gd,bk,2*sm,s>>>(x1, x0);
}

// softmax - backward
template<typename T>
__global__ void kckGSoftmax(kckMat4<T> gx0, const kckMat4<T> gx1, const kckMat4<T> x1)
{
	// declare shared memory
	extern __shared__ float sm[];

	KC_RDC_INDEX_EXDM1(x1); // ix1, ix2, ie1

	const int i4 = ie1;

	/////////////////////////////// step1
	// get sum_x1gx1

	// pre-reduction
	float val = 0;

	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2; if(i1 >= x1.n1) continue;

		for(int i2 = 0; i2 < x1.n2; ++i2)
		for(int i3 = 0; i3 < x1.n3; ++i3)
		{
			val += float(x1(i1,i2,i3,i4))*float(gx1(i1,i2,i3,i4));
		}
	}

	// reduction
	sm[ix1] = val;

	for(int size_s = nx1>>1; size_s > 0; size_s >>= 1)
	{
		__syncthreads();

		if(ix1 < size_s) sm[ix1] += sm[ix1 + size_s];
	}

	// update
	const float sum_x1gx1 = sm[0];

	/////////////////////////////// step2
	// update gx0

	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2; if(i1 >= x1.n1) continue;

		for(int i2 = 0; i2 < x1.n2; ++i2)
		for(int i3 = 0; i3 < x1.n3; ++i3)
		{
			T& gx0r = gx0(i1,i2,i3,i4);

			gx0r = float(gx0r) + float(x1(i1,i2,i3,i4))*(float(gx1(i1,i2,i3,i4)) - sum_x1gx1);
		}
	}
}

void kmDnnLyActf::cBackward_Smax(kcMat4f32& gx0, const kcMat4f32& gx1, const kcMat4f32& x1)
{
	dim3 bk, gd; int sm; gx1.GetBkGdRdc(bk, gd, sm, 4);
		
	kckGSoftmax<float><<<gd,bk,sm>>>(gx0, gx1, x1);
}

void kmDnnLyActf::cBackward_Smax(kcMat4f16& gx0, const kcMat4f16& gx1, const kcMat4f16& x1)
{
	dim3 bk, gd; int sm; gx1.GetBkGdRdc(bk, gd, sm, 4);
		
	kckGSoftmax<half><<<gd,bk,2*sm>>>(gx0, gx1, x1);
}

////////////////////////////////////
// kmDnnLyFull's members

// x1 : n1, nb
// x0 : n0, nb
// w0 : n1, n0
// b0 : n1, 1
template<typename T>
__global__ void kckDropoutFull(kckMat2<T> x1, const kckMat1i8 mask0)
{
	// i1(x1i1), i2(ib)
	KC2_STD_INDEX(x1);

	if(mask0(i1) == 1) x1(i1,i2) = 0;
}

void kmDnnLyFull::cDropout(kmMat1<kcMat4f32>& x, const kcMat3i8& mask0)
{
	// init variables and parameters		
	const int64     n_bat = x(_x0_idx).N4();		
	      kcMat2f32 x1(x(_x1_idx).P(), _nx1, n_bat);
	const kcMat1i8  mask(mask0.P(), _nx1);

	dim3 bk, gd; x1.GetBkGd(bk, gd);

	kckDropoutFull<float><<<gd,bk>>>(x1, mask);
}

void kmDnnLyFull::cDropout(kmMat1<kcMat4f16>& x, const kcMat3i8& mask0)
{
	// init variables and parameters		
	const int64     n_bat = x(_x0_idx).N4();		
	      kcMat2f16 x1(x(_x1_idx).P(), _nx1, n_bat);
	const kcMat1i8  mask(mask0.P(), _nx1);

	dim3 bk, gd; x1.GetBkGd(bk, gd);

	kckDropoutFull<half><<<gd,bk>>>(x1, mask);
}

// x1 : n1, nb
// x0 : n0, nb
// w0 : n1, n0
// b0 : n1, 1
template<typename T>
__global__ void kckForwardFull(kckMat2<T> x1, const kckMat2<T> x0, 
	                     const kckMat2<T> w0, const kckMat2<T> b0)
{
	// i1(x1i1), i2(ib)... loop(x0i1)
	KC2_STD_INDEX(x1);

	float val = b0(i1,0);

	for(int j = 0; j < x0.n1; ++j) val += float(w0(i1, j))*float(x0(j, i2));
		
	x1(i1,i2) = val;
}

void kmDnnLyFull::cForward(kmMat1<kcMat4f32>& x, const kcMat4f32& w0, const kcMat2f32& b0, const bool istrain)
{
	const int64     n_bat = x(_x0_idx).N4();
	const int64     nx0   = x(_x0_idx).N()/n_bat;
	const kcMat2f32 w0_(w0.P(), _nx1, nx0);
	const kcMat2f32 x0(x(_x0_idx).P(),  nx0, n_bat);
	      kcMat2f32 x1(x(_x1_idx).P(), _nx1, n_bat);

	dim3 bk, gd; x1.GetBkGd(bk, gd);
		
	kckForwardFull<float><<<gd,bk>>>(x1, x0, w0_, b0);
}

void kmDnnLyFull::cForward(kmMat1<kcMat4f16>& x, const kcMat4f16& w0, const kcMat2f16& b0, const bool istrain)
{	
	const int64     n_bat = x(_x0_idx).N4();
	const int64     nx0   = x(_x0_idx).N()/n_bat;
	const kcMat2f16 w0_(w0.P(), _nx1, nx0);
	const kcMat2f16 x0(x(_x0_idx).P(),  nx0, n_bat);
	      kcMat2f16 x1(x(_x1_idx).P(), _nx1, n_bat);

	dim3 bk, gd; x1.GetBkGd(bk, gd);

	kckForwardFull<half><<<gd,bk>>>(x1, x0, w0_, b0);
}

// gx1, x1 : n1, nb
// gx0, x0 : n0, nb
// gw0, w0 : n1, n0
// gb0, b0 : n1, 1
template<typename T>
__global__ void kckBackwardFull(kckMat2<T> gx0,       kckMat2<T> gw0,       kckMat2<T> gb0, 
	                      const kckMat2<T>  x0, const kckMat2<T>  w0, const kckMat2<T> gx1)
{
	// i1(ix1), i2(ix0)... loop(ib)
	KC2_STD_INDEX(gw0);

	// init parameters
	const int nb = gx0.n2;

	// calc gb0
	if(i2 == 0)
	{
		float gb0v = 0;
		for(int ib = 0; ib < nb; ++ib) gb0v += gx1(i1, ib);
		gb0(i1, 0) = gb0v;
	}

	// calc gw0, gx0
	const float  w0v = w0(i1,i2);
	      float gw0v = 0;

	for(int ib = 0; ib < nb; ++ib)
	{
		const float gx1v = gx1(i1, ib);

		gw0v += gx1v*x0(i2, ib);

		atomicAdd(&gx0(i2, ib), gx1v*w0v);
	}
	gw0(i1,i2) = gw0v;
}

__global__ void kckBackwardFull_wb(kckMat2f16 gw0,       kckMat2f16 gb0, 
	                         const kckMat2f16  x0, const kckMat2f16 w0, const kckMat2f16 gx1)
{
	// i1(ix1), i2(ix0)... loop(ib)
	KC2_STD_INDEX(gw0);

	// init parameters
	const int nb = gx1.n2;

	// calc gb0
	if(i2 == 0)
	{
		float gb0v = 0.f;
		for(int ib = 0; ib < nb; ++ib) gb0v += float(gx1(i1, ib));
		gb0(i1, 0) = gb0v;
	}

	// calc gw0, gb0
	const float  w0v = w0(i1,i2);
	      float gw0v = 0.f;

	for(int ib = 0; ib < nb; ++ib)
	{
		const float gx1v = gx1(i1, ib);

		gw0v += gx1v*float(x0(i2, ib));
	}
	gw0(i1,i2) = gw0v;
}

__global__ void kckBackwardFull_x(kckMat2f16 gx0,
	                        const kckMat2f16  w0, const kckMat2f16 gx1)
{
	// i1(ix0), i2(ib)... loop(ix1)
	KC2_STD_INDEX(gx0);
		
	// calc gw0, gb0
	const float  w0v = w0(i1,i2);
	      float gx0v = 0.f;

	for(int ix1 = 0; ix1 < gx1.n1; ++ix1)
	{
		gx0v += float(gx1(ix1,i2))*float(w0(ix1, i1));
	}
	half& gx0r = gx0(i1,i2);

	gx0r = float(gx0r) + gx0v;
}

void kmDnnLyFull::cBackward(kmMat1<kcMat4f32>& gx,       kcMat4f32& gw0,       kcMat2f32& gb0,
                      const kmMat1<kcMat4f32>&  x, const kcMat4f32&  w0, const kcMat2f32&  b0)
{
	// init variable and parameters
	const int64     n_bat = x(_x0_idx).N4();
	const int64     nx0   = x(_x0_idx).N()/n_bat;
	const kcMat2f32  w0_( w0.P(), _nx1, nx0);
	      kcMat2f32 gw0_(gw0.P(), _nx1, nx0);

	const kcMat2f32  x0( x(_x0_idx).P(),  nx0, n_bat);
	      kcMat2f32 gx0(gx(_x0_idx).P(),  nx0, n_bat);
	const kcMat2f32 gx1(gx(_x1_idx).P(), _nx1, n_bat);

	// calc dl/dw, dl/db and dl/dx0... gw0, gb0, gx0
	// * Note that gx0 was already set as zero in kmDnn::Backward()
	//KC_CHECK_TIME_START;

	dim3 bk, gd; gw0.GetBkGd(bk, gd);

	kckBackwardFull<float><<<gd,bk>>>(gx0, gw0, gb0, x0, w0, gx1);

	//KC_CHECK_TIME_END("kckBackwardFull");

	KC_CHECK_ERROR("kmDnnLyFull::cBackwardFull");
}

void kmDnnLyFull::cBackward(kmMat1<kcMat4f16>& gx,       kcMat4f16& gw0,       kcMat2f16& gb0,
                      const kmMat1<kcMat4f16>&  x, const kcMat4f16&  w0, const kcMat2f16&  b0)
{
	// init variable and parameters
	const int64     n_bat = x(_x0_idx).N4();
	const int64     nx0   = x(_x0_idx).N()/n_bat;
	const kcMat2f16  w0_( w0.P(), _nx1, nx0);
	      kcMat2f16 gw0_(gw0.P(), _nx1, nx0);

	const kcMat2f16  x0( x(_x0_idx).P(),  nx0, n_bat);
	      kcMat2f16 gx0(gx(_x0_idx).P(),  nx0, n_bat);
	const kcMat2f16 gx1(gx(_x1_idx).P(), _nx1, n_bat);

	// calc dl/dw, dl/db .. gw0, gb0	
	//KC_CHECK_TIME_START;
	{
		dim3 bk, gd; gw0.GetBkGd(bk, gd);

		kckBackwardFull_wb<<<gd,bk>>>(gw0, gb0, x0, w0, gx1);
	}

	// calc dl/dx0... gx0
	// * Note that gx0 was already set as zero in kmDnn::Backward()
	{
		dim3 bk, gd; gx0.GetBkGd(bk, gd);

		kckBackwardFull_x<<<gd,bk>>>(gx0, w0, gx1);
	}
	//KC_CHECK_TIME_END("kckBackwardFull");

	KC_CHECK_ERROR("kmDnnLyFull::cBackwardFull");
}

////////////////////////////////////
// kmDnnLyConv's members

// x1 : x1n1, x1n2, nf, nb
// x0 : x0n1, x0n2, nc, nb
// w0 :   nw,   nh, nc, nf
// b0 :   nf,    1
__global__ void kckForwardConv0(kckMat4f32 x1, const kckMat4f32 x0, 
	                      const kckMat4f32 w0, const kckMat2f32 b0,
	                      const int      strd, const int      zpad)
{
	// i1(x1i1), i2(x1i2), i3(if), i4(ib) ... loop(ic, ih, iw)
	KC4_STD_INDEX(x1);

	// init parameters
	const int nw = w0.n1, nh = w0.n2, nc = w0.n3;
	
	// calc convolution... main loop
	float x1v = 0;

	for(int ic = 0; ic < nc; ++ic)
	{
		int k2 = strd*i2 - zpad; // x0i2

		for(int ih = 0; ih < nh; ++ih, ++k2)
		{	
			if(k2 < 0 || k2 >= x0.n2) continue; // check ouf of boundary

			int k1 = strd*i1 - zpad; // x0i1

			for(int iw = 0; iw < nw; ++iw, ++k1)
			{
				if(k1 < 0 || k1 >= x0.n1) continue; // check out of boundary

				// sum weight-multification
				x1v += w0(iw,ih,ic,i3)*x0(k1,k2,ic,i4);
			}
		}
	}
	// sum bias-multification
	x1(i1,i2,i3,i4) = x1v + b0(i3,0);
}

// x1 : x1n1, x1n2, nf, nb
// x0 : x0n1, x0n2, nc, nb
// w0 :   nw,   nh, nc, nf
// b0 :   nf,    1
template<bool no_bias, typename T>
__global__ void kckForwardConv(kckMat4<T> x1, const kckMat4<T> x0, 
	                     const kckMat4<T> w0, const kckMat2<T> b0,
	                     const int      strd, const int      zpad)
{	
	// i1(x1i1), i2(x1i2), i3(if), i4(ib)... loop(ic, ih, iw)
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, i3 = blockIdx.y, i4 = blockIdx.z;
	const int nx1 = blockDim .x;
	const int ix  = ix1 + nx1*ix2;
	if(ix >= x1.n1*x1.n2) return;

	const int i2 = ix/x1.n1;
	const int i1 = ix - i2*x1.n1;

	// init parameters
	const int nw = w0.n1, nh = w0.n2, nc = w0.n3;
	
	// calc convolution... main loop
	float x1v = 0;

	for(int ic = 0; ic < nc; ++ic)
	{
		int k2 = strd*i2 - zpad; // x0i2

		for(int ih = 0; ih < nh; ++ih, ++k2)
		{	
			if(k2 < 0 || k2 >= x0.n2) continue; // check ouf of boundary

			int k1 = strd*i1 - zpad; // x0i1

			for(int iw = 0; iw < nw; ++iw, ++k1)
			{
				if(k1 < 0 || k1 >= x0.n1) continue; // check out of boundary

				// sum weight-multification
				x1v += float(w0(iw,ih,ic,i3))*float(x0(k1,k2,ic,i4));
			}
		}
	}
	// sum bias-multification	
	if(no_bias) x1(i1,i2,i3,i4) = x1v;
	else        x1(i1,i2,i3,i4) = x1v + float(b0(i3,0));
}

void kmDnnLyConv::cForward(kmMat1<kcMat4f32>& x, const kcMat4f32& w0, 
	                        const kcMat2f32& b0, const bool istrain)
{
	kcMat4f32& x0   = x(_x0_idx);
	kcMat4f32& x1   = x(_x1_idx);
	const uint n1n2 = (uint) (x1.N1()*x1.N2());
	const uint nx1  = MIN(n1n2, 1024);

	dim3 bk = {nx1, 1, 1};
	dim3 gd = {(nx1 -1)/n1n2 + 1, (uint)x1.N3(), (uint)x1.N4()};

	if(IsNoBias()) kckForwardConv<1,float><<<gd,bk>>>(x1, x0, w0, b0, (int)_strd, (int)_zpad);
	else           kckForwardConv<0,float><<<gd,bk>>>(x1, x0, w0, b0, (int)_strd, (int)_zpad);

	KC_CHECK_ERROR("kmDnnLyConv::cForward");
}

void kmDnnLyConv::cForward(kmMat1<kcMat4f16>& x, const kcMat4f16& w0, 
	                        const kcMat2f16& b0, const bool istrain)
{
	kcMat4f16& x0   = x(_x0_idx);
	kcMat4f16& x1   = x(_x1_idx);
	const uint n1n2 = (uint) (x1.N1()*x1.N2());
	const uint nx1  = MIN(n1n2, 1024);

	dim3 bk = {nx1, 1, 1};
	dim3 gd = {(nx1 -1)/n1n2 + 1, (uint)x1.N3(), (uint)x1.N4()};

	if(IsNoBias()) kckForwardConv<1,half><<<gd,bk>>>(x1, x0, w0, b0, (int)_strd, (int)_zpad);
	else           kckForwardConv<0,half><<<gd,bk>>>(x1, x0, w0, b0, (int)_strd, (int)_zpad);

	KC_CHECK_ERROR("kmDnnLyConv::cForward");
}

// gx1, x1 : x1n1, x1n2, nf, nb
// gw0, w0 :   nw,   nh, nc, nf
// gx0, x0 : x0n1, x0n2, nc, nb
template<typename T>
__global__ void kckBackwardConv_gx0(kckMat4<T> gx0, const kckMat4<T>  x0,
	                          const kckMat4<T>  w0, const kckMat4<T> gx1,
	                          const int       strd, const int      zpad)
{
	// i1(x0i1), i2(x0i2), i3(ic), i4(ib)... loop(if, ih, iw)
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, i3 = blockIdx.y, i4 = blockIdx.z;
	const int nx1 = blockDim .x;
	const int ix  = ix1 + nx1*ix2;
	if(ix >= gx0.n1*gx0.n2) return;

	const int i2 = ix/gx0.n1;
	const int i1 = ix - i2*gx0.n1;

	// init parameters
	const int nw = w0.n1, nh = w0.n2, nf = w0.n4;

	// calc gx0... main loop
	float gx0v = 0;

	if(strd == 1)
	{
		for(int i_f = 0; i_f < nf; ++i_f)
		for(int ih = 0, j2 = i2 + zpad; ih < nh; ++ih, --j2)
		{
			// set and check index... x1i2
			if(j2 < 0 || j2 >= gx1.n2) continue;
		
			for(int iw = 0, j1 = i1 + zpad; iw < nw; ++iw, --j1)
			{
				// set and check index... x1i1
				if(j1 < 0 || j1 >= gx1.n1) continue;
		
				// sum gx0v
				gx0v += float(gx1(j1,j2,i_f,i4))*float(w0(iw,ih,i3,i_f));
			}
		}
	}
	else
	{
		for(int i_f = 0; i_f < nf; ++i_f)
		for(int ih = 0, j2 = i2 + zpad; ih < nh; ++ih, --j2)
		{
			// set and check index... x1i2
			if(j2 % strd != 0 || j2 < 0) continue; else j2 /= strd;
			if(j2 >= gx1.n2) continue;

			for(int iw = 0, j1 = i1 + zpad; iw < nw; ++iw, --j1)
			{
				// set and check index... x1i1
				if(j1 % strd != 0 || j1 < 0) continue; else j1 /= strd;
				if(j1 >= gx1.n1) continue;

				// sum gx0v
				gx0v += float(gx1(j1,j2,i_f,i4))*float(w0(iw,ih,i3,i_f));
			}
		}
	}
	// update gx0
	T& gx0r = gx0(i1,i2,i3,i4);

	gx0r = float(gx0r) + gx0v;	
}

// gx1, x1 : x1n1, x1n2, nf, nb
// gx0, x0 : x0n1, x0n2, nc, nb
// gw0, w0 :   nw,   nh, nc, nf
template<typename T>
__global__ void kckBackwardConv_gw0(kckMat4<T> gw0,
	                          const kckMat4<T>  x0, const kckMat4<T> gx1,
	                          const int       strd, const int      zpad)
{
	// i1(iw), i2(ih), i3(ic), i4(if)... loop(ib, x1i2, x1i1)
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, i3 = blockIdx.y;
	const int nx1 = blockDim .x;
	const int ix  = ix1 + nx1*ix2;
	if(ix >= gw0.n1*gw0.n2*gw0.n4) return;

	const int n12 = gw0.n1*gw0.n2;
	      int i1  = ix;
	const int i4  = i1/n12;    i1 -= i4*n12;
	const int i2  = i1/gw0.n1; i1 -= i2*gw0.n1;
	
	// calc gw0... main loop
	float gw0v = 0;

	for(int ib = 0; ib < gx1.n4; ++ib)
	for(int j2 = 0, k2 = i2 - zpad; j2 < gx1.n2; ++j2, k2 += strd)
	{
		// check index... x0i2
		// k2 = strd*j2 - zpad + i2;
		if(k2 < 0 || k2 >= x0.n2) continue;

		for(int j1 = 0, k1 = i1 - zpad; j1 < gx1.n1; ++j1, k1 += strd)
		{
			// check index... x0i1
			// k1 = strd*j1 - zpad + i1;
			if(k1 < 0 || k1 >= x0.n1) continue;

			// sum gw0v
			gw0v += float(gx1(j1,j2,i4,ib))*float(x0(k1,k2,i3,ib));
		}
	}
	// update
	gw0(i1,i2,i3,i4) = gw0v;
}

// gx1, x1 : x1n1, x1n2, nf, nb
// gb0, b0 :   nf,    1
template<typename T>
__global__ void kckBackwardConv_gb0(kckMat2<T> gb0,const kckMat4<T> gx1)
{
	// declare shared memory
	extern __shared__ float sm[];

	KC_RDC_INDEX_EXDM1(gx1); // ix1, ix2, nx1, ie1

	const int i_f = ie1;

	// pre-reduction
	float val = 0;

	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2; if(i1 >= gx1.n1) continue;

		for(int i4 = 0; i4 < gx1.n4; ++i4)
		for(int i2 = 0; i2 < gx1.n2; ++i2)
		{
			val += float(gx1(i1,i2,i_f,i4));
		}
	}
	// reduction
	KC_RDC_REDUCTION(val);
		
	// update
	if(ix1 == 0) gb0(i_f,0) = val;
}

void kmDnnLyConv::cBackward(kmMat1<kcMat4f32>& gx,       kcMat4f32& gw0,       kcMat2f32& gb0,
		              const kmMat1<kcMat4f32>&  x, const kcMat4f32&  w0, const kcMat2f32&  b0)
{
	// init variable and parameters
	const kcMat4f32&  x0 =  x(_x0_idx);
	      kcMat4f32& gx0 = gx(_x0_idx);
	const kcMat4f32& gx1 = gx(_x1_idx);

	//KC_CHECK_TIME_START
	{	
		const uint n1n2 = (uint) (gx0.N1()*gx0.N2());
		const uint nx1  = MIN(n1n2, 1024);

		dim3 bk = {nx1, 1, 1};
		dim3 gd = {(nx1 -1)/n1n2 + 1, (uint)gx0.N3(), (uint)gx0.N4()};

		kckBackwardConv_gx0<float><<<gd,bk>>>(gx0, x0, w0, gx1, (int)_strd, (int)_zpad);
	}
	//KC_CHECK_TIME_END("BackwardConv_gx0");
	KC_CHECK_ERROR("kmDnnLyConv::cBackwardConv_gx0");

	// calc gw0
	//KC_CHECK_TIME_START
	if(!IsFixed())
	{
		const uint n124 = (uint) (gw0.N1()*gw0.N2()*gw0.N4());
		const uint nx1  = MIN(n124, 1024);

		dim3 bk = {nx1, 1, 1};
		dim3 gd = {(nx1 -1)/n124 + 1, (uint)gw0.N3(), 1};

		kckBackwardConv_gw0<float><<<gd,bk>>>(gw0, x0, gx1, (int)_strd, (int)_zpad);
	}
	//KC_CHECK_TIME_END("BackwardConv_gw0");
	KC_CHECK_ERROR("kmDnnLyConv::cBackwardConv_gw0");

	// calc gb
	if(!IsNoBias() && !IsFixed())
	{
		dim3 bk, gd; int sm; gx1.GetBkGdRdc(bk, gd, sm, 3);

		kckBackwardConv_gb0<float><<<gd,bk,sm>>>(gb0, gx1);
	}
}

void kmDnnLyConv::cBackward(kmMat1<kcMat4f16>& gx,       kcMat4f16& gw0,       kcMat2f16& gb0,
		              const kmMat1<kcMat4f16>&  x, const kcMat4f16&  w0, const kcMat2f16&  b0)
{
	// init variable and parameters
	const kcMat4f16&  x0 =  x(_x0_idx);
	      kcMat4f16& gx0 = gx(_x0_idx);
	const kcMat4f16& gx1 = gx(_x1_idx);

	//KC_CHECK_TIME_START
	{	
		const uint n1n2 = (uint) (gx0.N1()*gx0.N2());
		const uint nx1  = MIN(n1n2, 1024);

		dim3 bk = {nx1, 1, 1};
		dim3 gd = {(nx1 -1)/n1n2 + 1, (uint)gx0.N3(), (uint)gx0.N4()};

		kckBackwardConv_gx0<half><<<gd,bk>>>(gx0, x0, w0, gx1, (int)_strd, (int)_zpad);
	}
	//KC_CHECK_TIME_END("BackwardConv_gx0");
	KC_CHECK_ERROR("kmDnnLyConv::cBackwardConv_gx0");

	// calc gw0
	//KC_CHECK_TIME_START
	if(!IsFixed())
	{
		const uint n124 = (uint) (gw0.N1()*gw0.N2()*gw0.N4());
		const uint nx1  = MIN(n124, 1024);

		dim3 bk = {nx1, 1, 1};
		dim3 gd = {(nx1 -1)/n124 + 1, (uint)gw0.N3(), 1};

		kckBackwardConv_gw0<half><<<gd,bk>>>(gw0, x0, gx1, (int)_strd, (int)_zpad);
	}
	//KC_CHECK_TIME_END("BackwardConv_gw0");
	KC_CHECK_ERROR("kmDnnLyConv::cBackwardConv_gw0");

	// calc gb
	if(!IsNoBias() && !IsFixed())
	{
		dim3 bk, gd; int sm; gx1.GetBkGdRdc(bk, gd, sm, 3);

		kckBackwardConv_gb0<half><<<gd,bk,2*sm>>>(gb0, gx1);
	}
}

////////////////////////////////////
// kmDnnLyCnvf's members

// x1 : x1n1, x1n2, ncf, nb
// x0 : x0n1, x0n2,  nc, nb
// w0 :   nw,   nh,  nf, 1
// b0 :   nf,    1
template<bool no_bias, typename T>
__global__ void kckForwardCnvf(kckMat4<T> x1, const kckMat4<T> x0, 
	                     const kckMat4<T> w0, const kckMat2<T> b0,
	                     const int      strd, const int      zpad)
{	
	// i1(x1i1), i2(x1i2), i3(icf), i4(ib)... loop(ih, iw)
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, i3 = blockIdx.y, i4 = blockIdx.z;
	const int nx1 = blockDim .x;
	const int ix  = ix1 + nx1*ix2;
	if(ix >= x1.n1*x1.n2) return;

	const int i2 = ix/x1.n1;
	const int i1 = ix - i2*x1.n1;

	// init parameters
	const int nw = w0.n1, nh = w0.n2, nc = x0.n3; // nf = w0.n3

	const int it = i3/nc; // index of filter
	const int ic = i3 - it*nc;
	
	// calc convolution... main loop
	float x1v = 0;

	int k2 = strd*i2 - zpad; // x0i2

	for(int ih = 0; ih < nh; ++ih, ++k2)
	{	
		if(k2 < 0 || k2 >= x0.n2) continue; // check ouf of boundary

		int k1 = strd*i1 - zpad; // x0i1

		for(int iw = 0; iw < nw; ++iw, ++k1)
		{
			if(k1 < 0 || k1 >= x0.n1) continue; // check out of boundary

			// sum weight-multification
			x1v += float(w0(iw,ih,it,0))*float(x0(k1,k2,ic,i4));
		}
	}
	
	// sum bias-multification
	if(no_bias) x1(i1,i2,i3,i4) = T(x1v);
	else        x1(i1,i2,i3,i4) = T(x1v + (float)b0(i3,0));
}

void kmDnnLyCnvf::cForward(kmMat1<kcMat4f32>& x, const kcMat4f32& w0, const kcMat2f32&  b0, const bool istrain)
{
	kcMat4f32& x0   = x(_x0_idx);
	kcMat4f32& x1   = x(_x1_idx); // * Note that x1 doesn't need to be set as zero.
	const uint n1n2 = (uint) (x1.N1()*x1.N2());
	const uint nx1  = MIN(n1n2, 1024);

	dim3 bk = {nx1, 1, 1};
	dim3 gd = {(nx1 -1)/n1n2 + 1, (uint)x1.N3(), (uint)x1.N4()};

	if(IsNoBias()) kckForwardCnvf<1,float><<<gd,bk>>>(x1, x0, w0, b0, (int)_strd, (int)_zpad);
	else           kckForwardCnvf<0,float><<<gd,bk>>>(x1, x0, w0, b0, (int)_strd, (int)_zpad);

	KC_CHECK_ERROR("kmDnnLyCnvf::cForward");
}

void kmDnnLyCnvf::cForward(kmMat1<kcMat4f16>& x, const kcMat4f16& w0, const kcMat2f16&  b0, const bool istrain)
{
	kcMat4f16& x0   = x(_x0_idx);
	kcMat4f16& x1   = x(_x1_idx); // * Note that x1 doesn't need to be set as zero.
	const uint n1n2 = (uint) (x1.N1()*x1.N2());
	const uint nx1  = MIN(n1n2, 1024);

	dim3 bk = {nx1, 1, 1};
	dim3 gd = {(nx1 -1)/n1n2 + 1, (uint)x1.N3(), (uint)x1.N4()};

	if(IsNoBias()) kckForwardCnvf<1,half><<<gd,bk>>>(x1, x0, w0, b0, (int)_strd, (int)_zpad);
	else           kckForwardCnvf<0,half><<<gd,bk>>>(x1, x0, w0, b0, (int)_strd, (int)_zpad);

	KC_CHECK_ERROR("kmDnnLyCnvf::cForward");
}

// gx1, x1 : x1n1, x1n2, ncf, nb
// gx0, x0 : x0n1, x0n2,  nc, nb
// gw0, w0 :   nw,   nh,  nf, 1
template<typename T>
__global__ void kckBackwardCnvf_gx0(kckMat4<T> gx0, const kckMat4<T>  x0,
	                          const kckMat4<T>  w0, const kckMat4<T> gx1,
	                          const int       strd, const int      zpad)
{
	// i1(x0i1), i2(x0i2), i3(ic), i4(ib)... loop(if, ih, iw)
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, i3 = blockIdx.y, i4 = blockIdx.z;
	const int nx1 = blockDim .x;
	const int ix  = ix1 + nx1*ix2;
	if(ix >= gx0.n1*gx0.n2) return;

	const int i2 = ix/gx0.n1;
	const int i1 = ix - i2*gx0.n1;

	// init parameters
	const int nw = w0.n1, nh = w0.n2, nt = w0.n3, nc = gx0.n3;

	// calc gx0... main loop
	float gx0v = 0;

	if(strd == 1)
	{
		for(int it = 0, ict = i3; it < nt; ++it, ict += nc)
		{
			for(int ih = 0, j2 = i2 + zpad; ih < nh; ++ih, --j2)
			{
				// set and check index... x1i2
				if(j2 < 0 || j2 >= gx1.n2) continue;
			
				for(int iw = 0, j1 = i1 + zpad; iw < nw; ++iw, --j1)
				{
					// set and check index... x1i1
					if(j1 < 0 || j1 >= gx1.n1) continue;
			
					// sum gx0v
					gx0v += float(gx1(j1,j2,ict,i4))*float(w0(iw,ih,it,0));
				}
			}
		}
	}
	else
	{
		for(int it = 0, ict = i3; it < nt; ++it, ict += nc)
		for(int ih = 0, j2 = i2 + zpad; ih < nh; ++ih, --j2)
		{
			// set and check index... x1i2
			if(j2 % strd != 0 || j2 < 0) continue; else j2 /= strd;
			if(j2 >= gx1.n2) continue;

			for(int iw = 0, j1 = i1 + zpad; iw < nw; ++iw, --j1)
			{
				// set and check index... x1i1
				if(j1 % strd != 0 || j1 < 0) continue; else j1 /= strd;
				if(j1 >= gx1.n1) continue;

				// sum gx0v
				gx0v += float(gx1(j1,j2,ict,i4))*float(w0(iw,ih,it,0));
			}
		}
	}
	// update gx0
	T& gx0r = gx0(i1,i2,i3,i4);

	gx0r = float(gx0r) + gx0v;
}

// gx1, x1 : x1n1, x1n2, ncf, nb
// gx0, x0 : x0n1, x0n2,  nc, nb
// gw0, w0 :   nw,   nh,  nf, 1
template<typename T>
__global__ void kckBackwardCnvf_gw0(kckMat4<T> gw0,
	                          const kckMat4<T>  x0, const kckMat4<T> gx1,
	                          const int       strd, const int       zpad)
{
	// i1(iw), i2(ih), i3(if)... loop(ib, ic, x1i2, x1i1)
	const int ix1 = threadIdx.x;
	const int ix2 = blockIdx .x, i3 = blockIdx.y;
	const int nx1 = blockDim .x;
	const int ix  = ix1 + nx1*ix2;
	if(ix >= gw0.n1*gw0.n2) return;
		
	      int i1  = ix;	
	const int i2  = i1/gw0.n1; i1 -= i2*gw0.n1;

	// init parameters
	const int nc =  x0.n3, nb = x0.n4;
	
	// calc gw0... main loop
	float gw0v = 0;

	for(int ib = 0; ib < nb; ++ib)
	for(int ic = 0, ict = nc*i3; ic < nc; ++ic, ++ict)
	for(int j2 = 0, k2 = i2 - zpad; j2 < gx1.n2; ++j2, k2 += strd)
	{
		// check index... x0i2
		// k2 = strd*j2 - zpad + i2;
		if(k2 < 0 || k2 >= x0.n2) continue;

		for(int j1 = 0, k1 = i1 - zpad; j1 < gx1.n1; ++j1, k1 += strd)
		{
			// check index... x0i1
			// k1 = strd*j1 - zpad + i1;
			if(k1 < 0 || k1 >= x0.n1) continue;

			// sum gw0v
			gw0v += float(gx1(j1,j2,ict,ib))*float(x0(k1,k2,ic,ib));
		}
	}
	// update
	gw0(i1,i2,i3,0) = gw0v;
}

// gx1, x1 : x1n1, x1n2, ncf, nb
// gb0, b0 :   nf,    1
template<typename T>
__global__ void kckBackwardCnvf_gb0(kckMat2<T> gb0,const kckMat4<T> gx1, const int nc)
{
	// declare shared memory
	extern __shared__ float sm[];

	KC_RDC_INDEX_EXDM1(gx1); // ix1, nx1, nx2, ie1

	const int it = ie1;

	// pre-reduction
	float val = 0;

	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2; if(i1 >= gx1.n1) continue;

		for(int i4 = 0; i4 < gx1.n4; ++i4) 
		for(int ic = 0, ict = nc*it; ic < nc; ++ic, ++ict)
		for(int i2 = 0; i2 < gx1.n2; ++i2)
		{
			val += float(gx1(i1,i2,ict,i4));
		}
	}
	// reduction
	KC_RDC_REDUCTION(val);
		
	// update
	if(ix1 == 0) gb0(it,0) = val;
}

void kmDnnLyCnvf::cBackward(kmMat1<kcMat4f32>& gx,       kcMat4f32& gw0,       kcMat2f32& gb0,
		              const kmMat1<kcMat4f32>&  x, const kcMat4f32&  w0, const kcMat2f32&  b0)
{
	// init variable and parameters
	const kcMat4f32&  x0 =  x(_x0_idx);
	      kcMat4f32& gx0 = gx(_x0_idx);
	const kcMat4f32& gx1 = gx(_x1_idx);

	// calc gx0
	//KC_CHECK_TIME_START
	{	
		const uint n12 = (uint) (gx0.N1()*gx0.N2());
		const uint nx1  = MIN(n12, 1024);

		dim3 bk = {nx1, 1, 1};
		dim3 gd = {(nx1 -1)/n12 + 1, (uint)gx0.N3(), (uint)gx0.N4()};

		kckBackwardCnvf_gx0<float><<<gd,bk>>>(gx0, x0, w0, gx1, (int)_strd, (int)_zpad);
	}
	//KC_CHECK_TIME_END("BackwardConv_gx0");
	KC_CHECK_ERROR("kmDnnLyCnvf::cBackwardCnvf_gx0");

	// calc gw0
	//KC_CHECK_TIME_START
	if(!IsFixed())
	{
		const uint n12 = (uint) (gw0.N1()*gw0.N2());
		const uint nx1  = MIN(n12, 1024);

		dim3 bk = {nx1, 1, 1};
		dim3 gd = {(nx1 -1)/n12 + 1, (uint)gw0.N3(), 1};

		kckBackwardCnvf_gw0<float><<<gd,bk>>>(gw0, x0, gx1, (int)_strd, (int)_zpad);
	}
	//KC_CHECK_TIME_END("BackwardConv_gw0");
	KC_CHECK_ERROR("kmDnnLyCnvf::cBackwardCnvf_gw0");

	// calc gb
	if(!IsNoBias() && !IsFixed())
	{	
		dim3 bk, gd; int sm; gx1.GetBkGdRdc(bk, gd, sm, 3); gd = (uint)gb0.N1();

		kckBackwardCnvf_gb0<float><<<gd,bk,sm>>>(gb0, gx1, (int)x0.N3());
	}
	KC_CHECK_ERROR("kmDnnLyCnvf::cBackwardCnvf_gb0");
}

void kmDnnLyCnvf::cBackward(kmMat1<kcMat4f16>& gx,       kcMat4f16& gw0,       kcMat2f16& gb0,
		              const kmMat1<kcMat4f16>&  x, const kcMat4f16&  w0, const kcMat2f16&  b0)
{
	// init variable and parameters
	const kcMat4f16&  x0 =  x(_x0_idx);
	      kcMat4f16& gx0 = gx(_x0_idx);
	const kcMat4f16& gx1 = gx(_x1_idx);

	// calc gx0
	//KC_CHECK_TIME_START
	{	
		const uint n12 = (uint) (gx0.N1()*gx0.N2());
		const uint nx1  = MIN(n12, 1024);

		dim3 bk = {nx1, 1, 1};
		dim3 gd = {(nx1 -1)/n12 + 1, (uint)gx0.N3(), (uint)gx0.N4()};

		kckBackwardCnvf_gx0<half><<<gd,bk>>>(gx0, x0, w0, gx1, (int)_strd, (int)_zpad);
	}
	//KC_CHECK_TIME_END("BackwardConv_gx0");
	KC_CHECK_ERROR("kmDnnLyCnvf::cBackwardCnvf_gx0");

	// calc gw0
	//KC_CHECK_TIME_START
	if(!IsFixed())
	{
		const uint n12 = (uint) (gw0.N1()*gw0.N2());
		const uint nx1  = MIN(n12, 1024);

		dim3 bk = {nx1, 1, 1};
		dim3 gd = {(nx1 -1)/n12 + 1, (uint)gw0.N3(), 1};

		kckBackwardCnvf_gw0<half><<<gd,bk>>>(gw0, x0, gx1, (int)_strd, (int)_zpad);
	}
	//KC_CHECK_TIME_END("BackwardConv_gw0");
	KC_CHECK_ERROR("kmDnnLyCnvf::cBackwardCnvf_gw0");

	// calc gb
	if(!IsNoBias() && !IsFixed())
	{	
		dim3 bk, gd; int sm; gx1.GetBkGdRdc(bk, gd, sm, 3); gd = (uint)gb0.N1();

		kckBackwardCnvf_gb0<half><<<gd,bk,2*sm>>>(gb0, gx1, (int)x0.N3());
	}
	KC_CHECK_ERROR("kmDnnLyCnvf::cBackwardCnvf_gb0");
}

////////////////////////////////////
// kmDnnLyPool's members

// x1 : x1n1, x1n2, n3, nb
// x0 : x0n1, x0n2, n3, nb
template<typename T>
__global__ void kckForwardPool_Max(kckMat4<T> x1, const kckMat4<T> x0, 
	                               const int nw, const int nh, const int strd)
{
	// i1, i2, i3, i4(ib)... loop(ih, iw)
	KC4_STD_INDEX(x1);

	// clac pooling... main loop
	float x1v = FLOAT_MIN;

	for(int ih = 0, k2 = strd*i2; ih < nh; ++ih, ++k2)
	{
		if(k2 >= x0.n2) continue; // check out of boundary

		for(int iw = 0, k1 = strd*i1; iw < nw; ++iw,  ++k1)
		{
			if(k1 >= x0.n1) continue; // check out of boundary

			x1v = MAX(x1v, float(x0(k1,k2,i3,i4)));
		}
	}
	// update
	x1(i1,i2,i3,i4) = x1v;
}

// x1 : x1n1, x1n2, n3, nb
// x0 : x0n1, x0n2, n3, nb
template<typename T>
__global__ void kckForwardPool_Avg(kckMat4<T> x1, const kckMat4<T> x0, 
	                               const int nw, const int nh, const int strd)
{
	// i1, i2, i3, i4(ib)... loop(ih, iw)
	KC4_STD_INDEX(x1);

	// clac pooling... main loop
	float x1v = 0; int num = 0;

	for(int ih = 0, k2 = strd*i2; ih < nh; ++ih, ++k2)
	{
		if(k2 >= x0.n2) continue; // check out of boundary

		for(int iw = 0, k1 = strd*i1; iw < nw; ++iw,  ++k1)
		{
			if(k1 >= x0.n1) continue; // check out of boundary

			x1v += float(x0(k1,k2,i3,i4)); ++num;
		}
	}
	// update
	x1(i1,i2,i3,i4) = x1v/float(num);
}

void kmDnnLyPool::cForward(kmMat1<kcMat4f32>& x, const kcMat4f32& w0, const kcMat2f32& b0, const bool istrain)
{
	// init variable and parameters
	const kcMat4f32& x0 = x(_x0_idx);
	      kcMat4f32& x1 = x(_x1_idx); x1.SetZero();

	dim3 bk, gd; x1.GetBkGd(bk, gd);

	int nw = (int)_nw, nh = (int)_nh, strd = (int)_strd;

	switch(_pool)
	{
	case POOL_MAX : kckForwardPool_Max<float><<<gd,bk>>>(x1, x0, nw, nh, strd); break;
	case POOL_AVG : kckForwardPool_Avg<float><<<gd,bk>>>(x1, x0, nw, nh, strd); break;
	}
}

void kmDnnLyPool::cForward(kmMat1<kcMat4f16>& x, const kcMat4f16& w0, const kcMat2f16& b0, const bool istrain)
{
	// init variable and parameters
	const kcMat4f16& x0 = x(_x0_idx);
	      kcMat4f16& x1 = x(_x1_idx); x1.SetZero();

	dim3 bk, gd; x1.GetBkGd(bk, gd);

	int nw = (int)_nw, nh = (int)_nh, strd = (int)_strd;

	switch(_pool)
	{
	case POOL_MAX : kckForwardPool_Max<half><<<gd,bk>>>(x1, x0, nw, nh, strd); break;
	case POOL_AVG : kckForwardPool_Avg<half><<<gd,bk>>>(x1, x0, nw, nh, strd); break;
	}
}

// gx1 : x1n1, x1n2, n3, nb
// gx0 : x0n1, x0n2, n3, nb
template<typename T >
__global__ void kckBackwardPool_Max(kckMat4<T> gx0, const kckMat4<T> x0, const kckMat4<T> gx1,
	                                  const int nw, const int nh, const int strd)
{
	// i1, i2, i3, i4(ib)... loop(ih, iw)
	KC4_STD_INDEX(gx1);

	// calc pooling... main loop
	float val = FLOAT_MIN; int max_k1 = 0, max_k2 = 0;

	for(int ih = 0, k2 = strd*i2; ih < nh; ++ih, ++k2)
	{
		if(k2 >= x0.n2) continue; // check out of boundary

		for(int iw = 0, k1 = strd*i1; iw < nw; ++iw,  ++k1)
		{
			if(k1 >= x0.n1) continue; // check out of boundary

			// find max
			const float x0v = x0(k1,k2,i3,i4);
			if(x0v > val)
			{
				val = x0v, max_k1 = k1, max_k2 = k2;
			}
		}
	}
	// update
	T& gx0r = gx0(max_k1,max_k2,i3,i4);
	
	gx0r = float(gx0r) + float(gx1(i1,i2,i3,i4));
}

// gx1 : x1n1, x1n2, n3, nb
// gx0 : x0n1, x0n2, n3, nb
template<typename T>
__global__ void kckBackwardPool_Avg(kckMat4<T> gx0, const kckMat4<T> gx1,
	                          const int nw, const int nh, const int strd)
{
	// i1, i2, i3, i4(ib)... loop(ih, iw)
	KC4_STD_INDEX(gx1);

	// calc num
	int num = 0;

	for(int ih = 0, k2 = strd*i2; ih < nh; ++ih, ++k2)
	{
		if(k2 >= gx0.n2) continue; // check out of boundary

		for(int iw = 0, k1 = strd*i1; iw < nw; ++iw,  ++k1)
		{
			if(k1 >= gx0.n1) continue; // check out of boundary
			++num;
		}
	}

	// calc gx0
	float gx1v = float(gx1(i1,i2,i3,i4))/float(num);

	for(int ih = 0, k2 = strd*i2; ih < nh; ++ih, ++k2)
	{
		if(k2 >= gx0.n2) continue; // check out of boundary

		for(int iw = 0, k1 = strd*i1; iw < nw; ++iw,  ++k1)
		{
			if(k1 >= gx0.n1) continue; // check out of boundary
			
			// update	
			T& gx0r = gx0(k1,k2,i3,i4);
			
			gx0r = float(gx0r) + gx1v;
		}
	}
}

void kmDnnLyPool::cBackward(kmMat1<kcMat4f32>& gx,       kcMat4f32& gw0,       kcMat2f32& gb0,
		              const kmMat1<kcMat4f32>&  x, const kcMat4f32&  w0, const kcMat2f32&  b0)
{	
	      kcMat4f32& gx0 = gx(_x0_idx);
	const kcMat4f32& gx1 = gx(_x1_idx);
	const kcMat4f32&  x0 =  x(_x0_idx);

	dim3 bk, gd; gx1.GetBkGd(bk, gd);

	int nw = (int)_nw, nh = (int)_nh, strd = (int)_strd;

	switch(_pool)
	{
	case POOL_MAX : kckBackwardPool_Max<float><<<gd,bk>>>(gx0, x0, gx1, nw, nh, strd); break;
	case POOL_AVG : kckBackwardPool_Avg<float><<<gd,bk>>>(gx0,     gx1, nw, nh, strd); break;
	}
}

void kmDnnLyPool::cBackward(kmMat1<kcMat4f16>& gx,       kcMat4f16& gw0,       kcMat2f16& gb0,
		              const kmMat1<kcMat4f16>&  x, const kcMat4f16&  w0, const kcMat2f16&  b0)
{	
	      kcMat4f16& gx0 = gx(_x0_idx);
	const kcMat4f16& gx1 = gx(_x1_idx);
	const kcMat4f16&  x0 =  x(_x0_idx);

	dim3 bk, gd; gx1.GetBkGd(bk, gd);

	int nw = (int)_nw, nh = (int)_nh, strd = (int)_strd;

	switch(_pool)
	{
	case POOL_MAX : kckBackwardPool_Max<half><<<gd,bk>>>(gx0, x0, gx1, nw, nh, strd); break;
	case POOL_AVG : kckBackwardPool_Avg<half><<<gd,bk>>>(gx0,     gx1, nw, nh, strd); break;
	}
}

////////////////////////////////////
// kmDnnLyBnrm's members

// x0 : n1, n2, nc, nb
// w0 :  8, nc,  1, 1
template<typename T>
__global__ void kckForwardBnrmBatch(kckMat2<T> w0, const kckMat4<T> x0)
{
	// declare shared memory
	extern __shared__ float sm[];
		
	// init index... ix1, nx1, nx2, ie1
	KC_RDC_INDEX_EXDM1(x0);

	const int ic = ie1;

	// pre-reduction
	float sum_x = 0, sum_x2 = 0;

	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2; if(i1 >=  x0.n1) continue;

		for(int i4 = 0; i4 < x0.n4; ++i4)
		for(int i2 = 0; i2 < x0.n2; ++i2)
		{
			const float x0v = x0(i1,i2,ic,i4);

			sum_x  += x0v;
			sum_x2 += x0v*x0v;
		}		
	}

	// reduction
	float* sm1 = sm;        sm1[ix1] = sum_x;
	float* sm2 = sm + nx1;  sm2[ix1] = sum_x2;

	for(int size_s = nx1>>1; size_s > 0; size_s >>=1)
	{
		__syncthreads();

		if(ix1 < size_s)
		{
			sm1[ix1] += sm1[ix1 + size_s];
			sm2[ix1] += sm2[ix1 + size_s];
		}
	}

	// update
	if(ix1 == 0)
	{
		// calc mean, var, m of mini-batch		
		const float m    = float(x0.n1*x0.n2*x0.n4);
		const float mean = sm1[0]/m;
		const float var  = sm2[0]/m - mean*mean;
		
		// calc mean, var, m for inference
		// * Note that w0(4) are m1/(n*1000) because of the narrow range of half float.
		// * Max of half float is 65504 which is much small value for number of nodes.
		const float n     = float(x0.n1*x0.n2)*1e3f;
		const float mean0 = w0(2,ic);
		const float var0  = w0(3,ic);
		const float m0    = float(w0(4,ic))*n;
		
		const float m1    = m + m0;
		const float mean1 = (m*mean + m0*mean0)/m1;
		const float var1  = (m*(var + mean*mean) + m0*(var0 + mean0*mean0))/m1 - mean1*mean1;
		
		// update 		
		w0(2,ic) = mean1; w0(3,ic) = var1; w0(4,ic) = m1/n; // for inference
		w0(5,ic) = mean;  w0(6,ic) = var;  w0(7,ic) = m /n; // of mini-batch
	}
}

// x1 : n1, n2, nc, nb
// x0 : n1, n2, nc, nb
template<typename T>
__global__ void kckForwardBnrm(kckMat4<T> x1, const kckMat4<T> x0, const kckMat2<T> w0, 
	                           const bool istrain)
{
	// i1, i2, i3(ic), i4(ib)
	KC4_STD_INDEX(x1);

	// get mean and var
	float mean, var;

	if(istrain) { mean = w0(5,i3); var = w0(6,i3);} // mini-batch while training
	else        { mean = w0(2,i3); var = w0(3,i3);} // for inference

	// get a and b
	const float a = float(w0(0,i3))/sqrtf(var + FLOAT_SMALL);
	const float b = float(w0(1,i3)) - a*mean;
	
	// update
	x1(i1,i2,i3,i4) = a*float(x0(i1,i2,i3,i4)) + b;
}

void kmDnnLyBnrm::cForward(kmMat1<kcMat4f32>& x, const kcMat4f32& w0, 
	                        const kcMat2f32& b0, const bool istrain)
{
	// init parameters and variables
	const kcMat4f32& x0  = x(_x0_idx);
	      kcMat4f32& x1  = x(_x1_idx);
	const int64      nb  = x1.N4(); // n of batch
	      kcMat2f32  w0_ = w0.Mat2(0,0);

	// calc mean, var, m of mini-batch
	if(istrain)
	{
		dim3 bk, gd; int sm; x0.GetBkGdRdc(bk, gd, sm, 3);
	
		kckForwardBnrmBatch<float><<<gd,bk,2*sm>>>(w0_, x0);
	}
	
	// normalize
	dim3 bk, gd; x1.GetBkGd(bk, gd);

	kckForwardBnrm<float><<<gd,bk>>>(x1, x0, w0_, istrain);
}

void kmDnnLyBnrm::cForward(kmMat1<kcMat4f16>& x, const kcMat4f16& w0, 
	                        const kcMat2f16& b0, const bool istrain)
{
	// init parameters and variables
	const kcMat4f16& x0  = x(_x0_idx);
	      kcMat4f16& x1  = x(_x1_idx);
	const int64      nb  = x1.N4(); // n of batch
	      kcMat2f16  w0_ = w0.Mat2(0,0);

	// calc mean, var, m of mini-batch
	if(istrain)
	{
		dim3 bk, gd; int sm; x0.GetBkGdRdc(bk, gd, sm, 3);
	
		kckForwardBnrmBatch<half><<<gd,bk,4*sm>>>(w0_, x0);
	}
	
	// normalize
	dim3 bk, gd; x1.GetBkGd(bk, gd);

	kckForwardBnrm<half><<<gd,bk>>>(x1, x0, w0_, istrain);
}

// gx1, x1 : n1, n2, nc, nb
// gx0, x0 : n1, n2, nc, nb
// gw0, w0 :  8, nc,  1,  1
template<typename T>
__global__ void kckBackwardBnrmSum(kckMat2<T> gw0, const kckMat4<T>  x0, 
	                         const kckMat2<T>  w0, const kckMat4<T> gx1, const float inv_m)
{
	// declare shared memory
	extern __shared__ float sm[];
		
	// init index... ix1, nx1, nx2, ie1
	KC_RDC_INDEX_EXDM1(x0);

	const int ic = ie1;

	// init parameters
	const float mean = w0(5,ic);

	// pre-reduction
	float sum_gy0 = 0, sum_xm0 = 0, sum_gyxm0 = 0;

	for(int ix2 = 0; ix2 < nx2; ++ix2)
	{
		const int i1 = ix1 + nx1*ix2;

		if(i1 < x0.n1)
		for(int i4 = 0; i4 < x0.n4; ++i4)
		for(int i2 = 0; i2 < x0.n2; ++i2)
		{
			const float gyi = gx1(i1,i2,ic,i4);
			const float xmi = float(x0(i1,i2,ic,i4)) - mean;
			
			sum_gy0   += float(gyi);
			sum_xm0   += float(xmi);
			sum_gyxm0 += float(gyi*xmi);
		}		
	}

	// reduction
	float* sm1 = sm;         sm1[ix1] = sum_gy0;
	float* sm2 = sm1 + nx1;  sm2[ix1] = sum_xm0;
	float* sm3 = sm2 + nx1;  sm3[ix1] = sum_gyxm0;
		
	for(int size_s = nx1>>1; size_s > 0; size_s >>=1)
	{
		__syncthreads();

		if(ix1 < size_s)
		{
			sm1[ix1] += sm1[ix1 + size_s];
			sm2[ix1] += sm2[ix1 + size_s];
			sm3[ix1] += sm3[ix1 + size_s];
		}
	}
	// update
	if(ix1 == 0)
	{
		const float sum_gy   = sm1[0];
		const float sum_xm   = sm2[0];
		const float sum_gyxm = sm3[0];

		const float gamma = w0(0,ic);
		const float var   = float(w0(6,ic)) + FLOAT_SMALL;

		const float inv_sqrt_v       = rsqrtf(var);
		const float gamma_inv_sqrt_v = gamma*inv_sqrt_v;

		const float gv = -gamma_inv_sqrt_v/var*sum_gyxm;
		const float gm = -gamma_inv_sqrt_v*sum_gy - gv*inv_m*sum_xm;
		
		// update gw0
		gw0(0,ic) = inv_sqrt_v*sum_gyxm;
		gw0(1,ic) = sum_gy;

		// update temporally gw0 to calc gx0
		gw0(2,ic) = gv;
		gw0(3,ic) = gm;	
		gw0(4,ic) = gamma_inv_sqrt_v;
		gw0(5,ic) = mean;
	}
}

// gx1, x1 : n1, n2, nc, nb
// gx0, x0 : n1, n2, nc, nb
// gw0, w0 :  8, nc,  1,  1
template<typename T>
__global__ void kckBackwardBnrmGx0(kckMat4<T> gx0,       kckMat2<T> gw0,
	                         const kckMat4<T>  x0, const kckMat4<T> gx1, const float inv_m)
{
	// i1, i2, i3(ic), i4(ib)
	KC4_STD_INDEX(gx0);

	// init parameters
	const float gv               = gw0(2,i3);
	const float gm               = gw0(3,i3);
	const float gamma_inv_sqrt_v = gw0(4,i3);
	const float mean             = gw0(5,i3);

	// calc and update gx0
	const float gyi = gx1(i1,i2,i3,i4);
	const float xmi = float(x0(i1,i2,i3,i4)) - mean;
	
	T& gx0r = gx0(i1,i2,i3,i4);

	gx0r = float(gx0r) + gyi*gamma_inv_sqrt_v + inv_m*(gv*xmi + gm);
}

// gw0_ :  4, nc
template<typename T>
__global__ void kckBackwardBnrmClear(kckMat2<T> gw0_)
{
	// i1, i2(ic)
	KC2_STD_INDEX(gw0_);

	if(i1 > 1) gw0_(i1,i2) = 0;
}

void kmDnnLyBnrm::cBackward(kmMat1<kcMat4f32>& gx,       kcMat4f32& gw0,       kcMat2f32& gb0,
		              const kmMat1<kcMat4f32>&  x, const kcMat4f32&  w0, const kcMat2f32&  b0)
{

	// init variable and parameters	
	      kcMat4f32& gx0 = gx(_x0_idx);
	const kcMat4f32& gx1 = gx(_x1_idx);
	const kcMat4f32&  x0 =  x(_x0_idx);

	      kcMat2f32  w0_ =  w0.Mat2(0,0);
	      kcMat2f32 gw0_ = gw0.Mat2(0,0);

	const float inv_m = 1.f/(x0.N1()*x0.N2()*x0.N4());

	// update gw0(0:1,ic,0,0)
	// calc and temporally save gv, gm, gamma_inv_sqrt_v, mean to gw0(2:5,ic,0,0)
	// * Note that gw0(2:5,ic,0,0) will be set as zero later.
	{
		dim3 bk, gd; int sm; x0.GetBkGdRdc(bk, gd, sm, 3);

		kckBackwardBnrmSum<float><<<gd,bk,(3*sm)>>>(gw0_, x0, w0_, gx1, inv_m);
	}

	// calc gx0
	{
		dim3 bk, gd; gx0.GetBkGd(bk, gd);

		kckBackwardBnrmGx0<float><<<gd,bk>>>(gx0, gw0_, x0, gx1, inv_m);
	}	

	// clear the temporal elements of gw0
	{
		dim3 bk, gd; gw0_.GetBkGd(bk, gd);

		kckBackwardBnrmClear<float><<<gd,bk>>>(gw0_);
	}
}

void kmDnnLyBnrm::cBackward(kmMat1<kcMat4f16>& gx,       kcMat4f16& gw0,       kcMat2f16& gb0,
		              const kmMat1<kcMat4f16>&  x, const kcMat4f16&  w0, const kcMat2f16&  b0)
{

	// init variable and parameters	
	      kcMat4f16& gx0 = gx(_x0_idx);
	const kcMat4f16& gx1 = gx(_x1_idx);
	const kcMat4f16&  x0 =  x(_x0_idx);

	      kcMat2f16  w0_ =  w0.Mat2(0,0);
	      kcMat2f16 gw0_ = gw0.Mat2(0,0);

	const float inv_m = 1.f/(x0.N1()*x0.N2()*x0.N4());

	// update gw0(0:1,ic,0,0)
	// calc and temporally save gv, gm, gamma_inv_sqrt_v, mean to gw0(2:5,ic,0,0)
	// * Note that gw0(2:5,ic,0,0) will be set as zero later.
	{
		dim3 bk, gd; int sm; x0.GetBkGdRdc(bk, gd, sm, 3);

		kckBackwardBnrmSum<half><<<gd,bk,(6*sm)>>>(gw0_, x0, w0_, gx1, inv_m);
	}

	// calc gx0
	{
		dim3 bk, gd; gx0.GetBkGd(bk, gd);

		kckBackwardBnrmGx0<half><<<gd,bk>>>(gx0, gw0_, x0, gx1, inv_m);
	}	

	// clear the temporal elements of gw0
	{
		dim3 bk, gd; gw0_.GetBkGd(bk, gd);

		kckBackwardBnrmClear<half><<<gd,bk>>>(gw0_);
	}
}

///////////////////////////////////////////////////////////////
// optimization class's members

// Gradient descent
template<typename T>
__global__ void kckUpdateOpt1Gd(kckMat1<T> w, const kckMat1<T> gw, const float step)
{
	// i1
	KC1_STD_INDEX(w);

	// update w	
	w(i1) -= step*float(gw(i1));
 }

void kmOpt1Gd::cUpdate(kcMat1f32& w, const kcMat1f32& gw, float step)
{
	// init parameters
	if(step == 0) step = 0.01f;

	// increase itr
	++_itr;	
	
	// update w
	dim3 bk, gd; w.GetBkGd(bk, gd); 

	kckUpdateOpt1Gd<float><<<gd,bk>>>(w, gw, step);
}

void kmOpt1Gd::cUpdate(kcMat1f16& w, const kcMat1f16& gw, float step)
{
	// init parameters
	if(step == 0) step = 0.01f;

	// increase itr
	++_itr;	
	
	// update w
	dim3 bk, gd; w.GetBkGd(bk, gd);

	kckUpdateOpt1Gd<half><<<gd,bk>>>(w, gw, step);
}

// ADAM
template<typename T>
__global__ void kckUpdateOpt1Adam(kckMat1<T> w, kckMat1<T> gw, 
	                              kckMat1f32 v, kckMat1f32  s,
	                              float b1, float b2, float a, float c)
{
	// i1
	KC1_STD_INDEX(w);

	// calc v0 and s0	
	const float gw0 = gw(i1);
	const float v0  = b1*v(i1) + (1.f - b1)*gw0;
	const float s0  = b2*s(i1) + (1.f - b2)*gw0*gw0;

	// update w
	w(i1) -= (v0*a)/(sqrtf(s0*c) + 1e-8f);

	// updata v, s
	v(i1) = v0;
	s(i1) = s0;
 }

void kmOpt1Adam::cUpdate(kcMat1f32& w, const kcMat1f32& gw, float step)
{
	// init parameters
	if(step == 0) step = 0.01f;

	// increase itr
	++_itr;

	// pre-calc parameters
	const float a = step/(1.f - powf(_beta1, (float)_itr));
	const float c =  1.f/(1.f - powf(_beta2, (float)_itr));
		
	// update w
	dim3 bk, gd; w.GetBkGd(bk, gd);

	kckUpdateOpt1Adam<float><<<gd,bk>>>(w, gw, _cv, _cs, _beta1, _beta2, a, c);
}

void kmOpt1Adam::cUpdate(kcMat1f16& w, const kcMat1f16& gw, float step)
{
	// init parameters
	if(step == 0) step = 0.01f;

	// increase itr
	++_itr;

	// pre-calc parameters
	const float a = step/(1.f - powf(_beta1, (float)_itr));
	const float c =  1.f/(1.f - powf(_beta2, (float)_itr));
		
	// update w
	dim3 bk, gd; w.GetBkGd(bk, gd);

	kckUpdateOpt1Adam<half><<<gd,bk>>>(w, gw, _cv, _cs, _beta1, _beta2, a, c);
}

// SQN
template<typename T>
__global__ void kckUpdateOpt1Sqn(kckMat1<T> w,  kckMat1<T> g, 
	                             kckMat1f32 gm, kckMat1f32 s,  kckMat1f32 d)
{
	// i1
	KC1_STD_INDEX(w);

	// init parameters and variables
	const float b1 = 0.95f, etha = 1e-6f;
	const float g0 = g(i1), w0 = w(i1), d0 = d(i1);

	// update s
	const float dg = g0 - gm(i1);
	const float s0 = b1*s(i1) + (1 - b1)*dg*dg; s(i1) = s0;

	// update w
	const float dw = sqrt(d0 + etha)/sqrt(s0 + etha)*g0;
	w(i1) = w0 - dw;

	// updata d
	d(i1) = b1*d0 + (1 - b1)*dw*dw;

	// update gm
	gm(i1) = g0;
 }

void kmOpt1Sqn::cUpdate(kcMat1f32& w, const kcMat1f32& g, float step)
{
	// increase itr
	++_itr;

	// update w
	dim3 bk, gd; w.GetBkGd(bk, gd);

	kckUpdateOpt1Sqn<float><<<gd,bk>>>(w, g, _cgm, _cs, _cd);
}

void kmOpt1Sqn::cUpdate(kcMat1f16& w, const kcMat1f16& g, float step)
{
	// increase itr
	++_itr;

	// update w
	dim3 bk, gd; w.GetBkGd(bk, gd);

	kckUpdateOpt1Sqn<half><<<gd,bk>>>(w, g, _cgm, _cs, _cd);
}

// AdaDelta
template<typename T>
__global__ void kckUpdateOpt1Adadel(kckMat1<T> w, kckMat1<T> g, 
	                                kckMat1f32 d, kckMat1f32 s)
{
	// i1
	KC1_STD_INDEX(w);

	// init parameters and variables
	const float b1 = 0.95f, etha = 1e-6f;
	const float g0 = g(i1), w0 = w(i1), d0 = d(i1);

	// update s
	const float s0 = b1*s(i1) + (1.f - b1)*g0*g0; s(i1) = s0;

	// update w
	const float dw = sqrt(d0 + etha)/sqrt(s0 + etha)*g0; w(i1) = w0 - dw;

	// updata d
	d(i1) = b1*d0 + (1.f - b1)*dw*dw;
 }

void kmOpt1Adadel::cUpdate(kcMat1f32& w, const kcMat1f32& g, float step)
{	
	// increase itr
	++_itr;	
	
	// update w
	dim3 bk, gd; w.GetBkGd(bk, gd);

	kckUpdateOpt1Adadel<float><<<gd,bk>>>(w, g, _cd, _cs);
}

void kmOpt1Adadel::cUpdate(kcMat1f16& w, const kcMat1f16& g, float step)
{	
	// increase itr
	++_itr;	
	
	// update w
	dim3 bk, gd; w.GetBkGd(bk, gd);

	kckUpdateOpt1Adadel<half><<<gd,bk>>>(w, g, _cd, _cs);
}