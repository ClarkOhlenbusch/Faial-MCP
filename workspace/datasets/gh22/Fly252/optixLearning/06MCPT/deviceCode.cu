#include "deviceCode.h"
#include "deviceHelper.cuh"

#include <owl/common/math/random.h>

#include <optix_device.h>
#include <owl/owl_device.h>

#define __CUDACC__
#include <texture_fetch_functions.h>

extern "C" __constant__ LaunchParams optixLaunchParams;

typedef RayT<RAY_TYPE_RADIANCE, RAY_TYPE_COUNT> RadianceRay;
typedef RayT<RAY_TYPE_OCCLUSION, RAY_TYPE_COUNT> OcclusionRay;

typedef LCG<4> Random; // 随机数生成器

struct RadiancePRD
{
	vec3f Emitted;
	vec3f Radiance;
	vec3f Attenuation;
	vec3f ScatOrigin;
	vec3f ScatDirection;
	Random RandGen;
	int CountEmitted;
	int Done;
};

struct OcclusionPRD
{
	int Occluded;
};

//---- Lambert ----//

OPTIX_CLOSEST_HIT_PROGRAM(LambertOcclusion)()
{
	OcclusionPRD &prd = getPRD<OcclusionPRD>();
	prd.Occluded = true;
}

OPTIX_CLOSEST_HIT_PROGRAM(LambertRadiance)()
{
	const LaunchParams &params = optixLaunchParams;
	const LambertMeshData &progData = getProgramData<LambertMeshData>();
	const vec3i index = progData.Indices[optixGetPrimitiveIndex()];
	const vec3f &A = progData.Vertices[index.x];
	const vec3f &B = progData.Vertices[index.y];
	const vec3f &C = progData.Vertices[index.z];
	const vec3f &N0 = normalize(cross(B - A, C - A));

	const vec3f rayOrigin = toVec3f(optixGetWorldRayOrigin());
	const vec3f rayDir = toVec3f(optixGetWorldRayDirection());
	const vec3f N = faceforward(N0, -rayDir, N0);               // 保持法向量与射线方向相反
	const vec3f P = rayOrigin + optixGetRayTmax() * rayDir;     // 命中点

	RadiancePRD &prd = getPRD<RadiancePRD>();

	if (prd.RandGen() < params.PRR)
	{
		if (prd.CountEmitted)
			prd.Emitted = progData.EmissionColor / params.PRR;
		else
			prd.Emitted = vec3f(0.0f);

		{
			const float z1 = prd.RandGen();
			const float z2 = prd.RandGen();

			vec3f dir = sample_hemisphere(z1, z2);
			OrthonormalBasis onb(N);
			onb.InverseTransform(dir);

			prd.ScatOrigin = P;
			prd.ScatDirection = dir;
			prd.Attenuation *= progData.DiffuseColor;
			prd.CountEmitted = false;
		}

		const float z1 = prd.RandGen();
		const float z2 = prd.RandGen();

		ParallelogramLight light = params.Lights[0];  // TODO: 多光源
		const vec3f lightPos = light.Corner + light.V1 * z1 + light.V2 * z2;

		const float lightDist = length(lightPos - P);
		const vec3f L = normalize(lightPos - P);         // 指向光源
		const float NdotL = dot(N, L);
		const float lightNdotL = dot(-light.Normal, L);

		float weight = 0.0f;
		if (NdotL > 0.0f && lightNdotL > 0.0f)
		{
			OcclusionPRD occlusionPrd;
			occlusionPrd.Occluded = false;
			OcclusionRay occlusionRay;
			occlusionRay.origin = P;
			occlusionRay.direction = L;
			occlusionRay.tmin = 0.01f;
			occlusionRay.tmax = lightDist - 0.01f;
			traceRay(params.World, occlusionRay, occlusionPrd);

			if (!occlusionPrd.Occluded)
			{
				const float s = length(cross(light.V1, light.V2));  // 光源面积
				weight = NdotL * lightNdotL * s / (M_PI * lightDist * lightDist);
			}
		}

		prd.Radiance += light.Emission * weight / params.PRR;
	}
	else
	{
		prd.Emitted = vec3f(0.0f);
		prd.Radiance = vec3f(0.0f);
		prd.Done = true;
	}
}

//---- Metal ----//

OPTIX_CLOSEST_HIT_PROGRAM(MetalOcclusion)()
{
	OcclusionPRD &prd = getPRD<OcclusionPRD>();
	prd.Occluded = true;
}

OPTIX_CLOSEST_HIT_PROGRAM(MetalRadiance)()
{
	const LaunchParams &params = optixLaunchParams;
	const MetalMeshData &progData = getProgramData<MetalMeshData>();
	const vec3i index = progData.Indices[optixGetPrimitiveIndex()];
	const vec3f &A = progData.Vertices[index.x];
	const vec3f &B = progData.Vertices[index.y];
	const vec3f &C = progData.Vertices[index.z];
	const vec3f &N0 = normalize(cross(B - A, C - A));

	const vec3f rayOrigin = toVec3f(optixGetWorldRayOrigin());
	const vec3f rayDir = toVec3f(optixGetWorldRayDirection());
	const vec3f N = faceforward(N0, -rayDir, N0);               // 保持法向量与射线方向相反
	const vec3f P = rayOrigin + optixGetRayTmax() * rayDir;     // 命中点

	RadiancePRD &prd = getPRD<RadiancePRD>();

	// 轮盘赌
	if (prd.RandGen() < params.PRR)
	{
		if (prd.RandGen() > progData.ReflectProb)
		{ // 计算漫反射
			{
				const float z1 = prd.RandGen();
				const float z2 = prd.RandGen();

				vec3f dir = sample_hemisphere(z1, z2);
				OrthonormalBasis onb(N);
				onb.InverseTransform(dir);

				prd.ScatOrigin = P;
				prd.ScatDirection = dir;
				prd.Attenuation *= progData.DiffuseColor;
				prd.CountEmitted = false;
			}

			const float z1 = prd.RandGen();
			const float z2 = prd.RandGen();

			ParallelogramLight light = params.Lights[0];
			const vec3f lightPos = light.Corner + light.V1 * z1 + light.V2 * z2;

			const float lightDist = length(lightPos - P);
			const vec3f L = normalize(lightPos - P);         // 指向光源
			const float NdotL = dot(N, L);
			const float lightNdotL = dot(-light.Normal, L);

			float weight = 0.0f;
			if (NdotL > 0.0f && lightNdotL > 0.0f)
			{
				OcclusionPRD occlusionPrd;
				occlusionPrd.Occluded = false;
				OcclusionRay occlusionRay;
				occlusionRay.origin = P;
				occlusionRay.direction = L;
				occlusionRay.tmin = 0.01f;
				occlusionRay.tmax = lightDist - 0.01f;
				traceRay(params.World, occlusionRay, occlusionPrd);

				if (!occlusionPrd.Occluded)
				{
					const float s = length(cross(light.V1, light.V2));  // 光源面积
					weight = NdotL * lightNdotL * s / (M_PI * lightDist * lightDist);
				}
			}

			prd.Radiance += light.Emission * weight / (1.0f - progData.ReflectProb) / params.PRR;
		}
		else
		{ // 计算反射
			const float z1 = prd.RandGen();
			const float z2 = prd.RandGen();

			ParallelogramLight light = params.Lights[0];
			const vec3f lightPos = light.Corner + light.V1 * z1 + light.V2 * z2;

			const vec3f R = normalize(reflect(rayDir, N));
			const float lightDist = length(lightPos - P);
			const vec3f L = normalize(lightPos - P);         // 指向光源
			const float NdotL = dot(N, L);
			const float lightNdotL = dot(-light.Normal, L);

			{
				const float f1 = prd.RandGen();
				const float f2 = prd.RandGen();

				vec3f dir = sample_hemisphere(f1, f2);
				OrthonormalBasis onb(N);
				onb.InverseTransform(dir);

				prd.ScatOrigin = P;
				prd.ScatDirection = R + dir * progData.Fuzz;
				prd.Attenuation *= (1.0f - progData.Reflectivity);
			}

			float weight = 0.0f;
			if (NdotL > 0.0f && lightNdotL > 0.0f)
			{
				OcclusionPRD occlusionPrd;
				occlusionPrd.Occluded = false;
				OcclusionRay occlusionRay;
				occlusionRay.origin = P;
				occlusionRay.direction = L;
				occlusionRay.tmin = 0.01f;
				occlusionRay.tmax = lightDist - 0.01f;
				traceRay(params.World, occlusionRay, occlusionPrd);

				if (!occlusionPrd.Occluded)
				{
					const float s = length(cross(light.V1, light.V2));
					const float spec = fmaxf(dot(R, L), 0.0f);
					weight = spec * lightNdotL * s / (M_PI * lightDist * lightDist);
				}
			}

			prd.Radiance += light.Emission * weight / params.PRR / progData.ReflectProb;
		}
	}
	else
	{
		prd.Emitted = vec3f(0.0f);
		prd.Radiance = vec3f(0.0f);
		prd.Done = true;
	}
}

//---- RayGen and Miss ----//

OPTIX_MISS_PROGRAM(RadianceMiss)()
{
	const MissData &progData = getProgramData<MissData>();
	RadiancePRD &prd = getPRD<RadiancePRD>();

	prd.Radiance = progData.BackGround;
	prd.Done = true;
}

OPTIX_RAYGEN_PROGRAM(RayGen)()
{
	const LaunchParams &params = optixLaunchParams;
	const int width = params.FrameBufferSize.x;
	const vec3f eye = params.Camera.Pos;
	const vec3f N = params.Camera.N;
	const vec3f U = params.Camera.U;
	const vec3f V = params.Camera.V;
	const vec2i pixelIndex = getLaunchIndex();
	const int accID = params.AccID;
	const int sampleNum = 1;

	Random randGen;
	randGen.init(pixelIndex.x * accID, pixelIndex.y * accID);

	vec3f resultColor = vec3f(0.0f);
	for (int i = 0; i < sampleNum; ++i)
	{
		RadianceRay ray;
		vec2f screen = (vec2f(pixelIndex) + vec2f(randGen(), randGen())) / vec2f(params.FrameBufferSize);
		ray.origin = eye;
		ray.direction = normalize(N + screen.u * U + screen.v * V);

		RadiancePRD prd;
		prd.Emitted = vec3f(0.0f);
		prd.Radiance = vec3f(0.0f);
		prd.Attenuation = vec3f(1.0f);
		prd.ScatOrigin = ray.origin;
		prd.ScatDirection = ray.direction;
		prd.CountEmitted = true;
		prd.Done = false;
		prd.RandGen = randGen;

		vec3f color = vec3f(0.0f);
		while (!prd.Done)
		{
			ray.tmin = 0.01f;
			ray.tmax = 1e16f;
			traceRay(params.World, ray, prd);

			color += prd.Emitted;
			color += prd.Radiance * prd.Attenuation;

			ray = RadianceRay();
			ray.origin = prd.ScatOrigin;
			ray.direction = prd.ScatDirection;
		}
		resultColor += color;
	}

	// 将颜色输出到帧缓冲
	vec4f color4 = vec4f(resultColor, 1.0f);
	int bufOff = pixelIndex.x + pixelIndex.y * width;
	if (accID)
		color4 = color4 + params.AccBufferPtr[bufOff];
	params.AccBufferPtr[bufOff] = color4;
	params.FrameBufferPtr[bufOff] = owl::make_rgba(color4 * (1.0f / ((accID + 1) * sampleNum)));
}