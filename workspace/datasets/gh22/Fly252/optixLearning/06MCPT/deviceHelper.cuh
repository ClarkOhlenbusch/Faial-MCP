#pragma once

#include <owl/owl.h>
#include <owl/common/math/vec.h>

#include <optix_device.h>
#include <owl/owl_device.h>

using namespace owl;

inline __device__ vec3f reflect(const vec3f &i, const vec3f &n)
{
    return i - 2.0f * n * dot(n, i);
}

inline __device__ bool refract(const vec3f &v, const vec3f &n, float ni_over_nt, vec3f &refracted)
{
    vec3f uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0.f) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
    }
    else
        return false;
}

inline __device__ float fmaxf(const vec3f &a)
{
    return fmaxf(fmaxf(a.x, a.y), a.z);
}

inline __device__ float fresnel(float cosi, float cost, float etai, float etat) {
    float rs = (etat * cosi - etai * cost) / (etat * cosi + etai * cost);
    float rp = (etai * cosi - etat * cost) / (etai * cosi + etat * cost);
    return (rs * rs + rp * rp) * 0.5f;
}

inline __device__ float schlick(float cosi, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosi), 5.0f);
}

inline __device__ vec3f toVec3f(const float3 &rhs)
{
    return vec3f(rhs.x, rhs.y, rhs.z);
}

// 返回 n 指向与 i 相同方向的复制（以 nref 为基准）
//   通常直接让 n 和 nref 一致
inline __device__ vec3f faceforward(const vec3f &n, const vec3f &i, const vec3f &nref)
{
    return n * copysignf(1.0f, dot(i, nref));
}

struct OrthonormalBasis
{
    __forceinline__ __device__ OrthonormalBasis(const vec3f &normal)
    {
        mNormal = normal;

        if (fabsf(mNormal.x) > fabsf(mNormal.z))
        {
            mBinormal.x = -mNormal.y;
            mBinormal.y = mNormal.x;
            mBinormal.z = 0.0f;
        }
        else
        {
            mBinormal.x = 0.0f;
            mBinormal.y = -mNormal.z;
            mBinormal.z = mNormal.y;
        }

        mBinormal = normalize(mBinormal);
        mTangent = cross(mBinormal, mNormal);
    }

    __forceinline__ __device__ void InverseTransform(vec3f &p) const
    {
        p = p.x * mTangent + p.y * mBinormal + p.z * mNormal;
    }

    vec3f mTangent;   // U
    vec3f mBinormal;  // V
    vec3f mNormal;    // N
};

// 返回长为 1 的半球 cos 分布
//   u1: 影响底面半径
//   u2: 影响底面方向
inline __device__ vec3f sample_hemisphere(const float u1, const float u2)
{
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PI * u2;  // 角度 0 ~ 2π
    vec3f p;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // 保证长为 1
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
    return p;
}

inline __device__ float average(const vec3f &v)
{
    return (v.x + v.y + v.z) / 3.0f;
}