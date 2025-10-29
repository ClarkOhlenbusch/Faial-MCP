#include "blst_377_ops.h"
#include <stdio.h>
#include <stdint.h>
#include <algorithm>

static const uint32_t WINDOW_SIZE = 128;
// static const uint32_t BLST_WIDTH = 253;

extern "C" __global__ void msm6_pixel(blst_p1* bucket_lists, const blst_p1_affine* bases_in, const blst_scalar* scalars, const uint32_t* window_lengths, const uint32_t window_count) {
    limb_t index = threadIdx.x / 64;
    size_t shift = threadIdx.x - (index * 64);
    limb_t mask = (limb_t) 1 << (limb_t) shift;

    blst_p1 bucket;
    memcpy(&bucket, &BLS12_377_ZERO_PROJECTIVE, sizeof(blst_p1));

    uint32_t window_start = WINDOW_SIZE * blockIdx.x;
    uint32_t window_end = window_start + window_lengths[blockIdx.x];

    uint32_t activated_bases[WINDOW_SIZE];
    uint32_t activated_base_index = 0;

    // we delay the actual additions to a second loop because it reduces warp divergence (20% practical gain)
    for (uint32_t i = window_start; i < window_end; ++i) {
        limb_t bit = (scalars[i][index] & mask);
        if (bit == 0) {
            continue;
        }
        activated_bases[activated_base_index++] = i;
    }
    uint32_t i = 0;
    for (; i < (activated_base_index / 2 * 2); i += 2) {
        blst_p1 intermediate;
        blst_p1_add_affines_into_projective(&intermediate, &bases_in[activated_bases[i]], &bases_in[activated_bases[i + 1]]);
        blst_p1_add_projective_to_projective(&bucket, &bucket, &intermediate);
    }
    for (; i < activated_base_index; ++i) {
        blst_p1_add_affine_to_projective(&bucket, &bucket, &(bases_in[activated_bases[i]]));
    }
    //                    0..253     // scalars.len()/128  0..100
    memcpy(&bucket_lists[threadIdx.x * window_count + blockIdx.x], &bucket, sizeof(blst_p1));
}

extern "C" __global__ void msm6_collapse_rows(blst_p1* target, const blst_p1* bucket_lists, const uint32_t window_count) {
    blst_p1 temp_target;
    uint32_t base = threadIdx.x * window_count;
    uint32_t term = base + window_count;
    memcpy(&temp_target, &bucket_lists[base], sizeof(blst_p1));

    for (uint32_t i = base + 1; i < term; ++i) {
        blst_p1_add_projective_to_projective(&temp_target, &temp_target, &bucket_lists[i]);
    }
    
    memcpy(&target[threadIdx.x], &temp_target, sizeof(blst_p1));
}

extern "C" __device__ uint32_t each_window_coefficient(limb_t scalar,limb_t window, limb_t offset){
    limb_t remainder = scalar >> offset;
    limb_t mask = (1 << window) - 1;
    return remainder & mask;
}

extern "C" __global__ void variable_base_msm(blst_p1* bucket_lists, const blst_p1_affine* bases_in, const blst_scalar* scalars,const uint32_t c, const uint32_t* window_lengths, const uint32_t window_count) {
    limb_t each_limbs = 64 / c + 1;
    limb_t index = (threadIdx.x / each_limbs);
    limb_t offset = (threadIdx.x - (index * each_limbs)) * c;

    blst_p1 bucket;
    memcpy(&bucket, &BLS12_377_ZERO_PROJECTIVE, sizeof(blst_p1));

    uint32_t window_start = WINDOW_SIZE * blockIdx.x;
    uint32_t window_end = window_start + window_lengths[blockIdx.x];

    uint32_t size_of_bucket = ((1<< c)); // 包含了 0
    blst_p1 *bucketlist_to_sum = new blst_p1[size_of_bucket] ();

    for (uint32_t i = window_start; i < window_end; ++i) {
        limb_t sc = scalars[i][(uint32_t)index];
        uint32_t coeff = each_window_coefficient(sc,(limb_t)c,offset);
        blst_p1_add_affine_to_projective(&bucketlist_to_sum[coeff],&bucketlist_to_sum[coeff],&bases_in[i]);
    }

    // make a running sum of bucketlist_to_sum;
    for (uint32_t i = size_of_bucket - 2; i >= 1; --i) {
        blst_p1_add_projective_to_projective(&bucketlist_to_sum[i],&bucketlist_to_sum[i],&bucketlist_to_sum[i+1]);
    }
//    bucketlist_to_sum.reverse();
//    for (uint32_t i = 1; i < size_of_bucket; ++i) {
//        //bucketlist_to_sum[i] = bucketlist_to_sum[i] + bucketlist_to_sum[i-1];
//        blst_p1_add_projective_to_projective(&bucketlist_to_sum[i],&bucketlist_to_sum[i],&bucketlist_to_sum[i-1]);
//    }

    for (uint32_t i = 1; i < size_of_bucket; ++i) {
        //bucket = bucket + bucketlist_to_sum[i];
        blst_p1_add_projective_to_projective(&bucket,&bucket,&bucketlist_to_sum[i]);
    }

    memcpy(&bucket_lists[threadIdx.x * window_count + blockIdx.x], &bucket, sizeof(blst_p1));
}

extern "C" __global__ void variable_base_msm_collapse_rows(blst_p1* target, const blst_p1* bucket_lists, const uint32_t window_count) {
    blst_p1 temp_target;
    uint32_t base = threadIdx.x * window_count;
    uint32_t term = base + window_count;
    memcpy(&temp_target, &bucket_lists[base], sizeof(blst_p1));

    for (uint32_t i = base + 1; i < term; ++i) {
        blst_p1_add_projective_to_projective(&temp_target, &temp_target, &bucket_lists[i]);
    }

    memcpy(&target[threadIdx.x], &temp_target, sizeof(blst_p1));
}