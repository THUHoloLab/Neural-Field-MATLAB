#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define POS_DIM 3

__global__ void hashEncoding_Fwd_kernel(
    const float3 * __restrict__ xys,
    const float4 * __restrict__ embedding,
    const float * __restrict__ bbox,
    const uint32_t * __restrict__ hash_offsets,
    const uint32_t * __restrict__ hash_map_sizes,
    const dim3 output_sz,
    const float log_scale, // (log(res_high) - log(res_low)) / (level - 1)
    const float base_res,
    const uint32_t feature_dim,
    // outputs
    float4 * __restrict__ output_embedding
){
    const uint32_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t level_id = blockIdx.y; // page id

    if (batch_id >= output_sz.y) {
        return;
    }

    float scale = grid_resolution((int) level_id, log_scale, base_res);
    uint32_t resolution = uint32_t(ceilf(scale)) + 1;
    float pos[POS_DIM] = {xys[batch_id].x, xys[batch_id].y, xys[batch_id].z};
    uint32_t pos_grid[POS_DIM];

    #pragma unroll
    for(uint32_t idx = 0; idx < POS_DIM; ++ idx){
        float bbox_min = bbox[idx * 2 + 0];
        float normalized_pos = (pos[idx] - bbox_min) / 
                               (bbox[idx * 2 + 1] - bbox_min) * scale;
        // normalized_pos = max(min(normalized_pos,1.0f),0.0f);

        float temp_pos = __floorf(normalized_pos);
        pos_grid[idx] = (uint32_t) temp_pos;
        pos[idx] = normalized_pos - temp_pos;
    }

    const uint32_t map_size = __ldg(&hash_map_sizes[level_id]);
    const uint32_t offsets  = __ldg(&hash_offsets[level_id]);

    float4 features = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll
    for(uint32_t idx = 0; idx < 8; ++idx){
        float w = 1.0f;
        uint32_t loc_pos[POS_DIM];
        #pragma unroll
        for(uint32_t dim = 0; dim < POS_DIM; ++dim){
            if ((idx & (1 << dim)) == 0){
                loc_pos[dim] = pos_grid[dim];
                w *= 1 - pos[dim];
            }else{
                loc_pos[dim] = pos_grid[dim] + 1;
                w *= pos[dim];
            }
        }

        uint32_t hash_idx = grid_index(
            map_size, resolution,
            loc_pos[0], loc_pos[1], loc_pos[2]
        );
        // uint32_t table_index = (offsets + hash_idx) * feature_dim;
        // features.x = fmaf(w, __ldg(&embedding[table_index + 0]), features.x);
        // features.y = fmaf(w, __ldg(&embedding[table_index + 1]), features.y);
        // features.z = fmaf(w, __ldg(&embedding[table_index + 2]), features.z);
        // features.w = fmaf(w, __ldg(&embedding[table_index + 3]), features.w);
        float4 this_embedding = __ldg(&embedding[offsets + hash_idx]);
        features.x = fmaf(w, this_embedding.x, features.x);
        features.y = fmaf(w, this_embedding.y, features.y);
        features.z = fmaf(w, this_embedding.z, features.z);
        features.w = fmaf(w, this_embedding.w, features.w);
    }

    output_embedding[batch_id * (output_sz.x / 4) + level_id] = features;
}

__global__ void hashEncoding_Bwd_kernel(
    const float3 * __restrict__ xys,
    const float4 * __restrict__ dl_doutput,
    const float * __restrict__ bbox,
    const uint32_t * __restrict__ hash_offsets,
    const uint32_t * __restrict__ hash_map_sizes,
    const dim3 output_sz,
    const float log_scale, // (log(res_high) - log(res_low)) / (level - 1)
    const float base_res,
    const uint32_t feature_dim,
    float * __restrict__ dl_dembedding
){
    const uint32_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t level_id = blockIdx.y; // page id

    if (batch_id >= output_sz.y) {
        return;
    }

    float scale = grid_resolution((int) level_id, log_scale, base_res);
    uint32_t resolution = (uint32_t) ceilf(scale) + 1;
    float pos[POS_DIM] = {xys[batch_id].x, xys[batch_id].y, xys[batch_id].z};

    uint32_t pos_grid[POS_DIM];
    #pragma unroll
    for(uint32_t idx = 0; idx < POS_DIM; ++ idx){
        float bbox_min = bbox[idx * 2 + 0];
        float normalized_pos = (pos[idx] - bbox_min) / 
                               (bbox[idx * 2 + 1] - bbox_min) * scale;
        // normalized_pos = max(min(normalized_pos,1.0f),0.0f);
        float temp_pos = floorf(normalized_pos);
        pos_grid[idx] = (uint32_t) temp_pos;
        pos[idx] = normalized_pos - temp_pos;
    }

    const uint32_t map_size = __ldg(&hash_map_sizes[level_id]);
    const uint32_t offsets  = __ldg(&hash_offsets[level_id]);

    float4 dl_outputs = __ldg(&dl_doutput[batch_id * (output_sz.x / 4) + level_id]);

    #pragma unroll
    for(uint32_t idx = 0; idx < 8; ++idx){
        float w = 1.0f;
        uint32_t loc_pos[POS_DIM];
        #pragma unroll
        for(uint32_t dim = 0; dim < POS_DIM; ++dim){
            if ((idx & (1 << dim)) == 0){
                loc_pos[dim] = pos_grid[dim];
                w *= 1 - pos[dim];
            }else{
                loc_pos[dim] = pos_grid[dim] + 1;
                w *= pos[dim];
            }
        }

        uint32_t hash_idx = grid_index(
            map_size, resolution,
            loc_pos[0], loc_pos[1], loc_pos[2]
        );

        uint32_t base_id = (offsets + hash_idx) * feature_dim;
        float *temp = (float *) dl_dembedding;
        atomicAdd(temp + base_id + 0, w * dl_outputs.x);
        atomicAdd(temp + base_id + 1, w * dl_outputs.y);
        atomicAdd(temp + base_id + 2, w * dl_outputs.z);
        atomicAdd(temp + base_id + 3, w * dl_outputs.w);
    }
}
