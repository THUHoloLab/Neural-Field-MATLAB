#include "addon/helpers.cuh"
#include "addon/mxGPUArray.h"
#include "mex.h"

__global__ void SH_Encoding_forward_kernel(
    const float3 * __restrict__ dirs,
    const dim3 dirs_sz,
    float * __restrict__ embedding
);

__global__ void SH_Encoding_backward_kernel(
    const float * __restrict__ dl_dembedding,  // dL/dembedding, shape [batch_size, 16]
    const float3 * __restrict__ dirs,           // 前向传播的输入方向
    const dim3 dirs_sz,                         // batch size
    float3 * __restrict__ dl_ddirs             // 输出梯度 dL/ddirs, shape [batch_size]
);