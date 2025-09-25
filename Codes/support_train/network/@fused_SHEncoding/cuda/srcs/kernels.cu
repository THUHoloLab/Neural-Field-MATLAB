#include "kernels.cuh"
#include <cuda_runtime.h>

__global__ void SH_Encoding_forward_kernel(
    const float3 * __restrict__ dirs,
    const dim3 dirs_sz,
    float * __restrict__ embedding
){
    unsigned batch_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_idx >= dirs_sz.x) return;
    float x = dirs[batch_idx].x;
    float y = dirs[batch_idx].y;
    float z = dirs[batch_idx].z;
    float xy = x * y;
    float xz = x * z;
    float yz = z * y;
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    // l = 0

    unsigned bathc_idx16 = batch_idx * 16;

    embedding[bathc_idx16 + 0] = (0.28209479177387814f);
    // l = 1
    embedding[bathc_idx16 + 1] = (-0.48860251190291987f * y);
    embedding[bathc_idx16 + 2] = (0.48860251190291987f * z);
    embedding[bathc_idx16 + 3] = (-0.48860251190291987f * x);
    // l = 2
    embedding[bathc_idx16 + 4] = (1.0925484305920792f * xy);
    embedding[bathc_idx16 + 5] = (-1.0925484305920792f * yz);
    embedding[bathc_idx16 + 6] = (0.94617469575755997f * z2 - 0.31539156525251999f);
    embedding[bathc_idx16 + 7] = (-1.0925484305920792f * xz);
    embedding[bathc_idx16 + 8] = (0.54627421529603959f * x2 - 0.54627421529603959f * y2);
    // l = 3
    embedding[bathc_idx16 + 9] = (0.59004358992664352f * y * (-3.0 * x2 + y2));
    embedding[bathc_idx16 + 10] = (2.8906114426405538f * xy * z);
    embedding[bathc_idx16 + 11] = (0.45704579946446572f * y * (1.0 - 5.0 * z2));
    embedding[bathc_idx16 + 12] = (0.3731763325901154f * z * (5.0 * z2 - 3.0));
    embedding[bathc_idx16 + 13] = (0.45704579946446572f * x * (1.0 - 5.0 * z2));
    embedding[bathc_idx16 + 14] = (1.4453057213202769f * z * (x2 - y2));
    embedding[bathc_idx16 + 15] = (0.59004358992664352f * x * (-x2 + 3.0 * y2));
}

__global__ void SH_Encoding_backward_kernel(
    const float * __restrict__ dl_dembedding,  // dL/dembedding, shape [16,batch_size]
    const float3 * __restrict__ dirs,           // input direction
    const dim3 dirs_sz,                         // batch size
    float3 * __restrict__ dl_ddirs             // output dL/ddirs, shape [batch_size]
){
    unsigned batch_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_idx >= dirs_sz.x) return;

    float x = dirs[batch_idx].x;
    float y = dirs[batch_idx].y;
    float z = dirs[batch_idx].z;

    // collection current batch's gradient 16 channels
    float g[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        g[i] = dl_dembedding[batch_idx * 16 + i];
    }

    // preloading intermediate variables
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    // for dL/dx, dL/dy, dL/dz
    float dL_dx = 0.0f;
    float dL_dy = 0.0f;
    float dL_dz = 0.0f;

    // l=0: 
    // l=1
    dL_dx += -0.48860251190291987f * g[3];  // d(embedding[3])/dx = -0.488...
    dL_dy += -0.48860251190291987f * g[1];  // d(embedding[1])/dy = -0.488...
    dL_dz += 0.48860251190291987f * g[2];   // d(embedding[2])/dz = 0.488...

    // l=2
    dL_dx += 1.0925484305920792f * y * g[4] - 1.0925484305920792f * z * g[7] + 2 * 0.54627421529603959f * x * g[8];
    dL_dy += 1.0925484305920792f * x * g[4] - 1.0925484305920792f * z * g[5] - 2 * 0.54627421529603959f * y * g[8];
    dL_dz += -1.0925484305920792f * y * g[5] + (0.94617469575755997f * 2 * z) * g[6] - 1.0925484305920792f * x * g[7];

    // l=3
    dL_dx += 0.59004358992664352f * y * (-6.0f * x) * g[9] + 
             2.8906114426405538f * y * z * g[10] + 
             0.45704579946446572f * (1.0f - 5.0f * z2) * g[13] + 
             1.4453057213202769f * z * 2 * x * g[14] + 
             0.59004358992664352f * (-3.0f * x2 + 3.0f * y2) * g[15];

    dL_dy += 0.59004358992664352f * (-3.0f * x2 + y2 + y * 2 * y) * g[9] + 
             2.8906114426405538f * x * z * g[10] + 
             0.45704579946446572f * (1.0f - 5.0f * z2) * g[11] - 
             1.4453057213202769f * z * 2 * y * g[14] + 
             0.59004358992664352f * x * 6.0f * y * g[15];

    dL_dz += 2.8906114426405538f * xy * g[10] + 
             0.45704579946446572f * y * (-10.0f * z) * g[11] + 
             0.3731763325901154f * (15.0f * z2 - 3.0f) * g[12] + 
             0.45704579946446572f * x * (-10.0f * z) * g[13] + 
             1.4453057213202769f * (x2 - y2) * g[14];

    // gradient
    dl_ddirs[batch_idx] = make_float3(dL_dx, dL_dy, dL_dz);
}