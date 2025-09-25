#include "addon/helpers.cuh"
#include "callfun.cuh"

__host__ dim3 size2dim3( const mxGPUArray * in){
    const mwSize *sz = mxGPUGetDimensions(in);
    const int  dim = (int) mxGPUGetNumberOfDimensions(in);
    dim3 imgSz;
    imgSz = {(unsigned) sz[0], (unsigned) sz[1], 1};
    if (dim > 2){
        imgSz.z = (unsigned) sz[2];
    }
    return imgSz;
}

void Call_Fwd(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const * prhs[]
){
    const mxGPUArray * dir_batch;
    mxGPUArray * output_embedding;

    mxInitGPU();
    dir_batch = mxGPUCreateFromMxArray(prhs[0]);

    const int feature_dim = (const int) mxGetPr(prhs[1])[0];
    const dim3 dir_sz = size2dim3(dir_batch);
    const float3 *d_dir_batch = (const float3 *) mxGPUGetDataReadOnly(dir_batch);

    CHECK_THROW((size_t) dir_sz.x == 3);
    CHECK_THROW(feature_dim == 4);
    // mexPrintf("dir_batch size is %u %u %u \n", dir_sz.x, dir_sz.y, dir_sz.z);
    dim3 N_BLOCKS = {
        (unsigned) (dir_sz.y + N_THREADS - 1) / N_THREADS,
        1
    };

    const mwSize sz[2] = {16, dir_sz.y};
    dim3 output_sz = {(unsigned) sz[0], (unsigned) sz[1], 1};
    output_embedding = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(dir_batch),
        sz,
        mxSINGLE_CLASS,
        mxREAL,
        MX_GPU_INITIALIZE_VALUES
    );  

    float * d_output_embedding = (float *) mxGPUGetData(output_embedding);

    SH_Encoding_forward_kernel<<<N_BLOCKS, N_THREADS>>>(
        d_dir_batch,
        dir_sz,
        /// output
        d_output_embedding
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(output_embedding);

    mxGPUDestroyGPUArray(dir_batch);
    mxGPUDestroyGPUArray(output_embedding);
}

void Call_Bwd(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const * prhs[]
){
    const mxGPUArray * dir_batch;
    const mxGPUArray * dl_doutput;

    mxGPUArray * dl_ddir;

    mxInitGPU();
    dir_batch   = mxGPUCreateFromMxArray(prhs[0]);
    dl_doutput  = mxGPUCreateFromMxArray(prhs[1]);

    int feature_dim = (int) mxGetPr(prhs[2])[0];


    dim3 dir_sz = size2dim3(dir_batch);
    dim3 dl_doutput_sz = size2dim3(dl_doutput);
    // mexPrintf("dir_batch size is %u %u %u \n", dir_sz.x, dir_sz.y, dir_sz.z);
    // mexPrintf("dl_doutput_sz size is %u %u %u \n", dl_doutput_sz.x, dl_doutput_sz.y, dl_doutput_sz.z);
    CHECK_THROW((size_t) dir_sz.x == 3);
    CHECK_THROW((size_t) dir_sz.y == (size_t) dl_doutput_sz.y);
    CHECK_THROW((size_t) dl_doutput_sz.x == (size_t) 16);

    const float3 * d_xyz_batch = (const float3 *) mxGPUGetDataReadOnly(dir_batch);
    const float * d_dl_doutput = (const float *) mxGPUGetDataReadOnly(dl_doutput);

    dim3 N_BLOCKS = {
        (unsigned) (dir_sz.y + N_THREADS - 1) / N_THREADS,
        1
    };
    // gradient of embedding
    dl_ddir = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(dir_batch),
        mxGPUGetDimensions(dir_batch),
        mxSINGLE_CLASS,
        mxREAL,
        MX_GPU_INITIALIZE_VALUES
    );  

    float3 * d_dl_ddir = (float3 *) mxGPUGetData(dl_ddir);

    SH_Encoding_backward_kernel<<<N_BLOCKS, N_THREADS>>>(
        d_dl_doutput,
        d_xyz_batch,
        dir_sz,
        /// output
        d_dl_ddir
    );

    plhs[0] = mxGPUCreateMxArrayOnGPU(dl_ddir);
    mxGPUDestroyGPUArray(dir_batch);
    mxGPUDestroyGPUArray(dl_doutput);
    mxGPUDestroyGPUArray(dl_ddir);
}