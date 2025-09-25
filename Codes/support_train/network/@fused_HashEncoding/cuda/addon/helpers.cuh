#pragma once
#include <mxGPUArray.h>
#include <string>
#include <stdexcept>
#include <vector>

#define N_THREADS 256

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x) \
	do { if (!(x)) throw std::runtime_error(std::string(FILE_LINE " check failed " #x)); } while(0)

inline __device__ unsigned fast_hash(int x, int y, int z){
    // static const unsigned PRIMES[3] = {1, 2654435761, 805459861};
    // unsigned results = 0;
    // results = ((unsigned) pos_grid_local[0] * PRIMES[0]);
    // results = results ^ ((unsigned) pos_grid_local[1] * PRIMES[1]);
    // results = results ^ ((unsigned) pos_grid_local[2] * PRIMES[2]);
    return (x * 1u) ^ (y * 2654435761u) ^ (z * 805459861u);
}
    
inline __device__ float grid_resolution(
    const int level,
    const float log_scale,
    const float base_res
){
    float exp_scale = (float) base_res * __expf(level * log_scale);
    return floorf(exp_scale);
}

inline __device__ uint32_t grid_index(    
    uint32_t hash_map_sizes,
    uint32_t grid_resolution,
    uint32_t x,
    uint32_t y,
    uint32_t z
){
	// The second part of the loop condition is needed to avoid integer overflows in finer levels.
    // uint32_t index  = x * grid_resolution + y;
    // uint32_t stride = grid_resolution * grid_resolution;

	// if (hash_map_sizes < stride) {
	//     index = fast_hash(x,y);
	// }
    uint32_t index = fast_hash(x,y,z);
	return index % hash_map_sizes;
}