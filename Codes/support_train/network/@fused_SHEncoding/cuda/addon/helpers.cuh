#pragma once
#include <string>
#include <stdexcept>
#include <vector>

#define N_THREADS 256

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x) \
	do { if (!(x)) throw std::runtime_error(std::string(FILE_LINE " check failed " #x)); } while(0)

inline __device__ unsigned fast_hash(int pos_grid_local[3]){
    static const unsigned int PRIMES[3] = {1, 1441049, 2097143};
    
    unsigned results = 0;
    results = ((unsigned) pos_grid_local[0] * PRIMES[0]);
    results = results ^ ((unsigned) pos_grid_local[1] * PRIMES[1]);
    results = results ^ ((unsigned) pos_grid_local[2] * PRIMES[2]);
    return results;
}
    
inline __device__ float grid_resolution(
    const int level,
    const float log_scale,
    const float base_res
){
    float exp_scale = (float) base_res * __expf(level * log_scale);
    return floorf(exp_scale);
}