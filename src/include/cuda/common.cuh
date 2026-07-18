#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <utility>
#include "core/types.hpp"

inline constexpr int TILE_A = 256;
inline constexpr int TILE_B = 32;
inline constexpr int BATCH_SIZE_SH1 = 64;
inline constexpr int BATCH_SIZE_SH2 = 32;
inline constexpr int BATCH_SIZE_SH3 = 1;
inline constexpr int KERNEL_MAX_RANK = 128;

__device__ __forceinline__ int count_ones(uint32 v) { return __popc(v); }
__device__ __forceinline__ int count_ones(uint64 v) { return __popcll(v); }

template <typename T>
__device__ __forceinline__ T dev_conj(const T &x)
{
    if constexpr (std::is_arithmetic_v<T>)
        return x;
    else
        return T(x.real(), -x.imag());
}

template <typename Tv>
__device__ __forceinline__ Tv bcast_Tv(const Tv &val)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
        return __shfl_sync(0xffffffff, val, 0); // 直接广播标量
    }
    else
    {
        // 针对复数，分别广播实部和虚部
        auto r = __shfl_sync(0xffffffff, val.real(), 0);
        auto i = __shfl_sync(0xffffffff, val.imag(), 0);
        return Tv(r, i);
    }
}

void check_cuda(cudaError_t err, const char *f, int l)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", f, l, cudaGetErrorString(err));
        exit(1);
    }
}

#define CUDA_CHECK(err) check_cuda(err, __FILE__, __LINE__)

template <typename T>
static const T *up(const T *h, int64 n)
{
    T *dev_ptr = nullptr;
    if (n > 0 && h != nullptr)
    {
        CUDA_CHECK(cudaMalloc(&dev_ptr, n * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(dev_ptr, h, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    return dev_ptr;
}

template <typename T>
static T *up(T *h, int64 n)
{
    T *dev_ptr = nullptr;
    if (n > 0 && h != nullptr)
    {
        CUDA_CHECK(cudaMalloc(&dev_ptr, n * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(dev_ptr, h, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    return dev_ptr;
}

template <typename Tv>
__device__ __forceinline__ Tv warp_reduce_sum(Tv val)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
        return val;
    }
    else
    {
        double re = val.real();
        double im = val.imag();
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            re += __shfl_down_sync(0xffffffff, re, offset);
            im += __shfl_down_sync(0xffffffff, im, offset);
        }
        return Tv(re, im);
    }
}

template <typename Tv>
__device__ __forceinline__ void atomicAdd_Tv(Tv *address, Tv val)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
        atomicAdd(address, val);
    }
    else
    {
        double *p = reinterpret_cast<double *>(address);
        atomicAdd(p, val.real());
        atomicAdd(p + 1, val.imag());
    }
}
