#pragma once
#include "cuda_common.cuh"

template <int Rank, typename Ti, typename Tv>
__device__ __forceinline__ void compute_phase_dev(
    Ti str,
    const Ti *__restrict__ zs, int num_zs,
    const Tv *__restrict__ w0, Tv *p0, int stride, int rank)
{
    if constexpr (Rank == 1)
    {
        Tv v0 = {};
        for (int k = 0; k < num_zs; ++k)
        {
            bool parity = count_ones(str & zs[k]) & 1;
            v0 += parity ? -w0[k] : w0[k];
        }
        p0[0] = v0;
    }
    else if constexpr (Rank == 2)
    {
        const Tv *w1 = w0 + num_zs;
        Tv v0 = {};
        Tv v1 = {};
        for (int k = 0; k < num_zs; ++k)
        {
            bool parity = count_ones(str & zs[k]) & 1;
            v0 += parity ? -w0[k] : w0[k];
            v1 += parity ? -w1[k] : w1[k];
        }
        p0[0] = v0;
        p0[stride] = v1;
    }
    else
    {
        for (int r = 0; r < rank; ++r)
        {
            const Tv *wr = w0 + r * num_zs;
            Tv vr = {};
            for (int k = 0; k < num_zs; ++k)
            {
                bool parity = count_ones(str & zs[k]) & 1;
                vr += parity ? -wr[k] : wr[k];
            }
            p0[r * stride] = vr;
        }
    }
}

template <int Rank, typename Tv>
__device__ __forceinline__ Tv compute_coeff_dev(
    const Tv *__restrict__ pa,
    const Tv *__restrict__ pb, int stride, int rank, int b)
{
    if constexpr (Rank == 1)
        return pa[0] * pb[b];
    else if constexpr (Rank == 2)
        return pa[0] * pb[b] + pa[1] * pb[stride + b];
    else
    {
        Tv vt = {};
        for (int r = 0; r < rank; ++r)
            vt += pa[r] * pb[r * stride + b];
        return vt;
    }
}
