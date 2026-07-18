#pragma once
#include "cuda/common.cuh"

template <typename Tv>
__device__ __forceinline__ Tv fast_diag_exp_dev(Tv vt, double theta)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
        return static_cast<Tv>(1.0);
    }
    else
    {
        const double val = vt.imag() * theta;
        return Tv(std::cos(val), std::sin(val));
    }
}

template <typename Tv>
__device__ __forceinline__ Tv fast_diag_grad_dev(Tv vt, double theta)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
        return static_cast<Tv>(0.0);
    }
    else
    {
        const double val = vt.imag() * theta;
        const Tv u(std::cos(val), std::sin(val));
        return vt * u;
    }
}

template <int Rank, typename Ti, typename Tv>
__device__ __forceinline__ void compute_phase_dev(Ti str, const Ti *__restrict__ zs, int num_zs, const Tv *__restrict__ w0, Tv *p0, int stride, int rank)
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
__device__ __forceinline__ Tv compute_coeff_dev(const Tv *__restrict__ pa, const Tv *__restrict__ pb, int stride, int rank, int b)
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

template <int Rank, typename Tv>
__device__ __forceinline__ Tv compute_coeff_dev(const Tv *__restrict__ pa, const Tv *__restrict__ pb, int rank)
{
    if constexpr (Rank == 1)
        return pa[0] * pb[0];
    else if constexpr (Rank == 2)
        return pa[0] * pb[0] + pa[1] * pb[1];
    else
    {
        Tv vt = {};
        for (int r = 0; r < rank; ++r)
            vt += pa[r] * pb[r];
        return vt;
    }
}

template <typename Tv>
__device__ __forceinline__ void expm_update_dev(Tv *sp, Tv *dp, Tv vt, double cd, double co)
{
    const Tv vt_c = dev_conj(vt);

    const Tv vd = 1.0 + cd * (vt * vt_c);
    const Tv vo_fwd = co * vt;
    const Tv vo_rev = co * vt_c;

    const Tv vi = *sp;
    const Tv vj = *dp;

    *sp = vi * vd - vj * vo_rev;
    *dp = vj * vd + vi * vo_fwd;
}

template <typename Tv>
__device__ __forceinline__ void grad_update_dev(Tv &res, const Tv *ls, const Tv *ld, const Tv *rs, const Tv *rd, Tv vt, double cd, double co)
{
    const Tv vt_c = dev_conj(vt);

    const Tv vd = cd * (vt * vt_c);
    const Tv vo_fwd = co * vt;
    const Tv vo_rev = co * vt_c;

    const Tv rvi = *rs;
    const Tv rvj = *rd;

    res += dev_conj(*ls) * (rvi * vd + rvj * vo_rev) +
           dev_conj(*ld) * (rvj * vd - rvi * vo_fwd);
}

template <typename Tv>
__device__ __forceinline__ void tvec_update_dev(const Tv *ss, const Tv *sd, Tv *ds, Tv *dd, Tv vt)
{
    *ds = *sd * (-dev_conj(vt));
    *dd = *ss * vt;
}

template <typename Tv>
__device__ __forceinline__ void tvec_update_dev(Tv *sp, Tv *dp, Tv vt)
{
    const Tv vi = *sp;
    const Tv vj = *dp;

    *sp = vj * (-dev_conj(vt));
    *dp = vi * vt;
}

template <typename Tv>
__device__ __forceinline__ void tran_update_dev(Tv &res, const Tv *ls, const Tv *ld, const Tv *rs, const Tv *rd, Tv vt)
{
    res += dev_conj(*ls * vt) * *rd - dev_conj(*ld) * vt * *rs;
}

template <typename Tv>
__device__ __forceinline__ void backgrad_update_dev(Tv &res, Tv *ls, Tv *ld, Tv *rs, Tv *rd, Tv vt, double ecd, double eco, double gcd, double gco)
{
    const Tv vt_c = dev_conj(vt);

    const Tv evd = 1.0 + ecd * (vt * vt_c);
    const Tv evo_fwd = eco * vt;
    const Tv evo_rev = eco * vt_c;

    const Tv gvd = gcd * (vt * vt_c);
    const Tv gvo_fwd = gco * vt;
    const Tv gvo_rev = gco * vt_c;

    const Tv lvi = *ls;
    const Tv lvj = *ld;
    const Tv rvi = *rs;
    const Tv rvj = *rd;

    const Tv els = lvi * evd - lvj * evo_rev;
    const Tv eld = lvj * evd + lvi * evo_fwd;

    res += dev_conj(els) * (rvi * gvd + rvj * gvo_rev) +
           dev_conj(eld) * (rvj * gvd - rvi * gvo_fwd);

    *ls = els;
    *ld = eld;
    *rs = rvi * evd - rvj * evo_rev;
    *rd = rvj * evd + rvi * evo_fwd;
}

template <typename Tv>
__device__ __forceinline__ void backtran_update_dev(Tv *ls, Tv *ld, Tv *rs, Tv *rd, Tv *bs, Tv *bd, Tv vt, double ecd, double eco)
{
    expm_update(ls, ld, vt, ecd, eco);
    expm_update(rs, rd, vt, ecd, eco);
    tvec_update(ls, ld, bs, bd, vt);
}

// ========== 块归约工具 (tile-based, 1D block) ==========

template <typename Tv>
__device__ __forceinline__ void block_reduce_atomic_add_tile(Tv local_res, Tv *__restrict__ d_res)
{
    local_res = warp_reduce_sum(local_res);
    __shared__ Tv shared_sums[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0)
        shared_sums[warp] = local_res;
    __syncthreads();

    if (warp == 0)
    {
        const int num_warps = (blockDim.x + 31) >> 5;
        local_res = (lane < num_warps) ? shared_sums[lane] : Tv{};
        local_res = warp_reduce_sum(local_res);
        if (lane == 0)
            atomicAdd_Tv(d_res, local_res);
    }
}

// ========== 块归约工具 (2D block 32x16) ==========

template <typename Tv>
__device__ __forceinline__ void block_reduce_atomic_add_2d(Tv local_res, Tv *__restrict__ d_res)
{
    local_res = warp_reduce_sum(local_res);

    __shared__ Tv shared_sums[16];
    if (threadIdx.x == 0)
        shared_sums[threadIdx.y] = local_res;

    __syncthreads();

    if (threadIdx.y == 0)
    {
        local_res = (threadIdx.x < 16) ? shared_sums[threadIdx.x] : Tv{};
        local_res = warp_reduce_sum(local_res);

        if (threadIdx.x == 0)
            atomicAdd_Tv(d_res, local_res);
    }
}

// ========== 对称性 Hash 公式 (off-diag TypeCode 1/2/3) ==========

template <int TypeCode>
__device__ __forceinline__ int compute_sym_hash(int dst_asym, int dst_bsym, int group_asym, int group_bsym, int num_irreps)
{
    if constexpr (TypeCode == 1)
        return (dst_asym ^ group_asym) * num_irreps + dst_bsym;
    else if constexpr (TypeCode == 2)
        return dst_asym * num_irreps + (dst_bsym ^ group_bsym);
    else
        return (dst_asym ^ group_asym) * num_irreps + (dst_bsym ^ group_bsym);
}

// ========== Batch group 共享内存 / chunk 大小常量 ==========

template <int Rank>
inline constexpr int BATCH_GROUP_SIZE = Rank == 1 ? BATCH_SIZE_SH1 : Rank == 2 ? BATCH_SIZE_SH2 : BATCH_SIZE_SH3;

template <int Rank>
inline constexpr int BATCH_GROUP_RANK_FACTOR = Rank == 1 ? 1 : Rank == 2 ? 2 : KERNEL_MAX_RANK;

template <int Rank>
inline constexpr int BATCH_GROUP_SHARED_MEM = BATCH_GROUP_SIZE<Rank> * TILE_B * BATCH_GROUP_RANK_FACTOR<Rank>;

template <int Rank>
inline constexpr int BATCH_GROUP_IDX_MEM = BATCH_GROUP_SIZE<Rank> * TILE_B;
