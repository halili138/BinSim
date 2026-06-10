#pragma once
#include "cuda_common.cuh"

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
__device__ __forceinline__ void expm_update(Tv *sp, Tv *dp, Tv vt, double cd, double co)
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
__device__ __forceinline__ void grad_update(Tv &res, const Tv *ls, const Tv *ld, const Tv *rs, const Tv *rd, Tv vt, double cd, double co)
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
__device__ __forceinline__ void tvec_update(const Tv *ss, const Tv *sd, Tv *ds, Tv *dd, Tv vt)
{
    *ds = *sd * (-dev_conj(vt));
    *dd = *ss * vt;
}

template <typename Tv>
__device__ __forceinline__ void tvec_update(Tv *sp, Tv *dp, Tv vt)
{
    const Tv vi = *sp;
    const Tv vj = *dp;

    *sp = vj * (-dev_conj(vt));
    *dp = vi * vt;
}

template <typename Tv>
__device__ __forceinline__ void tran_update(Tv &res, const Tv *ls, const Tv *ld, const Tv *rs, const Tv *rd, Tv vt)
{
    res += dev_conj(*ls * vt) * *rd - dev_conj(*ld) * vt * *rs;
}

template <typename Tv>
__device__ __forceinline__ void backgrad_update(Tv &res, Tv *ls, Tv *ld, Tv *rs, Tv *rd, Tv vt, double ecd, double eco, double gcd, double gco)
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
__device__ __forceinline__ void backtran_update(Tv *ls, Tv *ld, Tv *rs, Tv *rd, Tv *bs, Tv *bd, Tv vt, double ecd, double eco)
{
    expm_update(ls, ld, vt, ecd, eco);
    expm_update(rs, rd, vt, ecd, eco);
    tvec_update(ls, ld, bs, bd, vt);
}
