#pragma once
#include "common.hpp"

template <typename Tv>
FORCE_INLINE Tv fast_diag_exp(const Tv &vt, double theta)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
        return static_cast<Tv>(1.0);
    }
    else
    {
        double val = vt.imag() * theta;
        return Tv(std::cos(val), std::sin(val));
    }
}

template <typename Tv>
FORCE_INLINE Tv fast_diag_grad(const Tv &vt, double theta)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
        return static_cast<Tv>(0.0);
    }
    else
    {
        double val = vt.imag() * theta;
        Tv u(std::cos(val), std::sin(val));
        return vt * u;
    }
}

template <int Rank, typename Ti, typename Tv>
FORCE_INLINE void precompute_phase(Ti str, const Ti *zs, int num_zs, const Tv *w0, Tv *p0, int stride, int rank)
{
    static_assert(is_supported_bit_uint_v<Ti>, "precompute_phase<Ti> requires a supported unsigned bit-integer type");
    if constexpr (Rank == 1)
    {
        Tv v0 = {};
        for (int k = 0; k < num_zs; ++k)
        {
            bool parity = popcnt(str & zs[k]) & 1;
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
            bool parity = popcnt(str & zs[k]) & 1;
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
                bool parity = popcnt(str & zs[k]) & 1;
                vr += parity ? -wr[k] : wr[k];
            }
            p0[r * stride] = vr;
        }
    }
}

template <int Rank, typename Tv>
FORCE_INLINE Tv compute_coeff(int b, const Tv *pa, const Tv *pb, int stride, int rank)
{
    Tv vt = {};
    if constexpr (Rank == 1)
    {
        vt = pa[0] * pb[b];
    }
    else if constexpr (Rank == 2)
    {
        vt = pa[0] * pb[b] + pa[1] * pb[b + stride];
    }
    else
    {
        for (int r = 0; r < rank; ++r)
        {
            vt += pa[r] * pb[b + r * stride];
        }
    }

    return vt;
}

template <int Rank, typename Tv>
FORCE_INLINE Tv compute_coeff(int a, int b, const Tv *pa, const Tv *pb, int stride_a, int stride_b, int rank)
{
    Tv vt = {};
    if constexpr (Rank == 1)
    {
        vt = pa[a] * pb[b];
    }
    else if constexpr (Rank == 2)
    {
        vt = pa[a] * pb[b] + pa[a + stride_a] * pb[b + stride_b];
    }
    else
    {
        for (int r = 0; r < rank; ++r)
        {
            vt += pa[a + r * stride_a] * pb[b + r * stride_b];
        }
    }

    return vt;
}

template <typename Tv>
FORCE_INLINE void hvec_update(const Tv *sp, Tv *dp, Tv vt)
{
    *dp += *sp * vt;
}

template <typename Tv>
FORCE_INLINE void expm_update(Tv *sp, Tv *dp, Tv vt, double cd, double co)
{
    const Tv vt_c = math_conj(vt);

    const Tv vd = 1.0 + cd * (vt * vt_c);
    const Tv vo_fwd = co * vt;
    const Tv vo_rev = co * vt_c;

    const Tv vi = *sp;
    const Tv vj = *dp;

    *sp = vi * vd - vj * vo_rev;
    *dp = vj * vd + vi * vo_fwd;
}

template <typename Tv>
FORCE_INLINE void grad_update(Tv &res, const Tv *ls, const Tv *ld, const Tv *rs, const Tv *rd, Tv vt, double cd, double co)
{
    const Tv vt_c = math_conj(vt);

    const Tv vd = cd * (vt * vt_c);
    const Tv vo_fwd = co * vt;
    const Tv vo_rev = co * vt_c;

    const Tv rvi = *rs;
    const Tv rvj = *rd;

    res += math_conj(*ls) * (rvi * vd + rvj * vo_rev) +
           math_conj(*ld) * (rvj * vd - rvi * vo_fwd);
}

template <typename Tv>
FORCE_INLINE void tvec_update(const Tv *ss, const Tv *sd, Tv *ds, Tv *dd, Tv vt)
{
    *ds = *sd * (-math_conj(vt));
    *dd = *ss * vt;
}

template <typename Tv>
FORCE_INLINE void tvec_update(Tv *sp, Tv *dp, Tv vt)
{
    const Tv vi = *sp;
    const Tv vj = *dp;

    *sp = vj * (-math_conj(vt));
    *dp = vi * vt;
}

template <typename Tv>
FORCE_INLINE void tran_update(Tv &res, const Tv *ls, const Tv *ld, const Tv *rs, const Tv *rd, Tv vt)
{
    res += math_conj(*ls * vt) * *rd - math_conj(*ld) * vt * *rs;
}

template <typename Tv>
FORCE_INLINE void backgrad_update(Tv &res, Tv *ls, Tv *ld, Tv *rs, Tv *rd, Tv vt, double ecd, double eco, double gcd, double gco)
{
    const Tv vt_c = math_conj(vt);

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

    res += math_conj(els) * (rvi * gvd + rvj * gvo_rev) +
           math_conj(eld) * (rvj * gvd - rvi * gvo_fwd);

    *ls = els;
    *ld = eld;
    *rs = rvi * evd - rvj * evo_rev;
    *rd = rvj * evd + rvi * evo_fwd;
}

template <typename Tv>
FORCE_INLINE void backtran_update(
    Tv *ls, Tv *ld, Tv *rs, Tv *rd, Tv *bs, Tv *bd,
    Tv vt, double ecd, double eco)
{
    expm_update(ls, ld, vt, ecd, eco);
    expm_update(rs, rd, vt, ecd, eco);
    tvec_update(ls, ld, bs, bd, vt);
}

template <typename Tv>
FORCE_INLINE void expm_batch_update_matrix(
    Tv *__restrict__ matrix, // 矩阵的平铺指针
    int ld,                  // 矩阵的 Leading Dimension (在 Julia 列主序中就是 N)
    int num_vecs,            // 当前激活的波函数数量 (即你说的有效行数范围)
    int64 si, int64 di,      // 基底索引 i 和 j
    Tv vt, double cd, double co)
{
    const Tv vt_c = math_conj(vt);
    const Tv vd = 1.0 + cd * (vt * vt_c);
    const Tv vo_fwd = co * vt;
    const Tv vo_rev = co * vt_c;

    Tv *__restrict__ sp_base = matrix + si * ld;
    Tv *__restrict__ dp_base = matrix + di * ld;

#pragma omp simd
    for (int k = 0; k < num_vecs; ++k)
    {
        const Tv vi = sp_base[k];
        const Tv vj = dp_base[k];

        sp_base[k] = vi * vd - vj * vo_rev;
        dp_base[k] = vj * vd + vi * vo_fwd;
    }
}

template <typename Tv>
FORCE_INLINE void expm_batch_update_diag(
    Tv *__restrict__ matrix, // 矩阵的平铺指针
    int ld,                  // 矩阵 Leading Dimension (即 N)
    int num_vecs,            // 当前激活的波函数数量
    int64 i,                 // 基底索引
    Tv u)                    // 标量系数 (相移或缩放)
{
    Tv *__restrict__ p_base = matrix + i * ld;

#pragma omp simd
    for (int k = 0; k < num_vecs; ++k)
    {
        p_base[k] *= u;
    }
}
