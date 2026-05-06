#pragma once
#include "otf.hpp"

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void compute_phases_direct(
    const Ti *__restrict__ strs, int num_strs,
    const Tv *__restrict__ w0,
    const Ti *__restrict__ zs, int num_zs,
    Tv *__restrict__ p0, int max_count,
    uint16 rank)
{
    if constexpr (Rank == 1)
    {
        for (int i = 0; i < num_strs; ++i)
        {
            const Ti str = strs[i];
            Tv pt0 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
            }
            p0[i] = pt0;
        }
    }
    else if constexpr (Rank == 2)
    {
        const Tv *w1 = w0 + num_zs;
        Tv *p1 = p0 + max_count;

        for (int i = 0; i < num_strs; ++i)
        {
            const Ti str = strs[i];
            Tv pt0 = {};
            Tv pt1 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
                pt1 += parity ? -w1[k] : w1[k];
            }
            p0[i] = pt0;
            p1[i] = pt1;
        }
    }
    else
    {
        for (uint16 r = 0; r < rank; ++r)
        {
            const Tv *wr = w0 + r * num_zs;
            Tv *pr = p0 + r * max_count;

            for (int i = 0; i < num_strs; ++i)
            {
                const Ti str = strs[i];
                Tv ptn = {};
                for (int k = 0; k < num_zs; ++k)
                {
                    const bool parity = std::popcount(str & zs[k]) & 1;
                    ptn += parity ? -wr[k] : wr[k];
                }
                pr[i] = ptn;
            }
        }
    }
}

template <int Rank, typename Tv>
static FORCE_INLINE Tv compute_coeff(
    int a, int b,
    const Tv *__restrict__ pa,
    const Tv *__restrict__ pb,
    int max_a_count, int max_b_count, uint16 rank)
{
    Tv vt = {};
    if constexpr (Rank == 1)
    {
        vt = pa[a] * pb[b];
    }
    else if constexpr (Rank == 2)
    {
        vt = pa[a] * pb[b] + pa[a + max_a_count] * pb[b + max_b_count];
    }
    else
    {
        for (uint16 r = 0; r < rank; ++r)
        {
            vt += pa[a + r * max_a_count] * pb[b + r * max_b_count];
        }
    }

    return vt;
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE int compute_phases_indirect(
    Ti x,
    const int *idx_map,
    const Ti *__restrict__ strs, int num_strs,
    const Tv *__restrict__ w0,
    const Ti *__restrict__ zs, int num_zs,
    int *__restrict__ src_idxs,
    int *__restrict__ dst_idxs,
    Tv *__restrict__ p0, int max_count,
    uint16 rank,
    bool is_same_block, bool enforce_upper_triangle)
{
    int count = 0;
    for (int i = 0; i < num_strs; ++i)
    {
        Ti dst_str = strs[i];
        Ti src_str = dst_str ^ x;
        int src_idx = idx_map[src_str];

        if (src_idx == -1)
            continue;

        if (is_same_block && enforce_upper_triangle && src_idx < i)
            continue;

        src_idxs[count] = src_idx;
        dst_idxs[count] = i;

        if constexpr (Rank == 1)
        {
            Tv pt0 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(src_str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
            }
            p0[count] = pt0;
        }
        else if constexpr (Rank == 2)
        {
            const Tv *w1 = w0 + num_zs;

            Tv pt0 = {}, pt1 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(src_str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
                pt1 += parity ? -w1[k] : w1[k];
            }

            p0[count] = pt0;
            p0[count + max_count] = pt1;
        }
        else
        {
            for (int r = 0; r < rank; ++r)
            {
                const Tv *wr = w0 + r * num_zs;

                Tv ptr = {};
                for (int k = 0; k < num_zs; ++k)
                {
                    const bool parity = std::popcount(src_str & zs[k]) & 1;
                    ptr += parity ? -wr[k] : wr[k];
                }

                p0[count + r * max_count] = ptr;
            }
        }
        count++;
    }

    return count;
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void expm_contract_diag_otf_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    Tv *__restrict__ vec)
{
    const uint16 rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const Tv *wa1 = wa0 + num_za;
    const Tv *wb1 = wb0 + num_zb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

#pragma omp parallel
    {
        std::vector<Tv> local_a_phase(max_b_count * rank);
        std::vector<Tv> local_b_phase(max_b_count * rank);

        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            const int a_count = block.num_a;
            const int b_count = block.num_b;
            Tv *__restrict__ pa0 = local_a_phase.data();
            Tv *__restrict__ pb0 = local_b_phase.data();

            compute_phases_direct<Rank, Ti, Tv>(
                block.astrs, a_count, wa0, zas, num_za, pa0, max_a_count, rank);

            compute_phases_direct<Rank, Ti, Tv>(
                block.bstrs, b_count, wb0, zbs, num_zb, pb0, max_b_count, rank);

            const Tv *__restrict__ pa = local_a_phase.data();
            const Tv *__restrict__ pb = local_b_phase.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < a_count; ++a)
            {
                for (int b = 0; b < b_count; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const Tv u = fast_diag_exp<Tv>(vt, theta);
                    const int64 i = block.offset + (int64)a * b_count + b;
                    vec[i] *= u;
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE Tv grad_contract_diag_otf_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const uint16 rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const Tv *wa1 = wa0 + num_za;
    const Tv *wb1 = wb0 + num_zb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    Tv res = {};

#pragma omp parallel reduction(+ : res)
    {
        std::vector<Tv> local_a_phase(max_b_count * rank);
        std::vector<Tv> local_b_phase(max_b_count * rank);

        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            const int a_count = block.num_a;
            const int b_count = block.num_b;
            Tv *__restrict__ pa0 = local_a_phase.data();
            Tv *__restrict__ pb0 = local_b_phase.data();

            compute_phases_direct<Rank, Ti, Tv>(
                block.astrs, a_count, wa0, zas, num_za, pa0, max_a_count, rank);

            compute_phases_direct<Rank, Ti, Tv>(
                block.bstrs, b_count, wb0, zbs, num_zb, pb0, max_b_count, rank);

            const Tv *__restrict__ pa = local_a_phase.data();
            const Tv *__restrict__ pb = local_b_phase.data();

#pragma omp for schedule(dynamic) nowait
            for (int a = 0; a < a_count; ++a)
            {
                for (int b = 0; b < b_count; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const Tv du = fast_diag_grad<Tv>(vt, theta);
                    const int64 i = block.offset + (int64)a * b_count + b;
                    const Tv lv = lp[i];
                    const Tv rv = rp[i];
                    res += math_conj(lv * du) * rv;
                }
            }
        }
    }

    return res;
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void expm_contract_pure_a_otf_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = std::cos(theta) - 1.0;
    const double co = std::sin(theta);
    const uint16 rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const Tv *wa1 = wa0 + num_za;
    const Tv *wb1 = wb0 + num_zb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

#pragma omp parallel
    {
        std::vector<int> src_a(max_a_count);
        std::vector<int> dst_a(max_a_count);
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int64 axsym = get_string_sym(group.ax, orbsym);
            const int64 bid = (dst_block.asym ^ axsym) * num_irreps + dst_block.bsym;
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

            int valid_na = compute_phases_indirect<Rank, Ti, Tv>(
                group.ax, idx_map.a_idx_map,
                dst_block.astrs, dst_block.num_a, wa0, zas, num_za,
                src_a.data(), dst_a.data(), phase_a.data(),
                max_a_count, rank, is_same_block, true);

            if (valid_na == 0)
                continue;

            compute_phases_direct<Rank, Ti, Tv>(
                dst_block.bstrs, dst_block.num_b, wb0, zbs, num_zb,
                phase_b.data(),
                max_b_count, rank);

            const Tv *__restrict__ pa = phase_a.data();
            const Tv *__restrict__ pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < dst_block.num_b; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                    const Tv vo_fwd = co * vt;
                    const Tv vo_rev = co * math_conj(vt);

                    const int64 si = src_block.offset + src_a[a] * src_block.num_b + b;
                    const int64 di = dst_block.offset + dst_a[a] * dst_block.num_b + b;

                    const Tv vi = vec[si];
                    const Tv vj = vec[di];

                    vec[si] = vi * vd - vj * vo_rev;
                    vec[di] = vj * vd + vi * vo_fwd;
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE Tv grad_contract_pure_a_otf_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -std::sin(theta);
    const double co = std::cos(theta);
    const uint16 rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const Tv *wa1 = wa0 + num_za;
    const Tv *wb1 = wb0 + num_zb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    Tv res = {};

#pragma omp parallel reduction(+ : res)
    {
        std::vector<int> src_a(max_a_count);
        std::vector<int> dst_a(max_a_count);
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int64 axsym = get_string_sym(group.ax, orbsym);
            const int64 bid = (dst_block.asym ^ axsym) * num_irreps + dst_block.bsym;
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

            int valid_na = compute_phases_indirect<Rank, Ti, Tv>(
                group.ax, idx_map.a_idx_map,
                dst_block.astrs, dst_block.num_a, wa0, zas, num_za,
                src_a.data(), dst_a.data(), phase_a.data(),
                max_a_count, rank, is_same_block, true);

            if (valid_na == 0)
                continue;

            compute_phases_direct<Rank, Ti, Tv>(
                dst_block.bstrs, dst_block.num_b, wb0, zbs, num_zb,
                phase_b.data(),
                max_b_count, rank);

            const Tv *__restrict__ pa = phase_a.data();
            const Tv *__restrict__ pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < dst_block.num_b; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const Tv vd = cd * (vt * math_conj(vt));
                    const Tv vo_fwd = co * vt;
                    const Tv vo_rev = co * math_conj(vt);

                    const int64 si = src_block.offset + src_a[a] * src_block.num_b + b;
                    const int64 di = dst_block.offset + dst_a[a] * dst_block.num_b + b;

                    const Tv r0 = rp[si];
                    const Tv r1 = rp[di];

                    res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                           math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                }
            }
        }
    }

    return res;
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void expm_contract_pure_b_otf_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = std::cos(theta) - 1.0;
    const double co = std::sin(theta);
    const uint16 rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const Tv *wa1 = wa0 + num_za;
    const Tv *wb1 = wb0 + num_zb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

#pragma omp parallel
    {
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<int> src_b(max_b_count);
        std::vector<int> dst_b(max_b_count);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int64 bxsym = get_string_sym(group.bx, orbsym);
            const int64 bid = dst_block.asym * num_irreps + (dst_block.bsym ^ bxsym);
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

            int valid_nb = compute_phases_indirect<Rank, Ti, Tv>(
                group.bx, idx_map.b_idx_map,
                dst_block.bstrs, dst_block.num_b, wb0, zbs, num_zb,
                src_b.data(), dst_b.data(), phase_b.data(),
                max_b_count, rank, is_same_block, true);

            if (valid_nb == 0)
                continue;

            compute_phases_direct<Rank, Ti, Tv>(
                dst_block.astrs, dst_block.num_a, wa0, zas, num_za,
                phase_a.data(),
                max_a_count, rank);

            const Tv *__restrict__ pa = phase_a.data();
            const Tv *__restrict__ pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < dst_block.num_a; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                    const Tv vo_fwd = co * vt;
                    const Tv vo_rev = co * math_conj(vt);

                    const int64 si = src_block.offset + a * src_block.num_b + src_b[b];
                    const int64 di = dst_block.offset + a * dst_block.num_b + dst_b[b];

                    const Tv vi = vec[si];
                    const Tv vj = vec[di];

                    vec[si] = vi * vd - vj * vo_rev;
                    vec[di] = vj * vd + vi * vo_fwd;
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE Tv grad_contract_pure_b_otf_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -std::sin(theta);
    const double co = std::cos(theta);
    const uint16 rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const Tv *wa1 = wa0 + num_za;
    const Tv *wb1 = wb0 + num_zb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    Tv res = {};

#pragma omp parallel reduction(+ : res)
    {
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<int> src_b(max_b_count);
        std::vector<int> dst_b(max_b_count);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int64 bxsym = get_string_sym(group.bx, orbsym);
            const int64 bid = dst_block.asym * num_irreps + (dst_block.bsym ^ bxsym);
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

            int valid_nb = compute_phases_indirect<Rank, Ti, Tv>(
                group.bx, idx_map.b_idx_map,
                dst_block.bstrs, dst_block.num_b, wb0, zbs, num_zb,
                src_b.data(), dst_b.data(), phase_b.data(),
                max_b_count, rank, is_same_block, true);

            if (valid_nb == 0)
                continue;

            compute_phases_direct<Rank, Ti, Tv>(
                dst_block.astrs, dst_block.num_a, wa0, zas, num_za,
                phase_a.data(),
                max_a_count, rank);

            const Tv *__restrict__ pa = phase_a.data();
            const Tv *__restrict__ pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < dst_block.num_a; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const Tv vd = cd * (vt * math_conj(vt));
                    const Tv vo_fwd = co * vt;
                    const Tv vo_rev = co * math_conj(vt);

                    const int64 si = src_block.offset + a * src_block.num_b + src_b[b];
                    const int64 di = dst_block.offset + a * dst_block.num_b + dst_b[b];

                    const Tv r0 = rp[si];
                    const Tv r1 = rp[di];

                    res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                           math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                }
            }
        }
    }

    return res;
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void expm_contract_mixed_otf_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = std::cos(theta) - 1.0;
    const double co = std::sin(theta);
    const uint16 rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const Tv *wa1 = wa0 + num_za;
    const Tv *wb1 = wb0 + num_zb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

#pragma omp parallel
    {
        std::vector<int> src_a(max_a_count);
        std::vector<int> dst_a(max_a_count);
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<int> src_b(max_b_count);
        std::vector<int> dst_b(max_b_count);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int64 axsym = get_string_sym(group.ax, orbsym);
            const int64 bxsym = get_string_sym(group.bx, orbsym);
            const int64 bid = (dst_block.asym ^ axsym) * num_irreps + (dst_block.bsym ^ bxsym);
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

            const int valid_na = compute_phases_indirect<Rank, Ti, Tv>(
                group.ax, idx_map.a_idx_map,
                dst_block.astrs, dst_block.num_a, wa0, zas, num_za,
                src_a.data(), dst_a.data(), phase_a.data(),
                max_a_count, rank, is_same_block, true);

            if (valid_na == 0)
                continue;

            const int valid_nb = compute_phases_indirect<Rank, Ti, Tv>(
                group.bx, idx_map.b_idx_map,
                dst_block.bstrs, dst_block.num_b, wb0, zbs, num_zb,
                src_b.data(), dst_b.data(), phase_b.data(),
                max_b_count, rank, is_same_block, false);

            if (valid_nb == 0)
                continue;

            const Tv *__restrict__ pa = phase_a.data();
            const Tv *__restrict__ pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                    const Tv vo_fwd = co * vt;
                    const Tv vo_rev = co * math_conj(vt);

                    const int64 si = src_block.offset + src_a[a] * src_block.num_b + src_b[b];
                    const int64 di = dst_block.offset + dst_a[a] * dst_block.num_b + dst_b[b];

                    const Tv vi = vec[si];
                    const Tv vj = vec[di];

                    vec[si] = vi * vd - vj * vo_rev;
                    vec[di] = vj * vd + vi * vo_fwd;
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE Tv grad_contract_mixed_otf_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -std::sin(theta);
    const double co = std::cos(theta);
    const uint16 rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const Tv *wa1 = wa0 + num_za;
    const Tv *wb1 = wb0 + num_zb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    Tv res = {};

#pragma omp parallel reduction(+ : res)
    {
        std::vector<int> src_a(max_a_count);
        std::vector<int> dst_a(max_a_count);
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<int> src_b(max_b_count);
        std::vector<int> dst_b(max_b_count);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int64 axsym = get_string_sym(group.ax, orbsym);
            const int64 bxsym = get_string_sym(group.bx, orbsym);
            const int64 bid = (dst_block.asym ^ axsym) * num_irreps + (dst_block.bsym ^ bxsym);
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

            const int valid_na = compute_phases_indirect<Rank, Ti, Tv>(
                group.ax, idx_map.a_idx_map,
                dst_block.astrs, dst_block.num_a, wa0, zas, num_za,
                src_a.data(), dst_a.data(), phase_a.data(),
                max_a_count, rank, is_same_block, true);

            if (valid_na == 0)
                continue;

            const int valid_nb = compute_phases_indirect<Rank, Ti, Tv>(
                group.bx, idx_map.b_idx_map,
                dst_block.bstrs, dst_block.num_b, wb0, zbs, num_zb,
                src_b.data(), dst_b.data(), phase_b.data(),
                max_b_count, rank, is_same_block, false);

            if (valid_nb == 0)
                continue;

            const Tv *__restrict__ pa = phase_a.data();
            const Tv *__restrict__ pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const Tv vd = cd * (vt * math_conj(vt));
                    const Tv vo_fwd = co * vt;
                    const Tv vo_rev = co * math_conj(vt);

                    const int64 si = src_block.offset + src_a[a] * src_block.num_b + src_b[b];
                    const int64 di = dst_block.offset + dst_a[a] * dst_block.num_b + dst_b[b];

                    const Tv r0 = rp[si];
                    const Tv r1 = rp[di];

                    res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                           math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                }
            }
        }
    }

    return res;
}

template <typename Ti, typename Tv>
void *build_pool_network_otf(
    const BasisManager<Ti> *basis,
    int64 norb,
    int64 ngs,
    const Ti *axs,
    const Ti *bxs,
    const int64 *ranks,
    const int64 *num_zas,
    const int64 *num_zbs,
    const Ti *flat_zas,
    const Ti *flat_zbs,
    const Tv *flat_wa,
    const Tv *flat_wb)
{
    Network_OTF<Ti, Tv> *net = new Network_OTF<Ti, Tv>();
    net->num_groups = ngs;

    int32 map_size = 1 << norb;
    int32 *a_map = new int32[map_size];
    int32 *b_map = new int32[map_size];
    std::fill(a_map, a_map + map_size, -1);
    std::fill(b_map, b_map + map_size, -1);

    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        for (int32 a = 0; a < basis->blocks[i].num_a; ++a)
            a_map[basis->blocks[i].astrs[a]] = a;
        for (int32 b = 0; b < basis->blocks[i].num_b; ++b)
            b_map[basis->blocks[i].bstrs[b]] = b;
    }
    net->map.a_idx_map = a_map;
    net->map.b_idx_map = b_map;

    net->excit_types = new uint8[ngs];
    net->flat_groups = new SVDGroup_OTF<Ti, Tv>[ngs];

    uint64 z_offset_a = 0, z_offset_b = 0;
    uint64 w_offset_a = 0, w_offset_b = 0;

    for (int64 g = 0; g < ngs; ++g)
    {
        SVDGroup_OTF<Ti, Tv> &group = net->flat_groups[g];
        group.ax = axs[g];
        group.bx = bxs[g];
        group.rank = ranks[g];
        group.num_za = num_zas[g];
        group.num_zb = num_zbs[g];

        if (group.ax == 0 && group.bx == 0)
            net->excit_types[g] = 0; // Diag
        else if (group.ax != 0 && group.bx == 0)
            net->excit_types[g] = 1; // Pure A
        else if (group.ax == 0 && group.bx != 0)
            net->excit_types[g] = 2; // Pure B
        else
            net->excit_types[g] = 3; // Mixed

        // 拷贝数据
        group.unique_zas = new Ti[group.num_za];
        std::copy(flat_zas + z_offset_a, flat_zas + z_offset_a + group.num_za, group.unique_zas);
        z_offset_a += group.num_za;

        group.unique_zbs = new Ti[group.num_zb];
        std::copy(flat_zbs + z_offset_b, flat_zbs + z_offset_b + group.num_zb, group.unique_zbs);
        z_offset_b += group.num_zb;

        uint64 wa_size = group.num_za * group.rank;
        group.wa = new Tv[wa_size];
        std::copy(flat_wa + w_offset_a, flat_wa + w_offset_a + wa_size, group.wa);
        w_offset_a += wa_size;

        uint64 wb_size = group.num_zb * group.rank;
        group.wb = new Tv[wb_size];
        std::copy(flat_wb + w_offset_b, flat_wb + w_offset_b + wb_size, group.wb);
        w_offset_b += wb_size;
    }

    return static_cast<void *>(net);
}

template <typename Ti,
          typename Tv>
void expm_svd_network_otf(
    const BasisManager<Ti> *__restrict__ basis,
    const Network_OTF<Ti, Tv> *__restrict__ net,
    const int64 idx,
    const double theta,
    Tv *__restrict__ vec)
{
    const uint8 type = net->excit_types[idx];
    const SVDGroup_OTF<Ti, Tv> &group = net->flat_groups[idx];
    const uint16 rank = group.rank;

    switch (type)
    {
    case 0: // Diag
        if (rank == 1)
            expm_contract_diag_otf_impl<1>(basis, net->map, group, theta, vec);
        else if (rank == 2)
            expm_contract_diag_otf_impl<2>(basis, net->map, group, theta, vec);
        else
            expm_contract_diag_otf_impl<0>(basis, net->map, group, theta, vec);
        break;
    case 1: // Pure A
        if (rank == 1)
            expm_contract_pure_a_otf_impl<1>(basis, net->map, group, theta, vec);
        else if (rank == 2)
            expm_contract_pure_a_otf_impl<2>(basis, net->map, group, theta, vec);
        else
            expm_contract_pure_a_otf_impl<0>(basis, net->map, group, theta, vec);
        break;
    case 2: // Pure B
        if (rank == 1)
            expm_contract_pure_b_otf_impl<1>(basis, net->map, group, theta, vec);
        else if (rank == 2)
            expm_contract_pure_b_otf_impl<2>(basis, net->map, group, theta, vec);
        else
            expm_contract_pure_b_otf_impl<0>(basis, net->map, group, theta, vec);
        break;
    case 3: // Mixed
        if (rank == 1)
            expm_contract_mixed_otf_impl<1>(basis, net->map, group, theta, vec);
        else if (rank == 2)
            expm_contract_mixed_otf_impl<2>(basis, net->map, group, theta, vec);
        else
            expm_contract_mixed_otf_impl<0>(basis, net->map, group, theta, vec);
        break;
    default:
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in expm_svd" << std::endl;
        break;
    }
}

template <typename Ti,
          typename Tv>
Tv grad_svd_network_otf(
    const BasisManager<Ti> *__restrict__ basis,
    const Network_OTF<Ti, Tv> *__restrict__ net,
    const int64 idx,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const uint8 type = net->excit_types[idx];
    const SVDGroup_OTF<Ti, Tv> &group = net->flat_groups[idx];
    const uint16 rank = group.rank;

    Tv res = {};

    switch (type)
    {
    case 0:
        if (rank == 1)
            res = grad_contract_diag_otf_impl<1>(basis, net->map, group, theta, lp, rp);
        else if (rank == 2)
            res = grad_contract_diag_otf_impl<2>(basis, net->map, group, theta, lp, rp);
        else
            res = grad_contract_diag_otf_impl<0>(basis, net->map, group, theta, lp, rp);
        break;
    case 1:
        if (rank == 1)
            res = grad_contract_pure_a_otf_impl<1>(basis, net->map, group, theta, lp, rp);
        else if (rank == 2)
            res = grad_contract_pure_a_otf_impl<2>(basis, net->map, group, theta, lp, rp);
        else
            res = grad_contract_pure_a_otf_impl<0>(basis, net->map, group, theta, lp, rp);
        break;
    case 2:
        if (rank == 1)
            res = grad_contract_pure_b_otf_impl<1>(basis, net->map, group, theta, lp, rp);
        else if (rank == 2)
            res = grad_contract_pure_b_otf_impl<2>(basis, net->map, group, theta, lp, rp);
        else
            res = grad_contract_pure_b_otf_impl<0>(basis, net->map, group, theta, lp, rp);
        break;
    case 3:
        if (rank == 1)
            res = grad_contract_mixed_otf_impl<1>(basis, net->map, group, theta, lp, rp);
        else if (rank == 2)
            res = grad_contract_mixed_otf_impl<2>(basis, net->map, group, theta, lp, rp);
        else
            res = grad_contract_mixed_otf_impl<0>(basis, net->map, group, theta, lp, rp);
        break;
    default:
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in grad_svd" << std::endl;
        break;
    }

    return res;
}
