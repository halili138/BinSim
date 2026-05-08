#pragma once
#include "otf.hpp"

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void expm_contract_diag_otf_impl(
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    Tv *__restrict__ vec)
{
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
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
        std::vector<Tv> local_a_phase(max_a_count * rank);
        std::vector<Tv> local_b_phase(max_b_count * rank);

        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            const int a_count = block.num_a;
            const int b_count = block.num_b;
            Tv *pa0 = local_a_phase.data();
            Tv *pb0 = local_b_phase.data();

            compute_phases<Rank, 0, Ti, Tv>(
                block.astrs, a_count, zas, num_za, wa0, pa0, max_a_count, rank);

            compute_phases<Rank, 0, Ti, Tv>(
                block.bstrs, b_count, zbs, num_zb, wb0, pb0, max_b_count, rank);

            const Tv *pa = local_a_phase.data();
            const Tv *pb = local_b_phase.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < a_count; ++a)
            {
                for (int b = 0; b < b_count; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
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
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
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
        std::vector<Tv> local_a_phase(max_a_count * rank);
        std::vector<Tv> local_b_phase(max_b_count * rank);

        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            const int a_count = block.num_a;
            const int b_count = block.num_b;
            Tv *pa0 = local_a_phase.data();
            Tv *pb0 = local_b_phase.data();

            compute_phases<Rank, 0, Ti, Tv>(
                block.astrs, a_count, zas, num_za, wa0, pa0, max_a_count, rank);

            compute_phases<Rank, 0, Ti, Tv>(
                block.bstrs, b_count, zbs, num_zb, wb0, pb0, max_b_count, rank);

            const Tv *pa = local_a_phase.data();
            const Tv *pb = local_b_phase.data();

#pragma omp for schedule(dynamic) nowait
            for (int a = 0; a < a_count; ++a)
            {
                for (int b = 0; b < b_count; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const Tv du = fast_diag_grad<Tv>(vt, theta);
                    const int64 i = block.offset + (int64)a * b_count + b;
                    res += math_conj(lp[i] * du) * rp[i];
                }
            }
        }
    }
    return res;
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void expm_contract_pure_a_otf_impl(
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = std::cos(theta) - 1.0;
    const double co = std::sin(theta);
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
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

            int valid_na = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.ax, idx_map.a_idx_map,
                dst_block.astrs, dst_block.num_a, zas, num_za,
                wa0, phase_a.data(), max_a_count, rank,
                src_a.data(), dst_a.data(), is_same_block, true);

            if (valid_na == 0)
                continue;

            compute_phases<Rank, 0, Ti, Tv>(
                dst_block.bstrs, dst_block.num_b, zbs, num_zb,
                wb0, phase_b.data(), max_b_count, rank);

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < dst_block.num_b; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
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
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -std::sin(theta);
    const double co = std::cos(theta);
    const int rank = group.rank;
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

            int valid_na = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.ax, idx_map.a_idx_map,
                dst_block.astrs, dst_block.num_a, zas, num_za,
                wa0, phase_a.data(), max_a_count, rank,
                src_a.data(), dst_a.data(), is_same_block, true);

            if (valid_na == 0)
                continue;

            compute_phases<Rank, 0, Ti, Tv>(
                dst_block.bstrs, dst_block.num_b, zbs, num_zb,
                wb0, phase_b.data(), max_b_count, rank);

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < dst_block.num_b; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
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
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = std::cos(theta) - 1.0;
    const double co = std::sin(theta);
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
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

            int valid_nb = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.bx, idx_map.b_idx_map, 
                dst_block.bstrs, dst_block.num_b, zbs, num_zb, 
                wb0, phase_b.data(), max_b_count, rank, 
                src_b.data(), dst_b.data(), is_same_block, true);

            if (valid_nb == 0)
                continue;

            compute_phases<Rank, 0, Ti, Tv>(
                dst_block.astrs, dst_block.num_a, zas, num_za, 
                wa0, phase_a.data(), max_a_count, rank);

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < dst_block.num_a; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
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
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -std::sin(theta);
    const double co = std::cos(theta);
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
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

            int valid_nb = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.bx, idx_map.b_idx_map, 
                dst_block.bstrs, dst_block.num_b, zbs, num_zb, 
                wb0, phase_b.data(), max_b_count, rank, 
                src_b.data(), dst_b.data(), is_same_block, true);

            if (valid_nb == 0)
                continue;

            compute_phases<Rank, 0, Ti, Tv>(
                dst_block.astrs, dst_block.num_a, zas, num_za, 
                wa0, phase_a.data(), max_a_count, rank);

            const Tv *__restrict__ pa = phase_a.data();
            const Tv *__restrict__ pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < dst_block.num_a; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
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
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = std::cos(theta) - 1.0;
    const double co = std::sin(theta);
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
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

            int valid_na = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.ax, idx_map.a_idx_map,
                dst_block.astrs, dst_block.num_a, zas, num_za,
                wa0, phase_a.data(), max_a_count, rank,
                src_a.data(), dst_a.data(), is_same_block, true);

            if (valid_na == 0)
                continue;

            int valid_nb = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.bx, idx_map.b_idx_map, 
                dst_block.bstrs, dst_block.num_b, zbs, num_zb, 
                wb0, phase_b.data(), max_b_count, rank, 
                src_b.data(), dst_b.data(), is_same_block, false);

            if (valid_nb == 0)
                continue;

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
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
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -std::sin(theta);
    const double co = std::cos(theta);
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
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

            int valid_na = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.ax, idx_map.a_idx_map,
                dst_block.astrs, dst_block.num_a, zas, num_za,
                wa0, phase_a.data(), max_a_count, rank,
                src_a.data(), dst_a.data(), is_same_block, true);

            if (valid_na == 0)
                continue;

            int valid_nb = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.bx, idx_map.b_idx_map, 
                dst_block.bstrs, dst_block.num_b, zbs, num_zb, 
                wb0, phase_b.data(), max_b_count, rank, 
                src_b.data(), dst_b.data(), is_same_block, false);

            if (valid_nb == 0)
                continue;

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
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

template <typename Ti,
          typename Tv>
void expm_svd_network_otf(
    const BasisManager<Ti> *basis,
    const Network_OTF<Ti, Tv> *__restrict__ net,
    const int64 idx,
    const double theta,
    Tv *__restrict__ vec)
{
    const uint8 type = net->excit_types[idx];
    const SVDGroup_OTF<Ti, Tv> &group = net->flat_groups[idx];
    const int rank = group.rank;

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
    const BasisManager<Ti> *basis,
    const Network_OTF<Ti, Tv> *__restrict__ net,
    const int64 idx,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const uint8 type = net->excit_types[idx];
    const SVDGroup_OTF<Ti, Tv> &group = net->flat_groups[idx];
    const int rank = group.rank;

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

template <typename Ti, typename Tv>
void *build_pool_network_otf(
    const BasisManager<Ti> *basis,
    int64 norb, int64 ngs,
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
