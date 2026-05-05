#pragma once
#include "otf.hpp"

template <int Rank, typename Ti, typename Tv>
static inline void expm_contract_diag_otf_impl(
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

    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    std::vector<Tv> shared_b_phase(max_b_count * rank);

#pragma omp parallel
    {
        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            const int a_count = block.num_a;
            const int b_count = block.num_b;
#pragma omp single
            {
                Tv *pb0 = shared_b_phase.data();
                if constexpr (Rank == 1)
                {
                    for (int b = 0; b < b_count; ++b)
                    {
                        Ti str_b = block.bstrs[b];
                        Tv pt0 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                        }
                        pb0[b] = pt0;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv *pb1 = pb0 + max_b_count;
                    for (int b = 0; b < b_count; ++b)
                    {
                        Ti str_b = block.bstrs[b];
                        Tv pt0 = {}, pt1 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                            pt1 += parity ? -wb1[k] : wb1[k];
                        }
                        pb0[b] = pt0;
                        pb1[b] = pt1;
                    }
                }
                else
                {
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv *pbr = pb0 + r * max_b_count;
                        const Tv *wr = wb0 + r * num_zb;
                        for (int b = 0; b < b_count; ++b)
                        {
                            Ti str_b = block.bstrs[b];
                            Tv ptn = {};
                            for (int k = 0; k < num_zb; ++k)
                            {
                                const bool parity = std::popcount(str_b & zbs[k]) & 1;
                                ptn += parity ? -wr[k] : wr[k];
                            }
                            pbr[b] = ptn;
                        }
                    }
                }
            }

#pragma omp for schedule(dynamic)
            for (int a = 0; a < a_count; ++a)
            {
                const Ti str_a = block.astrs[a];
                Tv *vp = vec + block.offset + (int64)a * b_count;
                const Tv *pb0 = shared_b_phase.data();

                if constexpr (Rank == 1)
                {
                    Tv pa0 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                    }
#pragma omp simd
                    for (int b = 0; b < b_count; ++b)
                    {
                        const Tv vt = pa0 * pb0[b];
                        const Tv u = fast_diag_exp<Tv>(vt, theta);
                        *(vp + b) *= u;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv pa0 = {}, pa1 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                        pa1 += parity ? -wa1[k] : wa1[k];
                    }
                    const Tv *pb1 = pb0 + max_b_count;
#pragma omp simd
                    for (int b = 0; b < b_count; ++b)
                    {
                        const Tv vt = pa0 * pb0[b] + pa1 * pb1[b];
                        const Tv u = fast_diag_exp<Tv>(vt, theta);
                        *(vp + b) *= u;
                    }
                }
                else
                {
                    Tv pan[64] = {};
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv ptn = {};
                        const Tv *wr = wa0 + r * num_za;
                        for (int k = 0; k < num_za; ++k)
                        {
                            const bool parity = std::popcount(str_a & zas[k]) & 1;
                            ptn += parity ? -wr[k] : wr[k];
                        }
                        pan[r] = ptn;
                    }
#pragma omp simd
                    for (int b = 0; b < b_count; ++b)
                    {
                        Tv vt = {};
                        for (uint16 r = 0; r < rank; ++r)
                        {
                            vt += pan[r] * pb0[r * max_b_count + b];
                        }
                        const Tv u = fast_diag_exp<Tv>(vt, theta);
                        *(vp + b) *= u;
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline Tv grad_contract_diag_otf_impl(
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

    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    std::vector<Tv> shared_b_phase(max_b_count * rank);

    Tv global_res = {};

#pragma omp parallel reduction(+ : global_res)
    {
        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            const int a_count = block.num_a;
            const int b_count = block.num_b;
#pragma omp single
            {
                Tv *pb0 = shared_b_phase.data();
                if constexpr (Rank == 1)
                {
                    for (int b = 0; b < b_count; ++b)
                    {
                        Ti str_b = block.bstrs[b];
                        Tv pt0 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                        }
                        pb0[b] = pt0;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv *pb1 = pb0 + max_b_count;
                    for (int b = 0; b < b_count; ++b)
                    {
                        Ti str_b = block.bstrs[b];
                        Tv pt0 = {}, pt1 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                            pt1 += parity ? -wb1[k] : wb1[k];
                        }
                        pb0[b] = pt0;
                        pb1[b] = pt1;
                    }
                }
                else
                {
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv *pbr = pb0 + r * max_b_count;
                        const Tv *wr = wb0 + r * num_zb;
                        for (int b = 0; b < b_count; ++b)
                        {
                            Ti str_b = block.bstrs[b];
                            Tv ptn = {};
                            for (int k = 0; k < num_zb; ++k)
                            {
                                const bool parity = std::popcount(str_b & zbs[k]) & 1;
                                ptn += parity ? -wr[k] : wr[k];
                            }
                            pbr[b] = ptn;
                        }
                    }
                }
            }

#pragma omp for schedule(dynamic)
            for (int a = 0; a < a_count; ++a)
            {
                const Ti str_a = block.astrs[a];
                const int64 cur_idx = block.offset + (int64)a * b_count;
                const Tv *l = lp + cur_idx;
                const Tv *r = rp + cur_idx;
                const Tv *pb0 = shared_b_phase.data();

                Tv local_res = {};

                if constexpr (Rank == 1)
                {
                    Tv pa0 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                    }
#pragma omp simd reduction(+ : local_res)
                    for (int b = 0; b < b_count; ++b)
                    {
                        const Tv vt = pa0 * pb0[b];
                        const Tv du = fast_diag_grad<Tv>(vt, theta);
                        local_res += math_conj(l[b] * du) * r[b];
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv pa0 = {}, pa1 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                        pa1 += parity ? -wa1[k] : wa1[k];
                    }
                    const Tv *pb1 = pb0 + max_b_count;
#pragma omp simd reduction(+ : local_res)
                    for (int b = 0; b < b_count; ++b)
                    {
                        const Tv vt = pa0 * pb0[b] + pa1 * pb1[b];
                        const Tv du = fast_diag_grad<Tv>(vt, theta);
                        local_res += math_conj(l[b] * du) * r[b];
                    }
                }
                else
                {
                    Tv pan[64] = {};
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv ptn = {};
                        const Tv *wr = wa0 + r * num_za;
                        for (int k = 0; k < num_za; ++k)
                        {
                            const bool parity = std::popcount(str_a & zas[k]) & 1;
                            ptn += parity ? -wr[k] : wr[k];
                        }
                        pan[r] = ptn;
                    }
#pragma omp simd reduction(+ : local_res)
                    for (int b = 0; b < b_count; ++b)
                    {
                        Tv vt = {};
                        for (uint16 r = 0; r < rank; ++r)
                        {
                            vt += pan[r] * pb0[r * max_b_count + b];
                        }

                        const Tv du = fast_diag_grad<Tv>(vt, theta);
                        local_res += math_conj(l[b] * du) * r[b];
                    }
                }
                global_res += local_res;
            }
        }
    }
    return global_res;
}

template <int Rank, typename Ti, typename Tv>
static inline void expm_contract_pure_a_otf_impl(
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

    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    std::vector<Tv> shared_b_phase(max_b_count * rank);

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;
            const int64 axsym = get_string_sym(group.ax, orbsym);
            const int64 bid = (dst_block.asym ^ axsym) * num_irreps + dst_block.bsym;
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

#pragma omp single
            {
                Tv *pb0 = shared_b_phase.data();
                if constexpr (Rank == 1)
                {
                    for (int b = 0; b < dst_b_count; ++b)
                    {
                        Ti str_b = dst_block.bstrs[b];
                        Tv pt0 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                        }
                        pb0[b] = pt0;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv *pb1 = pb0 + max_b_count;
                    for (int b = 0; b < dst_b_count; ++b)
                    {
                        Ti str_b = dst_block.bstrs[b];
                        Tv pt0 = {}, pt1 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                            pt1 += parity ? -wb1[k] : wb1[k];
                        }
                        pb0[b] = pt0;
                        pb1[b] = pt1;
                    }
                }
                else
                {
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv *pbr = pb0 + r * max_b_count;
                        const Tv *wr = wb0 + r * num_zb;
                        for (int b = 0; b < dst_b_count; ++b)
                        {
                            Ti str_b = dst_block.bstrs[b];
                            Tv ptn = {};
                            for (int k = 0; k < num_zb; ++k)
                            {
                                const bool parity = std::popcount(str_b & zbs[k]) & 1;
                                ptn += parity ? -wr[k] : wr[k];
                            }
                            pbr[b] = ptn;
                        }
                    }
                }
            }

#pragma omp for schedule(dynamic)
            for (int dst_a_idx = 0; dst_a_idx < dst_a_count; ++dst_a_idx)
            {
                const Ti dst_str_a = dst_block.astrs[dst_a_idx];
                const Ti src_str_a = dst_str_a ^ group.ax;
                const int src_a_idx = idx_map.a_idx_map[src_str_a];

                if (src_a_idx == -1)
                    continue;

                if (is_same_block && src_a_idx < dst_a_idx)
                    continue;

                Tv *src = vec + src_block.offset + (int64)src_a_idx * src_block.num_b;
                Tv *dst = vec + dst_block.offset + (int64)dst_a_idx * dst_block.num_b;
                const Tv *pb0 = shared_b_phase.data();

                if constexpr (Rank == 1)
                {
                    Tv pa0 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                    }
#pragma omp simd
                    for (int b = 0; b < dst_b_count; ++b)
                    {
                        Tv *sp = src + b;
                        Tv *dp = dst + b;
                        const Tv vt = pa0 * pb0[b];
                        const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);
                        const Tv vi = *(sp);
                        const Tv vj = *(dp);
                        *(sp) = vi * vd - vj * vo_rev;
                        *(dp) = vj * vd + vi * vo_fwd;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv pa0 = {}, pa1 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                        pa1 += parity ? -wa1[k] : wa1[k];
                    }
                    const Tv *pb1 = pb0 + max_b_count;
#pragma omp simd
                    for (int b = 0; b < dst_b_count; ++b)
                    {
                        Tv *sp = src + b;
                        Tv *dp = dst + b;
                        const Tv vt = pa0 * pb0[b] + pa1 * pb1[b];
                        const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);
                        const Tv vi = *(sp);
                        const Tv vj = *(dp);
                        *(sp) = vi * vd - vj * vo_rev;
                        *(dp) = vj * vd + vi * vo_fwd;
                    }
                }
                else
                {
                    Tv pan[64] = {};
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv ptn = {};
                        const Tv *wr = wa0 + r * num_za;
                        for (int k = 0; k < num_za; ++k)
                        {
                            const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                            ptn += parity ? -wr[k] : wr[k];
                        }
                        pan[r] = ptn;
                    }
#pragma omp simd
                    for (int b = 0; b < dst_b_count; ++b)
                    {
                        Tv vt = {};
                        for (uint16 r = 0; r < rank; ++r)
                        {
                            vt += pan[r] * pb0[r * max_b_count + b];
                        }
                        Tv *sp = src + b;
                        Tv *dp = dst + b;
                        const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);
                        const Tv vi = *(sp);
                        const Tv vj = *(dp);
                        *(sp) = vi * vd - vj * vo_rev;
                        *(dp) = vj * vd + vi * vo_fwd;
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline Tv grad_contract_pure_a_otf_impl(
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

    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    std::vector<Tv> shared_b_phase(max_b_count * rank);

    Tv global_res = {};

#pragma omp parallel reduction(+ : global_res)
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;
            const int64 axsym = get_string_sym(group.ax, orbsym);
            const int64 bid = (dst_block.asym ^ axsym) * num_irreps + dst_block.bsym;
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

#pragma omp single
            {
                Tv *pb0 = shared_b_phase.data();
                if constexpr (Rank == 1)
                {
                    for (int b = 0; b < dst_b_count; ++b)
                    {
                        Ti str_b = dst_block.bstrs[b];
                        Tv pt0 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                        }
                        pb0[b] = pt0;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv *pb1 = pb0 + max_b_count;
                    for (int b = 0; b < dst_b_count; ++b)
                    {
                        Ti str_b = dst_block.bstrs[b];
                        Tv pt0 = {}, pt1 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                            pt1 += parity ? -wb1[k] : wb1[k];
                        }
                        pb0[b] = pt0;
                        pb1[b] = pt1;
                    }
                }
                else
                {
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv *pbr = pb0 + r * max_b_count;
                        const Tv *wr = wb0 + r * num_zb;
                        for (int b = 0; b < dst_b_count; ++b)
                        {
                            Ti str_b = dst_block.bstrs[b];
                            Tv ptn = {};
                            for (int k = 0; k < num_zb; ++k)
                            {
                                const bool parity = std::popcount(str_b & zbs[k]) & 1;
                                ptn += parity ? -wr[k] : wr[k];
                            }
                            pbr[b] = ptn;
                        }
                    }
                }
            }

#pragma omp for schedule(dynamic)
            for (int dst_a_idx = 0; dst_a_idx < dst_a_count; ++dst_a_idx)
            {
                const Ti dst_str_a = dst_block.astrs[dst_a_idx];
                const Ti src_str_a = dst_str_a ^ group.ax;
                const int src_a_idx = idx_map.a_idx_map[src_str_a];

                if (src_a_idx == -1)
                    continue;

                if (is_same_block && src_a_idx < dst_a_idx)
                    continue;

                const int64 src_row = src_block.offset + (int64)src_a_idx * src_block.num_b;
                const int64 dst_row = dst_block.offset + (int64)dst_a_idx * dst_block.num_b;

                const Tv *pb0 = shared_b_phase.data();
                Tv local_res = {};

                if constexpr (Rank == 1)
                {
                    Tv pa0 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                    }
#pragma omp simd reduction(+ : local_res)
                    for (int b = 0; b < dst_b_count; ++b)
                    {
                        const int64 si = src_row + b;
                        const int64 di = dst_row + b;
                        const Tv vt = pa0 * pb0[b];
                        const Tv vd = cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);

                        const Tv r0 = rp[si];
                        const Tv r1 = rp[di];

                        local_res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                                     math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv pa0 = {}, pa1 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                        pa1 += parity ? -wa1[k] : wa1[k];
                    }
                    const Tv *pb1 = pb0 + max_b_count;
#pragma omp simd reduction(+ : local_res)
                    for (int b = 0; b < dst_b_count; ++b)
                    {
                        const int64 si = src_row + b;
                        const int64 di = dst_row + b;
                        const Tv vt = pa0 * pb0[b] + pa1 * pb1[b];
                        const Tv vd = cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);

                        const Tv r0 = rp[si];
                        const Tv r1 = rp[di];

                        local_res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                                     math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                    }
                }
                else
                {
                    Tv pan[64] = {};
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv ptn = {};
                        const Tv *wr = wa0 + r * num_za;
                        for (int k = 0; k < num_za; ++k)
                        {
                            const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                            ptn += parity ? -wr[k] : wr[k];
                        }
                        pan[r] = ptn;
                    }
#pragma omp simd reduction(+ : local_res)
                    for (int b = 0; b < dst_b_count; ++b)
                    {
                        Tv vt = {};
                        for (uint16 r = 0; r < rank; ++r)
                        {
                            vt += pan[r] * pb0[r * max_b_count + b];
                        }

                        const int64 si = src_row + b;
                        const int64 di = dst_row + b;
                        const Tv vd = cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);

                        const Tv r0 = rp[si];
                        const Tv r1 = rp[di];

                        local_res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                                     math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                    }
                }
                global_res += local_res;
            }
        }
    }
    return global_res;
}

template <int Rank, typename Ti, typename Tv>
static inline void expm_contract_pure_b_otf_impl(
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

    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    std::vector<int> shared_src_b(max_b_count);
    std::vector<int> shared_dst_b(max_b_count);
    std::vector<Tv> shared_b_phase(max_b_count * rank);

    int valid_b_count = 0;

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;
            const int64 bxsym = get_string_sym(group.bx, orbsym);
            const int64 bid = dst_block.asym * num_irreps + (dst_block.bsym ^ bxsym);
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

#pragma omp single
            {
                int count = 0;
                for (int dst_b_idx = 0; dst_b_idx < dst_b_count; ++dst_b_idx)
                {
                    const Ti dst_str_b = dst_block.bstrs[dst_b_idx];
                    const Ti src_str_b = dst_str_b ^ group.bx;
                    const int src_b_idx = idx_map.b_idx_map[src_str_b];

                    if (src_b_idx == -1)
                        continue;

                    if (is_same_block && src_b_idx < dst_b_idx)
                        continue;

                    shared_dst_b[count] = dst_b_idx;
                    shared_src_b[count] = src_b_idx;
                    count++;
                }

                valid_b_count = count;

                Tv *pb0 = shared_b_phase.data();
                if constexpr (Rank == 1)
                {
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                        Tv pt0 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                        }
                        pb0[vb] = pt0;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv *pb1 = pb0 + max_b_count;
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                        Tv pt0 = {}, pt1 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                            pt1 += parity ? -wb1[k] : wb1[k];
                        }
                        pb0[vb] = pt0;
                        pb1[vb] = pt1;
                    }
                }
                else
                {
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv *pbr = pb0 + r * max_b_count;
                        const Tv *wr = wb0 + r * num_zb;
                        for (int vb = 0; vb < valid_b_count; ++vb)
                        {
                            Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                            Tv ptn = {};
                            for (int k = 0; k < num_zb; ++k)
                            {
                                const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                                ptn += parity ? -wr[k] : wr[k];
                            }
                            pbr[vb] = ptn;
                        }
                    }
                }
            }

            if (valid_b_count == 0)
            {
#pragma omp barrier
                continue;
            }

#pragma omp for schedule(dynamic)
            for (int a = 0; a < dst_a_count; ++a)
            {
                const Ti str_a = dst_block.astrs[a];
                Tv *src = vec + src_block.offset + (int64)a * src_block.num_b;
                Tv *dst = vec + dst_block.offset + (int64)a * dst_block.num_b;
                const Tv *pb0 = shared_b_phase.data();

                if constexpr (Rank == 1)
                {
                    Tv pa0 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                    }
#pragma omp simd
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Tv *sp = src + shared_src_b[vb];
                        Tv *dp = dst + shared_dst_b[vb];
                        const Tv vt = pa0 * pb0[vb];
                        const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);
                        const Tv vi = *(sp);
                        const Tv vj = *(dp);
                        *(sp) = vi * vd - vj * vo_rev;
                        *(dp) = vj * vd + vi * vo_fwd;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv pa0 = {}, pa1 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                        pa1 += parity ? -wa1[k] : wa1[k];
                    }
                    const Tv *pb1 = pb0 + max_b_count;
#pragma omp simd
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Tv *sp = src + shared_src_b[vb];
                        Tv *dp = dst + shared_dst_b[vb];
                        const Tv vt = pa0 * pb0[vb] + pa1 * pb1[vb];
                        const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);
                        const Tv vi = *(sp);
                        const Tv vj = *(dp);
                        *(sp) = vi * vd - vj * vo_rev;
                        *(dp) = vj * vd + vi * vo_fwd;
                    }
                }
                else
                {
                    Tv pan[64] = {};
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv ptn = {};
                        const Tv *wr = wa0 + r * num_za;
                        for (int k = 0; k < num_za; ++k)
                        {
                            const bool parity = std::popcount(str_a & zas[k]) & 1;
                            ptn += parity ? -wr[k] : wr[k];
                        }
                        pan[r] = ptn;
                    }
#pragma omp simd
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Tv vt = {};
                        for (uint16 r = 0; r < rank; ++r)
                        {
                            vt += pan[r] * pb0[r * max_b_count + vb];
                        }
                        Tv *sp = src + shared_src_b[vb];
                        Tv *dp = dst + shared_dst_b[vb];
                        const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);
                        const Tv vi = *(sp);
                        const Tv vj = *(dp);
                        *(sp) = vi * vd - vj * vo_rev;
                        *(dp) = vj * vd + vi * vo_fwd;
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline Tv grad_contract_pure_b_otf_impl(
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

    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    std::vector<int> shared_src_b(max_b_count);
    std::vector<int> shared_dst_b(max_b_count);
    std::vector<Tv> shared_b_phase(max_b_count * rank);

    int valid_b_count = 0;
    Tv global_res = {};

#pragma omp parallel reduction(+ : global_res)
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;
            const int64 bxsym = get_string_sym(group.bx, orbsym);
            const int64 bid = dst_block.asym * num_irreps + (dst_block.bsym ^ bxsym);
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

#pragma omp single
            {
                int count = 0;
                for (int dst_b_idx = 0; dst_b_idx < dst_b_count; ++dst_b_idx)
                {
                    const Ti dst_str_b = dst_block.bstrs[dst_b_idx];
                    const Ti src_str_b = dst_str_b ^ group.bx;
                    const int src_b_idx = idx_map.b_idx_map[src_str_b];

                    if (src_b_idx == -1)
                        continue;

                    if (is_same_block && src_b_idx < dst_b_idx)
                        continue;

                    shared_dst_b[count] = dst_b_idx;
                    shared_src_b[count] = src_b_idx;
                    count++;
                }

                valid_b_count = count;

                Tv *pb0 = shared_b_phase.data();
                if constexpr (Rank == 1)
                {
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                        Tv pt0 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                        }
                        pb0[vb] = pt0;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv *pb1 = pb0 + max_b_count;
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                        Tv pt0 = {}, pt1 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                            pt1 += parity ? -wb1[k] : wb1[k];
                        }
                        pb0[vb] = pt0;
                        pb1[vb] = pt1;
                    }
                }
                else
                {
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv *pbr = pb0 + r * max_b_count;
                        const Tv *wr = wb0 + r * num_zb;
                        for (int vb = 0; vb < valid_b_count; ++vb)
                        {
                            Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                            Tv ptn = {};
                            for (int k = 0; k < num_zb; ++k)
                            {
                                const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                                ptn += parity ? -wr[k] : wr[k];
                            }
                            pbr[vb] = ptn;
                        }
                    }
                }
            }

            if (valid_b_count == 0)
            {
#pragma omp barrier
                continue;
            }

#pragma omp for schedule(dynamic)
            for (int a = 0; a < dst_a_count; ++a)
            {
                const Ti str_a = dst_block.astrs[a];
                const int64 src_row = src_block.offset + (int64)a * src_block.num_b;
                const int64 dst_row = dst_block.offset + (int64)a * dst_block.num_b;

                const Tv *pb0 = shared_b_phase.data();
                Tv local_res = {};

                if constexpr (Rank == 1)
                {
                    Tv pa0 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                    }
#pragma omp simd reduction(+ : local_res)
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        const int64 si = src_row + shared_src_b[vb];
                        const int64 di = dst_row + shared_dst_b[vb];
                        const Tv vt = pa0 * pb0[vb];
                        const Tv vd = cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);

                        const Tv r0 = rp[si];
                        const Tv r1 = rp[di];

                        local_res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                                     math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv pa0 = {}, pa1 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                        pa1 += parity ? -wa1[k] : wa1[k];
                    }
                    const Tv *pb1 = pb0 + max_b_count;
#pragma omp simd reduction(+ : local_res)
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        const int64 si = src_row + shared_src_b[vb];
                        const int64 di = dst_row + shared_dst_b[vb];
                        const Tv vt = pa0 * pb0[vb] + pa1 * pb1[vb];
                        const Tv vd = cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);

                        const Tv r0 = rp[si];
                        const Tv r1 = rp[di];

                        local_res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                                     math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                    }
                }
                else
                {
                    Tv pan[64] = {};
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv ptn = {};
                        const Tv *wr = wa0 + r * num_za;
                        for (int k = 0; k < num_za; ++k)
                        {
                            const bool parity = std::popcount(str_a & zas[k]) & 1;
                            ptn += parity ? -wr[k] : wr[k];
                        }
                        pan[r] = ptn;
                    }
#pragma omp simd reduction(+ : local_res)
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Tv vt = {};
                        for (uint16 r = 0; r < rank; ++r)
                        {
                            vt += pan[r] * pb0[r * max_b_count + vb];
                        }

                        const int64 si = src_row + shared_src_b[vb];
                        const int64 di = dst_row + shared_dst_b[vb];
                        const Tv vd = cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);

                        const Tv r0 = rp[si];
                        const Tv r1 = rp[di];

                        local_res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                                     math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                    }
                }
                global_res += local_res;
            }
        }
    }
    return global_res;
}

template <int Rank, typename Ti, typename Tv>
static inline void expm_contract_mixed_otf_impl(
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

    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    std::vector<int> shared_src_b(max_b_count);
    std::vector<int> shared_dst_b(max_b_count);
    std::vector<Tv> shared_b_phase(max_b_count * rank);

    int valid_b_count = 0;

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;
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

#pragma omp single
            {
                int count = 0;
                for (int dst_b_idx = 0; dst_b_idx < dst_b_count; ++dst_b_idx)
                {
                    const Ti dst_str_b = dst_block.bstrs[dst_b_idx];
                    const Ti src_str_b = dst_str_b ^ group.bx;
                    const int src_b_idx = idx_map.b_idx_map[src_str_b];

                    if (src_b_idx == -1)
                        continue;

                    shared_dst_b[count] = dst_b_idx;
                    shared_src_b[count] = src_b_idx;
                    count++;
                }

                valid_b_count = count;

                Tv *pb0 = shared_b_phase.data();
                if constexpr (Rank == 1)
                {
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                        Tv pt0 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                        }
                        pb0[vb] = pt0;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv *pb1 = pb0 + max_b_count;
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                        Tv pt0 = {}, pt1 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                            pt1 += parity ? -wb1[k] : wb1[k];
                        }
                        pb0[vb] = pt0;
                        pb1[vb] = pt1;
                    }
                }
                else
                {
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv *pbr = pb0 + r * max_b_count;
                        const Tv *wr = wb0 + r * num_zb;
                        for (int vb = 0; vb < valid_b_count; ++vb)
                        {
                            Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                            Tv ptn = {};
                            for (int k = 0; k < num_zb; ++k)
                            {
                                const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                                ptn += parity ? -wr[k] : wr[k];
                            }
                            pbr[vb] = ptn;
                        }
                    }
                }
            }

            if (valid_b_count == 0)
            {
#pragma omp barrier
                continue;
            }

#pragma omp for schedule(dynamic)
            for (int dst_a_idx = 0; dst_a_idx < dst_a_count; ++dst_a_idx)
            {
                const Ti dst_str_a = dst_block.astrs[dst_a_idx];
                const Ti src_str_a = dst_str_a ^ group.ax;
                const int src_a_idx = idx_map.a_idx_map[src_str_a];

                if (src_a_idx == -1)
                    continue;

                if (is_same_block && src_a_idx < dst_a_idx)
                    continue;

                Tv *src = vec + src_block.offset + (int64)src_a_idx * src_block.num_b;
                Tv *dst = vec + dst_block.offset + (int64)dst_a_idx * dst_block.num_b;
                const Tv *pb0 = shared_b_phase.data();

                if constexpr (Rank == 1)
                {
                    Tv pa0 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                    }
#pragma omp simd
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Tv *sp = src + shared_src_b[vb];
                        Tv *dp = dst + shared_dst_b[vb];
                        const Tv vt = pa0 * pb0[vb];
                        const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);
                        const Tv vi = *(sp);
                        const Tv vj = *(dp);
                        *(sp) = vi * vd - vj * vo_rev;
                        *(dp) = vj * vd + vi * vo_fwd;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv pa0 = {}, pa1 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                        pa1 += parity ? -wa1[k] : wa1[k];
                    }
                    const Tv *pb1 = pb0 + max_b_count;
#pragma omp simd
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Tv *sp = src + shared_src_b[vb];
                        Tv *dp = dst + shared_dst_b[vb];
                        const Tv vt = pa0 * pb0[vb] + pa1 * pb1[vb];
                        const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);
                        const Tv vi = *(sp);
                        const Tv vj = *(dp);
                        *(sp) = vi * vd - vj * vo_rev;
                        *(dp) = vj * vd + vi * vo_fwd;
                    }
                }
                else
                {
                    Tv pan[64] = {};
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv ptn = {};
                        const Tv *wr = wa0 + r * num_za;
                        for (int k = 0; k < num_za; ++k)
                        {
                            const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                            ptn += parity ? -wr[k] : wr[k];
                        }
                        pan[r] = ptn;
                    }
#pragma omp simd
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Tv vt = {};
                        for (uint16 r = 0; r < rank; ++r)
                        {
                            vt += pan[r] * pb0[r * max_b_count + vb];
                        }
                        Tv *sp = src + shared_src_b[vb];
                        Tv *dp = dst + shared_dst_b[vb];
                        const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);
                        const Tv vi = *(sp);
                        const Tv vj = *(dp);
                        *(sp) = vi * vd - vj * vo_rev;
                        *(dp) = vj * vd + vi * vo_fwd;
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline Tv grad_contract_mixed_otf_impl(
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

    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    std::vector<int> shared_src_b(max_b_count);
    std::vector<int> shared_dst_b(max_b_count);
    std::vector<Tv> shared_b_phase(max_b_count * rank);

    int valid_b_count = 0;
    Tv global_res = {};

#pragma omp parallel reduction(+ : global_res)
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;
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

#pragma omp single
            {
                int count = 0;
                for (int dst_b_idx = 0; dst_b_idx < dst_b_count; ++dst_b_idx)
                {
                    const Ti dst_str_b = dst_block.bstrs[dst_b_idx];
                    const Ti src_str_b = dst_str_b ^ group.bx;
                    const int src_b_idx = idx_map.b_idx_map[src_str_b];

                    if (src_b_idx == -1)
                        continue;

                    shared_dst_b[count] = dst_b_idx;
                    shared_src_b[count] = src_b_idx;
                    count++;
                }

                valid_b_count = count;

                Tv *pb0 = shared_b_phase.data();
                if constexpr (Rank == 1)
                {
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                        Tv pt0 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                        }
                        pb0[vb] = pt0;
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv *pb1 = pb0 + max_b_count;
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                        Tv pt0 = {}, pt1 = {};
                        for (int k = 0; k < num_zb; ++k)
                        {
                            const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                            pt0 += parity ? -wb0[k] : wb0[k];
                            pt1 += parity ? -wb1[k] : wb1[k];
                        }
                        pb0[vb] = pt0;
                        pb1[vb] = pt1;
                    }
                }
                else
                {
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv *pbr = pb0 + r * max_b_count;
                        const Tv *wr = wb0 + r * num_zb;
                        for (int vb = 0; vb < valid_b_count; ++vb)
                        {
                            Ti src_str_b = dst_block.bstrs[shared_dst_b[vb]] ^ group.bx;
                            Tv ptn = {};
                            for (int k = 0; k < num_zb; ++k)
                            {
                                const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                                ptn += parity ? -wr[k] : wr[k];
                            }
                            pbr[vb] = ptn;
                        }
                    }
                }
            }

            if (valid_b_count == 0)
            {
#pragma omp barrier
                continue;
            }

#pragma omp for schedule(dynamic)
            for (int dst_a_idx = 0; dst_a_idx < dst_a_count; ++dst_a_idx)
            {
                const Ti dst_str_a = dst_block.astrs[dst_a_idx];
                const Ti src_str_a = dst_str_a ^ group.ax;
                const int src_a_idx = idx_map.a_idx_map[src_str_a];

                if (src_a_idx == -1)
                    continue;

                if (is_same_block && src_a_idx < dst_a_idx)
                    continue;

                const int64 src_row = src_block.offset + (int64)src_a_idx * src_block.num_b;
                const int64 dst_row = dst_block.offset + (int64)dst_a_idx * dst_block.num_b;

                const Tv *pb0 = shared_b_phase.data();
                Tv local_res = {};

                if constexpr (Rank == 1)
                {
                    Tv pa0 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                    }
#pragma omp simd reduction(+ : local_res)
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        const int64 si = src_row + shared_src_b[vb];
                        const int64 di = dst_row + shared_dst_b[vb];
                        const Tv vt = pa0 * pb0[vb];
                        const Tv vd = cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);

                        const Tv r0 = rp[si];
                        const Tv r1 = rp[di];

                        local_res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                                     math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                    }
                }
                else if constexpr (Rank == 2)
                {
                    Tv pa0 = {}, pa1 = {};
                    for (int k = 0; k < num_za; ++k)
                    {
                        const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                        pa0 += parity ? -wa0[k] : wa0[k];
                        pa1 += parity ? -wa1[k] : wa1[k];
                    }
                    const Tv *pb1 = pb0 + max_b_count;
#pragma omp simd reduction(+ : local_res)
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        const int64 si = src_row + shared_src_b[vb];
                        const int64 di = dst_row + shared_dst_b[vb];
                        const Tv vt = pa0 * pb0[vb] + pa1 * pb1[vb];
                        const Tv vd = cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);

                        const Tv r0 = rp[si];
                        const Tv r1 = rp[di];

                        local_res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                                     math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                    }
                }
                else
                {
                    Tv pan[64] = {};
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        Tv ptn = {};
                        const Tv *wr = wa0 + r * num_za;
                        for (int k = 0; k < num_za; ++k)
                        {
                            const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                            ptn += parity ? -wr[k] : wr[k];
                        }
                        pan[r] = ptn;
                    }
#pragma omp simd reduction(+ : local_res)
                    for (int vb = 0; vb < valid_b_count; ++vb)
                    {
                        Tv vt = {};
                        for (uint16 r = 0; r < rank; ++r)
                        {
                            vt += pan[r] * pb0[r * max_b_count + vb];
                        }

                        const int64 si = src_row + shared_src_b[vb];
                        const int64 di = dst_row + shared_dst_b[vb];
                        const Tv vd = cd * (vt * math_conj(vt));
                        const Tv vo_fwd = co * vt;
                        const Tv vo_rev = co * math_conj(vt);

                        const Tv r0 = rp[si];
                        const Tv r1 = rp[di];

                        local_res += math_conj(lp[si]) * (r0 * vd + r1 * vo_rev) +
                                     math_conj(lp[di]) * (r1 * vd - r0 * vo_fwd);
                    }
                }
                global_res += local_res;
            }
        }
    }
    return global_res;
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