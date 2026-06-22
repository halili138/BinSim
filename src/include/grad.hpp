#pragma once
#include "otf.hpp"
#include "utils.hpp"

template <int Rank, int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE Tv grad_contract_otf_impl(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, const Tv *lp, const Tv *rp)
{
    constexpr bool IsDiagonal = TypeCode == 0;
    constexpr bool UsesAExcitation = TypeCode == 1 || TypeCode == 3;
    constexpr bool UsesBExcitation = TypeCode == 2 || TypeCode == 3;

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
    const int64 num_irreps = basis->num_irreps;
    const int max_a_count = (int)basis->max_a_count;
    const int max_b_count = (int)basis->max_b_count;
    const int *a_idx_map = basis->a_idx_map;
    const int *b_idx_map = basis->b_idx_map;

    Tv res = {};
#pragma omp parallel reduction(+ : res)
    {
        std::vector<int> src_a(UsesAExcitation ? max_a_count : 1);
        std::vector<int> dst_a(UsesAExcitation ? max_a_count : 1);
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<int> src_b(UsesBExcitation ? max_b_count : 1);
        std::vector<int> dst_b(UsesBExcitation ? max_b_count : 1);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            int64 src_block_idx = dst_block_idx;
            if constexpr (!IsDiagonal)
            {
                int64 h;
                if constexpr (TypeCode == 1)
                    h = (dst_block.asym ^ group.asym) * num_irreps + dst_block.bsym;
                else if constexpr (TypeCode == 2)
                    h = dst_block.asym * num_irreps + (dst_block.bsym ^ group.bsym);
                else
                    h = (dst_block.asym ^ group.asym) * num_irreps + (dst_block.bsym ^ group.bsym);
                src_block_idx = block_map[h];
                if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                    continue;
            }

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);
            Tv *pa0 = phase_a.data();
            Tv *pb0 = phase_b.data();
            int valid_na = dst_block.num_a;
            if constexpr (UsesAExcitation)
            {
                valid_na = 0;
                for (int i = 0; i < dst_block.num_a; ++i)
                {
                    const Ti src_str = dst_block.astrs[i] ^ group.ax;
                    const int src_idx = a_idx_map[src_str];
                    if (src_idx == -1 || (is_same_block && src_idx < i))
                        continue;
                    src_a[valid_na] = src_idx;
                    dst_a[valid_na] = i;
                    precompute_phase<Rank, Ti, Tv>(src_str, zas, num_za, wa0, pa0 + valid_na, max_a_count, rank);
                    valid_na++;
                }
            }
            else
            {
                for (int i = 0; i < dst_block.num_a; ++i)
                    precompute_phase<Rank, Ti, Tv>(dst_block.astrs[i], zas, num_za, wa0, pa0 + i, max_a_count, rank);
            }
            if (valid_na == 0)
                continue;

            int valid_nb = dst_block.num_b;
            if constexpr (UsesBExcitation)
            {
                valid_nb = 0;
                for (int i = 0; i < dst_block.num_b; ++i)
                {
                    const Ti src_str = dst_block.bstrs[i] ^ group.bx;
                    const int src_idx = b_idx_map[src_str];
                    if (src_idx == -1)
                        continue;
                    if constexpr (TypeCode == 2)
                    {
                        if (is_same_block && src_idx < i)
                            continue;
                    }
                    src_b[valid_nb] = src_idx;
                    dst_b[valid_nb] = i;
                    precompute_phase<Rank, Ti, Tv>(src_str, zbs, num_zb, wb0, pb0 + valid_nb, max_b_count, rank);
                    valid_nb++;
                }
            }
            else
            {
                for (int i = 0; i < dst_block.num_b; ++i)
                    precompute_phase<Rank, Ti, Tv>(dst_block.bstrs[i], zbs, num_zb, wb0, pb0 + i, max_b_count, rank);
            }
            if (valid_nb == 0)
                continue;

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();
            const int64 src_offset = src_block.offset;
            const int64 dst_offset = dst_block.offset;
            const int64 src_num_b = src_block.num_b;
            const int64 dst_num_b = dst_block.num_b;

            if constexpr (IsDiagonal)
            {
#pragma omp for schedule(dynamic) nowait
                for (int a = 0; a < valid_na; ++a)
                    for (int b = 0; b < valid_nb; ++b)
                    {
                        const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                        const Tv du = fast_diag_grad<Tv>(vt, theta);
                        const int64 i = dst_offset + (int64)a * dst_num_b + b;
                        res += math_conj(lp[i] * du) * rp[i];
                    }
            }
            else if constexpr (UsesAExcitation && UsesBExcitation)
            {
#pragma omp for collapse(2) schedule(static) nowait
                for (int a = 0; a < valid_na; ++a)
                    for (int b = 0; b < valid_nb; ++b)
                    {
                        const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                        const int64 si = src_offset + (int64)src_a[a] * src_num_b + src_b[b];
                        const int64 di = dst_offset + (int64)dst_a[a] * dst_num_b + dst_b[b];
                        grad_update<Tv>(res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
                    }
            }
            else if constexpr (UsesAExcitation)
            {
#pragma omp for collapse(2) schedule(static) nowait
                for (int a = 0; a < valid_na; ++a)
                    for (int b = 0; b < valid_nb; ++b)
                    {
                        const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                        const int64 si = src_offset + (int64)src_a[a] * src_num_b + b;
                        const int64 di = dst_offset + (int64)dst_a[a] * dst_num_b + b;
                        grad_update<Tv>(res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
                    }
            }
            else
            {
#pragma omp for collapse(2) schedule(static) nowait
                for (int a = 0; a < valid_na; ++a)
                    for (int b = 0; b < valid_nb; ++b)
                    {
                        const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                        const int64 si = src_offset + (int64)a * src_num_b + src_b[b];
                        const int64 di = dst_offset + (int64)a * dst_num_b + dst_b[b];
                        grad_update<Tv>(res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
                    }
            }
        }
    }
    return res;
}

template <int Rank, int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE Tv launch_grad_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, const Tv *lp, const Tv *rp)
{
    return grad_contract_otf_impl<Rank, TypeCode>(basis, group, theta, lp, rp);
}

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE Tv dispatch_grad_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, const Tv *lp, const Tv *rp)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    switch (rank)
    {
    case 1:
        return launch_grad_contract_group<1, TypeCode>(basis, group, theta, lp, rp);
    case 2:
        return launch_grad_contract_group<2, TypeCode>(basis, group, theta, lp, rp);
    default:
        return launch_grad_contract_group<0, TypeCode>(basis, group, theta, lp, rp);
    }
}

template <typename Ti, typename Tv>
Tv grad_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, const Tv *lp, const Tv *rp)
{
    const uint8 type = net->excit_types[idx];
    const int64 pos = net->sorted_idxs[idx];

    const SVDGroup_OTF<Ti, Tv> *group_ptr;
    switch (type)
    {
    case 0:
        group_ptr = &net->diag_groups[pos];
        break;
    case 1:
        group_ptr = &net->pure_a_groups[pos];
        break;
    case 2:
        group_ptr = &net->pure_b_groups[pos];
        break;
    case 3:
        group_ptr = &net->mixed_groups[pos];
        break;
    default:
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in grad_svd" << std::endl;
        return {};
    }
    
    const SVDGroup_OTF<Ti, Tv> &group = *group_ptr;

    switch (type)
    {
    case 0:
        return dispatch_grad_contract_group<0>(basis, group, theta, lp, rp);
    case 1:
        return dispatch_grad_contract_group<1>(basis, group, theta, lp, rp);
    case 2:
        return dispatch_grad_contract_group<2>(basis, group, theta, lp, rp);
    case 3:
        return dispatch_grad_contract_group<3>(basis, group, theta, lp, rp);
    default:
        return {};
    }
}
