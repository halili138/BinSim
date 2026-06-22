#pragma once
#include "otf.hpp"
#include "utils.hpp"

template <int Rank, int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void expm_contract_batched_otf_impl(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group, double theta, Tv *matrix, int ld, int num_vecs)
{
    constexpr bool IsDiagonal = TypeCode == 0;
    constexpr bool UsesAExcitation = TypeCode == 1 || TypeCode == 3;
    constexpr bool UsesBExcitation = TypeCode == 2 || TypeCode == 3;

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
    const int64 num_irreps = basis->num_irreps;
    const int max_a_count = (int)basis->max_a_count;
    const int max_b_count = (int)basis->max_b_count;
    const int *a_idx_map = basis->a_idx_map;
    const int *b_idx_map = basis->b_idx_map;

#pragma omp parallel
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
                    const Ti dst_str = dst_block.astrs[i];
                    const Ti src_str = dst_str ^ group.ax;
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
                    const Ti dst_str = dst_block.bstrs[i];
                    const Ti src_str = dst_str ^ group.bx;
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
#pragma omp for collapse(2) schedule(static) nowait
                for (int a = 0; a < valid_na; ++a)
                {
                    for (int b = 0; b < valid_nb; ++b)
                    {
                        const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                        const int64 di = dst_offset + (int64)a * dst_num_b + b;
                        const Tv u = fast_diag_exp<Tv>(vt, theta);
                        expm_batch_update_diag<Tv>(matrix, ld, num_vecs, di, u);
                    }
                }
            }
            else if constexpr (UsesAExcitation && UsesBExcitation)
            {
#pragma omp for collapse(2) schedule(static) nowait
                for (int a = 0; a < valid_na; ++a)
                {
                    for (int b = 0; b < valid_nb; ++b)
                    {
                        const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                        const int64 si = src_offset + (int64)src_a[a] * src_num_b + src_b[b];
                        const int64 di = dst_offset + (int64)dst_a[a] * dst_num_b + dst_b[b];
                        expm_batch_update_matrix<Tv>(matrix, ld, num_vecs, si, di, vt, cd, co);
                    }
                }
            }
            else if constexpr (UsesAExcitation)
            {
#pragma omp for collapse(2) schedule(static) nowait
                for (int a = 0; a < valid_na; ++a)
                {
                    for (int b = 0; b < valid_nb; ++b)
                    {
                        const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                        const int64 si = src_offset + (int64)src_a[a] * src_num_b + b;
                        const int64 di = dst_offset + (int64)dst_a[a] * dst_num_b + b;
                        expm_batch_update_matrix<Tv>(matrix, ld, num_vecs, si, di, vt, cd, co);
                    }
                }
            }
            else
            {
#pragma omp for collapse(2) schedule(static) nowait
                for (int a = 0; a < valid_na; ++a)
                {
                    for (int b = 0; b < valid_nb; ++b)
                    {
                        const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                        const int64 si = src_offset + (int64)a * src_num_b + src_b[b];
                        const int64 di = dst_offset + (int64)a * dst_num_b + dst_b[b];
                        expm_batch_update_matrix<Tv>(matrix, ld, num_vecs, si, di, vt, cd, co);
                    }
                }
            }
        }
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void launch_expm_batched_contract_group(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> &group,
    double theta,
    Tv *matrix, int ld, int num_vecs)
{
    expm_contract_batched_otf_impl<Rank, TypeCode>(basis, group, theta, matrix, ld, num_vecs);
}

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_expm_batched_contract_group(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> &group,
    double theta,
    Tv *matrix, int ld, int num_vecs)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    switch (rank)
    {
    case 1:
        launch_expm_batched_contract_group<1, TypeCode>(basis, group, theta, matrix, ld, num_vecs);
        break;
    case 2:
        launch_expm_batched_contract_group<2, TypeCode>(basis, group, theta, matrix, ld, num_vecs);
        break;
    default:
        launch_expm_batched_contract_group<0, TypeCode>(basis, group, theta, matrix, ld, num_vecs);
        break;
    }
}

template <typename Ti, typename Tv>
void expm_svd_batched_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta,
    Tv *matrix, int ld, int num_vecs)
{
    const uint8 type = net->excit_types[idx];
    const int64 pos = net->sorted_idxs[idx];
    const SVDGroup_OTF<Ti, Tv> *group = group_by_type(net, type, pos);
    if (group == nullptr)
    {
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in expm_svd_batched_network_otf" << std::endl;
        return;
    }

    switch (type)
    {
    case 0:
        dispatch_expm_batched_contract_group<0>(basis, *group, theta, matrix, ld, num_vecs);
        break;
    case 1:
        dispatch_expm_batched_contract_group<1>(basis, *group, theta, matrix, ld, num_vecs);
        break;
    case 2:
        dispatch_expm_batched_contract_group<2>(basis, *group, theta, matrix, ld, num_vecs);
        break;
    case 3:
        dispatch_expm_batched_contract_group<3>(basis, *group, theta, matrix, ld, num_vecs);
        break;
    }
}
