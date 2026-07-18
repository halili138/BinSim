#pragma once
#include <vector>
#include <omp.h>
#include "ham/otf.hpp"
#include "core/math.hpp"

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void expm_contract_batched_diag_otf_impl(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group, double theta,
    Tv *matrix, int ld, int num_vecs)
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
    const int max_a_count = (int)basis->max_a_count;
    const int max_b_count = (int)basis->max_b_count;

#pragma omp parallel
    {
        std::vector<Tv> local_a_phase(max_a_count * rank);
        std::vector<Tv> local_b_phase(max_b_count * rank);

        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            const Ti *astrs = block.astrs;
            const Ti *bstrs = block.bstrs;
            const int a_count = block.num_a;
            const int b_count = block.num_b;
            Tv *pa0 = local_a_phase.data();
            Tv *pb0 = local_b_phase.data();

            for (int i = 0; i < a_count; ++i)
            {
                precompute_phase<Rank, Ti, Tv>(astrs[i], zas, num_za, wa0, pa0 + i, max_a_count, rank);
            }

            for (int i = 0; i < b_count; ++i)
            {
                precompute_phase<Rank, Ti, Tv>(bstrs[i], zbs, num_zb, wb0, pb0 + i, max_b_count, rank);
            }

            const Tv *pa = local_a_phase.data();
            const Tv *pb = local_b_phase.data();
            const int64 offset = block.offset;

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < a_count; ++a)
            {
                for (int b = 0; b < b_count; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const Tv u = fast_diag_exp<Tv>(vt, theta);
                    const int64 i = offset + (int64)a * b_count + b;
                    expm_batch_update_diag<Tv>(matrix, ld, num_vecs, i, u);
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void expm_contract_batched_pure_a_otf_impl(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group, double theta,
    Tv *matrix, int ld, int num_vecs)
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
    const int64 num_irreps = basis->num_irreps;
    const int max_a_count = (int)basis->max_a_count;
    const int max_b_count = (int)basis->max_b_count;
    const int *a_idx_map = basis->a_idx_map;

#pragma omp parallel
    {
        std::vector<int> src_a(max_a_count);
        std::vector<int> dst_a(max_a_count);
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int64 h = (dst_block.asym ^ group.asym) * num_irreps + dst_block.bsym;
            const int64 src_block_idx = block_map[h];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);
            Tv *pa0 = phase_a.data();
            Tv *pb0 = phase_b.data();

            int valid_na = 0;
            for (int i = 0; i < dst_block.num_a; ++i)
            {
                const Ti dst_str = dst_block.astrs[i];
                const Ti src_str = dst_str ^ group.ax;
                const int src_idx = a_idx_map[src_str];

                if (src_idx == -1)
                    continue;

                if (is_same_block && src_idx < i)
                    continue;

                src_a[valid_na] = src_idx;
                dst_a[valid_na] = i;

                precompute_phase<Rank, Ti, Tv>(src_str, zas, num_za, wa0, pa0 + valid_na, max_a_count, rank);

                valid_na++;
            }

            if (valid_na == 0)
                continue;

            for (int i = 0; i < dst_block.num_b; ++i)
            {
                precompute_phase<Rank, Ti, Tv>(dst_block.bstrs[i], zbs, num_zb, wb0, pb0 + i, max_b_count, rank);
            }

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();
            const int64 src_offset = src_block.offset;
            const int64 dst_offset = dst_block.offset;
            const int64 src_num_b = src_block.num_b;
            const int64 dst_num_b = dst_block.num_b;

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < dst_num_b; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const int64 si = src_offset + src_a[a] * src_num_b + b;
                    const int64 di = dst_offset + dst_a[a] * dst_num_b + b;
                    expm_batch_update_matrix<Tv>(matrix, ld, num_vecs, si, di, vt, cd, co);
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void expm_contract_batched_pure_b_otf_impl(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group, double theta,
    Tv *matrix, int ld, int num_vecs)
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
    const int64 num_irreps = basis->num_irreps;
    const int max_a_count = (int)basis->max_a_count;
    const int max_b_count = (int)basis->max_b_count;
    const int *b_idx_map = basis->b_idx_map;

#pragma omp parallel
    {
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<int> src_b(max_b_count);
        std::vector<int> dst_b(max_b_count);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int64 h = dst_block.asym * num_irreps + (dst_block.bsym ^ group.bsym);
            const int64 src_block_idx = block_map[h];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);
            Tv *pa0 = phase_a.data();
            Tv *pb0 = phase_b.data();

            int valid_nb = 0;
            for (int i = 0; i < dst_block.num_b; ++i)
            {
                const Ti dst_str = dst_block.bstrs[i];
                const Ti src_str = dst_str ^ group.bx;
                const int src_idx = b_idx_map[src_str];

                if (src_idx == -1)
                    continue;

                if (is_same_block && src_idx < i)
                    continue;

                src_b[valid_nb] = src_idx;
                dst_b[valid_nb] = i;

                precompute_phase<Rank, Ti, Tv>(src_str, zbs, num_zb, wb0, pb0 + valid_nb, max_b_count, rank);

                valid_nb++;
            }

            if (valid_nb == 0)
                continue;

            for (int i = 0; i < dst_block.num_a; ++i)
            {
                precompute_phase<Rank, Ti, Tv>(dst_block.astrs[i], zas, num_za, wa0, pa0 + i, max_a_count, rank);
            }

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();
            const int64 src_offset = src_block.offset;
            const int64 dst_offset = dst_block.offset;
            const int64 src_num_b = src_block.num_b;
            const int64 dst_num_b = dst_block.num_b;

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < dst_block.num_a; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const int64 si = src_offset + a * src_num_b + src_b[b];
                    const int64 di = dst_offset + a * dst_num_b + dst_b[b];
                    expm_batch_update_matrix<Tv>(matrix, ld, num_vecs, si, di, vt, cd, co);
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void expm_contract_batched_mixed_otf_impl(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group, double theta,
    Tv *matrix, int ld, int num_vecs)
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
    const int64 num_irreps = basis->num_irreps;
    const int max_a_count = (int)basis->max_a_count;
    const int max_b_count = (int)basis->max_b_count;
    const int *a_idx_map = basis->a_idx_map;
    const int *b_idx_map = basis->b_idx_map;

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
            const int64 h = (dst_block.asym ^ group.asym) * num_irreps + (dst_block.bsym ^ group.bsym);
            const int64 src_block_idx = block_map[h];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);
            Tv *pa0 = phase_a.data();
            Tv *pb0 = phase_b.data();

            int valid_na = 0;
            for (int i = 0; i < dst_block.num_a; ++i)
            {
                const Ti dst_str = dst_block.astrs[i];
                const Ti src_str = dst_str ^ group.ax;
                const int src_idx = a_idx_map[src_str];

                if (src_idx == -1)
                    continue;

                if (is_same_block && src_idx < i)
                    continue;

                src_a[valid_na] = src_idx;
                dst_a[valid_na] = i;

                precompute_phase<Rank, Ti, Tv>(src_str, zas, num_za, wa0, pa0 + valid_na, max_a_count, rank);

                valid_na++;
            }

            if (valid_na == 0)
                continue;

            int valid_nb = 0;
            for (int i = 0; i < dst_block.num_b; ++i)
            {
                const Ti dst_str = dst_block.bstrs[i];
                const Ti src_str = dst_str ^ group.bx;
                const int src_idx = b_idx_map[src_str];

                if (src_idx == -1)
                    continue;

                src_b[valid_nb] = src_idx;
                dst_b[valid_nb] = i;

                precompute_phase<Rank, Ti, Tv>(src_str, zbs, num_zb, wb0, pb0 + valid_nb, max_b_count, rank);

                valid_nb++;
            }

            if (valid_nb == 0)
                continue;

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();
            const int64 src_offset = src_block.offset;
            const int64 dst_offset = dst_block.offset;
            const int64 src_num_b = src_block.num_b;
            const int64 dst_num_b = dst_block.num_b;

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const int64 si = src_offset + src_a[a] * src_num_b + src_b[b];
                    const int64 di = dst_offset + dst_a[a] * dst_num_b + dst_b[b];
                    expm_batch_update_matrix<Tv>(matrix, ld, num_vecs, si, di, vt, cd, co);
                }
            }
        }
    }
}

template <typename Ti, typename Tv>
void expm_svd_batched_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta,
    Tv *matrix, int ld, int num_vecs)
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
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in expm_svd" << std::endl;
        return;
    }

    const SVDGroup_OTF<Ti, Tv> &group = *group_ptr;
    const int rank = group.rank;

    switch (type)
    {
    case 0:
        if (rank == 1)
            expm_contract_batched_diag_otf_impl<1>(basis, group, theta, matrix, ld, num_vecs);
        else if (rank == 2)
            expm_contract_batched_diag_otf_impl<2>(basis, group, theta, matrix, ld, num_vecs);
        else
            expm_contract_batched_diag_otf_impl<0>(basis, group, theta, matrix, ld, num_vecs);
        break;
    case 1:
        if (rank == 1)
            expm_contract_batched_pure_a_otf_impl<1>(basis, group, theta, matrix, ld, num_vecs);
        else if (rank == 2)
            expm_contract_batched_pure_a_otf_impl<2>(basis, group, theta, matrix, ld, num_vecs);
        else
            expm_contract_batched_pure_a_otf_impl<0>(basis, group, theta, matrix, ld, num_vecs);
        break;
    case 2:
        if (rank == 1)
            expm_contract_batched_pure_b_otf_impl<1>(basis, group, theta, matrix, ld, num_vecs);
        else if (rank == 2)
            expm_contract_batched_pure_b_otf_impl<2>(basis, group, theta, matrix, ld, num_vecs);
        else
            expm_contract_batched_pure_b_otf_impl<0>(basis, group, theta, matrix, ld, num_vecs);
        break;
    case 3:
        if (rank == 1)
            expm_contract_batched_mixed_otf_impl<1>(basis, group, theta, matrix, ld, num_vecs);
        else if (rank == 2)
            expm_contract_batched_mixed_otf_impl<2>(basis, group, theta, matrix, ld, num_vecs);
        else
            expm_contract_batched_mixed_otf_impl<0>(basis, group, theta, matrix, ld, num_vecs);
        break;
    }
}
