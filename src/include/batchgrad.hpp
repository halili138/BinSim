#pragma once
#include "otf.hpp"
#include "utils.hpp"

template <int Rank, int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void grad_contract_batched_impl(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> *groups,
    int64 num_groups, const double *thetas, const Tv *lp, const Tv *rp, Tv *grads)
{
    if (num_groups == 0)
        return;

    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;
    constexpr bool IsDiagonal = TypeCode == 0;
    constexpr bool UsesAExcitation = TypeCode == 1 || TypeCode == 3;
    constexpr bool UsesBExcitation = TypeCode == 2 || TypeCode == 3;

    const int max_b_count = (int)basis->max_b_count;
    const int shift = max_b_count * MAX_RANK;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 num_irreps = basis->num_irreps;
    const int *a_idx_map = basis->a_idx_map;
    const int *b_idx_map = basis->b_idx_map;

    std::vector<int> src_b_idxs(UsesBExcitation ? BATCH_SIZE * max_b_count : 1);
    std::vector<int> dst_b_idxs(UsesBExcitation ? BATCH_SIZE * max_b_count : 1);
    std::vector<Tv> phase_b(BATCH_SIZE * shift);
    std::vector<int> valid_b_counts(BATCH_SIZE);
    std::vector<int> src_block_idxs(BATCH_SIZE);

#pragma omp parallel
    {
        std::vector<Tv> thread_grads(num_groups, Tv{});
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min<int64>(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    int src_block_idx = dst_block_idx;
                    if constexpr (!IsDiagonal)
                    {
                        int64 h;
                        if constexpr (TypeCode == 1)
                            h = (dst_block.asym ^ group.asym) * num_irreps + dst_block.bsym;
                        else if constexpr (TypeCode == 2)
                            h = dst_block.asym * num_irreps + (dst_block.bsym ^ group.bsym);
                        else
                            h = (dst_block.asym ^ group.asym) * num_irreps + (dst_block.bsym ^ group.bsym);
                        src_block_idx = (int)block_map[h];
                    }
                    src_block_idxs[batch_idx] = src_block_idx;

                    if constexpr (!IsDiagonal)
                    {
                        if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                        {
                            valid_b_counts[batch_idx] = 0;
                            continue;
                        }
                    }

                    Tv *pb0 = phase_b.data() + batch_idx * shift;
                    if constexpr (UsesBExcitation)
                    {
                        const bool is_same_block = (src_block_idx == dst_block_idx);
                        int *sb_ptr = src_b_idxs.data() + batch_idx * max_b_count;
                        int *db_ptr = dst_b_idxs.data() + batch_idx * max_b_count;
                        int count = 0;
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
                            sb_ptr[count] = src_idx;
                            db_ptr[count] = i;
                            precompute_phase<Rank, Ti, Tv>(src_str, group.unique_zbs, group.num_zb, group.wb, pb0 + count, max_b_count, group.rank);
                            count++;
                        }
                        valid_b_counts[batch_idx] = count;
                    }
                    else
                    {
                        for (int i = 0; i < dst_block.num_b; ++i)
                            precompute_phase<Rank, Ti, Tv>(dst_block.bstrs[i], group.unique_zbs, group.num_zb, group.wb, pb0 + i, max_b_count, group.rank);
                        valid_b_counts[batch_idx] = dst_block.num_b;
                    }
                }

#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti dst_str_a = dst_block.astrs[a];
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_nb = valid_b_counts[batch_idx];
                        if (valid_nb == 0)
                            continue;

                        const int64 g = batch_start + batch_idx;
                        const SVDGroup_OTF<Ti, Tv> &group = groups[g];
                        const int src_block_idx = src_block_idxs[batch_idx];
                        const BlockDesc<Ti> &src_block = IsDiagonal ? dst_block : blocks[src_block_idx];
                        Ti src_str_a = dst_str_a;
                        int src_a_idx = a;

                        if constexpr (UsesAExcitation)
                        {
                            src_str_a = dst_str_a ^ group.ax;
                            src_a_idx = a_idx_map[src_str_a];
                            if (src_a_idx == -1 || (src_block_idx == dst_block_idx && src_a_idx < a))
                                continue;
                        }

                        Tv pa[MAX_RANK] = {};
                        precompute_phase<Rank, Ti, Tv>(src_str_a, group.unique_zas, group.num_za, group.wa, pa, 1, group.rank);

                        const int rank = group.rank;
                        const Tv *pb = phase_b.data() + batch_idx * shift;
                        const int64 sa = src_block.offset + (int64)src_a_idx * src_block.num_b;
                        const int64 da = dst_block.offset + (int64)a * dst_block.num_b;
                        const double theta = thetas[groups[g].original_idx];

                        Tv local_res = {};
                        if constexpr (IsDiagonal)
                        {
                            const Tv *la = lp + da;
                            const Tv *ra = rp + da;
#pragma omp simd reduction(+ : local_res)
                            for (int b = 0; b < valid_nb; ++b)
                            {
                                const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, max_b_count, rank);
                                const Tv du = fast_diag_grad<Tv>(vt, theta);
                                local_res += math_conj(la[b] * du) * ra[b];
                            }
                        }
                        else
                        {
                            const double cd = -std::sin(theta);
                            const double co = std::cos(theta);
                            if constexpr (UsesBExcitation)
                            {
                                const int *src_b_ptr = src_b_idxs.data() + batch_idx * max_b_count;
                                const int *dst_b_ptr = dst_b_idxs.data() + batch_idx * max_b_count;
#pragma omp simd reduction(+ : local_res)
                                for (int b = 0; b < valid_nb; ++b)
                                {
                                    const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, max_b_count, rank);
                                    const int64 si = sa + src_b_ptr[b];
                                    const int64 di = da + dst_b_ptr[b];
                                    grad_update<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
                                }
                            }
                            else
                            {
#pragma omp simd reduction(+ : local_res)
                                for (int b = 0; b < valid_nb; ++b)
                                {
                                    const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, max_b_count, rank);
                                    const int64 si = sa + b;
                                    const int64 di = da + b;
                                    grad_update<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
                                }
                            }
                        }
                        thread_grads[g] += local_res;
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int64 g = 0; g < num_groups; ++g)
                grads[groups[g].original_idx] += thread_grads[g];
        }
    }
}

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_grad_chunks_by_rank(
    const BasisManager<Ti> *basis, const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const double *thetas,
    const Tv *lp, const Tv *rp, Tv *grads)
{
    const int64 total_ngs = groups.size();
    if (total_ngs == 0)
        return;

    const SVDGroup_OTF<Ti, Tv> *groups_ptr = groups.data();

    int64 start = 0;
    while (start < total_ngs)
    {
        const int dispatch_rank = normalized_dispatch_rank(groups, start);
        const int64 end = next_rank_chunk_end(groups, start);
        const SVDGroup_OTF<Ti, Tv> *chunk_ptr = groups_ptr + start;
        const int64 chunk_size = end - start;

        switch (dispatch_rank)
        {
        case 1:
            grad_contract_batched_impl<1, TypeCode>(basis, chunk_ptr, chunk_size, thetas, lp, rp, grads);
            break;
        case 2:
            grad_contract_batched_impl<2, TypeCode>(basis, chunk_ptr, chunk_size, thetas, lp, rp, grads);
            break;
        default:
            grad_contract_batched_impl<0, TypeCode>(basis, chunk_ptr, chunk_size, thetas, lp, rp, grads);
            break;
        }

        start = end;
    }
}

template <typename Ti, typename Tv>
void grad_pool_network_batched_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net,
    const double *thetas, const Tv *lp, const Tv *rp, Tv *grads)
{
    std::fill(grads, grads + net->num_groups, Tv{});

    dispatch_grad_chunks_by_rank<0>(basis, net->diag_groups, thetas, lp, rp, grads);
    dispatch_grad_chunks_by_rank<1>(basis, net->pure_a_groups, thetas, lp, rp, grads);
    dispatch_grad_chunks_by_rank<2>(basis, net->pure_b_groups, thetas, lp, rp, grads);
    dispatch_grad_chunks_by_rank<3>(basis, net->mixed_groups, thetas, lp, rp, grads);
}
