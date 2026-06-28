#pragma once
#include "otf.hpp"
#include "utils.hpp"

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void tran_contract_diag_batched_impl(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> *groups,
    int64 num_groups, const Tv *lp, const Tv *rp, Tv *trans)
{
    if (num_groups == 0)
        return;

    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;

    const int max_a_count = (int)basis->max_a_count;
    const int max_b_count = (int)basis->max_b_count;
    const int shift = max_b_count * MAX_RANK;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;

    std::vector<Tv> phase_b(BATCH_SIZE * shift);

#pragma omp parallel
    {
        std::vector<Tv> thread_trans(num_groups, Tv{});
        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min<int64>(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    Tv *pb0 = phase_b.data() + batch_idx * shift;
                    for (int i = 0; i < block.num_b; ++i)
                    {
                        precompute_phase<Rank, Ti, Tv>(block.bstrs[i], group.unique_zbs, group.num_zb, group.wb, pb0 + i, max_b_count, group.rank);
                    }
                }

#pragma omp for schedule(dynamic)
                for (int a = 0; a < block.num_a; ++a)
                {
                    const Ti str_a = block.astrs[a];
                    const Tv *la = lp + block.offset + a * block.num_b;
                    const Tv *ra = rp + block.offset + a * block.num_b;
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int64 g = batch_start + batch_idx;
                        const SVDGroup_OTF<Ti, Tv> &group = groups[g];
                        const int rank = group.rank;
                        const Tv *pb = phase_b.data() + batch_idx * shift;

                        Tv pa[MAX_RANK] = {};
                        precompute_phase<Rank, Ti, Tv>(str_a, group.unique_zas, group.num_za, group.wa, pa, 1, group.rank);

                        Tv local_res = {};
#pragma omp simd reduction(+ : local_res)
                        for (int b = 0; b < block.num_b; ++b)
                        {
                            const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, max_b_count, rank);
                            local_res += math_conj(la[b] * vt) * ra[b];
                        }
                        thread_trans[g] += local_res;
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int64 g = 0; g < num_groups; ++g)
                trans[groups[g].original_idx] += thread_trans[g];
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void tran_contract_pure_a_batched_impl(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> *groups,
    int64 num_groups, const Tv *lp, const Tv *rp, Tv *trans)
{
    if (num_groups == 0)
        return;

    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;

    const int max_a_count = (int)basis->max_a_count;
    const int max_b_count = (int)basis->max_b_count;
    const int shift = max_b_count * MAX_RANK;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 num_irreps = basis->num_irreps;
    const int *a_idx_map = basis->a_idx_map;

    std::vector<Tv> phase_b(BATCH_SIZE * shift);
    std::vector<int> src_block_idxs(BATCH_SIZE);

#pragma omp parallel
    {
        std::vector<Tv> thread_trans(num_groups, Tv{});
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
                    const int64 h = (dst_block.asym ^ group.asym) * num_irreps + dst_block.bsym;
                    const int64 src_block_idx = block_map[h];
                    src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                    {
                        continue;
                    }

                    Tv *pb0 = phase_b.data() + batch_idx * shift;
                    for (int i = 0; i < dst_block.num_b; ++i)
                    {
                        precompute_phase<Rank, Ti, Tv>(dst_block.bstrs[i], group.unique_zbs, group.num_zb, group.wb, pb0 + i, max_b_count, group.rank);
                    }
                }

#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti dst_str_a = dst_block.astrs[a];
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int src_block_idx = src_block_idxs[batch_idx];

                        if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                            continue;

                        const int64 g = batch_start + batch_idx;
                        const SVDGroup_OTF<Ti, Tv> &group = groups[g];

                        const Ti src_str_a = dst_str_a ^ group.ax;
                        const int src_a_idx = a_idx_map[src_str_a];

                        if (src_a_idx == -1 || (src_block_idx == dst_block_idx && src_a_idx < a))
                            continue;

                        Tv pa[MAX_RANK] = {};
                        precompute_phase<Rank, Ti, Tv>(src_str_a, group.unique_zas, group.num_za, group.wa, pa, 1, group.rank);

                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const int rank = group.rank;
                        const Tv *pb = phase_b.data() + batch_idx * shift;
                        const int64 sa = src_block.offset + src_a_idx * src_block.num_b;
                        const int64 da = dst_block.offset + a * dst_block.num_b;

                        Tv local_res = {};
#pragma omp simd reduction(+ : local_res)
                        for (int b = 0; b < dst_block.num_b; ++b)
                        {
                            const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, max_b_count, rank);
                            const int64 si = sa + b;
                            const int64 di = da + b;
                            tran_update<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt);
                        }
                        thread_trans[g] += local_res;
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int64 g = 0; g < num_groups; ++g)
                trans[groups[g].original_idx] += thread_trans[g];
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void tran_contract_pure_b_batched_impl(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> *groups,
    int64 num_groups, const Tv *lp, const Tv *rp, Tv *trans)
{
    if (num_groups == 0)
        return;

    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;

    const int max_a_count = (int)basis->max_a_count;
    const int max_b_count = (int)basis->max_b_count;
    const int shift = max_b_count * MAX_RANK;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 num_irreps = basis->num_irreps;
    const int *b_idx_map = basis->b_idx_map;

    std::vector<int> src_b_idxs(BATCH_SIZE * max_b_count);
    std::vector<int> dst_b_idxs(BATCH_SIZE * max_b_count);
    std::vector<Tv> phase_b(BATCH_SIZE * shift);
    std::vector<int> valid_b_counts(BATCH_SIZE);
    std::vector<int> src_block_idxs(BATCH_SIZE);

#pragma omp parallel
    {
        std::vector<Tv> thread_trans(num_groups, Tv{});
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
                    const int64 h = dst_block.asym * num_irreps + (dst_block.bsym ^ group.bsym);
                    const int64 src_block_idx = block_map[h];
                    src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                    {
                        valid_b_counts[batch_idx] = 0;
                        continue;
                    }

                    const bool is_same_block = (src_block_idx == dst_block_idx);

                    Tv *pb0 = phase_b.data() + batch_idx * shift;
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

                        if (is_same_block && src_idx < i)
                            continue;

                        sb_ptr[count] = src_idx;
                        db_ptr[count] = i;

                        precompute_phase<Rank, Ti, Tv>(src_str, group.unique_zbs, group.num_zb, group.wb, pb0 + count, max_b_count, group.rank);

                        count++;
                    }

                    valid_b_counts[batch_idx] = count;
                }

#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti str_a = dst_block.astrs[a];
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_nb = valid_b_counts[batch_idx];

                        if (valid_nb == 0)
                            continue;

                        const int64 g = batch_start + batch_idx;
                        const SVDGroup_OTF<Ti, Tv> &group = groups[g];

                        Tv pa[MAX_RANK] = {};
                        precompute_phase<Rank, Ti, Tv>(str_a, group.unique_zas, group.num_za, group.wa, pa, 1, group.rank);

                        const int src_block_idx = src_block_idxs[batch_idx];
                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const int rank = group.rank;
                        const Tv *pb = phase_b.data() + batch_idx * shift;
                        const int *sb_ptr = src_b_idxs.data() + batch_idx * max_b_count;
                        const int *db_ptr = dst_b_idxs.data() + batch_idx * max_b_count;
                        const int64 sa = src_block.offset + a * src_block.num_b;
                        const int64 da = dst_block.offset + a * dst_block.num_b;

                        Tv local_res = {};
#pragma omp simd reduction(+ : local_res)
                        for (int b = 0; b < valid_nb; ++b)
                        {
                            const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, max_b_count, rank);
                            const int64 si = sa + sb_ptr[b];
                            const int64 di = da + db_ptr[b];
                            tran_update<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt);
                        }
                        thread_trans[g] += local_res;
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int64 g = 0; g < num_groups; ++g)
                trans[groups[g].original_idx] += thread_trans[g];
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void tran_contract_mixed_batched_impl(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> *groups,
    int64 num_groups, const Tv *lp, const Tv *rp, Tv *trans)
{
    if (num_groups == 0)
        return;

    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;

    const int max_a_count = (int)basis->max_a_count;
    const int max_b_count = (int)basis->max_b_count;
    const int shift = max_b_count * MAX_RANK;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 num_irreps = basis->num_irreps;
    const int *a_idx_map = basis->a_idx_map;
    const int *b_idx_map = basis->b_idx_map;

    std::vector<int> src_b_idxs(BATCH_SIZE * max_b_count);
    std::vector<int> dst_b_idxs(BATCH_SIZE * max_b_count);
    std::vector<Tv> phase_b(BATCH_SIZE * shift);
    std::vector<int> valid_b_counts(BATCH_SIZE);
    std::vector<int> src_block_idxs(BATCH_SIZE);

#pragma omp parallel
    {
        std::vector<Tv> thread_trans(num_groups, Tv{});
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
                    const int64 h = (dst_block.asym ^ group.asym) * num_irreps + (dst_block.bsym ^ group.bsym);
                    const int64 src_block_idx = block_map[h];
                    src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                    {
                        valid_b_counts[batch_idx] = 0;
                        continue;
                    }

                    Tv *pb0 = phase_b.data() + batch_idx * shift;
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

                        sb_ptr[count] = src_idx;
                        db_ptr[count] = i;

                        precompute_phase<Rank, Ti, Tv>(src_str, group.unique_zbs, group.num_zb, group.wb, pb0 + count, max_b_count, group.rank);

                        count++;
                    }

                    valid_b_counts[batch_idx] = count;
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

                        const int src_block_idx = src_block_idxs[batch_idx];
                        if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                            continue;

                        const int64 g = batch_start + batch_idx;
                        const SVDGroup_OTF<Ti, Tv> &group = groups[g];
                        const Ti src_str_a = dst_str_a ^ group.ax;
                        const int src_a_idx = a_idx_map[src_str_a];

                        if (src_a_idx == -1 || (src_block_idx == dst_block_idx && src_a_idx < a))
                            continue;

                        Tv pa[MAX_RANK] = {};
                        precompute_phase<Rank, Ti, Tv>(src_str_a, group.unique_zas, group.num_za, group.wa, pa, 1, group.rank);

                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const int rank = group.rank;
                        const Tv *pb = phase_b.data() + batch_idx * shift;
                        const int *sb_ptr = src_b_idxs.data() + batch_idx * max_b_count;
                        const int *db_ptr = dst_b_idxs.data() + batch_idx * max_b_count;
                        const int64 sa = src_block.offset + src_a_idx * src_block.num_b;
                        const int64 da = dst_block.offset + a * dst_block.num_b;

                        Tv local_res = {};
#pragma omp simd reduction(+ : local_res)
                        for (int b = 0; b < valid_nb; ++b)
                        {
                            const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, max_b_count, rank);
                            const int64 si = sa + sb_ptr[b];
                            const int64 di = da + db_ptr[b];
                            tran_update<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt);
                        }
                        thread_trans[g] += local_res;
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int64 g = 0; g < num_groups; ++g)
                trans[groups[g].original_idx] += thread_trans[g];
        }
    }
}

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_tran_chunks_by_rank(
    const BasisManager<Ti> *basis, const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const Tv *lp, const Tv *rp, Tv *trans)
{
    const int64 total_ngs = groups.size();
    if (total_ngs == 0)
        return;

    const SVDGroup_OTF<Ti, Tv> *groups_ptr = groups.data();

    int64 start = 0;
    while (start < total_ngs)
    {
        const int dispatch_rank = groups_ptr[start].rank;
        const int eff_rank = (dispatch_rank == 1 || dispatch_rank == 2) ? dispatch_rank : 0;

        int64 end = start + 1;
        while (end < total_ngs)
        {
            const int next_rank = groups_ptr[end].rank;
            const int next_eff_rank = (next_rank == 1 || next_rank == 2) ? next_rank : 0;
            if (next_eff_rank != eff_rank)
                break;
            end++;
        }

        const SVDGroup_OTF<Ti, Tv> *chunk_ptr = groups_ptr + start;
        const int64 chunk_size = end - start;

        if constexpr (TypeCode == 0)
        {
            if (eff_rank == 1)
                tran_contract_diag_batched_impl<1>(basis, chunk_ptr, chunk_size, lp, rp, trans);
            else if (eff_rank == 2)
                tran_contract_diag_batched_impl<2>(basis, chunk_ptr, chunk_size, lp, rp, trans);
            else
                tran_contract_diag_batched_impl<0>(basis, chunk_ptr, chunk_size, lp, rp, trans);
        }
        else if constexpr (TypeCode == 1)
        {
            if (eff_rank == 1)
                tran_contract_pure_a_batched_impl<1>(basis, chunk_ptr, chunk_size, lp, rp, trans);
            else if (eff_rank == 2)
                tran_contract_pure_a_batched_impl<2>(basis, chunk_ptr, chunk_size, lp, rp, trans);
            else
                tran_contract_pure_a_batched_impl<0>(basis, chunk_ptr, chunk_size, lp, rp, trans);
        }
        else if constexpr (TypeCode == 2)
        {
            if (eff_rank == 1)
                tran_contract_pure_b_batched_impl<1>(basis, chunk_ptr, chunk_size, lp, rp, trans);
            else if (eff_rank == 2)
                tran_contract_pure_b_batched_impl<2>(basis, chunk_ptr, chunk_size, lp, rp, trans);
            else
                tran_contract_pure_b_batched_impl<0>(basis, chunk_ptr, chunk_size, lp, rp, trans);
        }
        else if constexpr (TypeCode == 3)
        {
            if (eff_rank == 1)
                tran_contract_mixed_batched_impl<1>(basis, chunk_ptr, chunk_size, lp, rp, trans);
            else if (eff_rank == 2)
                tran_contract_mixed_batched_impl<2>(basis, chunk_ptr, chunk_size, lp, rp, trans);
            else
                tran_contract_mixed_batched_impl<0>(basis, chunk_ptr, chunk_size, lp, rp, trans);
        }
        start = end;
    }
}

template <typename Ti, typename Tv>
void tran_pool_network_batched_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net,
    const Tv *lp, const Tv *rp, Tv *trans)
{
    std::fill(trans, trans + net->num_groups, Tv{});

    dispatch_tran_chunks_by_rank<0>(basis, net->diag_groups, lp, rp, trans);
    dispatch_tran_chunks_by_rank<1>(basis, net->pure_a_groups, lp, rp, trans);
    dispatch_tran_chunks_by_rank<2>(basis, net->pure_b_groups, lp, rp, trans);
    dispatch_tran_chunks_by_rank<3>(basis, net->mixed_groups, lp, rp, trans);
}
