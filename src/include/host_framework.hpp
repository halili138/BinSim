#pragma once
#include "otf.hpp"
#include "utils.hpp"

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
static FORCE_INLINE typename Op::Result otf_contract_single_group_impl(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> &group,
    Op op)
{
    constexpr bool IsDiagonal = TypeCode == 0;
    constexpr bool UsesAExcitation = TypeCode == 1 || TypeCode == 3;
    constexpr bool UsesBExcitation = TypeCode == 2 || TypeCode == 3;

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

    typename Op::Result total = {};

#pragma omp parallel
    {
        typename Op::Result local = {};
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
                    for (int b = 0; b < valid_nb; ++b)
                    {
                        const Tv vt = compute_coeff<Rank, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                        const int64 di = dst_offset + (int64)a * dst_num_b + b;
                        op.diag(local, vt, di);
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
                        op.offdiag(local, vt, si, di);
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
                        op.offdiag(local, vt, si, di);
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
                        op.offdiag(local, vt, si, di);
                    }
            }
        }

        if constexpr (Op::Accumulates)
        {
#pragma omp critical
            total += local;
        }
    }

    return total;
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
static FORCE_INLINE void otf_contract_batched_groups_impl(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> *groups,
    int64 num_groups,
    const Op &op)
{
    if (num_groups == 0)
        return;

    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;
    constexpr bool IsDiagonal = TypeCode == 0;
    constexpr bool UsesAExcitation = TypeCode == 1 || TypeCode == 3;
    constexpr bool UsesBExcitation = TypeCode == 2 || TypeCode == 3;

    const BasisView<Ti> &view = basis->view;
    const int max_b_count = view.max_b_count;
    const int shift = max_b_count * MAX_RANK;
    const BlockDesc<Ti> *blocks = view.blocks;
    const int64 num_blocks = view.num_blocks;
    const int64 *block_map = view.block_map;
    const int64 num_irreps = view.num_irreps;
    const int *a_idx_map = view.a_idx_map;
    const int *b_idx_map = view.b_idx_map;

    std::vector<int> src_b_idxs(UsesBExcitation ? BATCH_SIZE * max_b_count : 1);
    std::vector<int> dst_b_idxs(UsesBExcitation ? BATCH_SIZE * max_b_count : 1);
    std::vector<Tv> phase_b(BATCH_SIZE * shift);
    std::vector<int> valid_b_counts(BATCH_SIZE);
    std::vector<int64> src_block_idxs(BATCH_SIZE);

#pragma omp parallel
    {
        auto thread_state = op.make_thread_state(num_groups);

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
                    }
                    src_block_idxs[batch_idx] = src_block_idx;

                    if constexpr (!IsDiagonal)
                    {
                        if (src_block_idx == -1 || (Op::SkipLowerBlocks && src_block_idx < dst_block_idx))
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
                                if (Op::SkipSameBlockBReverseForPureB && is_same_block && src_idx < i)
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
                        const int64 src_block_idx = src_block_idxs[batch_idx];
                        const BlockDesc<Ti> &src_block = IsDiagonal ? dst_block : blocks[src_block_idx];
                        Ti src_str_a = dst_str_a;
                        int src_a_idx = a;

                        if constexpr (UsesAExcitation)
                        {
                            src_str_a = dst_str_a ^ group.ax;
                            src_a_idx = a_idx_map[src_str_a];
                            if (src_a_idx == -1 || (Op::SkipSameBlockAReverse && src_block_idx == dst_block_idx && src_a_idx < a))
                                continue;
                        }

                        Tv pa[MAX_RANK] = {};
                        precompute_phase<Rank, Ti, Tv>(src_str_a, group.unique_zas, group.num_za, group.wa, pa, 1, group.rank);

                        const int rank = group.rank;
                        const Tv *pb = phase_b.data() + batch_idx * shift;
                        const int64 sa = src_block.offset + (int64)src_a_idx * src_block.num_b;
                        const int64 da = dst_block.offset + (int64)a * dst_block.num_b;

                        if constexpr (IsDiagonal)
                        {
                            Tv local_res = {};
#pragma omp simd reduction(+ : local_res)
                            for (int b = 0; b < valid_nb; ++b)
                            {
                                const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, max_b_count, rank);
                                op.diag(thread_state, g, group, local_res, vt, da + b);
                            }
                            op.commit_local(thread_state, g, local_res);
                        }
                        else if constexpr (UsesBExcitation)
                        {
                            const int *src_b_ptr = src_b_idxs.data() + batch_idx * max_b_count;
                            const int *dst_b_ptr = dst_b_idxs.data() + batch_idx * max_b_count;
                            Tv local_res = {};
#pragma omp simd reduction(+ : local_res)
                            for (int b = 0; b < valid_nb; ++b)
                            {
                                const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, max_b_count, rank);
                                const int64 si = sa + src_b_ptr[b];
                                const int64 di = da + dst_b_ptr[b];
                                op.offdiag(thread_state, g, group, local_res, vt, si, di);
                            }
                            op.commit_local(thread_state, g, local_res);
                        }
                        else
                        {
                            Tv local_res = {};
#pragma omp simd reduction(+ : local_res)
                            for (int b = 0; b < valid_nb; ++b)
                            {
                                const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, max_b_count, rank);
                                const int64 si = sa + b;
                                const int64 di = da + b;
                                op.offdiag(thread_state, g, group, local_res, vt, si, di);
                            }
                            op.commit_local(thread_state, g, local_res);
                        }
                    }
                }
            }
        }

        op.finish_thread(thread_state, groups, num_groups);
    }
}
