#pragma once
#include "otf.hpp"
#include "utils.hpp"

template <int Rank, int TypeCode, typename Ti, typename Tv>
static inline void gather_contract_batched_impl(
    const BasisView<Ti> &view,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups,
    const Tv *src_vec, Tv *dst_vec)
{
    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;
    constexpr bool UsesAExcitation = TypeCode == 1 || TypeCode == 3;
    constexpr bool UsesBExcitation = TypeCode == 2 || TypeCode == 3;
    constexpr bool IsDiagonal = TypeCode == 0;

    const int max_b_count = view.max_b_count;
    const int shift = max_b_count * MAX_RANK;
    const BlockDesc<Ti> *blocks = view.blocks;
    const int64 num_blocks = view.num_blocks;
    const int64 *block_map = view.block_map;
    const int64 num_irreps = view.num_irreps;
    const int *a_idx_map = view.a_idx_map;
    const int *b_idx_map = view.b_idx_map;

    std::vector<int> src_b_idxs(BATCH_SIZE * max_b_count);
    std::vector<int> dst_b_idxs(BATCH_SIZE * max_b_count);
    std::vector<Tv> batch_phase(BATCH_SIZE * shift);
    std::vector<int> valid_b_counts(BATCH_SIZE);
    std::vector<int64> src_block_idxs(BATCH_SIZE);

#pragma omp parallel
    {
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
                    if constexpr (!IsDiagonal)
                    {
                        int64 h;
                        if constexpr (TypeCode == 1)
                            h = (dst_block.asym ^ group.asym) * num_irreps + dst_block.bsym;
                        else if constexpr (TypeCode == 2)
                            h = dst_block.asym * num_irreps + (dst_block.bsym ^ group.bsym);
                        else
                            h = (dst_block.asym ^ group.asym) * num_irreps + (dst_block.bsym ^ group.bsym);

                        const int64 src_block_idx = block_map[h];
                        src_block_idxs[batch_idx] = src_block_idx;
                        if (src_block_idx == -1)
                        {
                            valid_b_counts[batch_idx] = 0;
                            continue;
                        }
                    }

                    Tv *pb0 = batch_phase.data() + batch_idx * shift;
                    int *sb_ptr = src_b_idxs.data() + batch_idx * max_b_count;
                    int *db_ptr = dst_b_idxs.data() + batch_idx * max_b_count;

                    int count = 0;
                    for (int i = 0; i < dst_block.num_b; ++i)
                    {
                        const Ti dst_str_b = dst_block.bstrs[i];
                        Ti src_str_b = dst_str_b;
                        int src_b_idx = i;

                        if constexpr (UsesBExcitation)
                        {
                            src_str_b = dst_str_b ^ group.bx;
                            src_b_idx = b_idx_map[src_str_b];
                            if (src_b_idx == -1)
                                continue;
                        }

                        sb_ptr[count] = src_b_idx;
                        db_ptr[count] = i;
                        precompute_phase<Rank, Ti, Tv>(src_str_b, group.unique_zbs, group.num_zb, group.wb, pb0 + count, max_b_count, group.rank);
                        count++;
                    }

                    valid_b_counts[batch_idx] = count;
                }

#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti dst_str_a = dst_block.astrs[a];
                    Tv *da = dst_vec + dst_block.offset + a * dst_block.num_b;
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_b_count = valid_b_counts[batch_idx];
                        if (valid_b_count == 0)
                            continue;

                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        Ti src_str_a = dst_str_a;
                        int src_a_idx = a;

                        if constexpr (UsesAExcitation)
                        {
                            src_str_a = dst_str_a ^ group.ax;
                            src_a_idx = a_idx_map[src_str_a];
                            if (src_a_idx == -1)
                                continue;
                        }

                        Tv pa[MAX_RANK] = {};
                        precompute_phase<Rank, Ti, Tv>(src_str_a, group.unique_zas, group.num_za, group.wa, pa, 1, group.rank);

                        const BlockDesc<Ti> &src_block = IsDiagonal ? dst_block : blocks[src_block_idxs[batch_idx]];
                        const int rank = group.rank;
                        const Tv *pb = batch_phase.data() + batch_idx * shift;
                        const int *si = src_b_idxs.data() + batch_idx * max_b_count;
                        const int *di = dst_b_idxs.data() + batch_idx * max_b_count;
                        const Tv *sa = src_vec + src_block.offset + src_a_idx * src_block.num_b;

#pragma omp simd
                        for (int b = 0; b < valid_b_count; ++b)
                        {
                            const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, max_b_count, rank);
                            hvec_update<Tv>(sa + si[b], da + di[b], vt);
                        }
                    }
                }
            }
        }
    }
}

template <int TypeCode, typename Ti, typename Tv>
static inline void dispatch_chunks_by_rank(
    const BasisView<Ti> &view,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const Tv *src_vec, Tv *dst_vec)
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
            gather_contract_batched_impl<1, TypeCode>(view, chunk_ptr, chunk_size, src_vec, dst_vec);
            break;
        case 2:
            gather_contract_batched_impl<2, TypeCode>(view, chunk_ptr, chunk_size, src_vec, dst_vec);
            break;
        default:
            gather_contract_batched_impl<0, TypeCode>(view, chunk_ptr, chunk_size, src_vec, dst_vec);
            break;
        }

        start = end;
    }
}

template <typename Ti, typename Tv>
void contract_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, const Tv *src_vec, Tv *dst_vec)
{
    const BasisView<Ti> &view = basis->view;

#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
    {
        dst_vec[i] = {};
    }

    dispatch_chunks_by_rank<0>(view, net->diag_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank<1>(view, net->pure_a_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank<2>(view, net->pure_b_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank<3>(view, net->mixed_groups, src_vec, dst_vec);
}

template <typename Ti, typename Tv>
void get_diags_elements(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, Tv *diags)
{
    const SVDGroup_OTF<Ti, Tv> &group = net->diag_groups[0];
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
                precompute_phase<0, Ti, Tv>(astrs[i], zas, num_za, wa0, pa0 + i, max_a_count, rank);
            }

            for (int i = 0; i < b_count; ++i)
            {
                precompute_phase<0, Ti, Tv>(bstrs[i], zbs, num_zb, wb0, pb0 + i, max_b_count, rank);
            }

            const Tv *pa = local_a_phase.data();
            const Tv *pb = local_b_phase.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < a_count; ++a)
            {
                for (int b = 0; b < b_count; ++b)
                {
                    const Tv vt = compute_coeff<0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const int64 i = block.offset + (int64)a * b_count + b;
                    diags[i] += vt;
                }
            }
        }
    }
}
