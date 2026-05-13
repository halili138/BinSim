#pragma once
#include "otf.hpp"

template <typename Tv>
struct SharedBatchBuffer
{
    std::vector<int> src_b_idxs;
    std::vector<int> dst_b_idxs;
    std::vector<Tv> batch_phase;

    std::vector<int> valid_b_counts;
    std::vector<int> src_block_idxs;

    int max_b_count;
    int max_rank;

    SharedBatchBuffer(int64 batch_size, int mb, int mr)
        : max_b_count(mb), max_rank(mr)
    {
        src_b_idxs.resize(batch_size * mb);
        dst_b_idxs.resize(batch_size * mb);
        batch_phase.resize(batch_size * mb * mr);
        valid_b_counts.resize(batch_size);
        src_block_idxs.resize(batch_size);
    }

    inline int *ptr_src_b(int64 batch_idx) { return src_b_idxs.data() + batch_idx * max_b_count; }
    inline int *ptr_dst_b(int64 batch_idx) { return dst_b_idxs.data() + batch_idx * max_b_count; }
    inline Tv *ptr_phase(int64 batch_idx) { return batch_phase.data() + batch_idx * max_b_count * max_rank; }
};

template <int Rank, typename Ti, typename Tv>
static inline void gather_contract_diag_batched_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> *__restrict__ groups,
    const int64 num_groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;

    int max_a_count = 0, max_b_count = 0, max_rank = 0;

    get_upper<Rank, Ti, Tv>(
        blocks, groups, num_blocks, num_groups, max_a_count, max_b_count, max_rank);

    std::vector<Tv> phase_b(BATCH_SIZE * max_b_count * max_rank);

#pragma omp parallel
    {
        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min(BATCH_SIZE, num_groups - batch_start);
#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    Tv *pb0 = phase_b.data() + batch_idx * max_b_count * max_rank;
                    compute_phases<Rank, 1, Ti, Tv>(block.bstrs, block.num_b,
                                                    group.unique_zbs, group.num_zb, group.wb,
                                                    pb0, max_b_count, group.rank);
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < block.num_a; ++a)
                {
                    const Ti str_a = block.astrs[a];
                    const Tv *src = src_vec + block.offset + a * block.num_b;
                    Tv *dst = dst_vec + block.offset + a * block.num_b;
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const Tv *pb0 = phase_b.data() + batch_idx * max_b_count * max_rank;
                        Tv pa0 = {}, pa1 = {}, pan[64] = {};
                        compute_a_phase<Rank, Ti, Tv>(str_a,
                                                      group.unique_zas, group.num_za,
                                                      group.wa, group.rank,
                                                      pa0, pa1, pan);
                        update_dst<Rank, 1, 0, Tv>(block.num_b,
                                                   group.rank, pa0, pa1, pan,
                                                   pb0, max_b_count,
                                                   nullptr, nullptr, src, dst);
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline void gather_contract_pure_a_batched_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> *__restrict__ groups,
    const int64 num_groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int max_a_count = 0, max_b_count = 0, max_rank = 0;

    get_upper<Rank, Ti, Tv>(
        blocks, groups, num_blocks, num_groups, max_a_count, max_b_count, max_rank);

    std::vector<Tv> phase_b(BATCH_SIZE * max_b_count * max_rank);
    std::vector<int> src_block_idxs(BATCH_SIZE);

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min(BATCH_SIZE, num_groups - batch_start);
#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    const int64 axsym = get_string_sym(group.ax, orbsym);
                    const int64 bid = (dst_block.asym ^ axsym) * num_irreps + dst_block.bsym;
                    const int64 src_block_idx = block_map[bid];

                    src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1)
                        continue;

                    Tv *pb0 = phase_b.data() + batch_idx * max_b_count * max_rank;
                    compute_phases<Rank, 1, Ti, Tv>(dst_block.bstrs, dst_block.num_b,
                                                    group.unique_zbs, group.num_zb, group.wb,
                                                    pb0, max_b_count, group.rank);
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti dst_str_a = dst_block.astrs[a];
                    Tv *dst = dst_vec + dst_block.offset + a * dst_block.num_b;
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int src_block_idx = src_block_idxs[batch_idx];

                        if (src_block_idx == -1)
                            continue;

                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const Ti src_str_a = dst_str_a ^ group.ax;
                        const int src_a_idx = idx_map.a_idx_map[src_str_a];

                        if (src_a_idx == -1)
                            continue;

                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const Tv *pb0 = phase_b.data() + batch_idx * max_b_count * max_rank;
                        const Tv *src = src_vec + src_block.offset + src_a_idx * src_block.num_b;
                        Tv pa0 = {}, pa1 = {}, pan[64] = {};
                        compute_a_phase<Rank, Ti, Tv>(src_str_a,
                                                      group.unique_zas, group.num_za,
                                                      group.wa, group.rank,
                                                      pa0, pa1, pan);
                        update_dst<Rank, 1, 0, Tv>(dst_block.num_b,
                                                   group.rank, pa0, pa1, pan,
                                                   pb0, max_b_count,
                                                   nullptr, nullptr, src, dst);
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline void gather_contract_pure_b_batched_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> *__restrict__ groups,
    const int64 num_groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int max_a_count = 0, max_b_count = 0, max_rank = 0;

    get_upper<Rank, Ti, Tv>(
        blocks, groups, num_blocks, num_groups, max_a_count, max_b_count, max_rank);

    SharedBatchBuffer<Tv> batch_buf(BATCH_SIZE, max_b_count, max_rank);

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min(BATCH_SIZE, num_groups - batch_start);
#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    const int64 bxsym = get_string_sym(group.bx, orbsym);
                    const int64 bid = dst_block.asym * num_irreps + (dst_block.bsym ^ bxsym);
                    const int64 src_block_idx = block_map[bid];

                    batch_buf.src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1)
                    {
                        batch_buf.valid_b_counts[batch_idx] = 0;
                        continue;
                    }

                    batch_buf.valid_b_counts[batch_idx] = compute_phases_symm<Rank, 1, Ti, Tv>(
                        group.bx, idx_map.b_idx_map,
                        dst_block.bstrs, dst_block.num_b,
                        group.unique_zbs, group.num_zb, group.wb,
                        batch_buf.ptr_phase(batch_idx), max_b_count, group.rank,
                        batch_buf.ptr_src_b(batch_idx), batch_buf.ptr_dst_b(batch_idx),
                        false, false);
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti str_a = dst_block.astrs[a];
                    Tv *dst = dst_vec + dst_block.offset + a * dst_block.num_b;
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_b_count = batch_buf.valid_b_counts[batch_idx];

                        if (valid_b_count == 0)
                            continue;

                        const int src_block_idx = batch_buf.src_block_idxs[batch_idx];
                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const Tv *pb0 = batch_buf.ptr_phase(batch_idx);
                        const int *src_b_idx = batch_buf.ptr_src_b(batch_idx);
                        const int *dst_b_idx = batch_buf.ptr_dst_b(batch_idx);
                        const Tv *src = src_vec + src_block.offset + a * src_block.num_b;
                        Tv pa0 = {}, pa1 = {}, pan[64] = {};
                        compute_a_phase<Rank, Ti, Tv>(str_a,
                                                      group.unique_zas, group.num_za,
                                                      group.wa, group.rank,
                                                      pa0, pa1, pan);
                        update_dst<Rank, 1, 1, Tv>(valid_b_count,
                                                   group.rank, pa0, pa1, pan,
                                                   pb0, max_b_count,
                                                   src_b_idx, dst_b_idx, src, dst);
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline void gather_contract_mixed_batched_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> *__restrict__ groups,
    const int64 num_groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int max_a_count = 0, max_b_count = 0, max_rank = 0;

    get_upper<Rank, Ti, Tv>(
        blocks, groups, num_blocks, num_groups, max_a_count, max_b_count, max_rank);

    SharedBatchBuffer<Tv> batch_buf(BATCH_SIZE, max_b_count, max_rank);

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min(BATCH_SIZE, num_groups - batch_start);
#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    const int64 axsym = get_string_sym(group.ax, orbsym);
                    const int64 bxsym = get_string_sym(group.bx, orbsym);
                    const int64 bid = (dst_block.asym ^ axsym) * num_irreps + (dst_block.bsym ^ bxsym);
                    const int64 src_block_idx = block_map[bid];

                    batch_buf.src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1)
                    {
                        batch_buf.valid_b_counts[batch_idx] = 0;
                        continue;
                    }

                    batch_buf.valid_b_counts[batch_idx] = compute_phases_symm<Rank, 1, Ti, Tv>(
                        group.bx, idx_map.b_idx_map,
                        dst_block.bstrs, dst_block.num_b,
                        group.unique_zbs, group.num_zb, group.wb,
                        batch_buf.ptr_phase(batch_idx), max_b_count, group.rank,
                        batch_buf.ptr_src_b(batch_idx), batch_buf.ptr_dst_b(batch_idx),
                        false, false);
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti dst_str_a = dst_block.astrs[a];
                    Tv *dst = dst_vec + dst_block.offset + a * dst_block.num_b;
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_b_count = batch_buf.valid_b_counts[batch_idx];

                        if (valid_b_count == 0)
                            continue;

                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const Ti src_str_a = dst_str_a ^ group.ax;
                        const int src_a_idx = idx_map.a_idx_map[src_str_a];

                        if (src_a_idx == -1)
                            continue;

                        const int src_block_idx = batch_buf.src_block_idxs[batch_idx];
                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const Tv *pb0 = batch_buf.ptr_phase(batch_idx);
                        const int *src_b_idx = batch_buf.ptr_src_b(batch_idx);
                        const int *dst_b_idx = batch_buf.ptr_dst_b(batch_idx);
                        const Tv *src = src_vec + src_block.offset + src_a_idx * src_block.num_b;
                        Tv pa0 = {}, pa1 = {}, pan[64] = {};
                        compute_a_phase<Rank, Ti, Tv>(src_str_a,
                                                      group.unique_zas, group.num_za,
                                                      group.wa, group.rank,
                                                      pa0, pa1, pan);
                        update_dst<Rank, 1, 1, Tv>(valid_b_count,
                                                   group.rank, pa0, pa1, pan,
                                                   pb0, max_b_count,
                                                   src_b_idx, dst_b_idx, src, dst);
                    }
                }
            }
        }
    }
}

template <int TypeCode, typename Ti, typename Tv>
static inline void dispatch_chunks_by_rank(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &map,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int64 total_ngs = groups.size();
    if (total_ngs == 0)
        return;

    const SVDGroup_OTF<Ti, Tv> *groups_ptr = groups.data();

    int64 start = 0;
    while (start < total_ngs)
    {
        const int current_rank = groups_ptr[start].rank;
        const int dispatch_rank = (current_rank == 1 || current_rank == 2) ? current_rank : 0;

        int64 end = start + 1;
        while (end < total_ngs)
        {
            const int next_rank = groups_ptr[end].rank;
            const int next_dispatch_rank = (next_rank == 1 || next_rank == 2) ? next_rank : 0;
            if (next_dispatch_rank != dispatch_rank)
                break;
            end++;
        }

        const int64 chunk_size = end - start;
        const SVDGroup_OTF<Ti, Tv> *chunk_ptr = groups_ptr + start;

        if constexpr (TypeCode == 0)
        {
            switch (dispatch_rank)
            {
            case 1:
                gather_contract_diag_batched_impl<1>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            case 2:
                gather_contract_diag_batched_impl<2>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            default:
                gather_contract_diag_batched_impl<0>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            }
        }
        else if constexpr (TypeCode == 1)
        {
            switch (dispatch_rank)
            {
            case 1:
                gather_contract_pure_a_batched_impl<1>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            case 2:
                gather_contract_pure_a_batched_impl<2>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            default:
                gather_contract_pure_a_batched_impl<0>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            }
        }
        else if constexpr (TypeCode == 2)
        {
            switch (dispatch_rank)
            {
            case 1:
                gather_contract_pure_b_batched_impl<1>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            case 2:
                gather_contract_pure_b_batched_impl<2>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            default:
                gather_contract_pure_b_batched_impl<0>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            }
        }
        else if constexpr (TypeCode == 3)
        {
            switch (dispatch_rank)
            {
            case 1:
                gather_contract_mixed_batched_impl<1>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            case 2:
                gather_contract_mixed_batched_impl<2>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            default:
                gather_contract_mixed_batched_impl<0>(basis, map, chunk_ptr, chunk_size, src_vec, dst_vec);
                break;
            }
        }
        start = end;
    }
}

template <typename Ti,
          typename Tv>
void contract_network_otf(
    const BasisManager<Ti> *__restrict__ basis,
    const Network_OTF<Ti, Tv> *__restrict__ net,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
    {
        dst_vec[i] = {};
    }

    dispatch_chunks_by_rank<0>(basis, net->map, net->diag_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank<1>(basis, net->map, net->pure_a_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank<2>(basis, net->map, net->pure_b_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank<3>(basis, net->map, net->mixed_groups, src_vec, dst_vec);
}
