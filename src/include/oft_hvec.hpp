#pragma once
#include "otf.hpp"

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
    int max_b_count = 0;
    int max_rank = 0;
    get_upper<Rank, Ti, Tv>(blocks, groups, num_blocks, num_groups, max_b_count, max_rank);
    std::vector<Tv> batch_b_phase(BATCH_SIZE * max_b_count * max_rank);

#pragma omp parallel
    {
        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            const int a_count = block.num_a;
            const int b_count = block.num_b;

            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    const uint16 rank = group.rank;
                    const int num_zb = group.num_zb;
                    const Ti *zbs = group.unique_zbs;
                    const Tv *wb0 = group.wb;
                    Tv *pb0 = batch_b_phase.data() + batch_idx * max_b_count * max_rank;

                    compute_phases_direct_aos<Rank, Ti, Tv>(
                        block.bstrs, b_count, wb0, zbs, num_zb, pb0, max_b_count, rank);
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < a_count; ++a)
                {
                    const Ti str_a = block.astrs[a];
                    const int64 ptr = block.offset + (int64)a * b_count;
                    const Tv *src = src_vec + ptr;
                    Tv *dst = dst_vec + ptr;

                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const uint16 rank = group.rank;
                        const int num_za = group.num_za;
                        const Ti *zas = group.unique_zas;
                        const Tv *wa0 = group.wa;
                        const Tv *pb0 = batch_b_phase.data() + batch_idx * max_b_count * max_rank;
                        const Tv *pb1 = pb0 + max_b_count;

                        Tv pa0 = {};
                        Tv pa1 = {};
                        Tv pan[64] = {};
                        compute_a_phase_impl<Rank, Ti, Tv>(rank, str_a, num_za, zas, wa0, pa0, pa1, pan);
                        update_vec_direct_aos<Rank, Tv>(b_count, max_b_count, rank, pa0, pa1, pan, pb0, pb1, src, dst);
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
    int max_b_count = 0;
    int max_rank = 0;
    get_upper<Rank, Ti, Tv>(blocks, groups, num_blocks, num_groups, max_b_count, max_rank);
    std::vector<Tv> batch_b_phase(BATCH_SIZE * max_b_count * max_rank);
    std::vector<int> batch_src_block_idxs(BATCH_SIZE);

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;

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

                    batch_src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1)
                        continue;

                    const uint16 rank = group.rank;
                    const int num_zb = group.num_zb;
                    const Ti *zbs = group.unique_zbs;
                    const Tv *wb0 = group.wb;
                    Tv *pb0 = batch_b_phase.data() + batch_idx * max_b_count * max_rank;

                    compute_phases_direct_aos<Rank, Ti, Tv>(
                        dst_block.bstrs, dst_b_count, wb0, zbs, num_zb, pb0, max_b_count, rank);
                }
#pragma omp for schedule(dynamic)
                for (int dst_a_idx = 0; dst_a_idx < dst_a_count; ++dst_a_idx)
                {
                    const Ti dst_str_a = dst_block.astrs[dst_a_idx];
                    Tv *dst = dst_vec + dst_block.offset + (int64)dst_a_idx * dst_b_count;

                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int src_block_idx = batch_src_block_idxs[batch_idx];

                        if (src_block_idx == -1)
                            continue;

                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const Ti src_str_a = dst_str_a ^ group.ax;
                        const int src_a_idx = idx_map.a_idx_map[src_str_a];

                        if (src_a_idx == -1)
                            continue;

                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const uint16 rank = group.rank;
                        const int num_za = group.num_za;
                        const Ti *zas = group.unique_zas;
                        const Tv *wa0 = group.wa;
                        const Tv *pb0 = batch_b_phase.data() + batch_idx * max_b_count * max_rank;
                        const Tv *pb1 = pb0 + max_b_count;
                        const Tv *src = src_vec + src_block.offset + (int64)src_a_idx * src_block.num_b;

                        Tv pa0 = {};
                        Tv pa1 = {};
                        Tv pan[64] = {};
                        compute_a_phase_impl<Rank, Ti, Tv>(rank, src_str_a, num_za, zas, wa0, pa0, pa1, pan);
                        update_vec_direct_aos<Rank, Tv>(dst_b_count, max_b_count, rank, pa0, pa1, pan, pb0, pb1, src, dst);
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
    int max_b_count = 0;
    int max_rank = 0;
    get_upper<Rank, Ti, Tv>(blocks, groups, num_blocks, num_groups, max_b_count, max_rank);
    SharedBatchBuffer<Tv> batch_buf(BATCH_SIZE, max_b_count, max_rank);

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;

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

                    const uint16 rank = group.rank;
                    const int num_zb = group.num_zb;
                    const Ti *zbs = group.unique_zbs;
                    const Tv *wb0 = group.wb;
                    int *src_b_ptr = batch_buf.ptr_src_b(batch_idx);
                    int *dst_b_ptr = batch_buf.ptr_dst_b(batch_idx);
                    Tv *pb0 = batch_buf.ptr_phase(batch_idx);

                    int valid_nb = compute_phases_indirect_aos<Rank, Ti, Tv>(
                        group.bx, idx_map.b_idx_map,
                        dst_block.bstrs, dst_block.num_b, wb0, zbs, num_zb,
                        src_b_ptr, dst_b_ptr, pb0,
                        max_b_count, rank, false, false);

                    batch_buf.valid_b_counts[batch_idx] = valid_nb;
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_a_count; ++a)
                {
                    const Ti str_a = dst_block.astrs[a];
                    Tv *dst = dst_vec + dst_block.offset + (int64)a * dst_b_count;

                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_b_count = batch_buf.valid_b_counts[batch_idx];
                        if (valid_b_count == 0)
                            continue;

                        const int src_block_idx = batch_buf.src_block_idxs[batch_idx];
                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const uint16 rank = group.rank;
                        const int num_za = group.num_za;
                        const Ti *zas = group.unique_zas;
                        const Tv *wa0 = group.wa;
                        const Tv *pb0 = batch_buf.ptr_phase(batch_idx);
                        const Tv *pb1 = pb0 + max_b_count;
                        const Tv *src = src_vec + src_block.offset + (int64)a * src_block.num_b;
                        const int *src_b_ptr = batch_buf.ptr_src_b(batch_idx);
                        const int *dst_b_ptr = batch_buf.ptr_dst_b(batch_idx);

                        Tv pa0 = {};
                        Tv pa1 = {};
                        Tv pan[64] = {};
                        compute_a_phase_impl<Rank, Ti, Tv>(rank, str_a, num_za, zas, wa0, pa0, pa1, pan);
                        update_vec_indirect_aos<Rank, Tv>(
                            valid_b_count, max_b_count, rank,
                            pa0, pa1, pan, pb0, pb1,
                            src_b_ptr, dst_b_ptr, src, dst);
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
    int max_b_count = 0;
    int max_rank = 0;
    get_upper<Rank, Ti, Tv>(blocks, groups, num_blocks, num_groups, max_b_count, max_rank);
    SharedBatchBuffer<Tv> batch_buf(BATCH_SIZE, max_b_count, max_rank);

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;

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

                    const uint16 rank = group.rank;
                    const int num_zb = group.num_zb;
                    const Ti *zbs = group.unique_zbs;
                    const Tv *wb0 = group.wb;
                    int *src_b_ptr = batch_buf.ptr_src_b(batch_idx);
                    int *dst_b_ptr = batch_buf.ptr_dst_b(batch_idx);
                    Tv *pb0 = batch_buf.ptr_phase(batch_idx);

                    int valid_nb = compute_phases_indirect_aos<Rank, Ti, Tv>(
                        group.bx, idx_map.b_idx_map,
                        dst_block.bstrs, dst_block.num_b, wb0, zbs, num_zb,
                        src_b_ptr, dst_b_ptr, pb0,
                        max_b_count, rank, false, false);

                    batch_buf.valid_b_counts[batch_idx] = valid_nb;
                }
#pragma omp for schedule(dynamic)
                for (int dst_a_idx = 0; dst_a_idx < dst_a_count; ++dst_a_idx)
                {
                    const Ti dst_str_a = dst_block.astrs[dst_a_idx];
                    Tv *dst = dst_vec + dst_block.offset + (int64)dst_a_idx * dst_b_count;

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
                        const uint16 rank = group.rank;
                        const int num_za = group.num_za;
                        const Ti *zas = group.unique_zas;
                        const Tv *wa0 = group.wa;
                        const Tv *pb0 = batch_buf.ptr_phase(batch_idx);
                        const Tv *pb1 = pb0 + max_b_count;
                        const Tv *src = src_vec + src_block.offset + (int64)src_a_idx * src_block.num_b;
                        const int *src_b_ptr = batch_buf.ptr_src_b(batch_idx);
                        const int *dst_b_ptr = batch_buf.ptr_dst_b(batch_idx);

                        Tv pa0 = {};
                        Tv pa1 = {};
                        Tv pan[64] = {};
                        compute_a_phase_impl<Rank, Ti, Tv>(rank, src_str_a, num_za, zas, wa0, pa0, pa1, pan);
                        update_vec_indirect_aos<Rank, Tv>(
                            valid_b_count, max_b_count, rank,
                            pa0, pa1, pan, pb0, pb1,
                            src_b_ptr, dst_b_ptr, src, dst);
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

template <typename Ti,
          typename Tv>
void *build_network_otf(
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

    uint64 z_offset_a = 0, z_offset_b = 0;
    uint64 w_offset_a = 0, w_offset_b = 0;

    for (int64 g = 0; g < ngs; ++g)
    {
        SVDGroup_OTF<Ti, Tv> group;
        group.ax = axs[g];
        group.bx = bxs[g];
        group.rank = ranks[g];
        group.num_za = num_zas[g];
        group.num_zb = num_zbs[g];

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

        if (group.ax == 0 && group.bx == 0)
            net->diag_groups.push_back(group);
        else if (group.ax != 0 && group.bx == 0)
            net->pure_a_groups.push_back(group);
        else if (group.ax == 0 && group.bx != 0)
            net->pure_b_groups.push_back(group);
        else
            net->mixed_groups.push_back(group);
    }

    auto rank_comparator = [](const SVDGroup_OTF<Ti, Tv> &a, const SVDGroup_OTF<Ti, Tv> &b)
    {
        return a.rank < b.rank;
    };

    std::sort(net->diag_groups.begin(), net->diag_groups.end(), rank_comparator);
    std::sort(net->pure_a_groups.begin(), net->pure_a_groups.end(), rank_comparator);
    std::sort(net->pure_b_groups.begin(), net->pure_b_groups.end(), rank_comparator);
    std::sort(net->mixed_groups.begin(), net->mixed_groups.end(), rank_comparator);

    return static_cast<void *>(net);
}
