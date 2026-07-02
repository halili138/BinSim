#pragma once
#include "sci_common.hpp"
#include "otf.hpp"
#include "utils.hpp"

template <int Rank, typename Ti, typename Tv>
static inline void gather_diag_for_block(
    const BlockDesc<Ti> &tgt_block,
    const SciBasisManager<Ti> *src_basis,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups,
    const Tv *src_vec, Tv *dst_acc)
{
    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;

    const int tgt_num_b = (int)tgt_block.num_b;
    const int shift = tgt_num_b * MAX_RANK;

    const int64 num_irreps = src_basis->num_irreps;
    const int64 src_block_idx = src_basis->block_map[tgt_block.asym * num_irreps + tgt_block.bsym];

    std::vector<int> src_a_idxs(tgt_block.num_a);
    std::vector<int> src_b_idxs(tgt_block.num_b);

    bool has_src = (src_block_idx != -1);
    if (has_src)
    {
        for (int a = 0; a < tgt_block.num_a; ++a)
        {
            auto it = src_basis->a_idx_map.find(tgt_block.astrs[a]);
            src_a_idxs[a] = (it != src_basis->a_idx_map.end()) ? it->second : -1;
        }
        for (int b = 0; b < tgt_block.num_b; ++b)
        {
            auto it = src_basis->b_idx_map.find(tgt_block.bstrs[b]);
            src_b_idxs[b] = (it != src_basis->b_idx_map.end()) ? it->second : -1;
        }
    }

    std::vector<Tv> phase_b(BATCH_SIZE * shift);

#pragma omp parallel
    {
        for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
        {
            const int64 cur_batch_size = std::min<int64>(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
            for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
            {
                const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                Tv *pb0 = phase_b.data() + batch_idx * shift;
                for (int i = 0; i < tgt_block.num_b; ++i)
                {
                    precompute_phase<Rank, Ti, Tv>(tgt_block.bstrs[i], group.unique_zbs,
                                                   group.num_zb, group.wb,
                                                   pb0 + i, tgt_num_b, group.rank);
                }
            }

#pragma omp for schedule(dynamic)
            for (int a = 0; a < tgt_block.num_a; ++a)
            {
                if (!has_src || src_a_idxs[a] == -1)
                    continue;

                const BlockDesc<Ti> &src_block = src_basis->blocks[src_block_idx];
                const Tv *sa_base = src_vec + src_block.offset
                                    + (int64)src_a_idxs[a] * src_block.num_b;
                Tv *da = dst_acc + a * tgt_block.num_b;

                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];

                    Tv pa[MAX_RANK] = {};
                    precompute_phase<Rank, Ti, Tv>(tgt_block.astrs[a], group.unique_zas,
                                                   group.num_za, group.wa,
                                                   pa, 1, group.rank);

                    const Tv *pb = phase_b.data() + batch_idx * shift;
                    const int rank = group.rank;

#pragma omp simd
                    for (int b = 0; b < tgt_block.num_b; ++b)
                    {
                        const int src_b_idx = src_b_idxs[b];
                        if (src_b_idx == -1)
                            continue;
                        const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, tgt_num_b, rank);
                        hvec_update<Tv>(sa_base + src_b_idx, da + b, vt);
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline void gather_pure_a_for_block(
    const BlockDesc<Ti> &tgt_block,
    const SciBasisManager<Ti> *src_basis,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups,
    const Tv *src_vec, Tv *dst_acc)
{
    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;

    const int tgt_num_a = (int)tgt_block.num_a;
    const int tgt_num_b = (int)tgt_block.num_b;
    const int shift = tgt_num_b * MAX_RANK;

    const int64 num_irreps = src_basis->num_irreps;

    std::vector<int> src_b_idxs(tgt_num_b);
    for (int b = 0; b < tgt_num_b; ++b)
    {
        auto it = src_basis->b_idx_map.find(tgt_block.bstrs[b]);
        src_b_idxs[b] = (it != src_basis->b_idx_map.end()) ? it->second : -1;
    }

    std::vector<Tv> phase_b(BATCH_SIZE * shift);
    std::vector<int> src_block_idxs(BATCH_SIZE);

    std::vector<int> src_a_idxs(BATCH_SIZE * tgt_num_a);
    std::vector<Tv> phase_a(BATCH_SIZE * tgt_num_a * MAX_RANK);

#pragma omp parallel
    {
        for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
        {
            const int64 cur_batch_size = std::min<int64>(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
            for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
            {
                const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                const int64 h = (tgt_block.asym ^ group.asym) * num_irreps + tgt_block.bsym;
                const int64 sidx = src_basis->block_map[h];
                src_block_idxs[batch_idx] = sidx;

                if (sidx == -1)
                    continue;

                Tv *pb0 = phase_b.data() + batch_idx * shift;
                for (int i = 0; i < tgt_num_b; ++i)
                {
                    precompute_phase<Rank, Ti, Tv>(tgt_block.bstrs[i], group.unique_zbs,
                                                   group.num_zb, group.wb,
                                                   pb0 + i, tgt_num_b, group.rank);
                }

                int *sa_ptr = src_a_idxs.data() + batch_idx * tgt_num_a;
                Tv *pa_ptr = phase_a.data() + batch_idx * tgt_num_a * MAX_RANK;
                for (int a = 0; a < tgt_num_a; ++a)
                {
                    const Ti src_a_str = tgt_block.astrs[a] ^ group.ax;
                    auto it = src_basis->a_idx_map.find(src_a_str);
                    if (it != src_basis->a_idx_map.end())
                    {
                        sa_ptr[a] = it->second;
                        precompute_phase<Rank, Ti, Tv>(src_a_str, group.unique_zas,
                                                       group.num_za, group.wa,
                                                       pa_ptr + a * MAX_RANK, 1, group.rank);
                    }
                    else
                    {
                        sa_ptr[a] = -1;
                    }
                }
            }

#pragma omp for schedule(dynamic)
            for (int a = 0; a < tgt_num_a; ++a)
            {
                Tv *da = dst_acc + a * tgt_num_b;
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const int64 src_block_idx = src_block_idxs[batch_idx];
                    if (src_block_idx == -1)
                        continue;

                    const int src_a_idx = src_a_idxs[batch_idx * tgt_num_a + a];
                    if (src_a_idx == -1)
                        continue;

                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    const Tv *pa = phase_a.data() + batch_idx * tgt_num_a * MAX_RANK + a * MAX_RANK;
                    const Tv *pb = phase_b.data() + batch_idx * shift;

                    const BlockDesc<Ti> &src_block = src_basis->blocks[src_block_idx];
                    const Tv *sa = src_vec + src_block.offset
                                   + (int64)src_a_idx * src_block.num_b;
                    const int rank = group.rank;

#pragma omp simd
                    for (int b = 0; b < tgt_num_b; ++b)
                    {
                        if (src_b_idxs[b] == -1)
                            continue;
                        const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, tgt_num_b, rank);
                        hvec_update<Tv>(sa + src_b_idxs[b], da + b, vt);
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline void gather_pure_b_for_block(
    const BlockDesc<Ti> &tgt_block,
    const SciBasisManager<Ti> *src_basis,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups,
    const Tv *src_vec, Tv *dst_acc)
{
    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;

    const int tgt_num_a = (int)tgt_block.num_a;
    const int tgt_num_b = (int)tgt_block.num_b;
    const int shift = tgt_num_b * MAX_RANK;

    const int64 num_irreps = src_basis->num_irreps;

    std::vector<int> src_a_idxs(tgt_num_a);
    for (int a = 0; a < tgt_num_a; ++a)
    {
        auto it = src_basis->a_idx_map.find(tgt_block.astrs[a]);
        src_a_idxs[a] = (it != src_basis->a_idx_map.end()) ? it->second : -1;
    }

    std::vector<int> src_b_idxs_v(BATCH_SIZE * tgt_num_b);
    std::vector<int> dst_b_idxs(BATCH_SIZE * tgt_num_b);
    std::vector<Tv> batch_phase(BATCH_SIZE * shift);
    std::vector<int> valid_b_counts(BATCH_SIZE);
    std::vector<int> src_block_idxs(BATCH_SIZE);

#pragma omp parallel
    {
        for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
        {
            const int64 cur_batch_size = std::min<int64>(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
            for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
            {
                const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                const int64 h = tgt_block.asym * num_irreps + (tgt_block.bsym ^ group.bsym);
                const int64 sidx = src_basis->block_map[h];
                src_block_idxs[batch_idx] = sidx;

                if (sidx == -1)
                {
                    valid_b_counts[batch_idx] = 0;
                    continue;
                }

                Tv *pb0 = batch_phase.data() + batch_idx * shift;
                int *sb_ptr = src_b_idxs_v.data() + batch_idx * tgt_num_b;
                int *db_ptr = dst_b_idxs.data() + batch_idx * tgt_num_b;

                int count = 0;
                for (int i = 0; i < tgt_num_b; ++i)
                {
                    const Ti src_b_str = tgt_block.bstrs[i] ^ group.bx;
                    auto it = src_basis->b_idx_map.find(src_b_str);
                    if (it == src_basis->b_idx_map.end())
                        continue;

                    sb_ptr[count] = it->second;
                    db_ptr[count] = i;

                    precompute_phase<Rank, Ti, Tv>(src_b_str, group.unique_zbs,
                                                   group.num_zb, group.wb,
                                                   pb0 + count, tgt_num_b, group.rank);
                    count++;
                }
                valid_b_counts[batch_idx] = count;
            }

#pragma omp for schedule(dynamic)
            for (int a = 0; a < tgt_num_a; ++a)
            {
                if (src_a_idxs[a] == -1)
                    continue;

                Tv *da = dst_acc + a * tgt_num_b;
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const int valid_count = valid_b_counts[batch_idx];
                    if (valid_count == 0)
                        continue;

                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];

                    Tv pa[MAX_RANK] = {};
                    precompute_phase<Rank, Ti, Tv>(tgt_block.astrs[a], group.unique_zas,
                                                   group.num_za, group.wa,
                                                   pa, 1, group.rank);

                    const int64 src_block_idx = src_block_idxs[batch_idx];
                    const BlockDesc<Ti> &src_block = src_basis->blocks[src_block_idx];
                    const Tv *pb = batch_phase.data() + batch_idx * shift;
                    const int *si = src_b_idxs_v.data() + batch_idx * tgt_num_b;
                    const int *di = dst_b_idxs.data() + batch_idx * tgt_num_b;
                    const Tv *sa = src_vec + src_block.offset
                                   + (int64)src_a_idxs[a] * src_block.num_b;
                    const int rank = group.rank;

#pragma omp simd
                    for (int b = 0; b < valid_count; ++b)
                    {
                        const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, tgt_num_b, rank);
                        hvec_update<Tv>(sa + si[b], da + di[b], vt);
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline void gather_mixed_for_block(
    const BlockDesc<Ti> &tgt_block,
    const SciBasisManager<Ti> *src_basis,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups,
    const Tv *src_vec, Tv *dst_acc)
{
    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;

    const int tgt_num_a = (int)tgt_block.num_a;
    const int tgt_num_b = (int)tgt_block.num_b;
    const int shift = tgt_num_b * MAX_RANK;

    const int64 num_irreps = src_basis->num_irreps;

    std::vector<int> src_b_idxs_v(BATCH_SIZE * tgt_num_b);
    std::vector<int> dst_b_idxs(BATCH_SIZE * tgt_num_b);
    std::vector<Tv> batch_phase(BATCH_SIZE * shift);
    std::vector<int> valid_b_counts(BATCH_SIZE);
    std::vector<int> src_block_idxs(BATCH_SIZE);

    std::vector<int> src_a_idxs(BATCH_SIZE * tgt_num_a);
    std::vector<Tv> phase_a(BATCH_SIZE * tgt_num_a * MAX_RANK);

#pragma omp parallel
    {
        for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
        {
            const int64 cur_batch_size = std::min<int64>(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
            for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
            {
                const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                const int64 h = (tgt_block.asym ^ group.asym) * num_irreps
                                + (tgt_block.bsym ^ group.bsym);
                const int64 sidx = src_basis->block_map[h];
                src_block_idxs[batch_idx] = sidx;

                if (sidx == -1)
                {
                    valid_b_counts[batch_idx] = 0;
                    continue;
                }

                Tv *pb0 = batch_phase.data() + batch_idx * shift;
                int *sb_ptr = src_b_idxs_v.data() + batch_idx * tgt_num_b;
                int *db_ptr = dst_b_idxs.data() + batch_idx * tgt_num_b;

                int count = 0;
                for (int i = 0; i < tgt_num_b; ++i)
                {
                    const Ti src_b_str = tgt_block.bstrs[i] ^ group.bx;
                    auto it = src_basis->b_idx_map.find(src_b_str);
                    if (it == src_basis->b_idx_map.end())
                        continue;

                    sb_ptr[count] = it->second;
                    db_ptr[count] = i;

                    precompute_phase<Rank, Ti, Tv>(src_b_str, group.unique_zbs,
                                                   group.num_zb, group.wb,
                                                   pb0 + count, tgt_num_b, group.rank);
                    count++;
                }
                valid_b_counts[batch_idx] = count;

                int *sa_ptr = src_a_idxs.data() + batch_idx * tgt_num_a;
                Tv *pa_ptr = phase_a.data() + batch_idx * tgt_num_a * MAX_RANK;
                for (int a = 0; a < tgt_num_a; ++a)
                {
                    const Ti src_a_str = tgt_block.astrs[a] ^ group.ax;
                    auto it = src_basis->a_idx_map.find(src_a_str);
                    if (it != src_basis->a_idx_map.end())
                    {
                        sa_ptr[a] = it->second;
                        precompute_phase<Rank, Ti, Tv>(src_a_str, group.unique_zas,
                                                       group.num_za, group.wa,
                                                       pa_ptr + a * MAX_RANK, 1, group.rank);
                    }
                    else
                    {
                        sa_ptr[a] = -1;
                    }
                }
            }

#pragma omp for schedule(dynamic)
            for (int a = 0; a < tgt_num_a; ++a)
            {
                Tv *da = dst_acc + a * tgt_num_b;
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const int valid_count = valid_b_counts[batch_idx];
                    if (valid_count == 0)
                        continue;

                    const int src_a_idx = src_a_idxs[batch_idx * tgt_num_a + a];
                    if (src_a_idx == -1)
                        continue;

                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    const Tv *pa = phase_a.data() + batch_idx * tgt_num_a * MAX_RANK + a * MAX_RANK;

                    const int64 src_block_idx = src_block_idxs[batch_idx];
                    const BlockDesc<Ti> &src_block = src_basis->blocks[src_block_idx];
                    const Tv *pb = batch_phase.data() + batch_idx * shift;
                    const int *si = src_b_idxs_v.data() + batch_idx * tgt_num_b;
                    const int *di = dst_b_idxs.data() + batch_idx * tgt_num_b;
                    const Tv *sa = src_vec + src_block.offset
                                   + (int64)src_a_idx * src_block.num_b;
                    const int rank = group.rank;

#pragma omp simd
                    for (int b = 0; b < valid_count; ++b)
                    {
                        const Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, tgt_num_b, rank);
                        hvec_update<Tv>(sa + si[b], da + di[b], vt);
                    }
                }
            }
        }
    }
}

template <int TypeCode, typename Ti, typename Tv>
static inline void dispatch_chunks_for_block(
    const BlockDesc<Ti> &tgt_block,
    const SciBasisManager<Ti> *src_basis,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const Tv *src_vec, Tv *dst_acc)
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
                gather_diag_for_block<1>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            case 2:
                gather_diag_for_block<2>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            default:
                gather_diag_for_block<0>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            }
        }
        else if constexpr (TypeCode == 1)
        {
            switch (dispatch_rank)
            {
            case 1:
                gather_pure_a_for_block<1>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            case 2:
                gather_pure_a_for_block<2>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            default:
                gather_pure_a_for_block<0>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            }
        }
        else if constexpr (TypeCode == 2)
        {
            switch (dispatch_rank)
            {
            case 1:
                gather_pure_b_for_block<1>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            case 2:
                gather_pure_b_for_block<2>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            default:
                gather_pure_b_for_block<0>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            }
        }
        else if constexpr (TypeCode == 3)
        {
            switch (dispatch_rank)
            {
            case 1:
                gather_mixed_for_block<1>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            case 2:
                gather_mixed_for_block<2>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            default:
                gather_mixed_for_block<0>(tgt_block, src_basis, chunk_ptr, chunk_size, src_vec, dst_acc);
                break;
            }
        }
        start = end;
    }
}

template <typename Ti, typename Tv>
static inline void contract_hvec_sci_for_desc(
    const BlockDesc<Ti> &tgt_block,
    const SciBasisManager<Ti> *src_basis,
    const Network_OTF<Ti, Tv> *net,
    const Tv *src_vec, Tv *dst_acc)
{
    const int64 block_size = tgt_block.num_a * tgt_block.num_b;
    std::fill_n(dst_acc, block_size, Tv{});

    dispatch_chunks_for_block<0>(tgt_block, src_basis, net->diag_groups, src_vec, dst_acc);
    dispatch_chunks_for_block<1>(tgt_block, src_basis, net->pure_a_groups, src_vec, dst_acc);
    dispatch_chunks_for_block<2>(tgt_block, src_basis, net->pure_b_groups, src_vec, dst_acc);
    dispatch_chunks_for_block<3>(tgt_block, src_basis, net->mixed_groups, src_vec, dst_acc);
}

template <typename Ti, typename Tv>
void contract_hvec_sci_for_block(
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisManager<Ti> *src_basis,
    const Network_OTF<Ti, Tv> *net,
    int64 tgt_block_idx,
    const Tv *src_vec, Tv *dst_acc)
{
    const BlockDesc<Ti> &tgt_block = tgt_basis->blocks[tgt_block_idx];

    contract_hvec_sci_for_desc(tgt_block, src_basis, net, src_vec, dst_acc);
}

template <typename Ti, typename Tv, typename AccumFunc>
void contract_hvec_sci_chunked(
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisManager<Ti> *src_basis,
    const Network_OTF<Ti, Tv> *net,
    int64 tgt_block_idx,
    const Tv *src_vec,
    int chunk_size,
    AccumFunc &&accum_func)
{
    const BlockDesc<Ti> &full_block = tgt_basis->blocks[tgt_block_idx];
    const int64 num_a_total = full_block.num_a;
    const int64 num_b = full_block.num_b;

    for (int64 a_start = 0; a_start < num_a_total; a_start += chunk_size)
    {
        const int64 a_end = std::min(a_start + (int64)chunk_size, num_a_total);
        const int64 cur_num_a = a_end - a_start;

        BlockDesc<Ti> chunk_desc = full_block;
        chunk_desc.astrs = full_block.astrs + a_start;
        chunk_desc.num_a = cur_num_a;
        chunk_desc.offset = 0;

        Tv *chunk_acc = new Tv[cur_num_a * num_b];
        std::fill_n(chunk_acc, cur_num_a * num_b, Tv{});

        dispatch_chunks_for_block<0>(chunk_desc, src_basis, net->diag_groups, src_vec, chunk_acc);
        dispatch_chunks_for_block<1>(chunk_desc, src_basis, net->pure_a_groups, src_vec, chunk_acc);
        dispatch_chunks_for_block<2>(chunk_desc, src_basis, net->pure_b_groups, src_vec, chunk_acc);
        dispatch_chunks_for_block<3>(chunk_desc, src_basis, net->mixed_groups, src_vec, chunk_acc);

        for (int a = 0; a < cur_num_a; ++a)
        {
            const int64 a_global = a_start + a;
            const Tv *row = chunk_acc + a * num_b;
            for (int b = 0; b < num_b; ++b)
            {
                if (row[b] != Tv{})
                    accum_func(a_global, b, row[b]);
            }
        }

        delete[] chunk_acc;
    }
}
