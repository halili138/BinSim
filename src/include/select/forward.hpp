#pragma once
#include <cassert>
#include <cstdlib>
#include <vector>
#include <utility>
#include <ankerl/unordered_dense.h>
#include "basis/sci_basis.hpp"

template <typename Tv>
struct ForwardShared
{
    int64 ngs;
    int64 num_blocks;
    std::vector<int64> offsets;       // [ngs+1]
    std::vector<int64> block_offsets; // [ngs * num_blocks + 1], grouped by (group, src block)
    std::vector<int> dst_idxs;        // flat: global b index into target array
    std::vector<int> src_idxs;        // flat: local b index within source block
    std::vector<int> src_blk_idxs;    // flat: source block index
    std::vector<Tv> phase0;           // flat: first phase component per entry
    std::vector<Tv> phase1;           // flat: second phase component per entry (rank-1 unused)

    std::pair<int64, int64> range(int64 group, int64 src_blk_idx) const
    {
        assert(group >= 0 && group < ngs);
        assert(src_blk_idx >= 0 && src_blk_idx < num_blocks);
        int64 key = group * num_blocks + src_blk_idx;
        return {block_offsets[key], block_offsets[key + 1]};
    }
};

template <typename Ti, bool IsAlpha>
static ankerl::unordered_dense::map<Ti, std::pair<int, int>> build_idx_map(const SciBasisManager<Ti> *basis)
{
    ankerl::unordered_dense::map<Ti, std::pair<int, int>> idx_map;
    for (int bi = 0; bi < basis->num_blocks; ++bi)
    {
        const auto &blk = basis->blocks[bi];
        const Ti *strs;
        int64 num;
        if constexpr (IsAlpha)
        {
            strs = blk.astrs;
            num = blk.num_a;
        }
        else
        {
            strs = blk.bstrs;
            num = blk.num_b;
        }
        for (int j = 0; j < num; ++j)
            idx_map[strs[j]] = {j, bi};
    }
    return idx_map;
}

template <typename Ti, typename Tv, bool IsAlpha>
static std::vector<std::vector<int>> build_old2new_link_chunk(
    const Ti *dst_chunk, int64 n_dst_chunk,
    const ankerl::unordered_dense::set<Ti> &new_set,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups, int64 g_begin, int64 g_end)
{
    const int64 ngs = std::max<int64>(0, g_end - g_begin);
    std::vector<std::vector<int>> link(n_dst_chunk);

#pragma omp parallel for schedule(dynamic)
    for (int64 i = 0; i < n_dst_chunk; ++i)
    {
        Ti dst = dst_chunk[i];
        for (int local_g = 0; local_g < ngs; ++local_g)
        {
            const auto &group = groups[g_begin + local_g];
            Ti exc;
            if constexpr (IsAlpha)
                exc = group.ax;
            else
                exc = group.bx;
            if (exc == 0)
                continue;
            if (new_set.find(dst ^ exc) == new_set.end())
                link[i].push_back(local_g);
        }
    }
    return link;
}

template <typename Ti, typename Tv, bool IsAlpha>
static ForwardShared<Tv> precompute_shared_chunk(
    const Ti *tgt_chunk, int64 n_tgt_chunk, int64 tgt_global_offset, int64 g_begin, int64 g_end,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_idx_map, int64 num_blocks,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups)
{
    int64 ngs = std::max<int64>(0, g_end - g_begin);
    ForwardShared<Tv> result;
    result.ngs = ngs;
    result.offsets.assign(ngs + 1, 0);

    num_blocks = std::max<int64>(0, num_blocks);
    result.num_blocks = num_blocks;

    const int64 num_buckets = ngs * num_blocks;
    std::vector<int64> cnts(num_buckets);

    struct SharedHit
    {
        int64 local_g;
        int src_blk_idx;
        int dst_idx;
        int src_idx;
        Tv phase0;
        Tv phase1;
    };
    std::vector<SharedHit> hits;

    for (int64 i = 0; i < n_tgt_chunk; ++i)
    {
        Ti dst = tgt_chunk[i];
        for (int64 local_g = 0; local_g < ngs; ++local_g)
        {
            const auto &group = all_groups[g_begin + local_g];
            Ti exc;
            if constexpr (IsAlpha)
                exc = group.ax;
            else
                exc = group.bx;
            Ti src = dst ^ exc;
            auto it = old_idx_map.find(src);
            if (it == old_idx_map.end())
                continue;

            const auto &src_pos = it->second;
            if (src_pos.second < 0 || src_pos.second >= num_blocks)
                continue;

            Tv phase_tmp[2] = {};
            if (exc == 0)
            {
                phase_tmp[0] = Tv(1);
            }
            else if constexpr (IsAlpha)
            {
                precompute_phase_select<Ti, Tv>(src, group.unique_zas, group.num_za, group.wa, phase_tmp, 1, group.rank);
            }
            else
            {
                precompute_phase_select<Ti, Tv>(src, group.unique_zbs, group.num_zb, group.wb, phase_tmp, 1, group.rank);
            }

            hits.push_back({local_g,
                            src_pos.second,
                            static_cast<int>(tgt_global_offset + i),
                            src_pos.first,
                            phase_tmp[0],
                            group.rank >= 2 ? phase_tmp[1] : Tv{}});
            ++cnts[local_g * num_blocks + src_pos.second];
        }
    }

    result.block_offsets.assign(num_buckets + 1, 0);
    for (int64 b = 0; b < num_buckets; ++b)
        result.block_offsets[b + 1] = result.block_offsets[b] + cnts[b];

    for (int64 local_g = 0; local_g < ngs; ++local_g)
        result.offsets[local_g + 1] = result.block_offsets[(local_g + 1) * num_blocks];

    int64 total = result.block_offsets[num_buckets];
    result.dst_idxs.assign(total, 0);
    result.src_idxs.assign(total, 0);
    result.src_blk_idxs.assign(total, 0);
    result.phase0.assign(total, Tv{});
    result.phase1.assign(total, Tv{});

    std::vector<int64> pos = result.block_offsets;

    for (const auto &hit : hits)
    {
        int64 p = pos[hit.local_g * num_blocks + hit.src_blk_idx]++;
        result.dst_idxs[p] = hit.dst_idx;
        result.src_idxs[p] = hit.src_idx;
        result.src_blk_idxs[p] = hit.src_blk_idx;
        result.phase0[p] = hit.phase0;
        result.phase1[p] = hit.phase1;
    }

    return result;
}

template <typename Ti, typename Tv>
static void select_pass_a(
    const Ti *new_α, int64 n_new_α,
    const Ti *old_β, int64 n_old_β,
    const Ti *new_β, int64 n_new_β,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_a_idx_map,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_b_idx_map,
    int64 old_b_num_blocks,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups,
    const Tv *src_psi,
    const BlockDesc<Ti> *src_blocks,
    const Tv *pa_diag, const Tv *pb_diag_old, const Tv *pb_diag_new, int diag_rank,
    Tv E_var, Tv eps,
    std::vector<std::pair<Ti, Ti>> &out_p1,
    std::vector<std::pair<Ti, Ti>> &out_p3)
{
    const int64 num_group_chunks = ((int64)all_groups.size() + GROUP_CHUNK_SIZE - 1) / GROUP_CHUNK_SIZE;
    const int64 num_a_chunks = (n_new_α + TARGET_CHUNK_SIZE - 1) / TARGET_CHUNK_SIZE;
    const ankerl::unordered_dense::set<Ti> new_a_set(new_α, new_α + n_new_α);

    auto run_beta_side = [&](const Ti *beta, int64 n_beta, const Tv *pb_diag, std::vector<std::pair<Ti, Ti>> &out)
    {
        for (int64 b_begin = 0; b_begin < n_beta; b_begin += TARGET_CHUNK_SIZE)
        {
            const int64 b_end = std::min<int64>(b_begin + TARGET_CHUNK_SIZE, n_beta);
            const int64 b_count = b_end - b_begin;
            std::vector<ForwardShared<Tv>> shared_chunks;
            shared_chunks.reserve(num_group_chunks);
            for (int64 g_begin = 0; g_begin < (int64)all_groups.size(); g_begin += GROUP_CHUNK_SIZE)
            {
                int64 g_end = std::min<int64>(g_begin + GROUP_CHUNK_SIZE, (int64)all_groups.size());
                shared_chunks.push_back(precompute_shared_chunk<Ti, Tv, false>(beta + b_begin, b_count, b_begin, g_begin, g_end, old_b_idx_map, old_b_num_blocks, all_groups));
            }

            for (int64 a_chunk = 0; a_chunk < num_a_chunks; ++a_chunk)
            {
                const int64 a_begin = a_chunk * TARGET_CHUNK_SIZE;
                const int64 a_end = std::min<int64>(a_begin + TARGET_CHUNK_SIZE, n_new_α);
                const int64 a_count = a_end - a_begin;
                std::vector<std::vector<std::vector<int>>> link_chunks;
                link_chunks.reserve(num_group_chunks);
                for (int64 g_begin = 0; g_begin < (int64)all_groups.size(); g_begin += GROUP_CHUNK_SIZE)
                {
                    int64 g_end = std::min<int64>(g_begin + GROUP_CHUNK_SIZE, (int64)all_groups.size());
                    link_chunks.push_back(build_old2new_link_chunk<Ti, Tv, true>(new_α + a_begin, a_count, new_a_set, all_groups, g_begin, g_end));
                }

#pragma omp parallel
                {
                    std::vector<Tv> accum(b_count);
                    std::vector<std::pair<Ti, Ti>> thread_out;

#pragma omp for schedule(dynamic)
                    for (int64 local_ia = 0; local_ia < a_count; ++local_ia)
                    {
                        Ti dst_a = new_α[a_begin + local_ia];
                        std::fill(accum.begin(), accum.end(), Tv{});

                        for (int64 chunk_id = 0; chunk_id < (int64)link_chunks.size(); ++chunk_id)
                        {
                            const int64 g_begin = chunk_id * GROUP_CHUNK_SIZE;
                            const auto &shared = shared_chunks[chunk_id];
                            for (int local_g : link_chunks[chunk_id][local_ia])
                            {
                                const int64 ig = g_begin + local_g;
                                const auto &group = all_groups[ig];
                                Ti src_a = dst_a ^ group.ax;
                                auto it = old_a_idx_map.find(src_a);
                                if (it == old_a_idx_map.end())
                                    continue;
                                int src_ia = it->second.first;
                                int src_a_blk_idx = it->second.second;

                                Tv pa[2] = {};
                                precompute_phase_select<Ti, Tv>(src_a, group.unique_zas, group.num_za, group.wa, pa, 1, group.rank);
                                if (group.rank == 1)
                                    pa[1] = Tv{};

                                const auto &blk = src_blocks[src_a_blk_idx];
                                const int64 row_base = blk.offset + src_ia * blk.num_b;
                                auto [off, end] = shared.range(local_g, src_a_blk_idx);
                                for (int64 j = off; j < end; ++j)
                                {
                                    int beta_idx = shared.dst_idxs[j];
                                    int src_ib = shared.src_idxs[j];
                                    Tv coeff = pa[0] * shared.phase0[j];
                                    if (group.rank >= 2)
                                        coeff += pa[1] * shared.phase1[j];
                                    accum[beta_idx - b_begin] += src_psi[row_base + src_ib] * coeff;
                                }
                            }
                        }

                        for (int64 local_b = 0; local_b < b_count; ++local_b)
                        {
                            Tv v = accum[local_b];
                            if (v == Tv{})
                                continue;
                            Tv Haa = {};
                            const Tv *par = pa_diag + (a_begin + local_ia) * diag_rank;
                            const Tv *pbr = pb_diag + (b_begin + local_b) * diag_rank;
                            for (int r = 0; r < diag_rank; ++r)
                                Haa += par[r] * pbr[r];
                            if (!sci_eps_check(v, Haa, E_var, eps))
                                continue;
                            thread_out.emplace_back(dst_a, beta[b_begin + local_b]);
                        }
                    }

#pragma omp critical
                    out.insert(out.end(), std::make_move_iterator(thread_out.begin()), std::make_move_iterator(thread_out.end()));
                }
            }
        }
    };

    run_beta_side(old_β, n_old_β, pb_diag_old, out_p1);
    run_beta_side(new_β, n_new_β, pb_diag_new, out_p3);
}

template <typename Ti, typename Tv>
static void select_pass_b(
    const Ti *new_β, int64 n_new_β,
    const Ti *old_α, int64 n_old_α,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_b_idx_map,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_a_idx_map,
    int64 old_a_num_blocks,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups,
    const Tv *src_psi,
    const BlockDesc<Ti> *src_blocks,
    const Tv *pb_diag, const Tv *pa_diag_old, int diag_rank,
    Tv E_var, Tv eps,
    std::vector<std::pair<Ti, Ti>> &out_p2)
{
    const int64 num_group_chunks = ((int64)all_groups.size() + GROUP_CHUNK_SIZE - 1) / GROUP_CHUNK_SIZE;
    const int64 num_b_chunks = (n_new_β + TARGET_CHUNK_SIZE - 1) / TARGET_CHUNK_SIZE;
    const ankerl::unordered_dense::set<Ti> new_b_set(new_β, new_β + n_new_β);

    for (int64 a_begin = 0; a_begin < n_old_α; a_begin += TARGET_CHUNK_SIZE)
    {
        const int64 a_end = std::min<int64>(a_begin + TARGET_CHUNK_SIZE, n_old_α);
        const int64 a_count = a_end - a_begin;
        std::vector<ForwardShared<Tv>> shared_chunks;
        shared_chunks.reserve(num_group_chunks);
        for (int64 g_begin = 0; g_begin < (int64)all_groups.size(); g_begin += GROUP_CHUNK_SIZE)
        {
            int64 g_end = std::min<int64>(g_begin + GROUP_CHUNK_SIZE, (int64)all_groups.size());
            shared_chunks.push_back(precompute_shared_chunk<Ti, Tv, true>(old_α + a_begin, a_count, a_begin, g_begin, g_end, old_a_idx_map, old_a_num_blocks, all_groups));
        }

        for (int64 b_chunk = 0; b_chunk < num_b_chunks; ++b_chunk)
        {
            const int64 b_begin = b_chunk * TARGET_CHUNK_SIZE;
            const int64 b_end = std::min<int64>(b_begin + TARGET_CHUNK_SIZE, n_new_β);
            const int64 b_count = b_end - b_begin;
            std::vector<std::vector<std::vector<int>>> link_chunks;
            link_chunks.reserve(num_group_chunks);
            for (int64 g_begin = 0; g_begin < (int64)all_groups.size(); g_begin += GROUP_CHUNK_SIZE)
            {
                int64 g_end = std::min<int64>(g_begin + GROUP_CHUNK_SIZE, (int64)all_groups.size());
                link_chunks.push_back(build_old2new_link_chunk<Ti, Tv, false>(new_β + b_begin, b_count, new_b_set, all_groups, g_begin, g_end));
            }

#pragma omp parallel
            {
                std::vector<Tv> accum(a_count);
                std::vector<std::pair<Ti, Ti>> thread_p2;

#pragma omp for schedule(dynamic)
                for (int64 local_ib = 0; local_ib < b_count; ++local_ib)
                {
                    Ti dst_b = new_β[b_begin + local_ib];
                    std::fill(accum.begin(), accum.end(), Tv{});

                    for (int64 chunk_id = 0; chunk_id < (int64)link_chunks.size(); ++chunk_id)
                    {
                        const int64 g_begin = chunk_id * GROUP_CHUNK_SIZE;
                        const auto &shared = shared_chunks[chunk_id];
                        for (int local_g : link_chunks[chunk_id][local_ib])
                        {
                            const int64 ig = g_begin + local_g;
                            const auto &group = all_groups[ig];
                            Ti src_b = dst_b ^ group.bx;
                            auto it = old_b_idx_map.find(src_b);
                            if (it == old_b_idx_map.end())
                                continue;
                            int src_ib = it->second.first;
                            int src_b_blk_idx = it->second.second;

                            Tv pb[2] = {};
                            precompute_phase_select<Ti, Tv>(src_b, group.unique_zbs, group.num_zb, group.wb, pb, 1, group.rank);
                            if (group.rank == 1)
                                pb[1] = Tv{};

                            const auto &blk = src_blocks[src_b_blk_idx];
                            const int64 col_or_row_base = blk.offset + src_ib;
                            auto [off, end] = shared.range(local_g, src_b_blk_idx);
                            for (int64 j = off; j < end; ++j)
                            {
                                int old_ia = shared.dst_idxs[j];
                                int src_ia = shared.src_idxs[j];
                                Tv coeff = shared.phase0[j] * pb[0];
                                if (group.rank >= 2)
                                    coeff += shared.phase1[j] * pb[1];
                                accum[old_ia - a_begin] += src_psi[col_or_row_base + src_ia * blk.num_b] * coeff;
                            }
                        }
                    }

                    for (int64 local_a = 0; local_a < a_count; ++local_a)
                    {
                        Tv v = accum[local_a];
                        if (v == Tv{})
                            continue;
                        Tv Haa = {};
                        const Tv *par = pa_diag_old + (a_begin + local_a) * diag_rank;
                        const Tv *pbr = pb_diag + (b_begin + local_ib) * diag_rank;
                        for (int r = 0; r < diag_rank; ++r)
                            Haa += par[r] * pbr[r];
                        if (!sci_eps_check(v, Haa, E_var, eps))
                            continue;
                        thread_p2.emplace_back(old_α[a_begin + local_a], dst_b);
                    }
                }

#pragma omp critical
                out_p2.insert(out_p2.end(), std::make_move_iterator(thread_p2.begin()), std::make_move_iterator(thread_p2.end()));
            }
        }
    }
}

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
                const Tv *sa_base = src_vec + src_block.offset + (int64)src_a_idxs[a] * src_block.num_b;
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
                    const Tv *sa = src_vec + src_block.offset + (int64)src_a_idx * src_block.num_b;
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
                    const Tv *sa = src_vec + src_block.offset + (int64)src_a_idxs[a] * src_block.num_b;
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
                const int64 h = (tgt_block.asym ^ group.asym) * num_irreps + (tgt_block.bsym ^ group.bsym);
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
                    const Tv *sa = src_vec + src_block.offset + (int64)src_a_idx * src_block.num_b;
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
static inline void dispatch_contract_chunks_for_block(
    const BlockDesc<Ti> &tgt_block,
    const SciBasisManager<Ti> *src_basis,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const Tv *src_vec, Tv *dst_acc)
{
    static_assert(TypeCode >= 0 && TypeCode <= 3);

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
        else
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
static inline void contract_hvec_sci(
    const BlockDesc<Ti> &tgt_block,
    const SciBasisManager<Ti> *src_basis,
    const Network_OTF<Ti, Tv> *net,
    const Tv *src_vec, Tv *dst_acc)
{
    const int64 block_size = tgt_block.num_a * tgt_block.num_b;
    std::fill_n(dst_acc, block_size, Tv{});

    dispatch_contract_chunks_for_block<0>(tgt_block, src_basis, net->diag_groups, src_vec, dst_acc);
    dispatch_contract_chunks_for_block<1>(tgt_block, src_basis, net->pure_a_groups, src_vec, dst_acc);
    dispatch_contract_chunks_for_block<2>(tgt_block, src_basis, net->pure_b_groups, src_vec, dst_acc);
    dispatch_contract_chunks_for_block<3>(tgt_block, src_basis, net->mixed_groups, src_vec, dst_acc);
}

template <typename Ti, typename Tv>
static void sci_select_bitstr_impl(
    const Ti *new_a, int64 n_new_a,
    const Ti *new_b, int64 n_new_b,
    const Ti *old_a, int64 n_old_a,
    const Ti *old_b, int64 n_old_b,
    void *src_basis,
    void *net,
    Tv *src_psi, Tv E_var, Tv eps,
    Ti **out_a, Ti **out_b, int64 *n_pairs)
{
    auto *basis = static_cast<SciBasisManager<Ti> *>(src_basis);
    auto *otf = static_cast<Network_OTF<Ti, Tv> *>(net);

    auto all_groups = flatten_groups<Ti, Tv>(otf);

    auto old_a_idx = build_idx_map<Ti, true>(basis);
    auto old_b_idx = build_idx_map<Ti, false>(basis);

    const auto &dg = otf->diag_groups;
    int diag_rank = 0;
    std::vector<Tv> pa_d, pb_do, pb_dn, pa_do;

    if (!dg.empty() && dg[0].rank > 0)
    {
        diag_rank = dg[0].rank;
        const auto &g = dg[0];

        auto alloc_diag = [&](int64 n)
        {
            return std::vector<Tv>((size_t)(n * diag_rank), Tv{});
        };

        pa_d = alloc_diag(n_new_a);
        pb_do = alloc_diag(n_old_b);
        pb_dn = alloc_diag(n_new_b);
        pa_do = alloc_diag(n_old_a);

        precompute_diag_phases<Ti, Tv>(new_a, n_new_a, g.unique_zas, g.wa, g.num_za, g.rank, pa_d.data());
        precompute_diag_phases<Ti, Tv>(old_b, n_old_b, g.unique_zbs, g.wb, g.num_zb, g.rank, pb_do.data());
        precompute_diag_phases<Ti, Tv>(new_b, n_new_b, g.unique_zbs, g.wb, g.num_zb, g.rank, pb_dn.data());
        precompute_diag_phases<Ti, Tv>(old_a, n_old_a, g.unique_zas, g.wa, g.num_za, g.rank, pa_do.data());
    }

    std::vector<std::pair<Ti, Ti>> p1, p2, p3;
    select_pass_a<Ti, Tv>(
        new_a, n_new_a, old_b, n_old_b, new_b, n_new_b,
        old_a_idx, old_b_idx,
        basis->num_blocks,
        all_groups, src_psi, basis->blocks,
        pa_d.data(), pb_do.data(), pb_dn.data(),
        diag_rank,
        E_var, eps, p1, p3);
    select_pass_b<Ti, Tv>(
        new_b, n_new_b, old_a, n_old_a,
        old_b_idx, old_a_idx,
        basis->num_blocks,
        all_groups, src_psi, basis->blocks,
        pb_dn.data(), pa_do.data(),
        diag_rank,
        E_var, eps, p2);

    *n_pairs = (int64)(p1.size() + p2.size() + p3.size());
    if (*n_pairs == 0)
    {
        *out_a = nullptr;
        *out_b = nullptr;
        return;
    }

    *out_a = (Ti *)malloc((size_t)(*n_pairs) * sizeof(Ti));
    *out_b = (Ti *)malloc((size_t)(*n_pairs) * sizeof(Ti));

    int64 idx = 0;
    for (const auto &[a, b] : p1)
    {
        (*out_a)[idx] = a;
        (*out_b)[idx] = b;
        ++idx;
    }
    for (const auto &[a, b] : p2)
    {
        (*out_a)[idx] = a;
        (*out_b)[idx] = b;
        ++idx;
    }
    for (const auto &[a, b] : p3)
    {
        (*out_a)[idx] = a;
        (*out_b)[idx] = b;
        ++idx;
    }
}
