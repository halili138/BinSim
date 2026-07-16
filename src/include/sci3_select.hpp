#pragma once
#include "sci_select_test.hpp"
#include <cassert>

template <typename Tv>
struct ForwardShared
{
    int64 ngs;
    int64 num_blocks;
    std::vector<int64> offsets;       // [ngs+1], group totals retained for diagnostics
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

template <typename Ti, typename Tv>
static std::vector<SVDGroup_OTF<Ti, Tv>> flatten_groups(const Network_OTF<Ti, Tv> *net)
{
    std::vector<SVDGroup_OTF<Ti, Tv>> all;
    all.insert(all.end(), net->pure_a_groups.begin(), net->pure_a_groups.end());
    all.insert(all.end(), net->pure_b_groups.begin(), net->pure_b_groups.end());
    all.insert(all.end(), net->mixed_groups.begin(), net->mixed_groups.end());
    return all;
}

template <typename Ti, bool IsAlpha>
static ankerl::unordered_dense::map<Ti, std::pair<int, int>> build_idx_map(const SciBasisManager<Ti> *basis)
{
    ankerl::unordered_dense::map<Ti, std::pair<int, int>> idx_map;
    for (int64 bi = 0; bi < basis->num_blocks; ++bi)
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
        for (int64 j = 0; j < num; ++j)
            idx_map[strs[j]] = {static_cast<int>(j), static_cast<int>(bi)};
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
        for (int64 local_g = 0; local_g < ngs; ++local_g)
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
                link[i].push_back(static_cast<int>(local_g));
        }
    }
    return link;
}

template <typename Ti, typename Tv, bool IsAlpha>
static std::vector<std::vector<int>> build_old2old_link_chunk(
    const Ti *dst_chunk, int64 n_dst_chunk,
    const ankerl::unordered_dense::set<Ti> &old_set,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups, int64 g_begin, int64 g_end)
{
    const int64 ngs = std::max<int64>(0, g_end - g_begin);
    std::vector<std::vector<int>> link(n_dst_chunk);

#pragma omp parallel for schedule(dynamic)
    for (int64 i = 0; i < n_dst_chunk; ++i)
    {
        Ti dst = dst_chunk[i];
        for (int64 local_g = 0; local_g < ngs; ++local_g)
        {
            const auto &group = groups[g_begin + local_g];
            Ti exc;
            if constexpr (IsAlpha)
                exc = group.ax;
            else
                exc = group.bx;
            if (old_set.find(dst ^ exc) != old_set.end())
                link[i].push_back(static_cast<int>(local_g));
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

inline constexpr int64 GROUP_CHUNK_SIZE = 1 << 9;
inline constexpr int64 TARGET_CHUNK_SIZE = 1 << 15;
template <typename Ti, typename Tv>
static void precompute_diag_phases(
    const Ti *strs, int64 num_strs,
    const Ti *zs, const Tv *w, int num_zs, int rank,
    Tv *ps)
{
    for (int64 i = 0; i < num_strs; ++i)
    {
        Ti str = strs[i];
        Tv *pi = ps + i;
        for (int r = 0; r < rank; ++r)
        {
            const Tv *wr = w + r * num_zs;
            Tv vr = {};
            for (int k = 0; k < num_zs; ++k)
            {
                bool parity = std::popcount(str & zs[k]) & 1;
                vr += parity ? -wr[k] : wr[k];
            }
            pi[r * num_strs] = vr;
        }
    }
}

template <typename Tv>
FORCE_INLINE Tv compute_haa(const Tv *a_diag_phase, const Tv *b_diag_phase, int rank, int64 a_stride, int64 b_stride)
{
    Tv haa = {};
    for (int r = 0; r < rank; ++r)
        haa += a_diag_phase[r * a_stride] * b_diag_phase[r * b_stride];
    return haa;
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
    const SVDGroup_OTF<Ti, Tv> *diag_group,
    const Tv *src_psi,
    const BlockDesc<Ti> *src_blocks,
    Tv E_var, Tv eps,
    std::vector<std::pair<Ti, Ti>> &out_p1,
    std::vector<std::pair<Ti, Ti>> &out_p3)
{
    const int64 num_group_chunks = ((int64)all_groups.size() + GROUP_CHUNK_SIZE - 1) / GROUP_CHUNK_SIZE;
    const int64 num_a_chunks = (n_new_α + TARGET_CHUNK_SIZE - 1) / TARGET_CHUNK_SIZE;
    const ankerl::unordered_dense::set<Ti> new_a_set(new_α, new_α + n_new_α);

    auto run_beta_side = [&](const Ti *beta, int64 n_beta,
                             std::vector<std::pair<Ti, Ti>> &out)
    {
        for (int64 b_begin = 0; b_begin < n_beta; b_begin += TARGET_CHUNK_SIZE)
        {
            const int64 b_end = std::min<int64>(b_begin + TARGET_CHUNK_SIZE, n_beta);
            const int64 b_count = b_end - b_begin;
            const int diag_rank = diag_group ? diag_group->rank : 0;
            std::vector<Tv> b_diag_phase(b_count * diag_rank);
            if (diag_rank > 0)
                precompute_diag_phases<Ti, Tv>(beta + b_begin, b_count,
                                               diag_group->unique_zbs, diag_group->wb,
                                               diag_group->num_zb, diag_rank,
                                               b_diag_phase.data());
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
                std::vector<Tv> a_diag_phase(a_count * diag_rank);
                if (diag_rank > 0)
                    precompute_diag_phases<Ti, Tv>(new_α + a_begin, a_count,
                                                   diag_group->unique_zas, diag_group->wa,
                                                   diag_group->num_za, diag_rank,
                                                   a_diag_phase.data());
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
                            const Tv haa = diag_rank > 0
                                ? compute_haa(a_diag_phase.data() + local_ia,
                                              b_diag_phase.data() + local_b, diag_rank, a_count, b_count)
                                : Tv{};
                            if (sci_eps_check(v, haa, E_var, eps))
                                thread_out.emplace_back(dst_a, beta[b_begin + local_b]);
                        }
                    }

#pragma omp critical
                    out.insert(out.end(), std::make_move_iterator(thread_out.begin()), std::make_move_iterator(thread_out.end()));
                }
            }
        }
    };

    run_beta_side(old_β, n_old_β, out_p1);
    run_beta_side(new_β, n_new_β, out_p3);
}

template <typename Ti, typename Tv>
static void select_pass_b(
    const Ti *new_β, int64 n_new_β,
    const Ti *old_α, int64 n_old_α,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_b_idx_map,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_a_idx_map,
    int64 old_a_num_blocks,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups,
    const SVDGroup_OTF<Ti, Tv> *diag_group,
    const Tv *src_psi,
    const BlockDesc<Ti> *src_blocks,
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
        const int diag_rank = diag_group ? diag_group->rank : 0;
        std::vector<Tv> a_diag_phase(a_count * diag_rank);
        if (diag_rank > 0)
            precompute_diag_phases<Ti, Tv>(old_α + a_begin, a_count,
                                           diag_group->unique_zas, diag_group->wa,
                                           diag_group->num_za, diag_rank,
                                           a_diag_phase.data());
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
            std::vector<Tv> b_diag_phase(b_count * diag_rank);
            if (diag_rank > 0)
                precompute_diag_phases<Ti, Tv>(new_β + b_begin, b_count,
                                               diag_group->unique_zbs, diag_group->wb,
                                               diag_group->num_zb, diag_rank,
                                               b_diag_phase.data());
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
                        const Tv haa = diag_rank > 0
                            ? compute_haa(a_diag_phase.data() + local_a,
                                          b_diag_phase.data() + local_ib, diag_rank, a_count, b_count)
                            : Tv{};
                        if (sci_eps_check(v, haa, E_var, eps))
                            thread_p2.emplace_back(old_α[a_begin + local_a], dst_b);
                    }
                }

#pragma omp critical
                out_p2.insert(out_p2.end(), std::make_move_iterator(thread_p2.begin()), std::make_move_iterator(thread_p2.end()));
            }
        }
    }
}
