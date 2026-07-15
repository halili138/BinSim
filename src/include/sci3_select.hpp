#pragma once
#include "sci_select_test.hpp"

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
        if (group < 0 || group >= ngs || src_blk_idx < 0 || src_blk_idx >= num_blocks)
            return {0, 0};
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

template <typename Ti>
static ankerl::unordered_dense::map<Ti, std::pair<int, int>> build_idx_map(
    const SciBasisManager<Ti> *basis, bool is_alpha)
{
    ankerl::unordered_dense::map<Ti, std::pair<int, int>> idx_map;
    for (int64 bi = 0; bi < basis->num_blocks; ++bi)
    {
        const auto &blk = basis->blocks[bi];
        const Ti *strs = is_alpha ? blk.astrs : blk.bstrs;
        const int64 num = is_alpha ? blk.num_a : blk.num_b;
        for (int64 j = 0; j < num; ++j)
            idx_map[strs[j]] = {static_cast<int>(j), static_cast<int>(bi)};
    }
    return idx_map;
}

template <typename Ti, typename Tv>
static std::vector<std::vector<int>> build_old2new_link(
    const Ti *new_strs, int64 n,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    bool is_alpha)
{
    ankerl::unordered_dense::set<Ti> new_set(new_strs, new_strs + n);
    std::vector<std::vector<int>> link(n);

#pragma omp parallel for schedule(dynamic)
    for (int64 i = 0; i < n; ++i)
    {
        Ti str = new_strs[i];
        for (int g = 0; g < (int)groups.size(); ++g)
        {
            Ti exc = is_alpha ? groups[g].ax : groups[g].bx;
            if (exc == 0)
                continue;
            // include if src (= str ^ exc) is NOT in new_strs
            if (new_set.find(str ^ exc) == new_set.end())
                link[i].push_back(g);
        }
    }
    return link;
}

template <typename Ti, typename Tv>
static std::vector<std::vector<int>> build_old2old_link(
    const Ti *old_strs, int64 n,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    bool is_alpha)
{
    ankerl::unordered_dense::set<Ti> old_set(old_strs, old_strs + n);
    std::vector<std::vector<int>> link(n);

#pragma omp parallel for schedule(dynamic)
    for (int64 i = 0; i < n; ++i)
    {
        Ti str = old_strs[i];
        for (int g = 0; g < (int)groups.size(); ++g)
        {
            Ti exc = is_alpha ? groups[g].ax : groups[g].bx;
            // include if src (= str ^ exc) IS in old_strs
            if (old_set.find(str ^ exc) != old_set.end())
                link[i].push_back(g);
        }
    }
    return link;
}


template <typename Ti, typename Tv>
static std::vector<std::vector<int>> build_old2new_link_chunk(
    const Ti *dst_chunk, int64 n_dst_chunk,
    const ankerl::unordered_dense::set<Ti> &new_set,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    int64 g_begin, int64 g_end,
    bool is_alpha)
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
            Ti exc = is_alpha ? group.ax : group.bx;
            if (exc == 0)
                continue;
            if (new_set.find(dst ^ exc) == new_set.end())
                link[i].push_back(static_cast<int>(local_g));
        }
    }
    return link;
}

template <typename Ti, typename Tv>
static std::vector<std::vector<int>> build_old2old_link_chunk(
    const Ti *dst_chunk, int64 n_dst_chunk,
    const ankerl::unordered_dense::set<Ti> &old_set,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    int64 g_begin, int64 g_end,
    bool is_alpha)
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
            Ti exc = is_alpha ? group.ax : group.bx;
            if (old_set.find(dst ^ exc) != old_set.end())
                link[i].push_back(static_cast<int>(local_g));
        }
    }
    return link;
}


template <typename Tv>
struct SelectPhaseCache
{
    int64 ngs = 0;
    std::vector<std::vector<int>> links; // valid local group ids for each local destination index
    std::vector<int> src_idxs;           // [local dst * ngs + local group]
    std::vector<int> src_blk_idxs;       // [local dst * ngs + local group]
    std::vector<Tv> phase0;              // [local dst * ngs + local group]
    std::vector<Tv> phase1;              // [local dst * ngs + local group] (rank-1 remains zero)

    int64 index(int64 local_i, int64 local_g) const
    {
        return local_i * ngs + local_g;
    }
};

template <typename Ti, typename Tv>
static SelectPhaseCache<Tv> precompute_select_phase_cache_chunk(
    const Ti *dst_chunk, int64 n_dst_chunk,
    const ankerl::unordered_dense::set<Ti> &new_set,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_idx_map,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    int64 g_begin, int64 g_end,
    bool is_alpha)
{
    SelectPhaseCache<Tv> cache;
    cache.ngs = std::max<int64>(0, g_end - g_begin);
    cache.links.resize(n_dst_chunk);

    const int64 total = n_dst_chunk * cache.ngs;
    cache.src_idxs.assign(total, 0);
    cache.src_blk_idxs.assign(total, 0);
    cache.phase0.assign(total, Tv{});
    cache.phase1.assign(total, Tv{});

#pragma omp parallel for schedule(dynamic)
    for (int64 local_i = 0; local_i < n_dst_chunk; ++local_i)
    {
        Ti dst = dst_chunk[local_i];
        for (int64 local_g = 0; local_g < cache.ngs; ++local_g)
        {
            const auto &group = groups[g_begin + local_g];
            Ti exc = is_alpha ? group.ax : group.bx;
            if (exc == 0)
                continue;
            Ti src = dst ^ exc;
            if (new_set.find(src) != new_set.end())
                continue;
            auto it = old_idx_map.find(src);
            if (it == old_idx_map.end())
                continue;

            const int64 idx = cache.index(local_i, local_g);
            cache.src_idxs[idx] = it->second.first;
            cache.src_blk_idxs[idx] = it->second.second;

            Tv phase_tmp[2] = {};
            if (is_alpha)
                precompute_phase_select<Ti, Tv>(src, group.unique_zas, group.num_za, group.wa, phase_tmp, 1, group.rank);
            else
                precompute_phase_select<Ti, Tv>(src, group.unique_zbs, group.num_zb, group.wb, phase_tmp, 1, group.rank);

            cache.phase0[idx] = phase_tmp[0];
            if (group.rank >= 2)
                cache.phase1[idx] = phase_tmp[1];
            cache.links[local_i].push_back(static_cast<int>(local_g));
        }
    }

    return cache;
}

template <typename Ti, typename Tv>
static ForwardShared<Tv> precompute_shared_chunk(
    const Ti *tgt_chunk, int64 n_tgt_chunk,
    int64 tgt_global_offset,
    int64 g_begin, int64 g_end,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_idx_map,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups,
    bool is_alpha)
{
    int64 ngs = std::max<int64>(0, g_end - g_begin);
    ForwardShared<Tv> result;
    result.ngs = ngs;
    result.offsets.assign(ngs + 1, 0);

    int64 num_blocks = 0;
    for (const auto &kv : old_idx_map)
        num_blocks = std::max<int64>(num_blocks, (int64)kv.second.second + 1);
    result.num_blocks = num_blocks;

    const int64 num_buckets = ngs * num_blocks;
    std::vector<int64> cnts(num_buckets);

    for (int64 i = 0; i < n_tgt_chunk; ++i)
    {
        Ti dst = tgt_chunk[i];
        for (int64 local_g = 0; local_g < ngs; ++local_g)
        {
            const auto &group = all_groups[g_begin + local_g];
            Ti exc = is_alpha ? group.ax : group.bx;
            Ti src = dst ^ exc;
            auto it = old_idx_map.find(src);
            if (it == old_idx_map.end())
                continue;
            ++cnts[local_g * num_blocks + it->second.second];
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

    for (int64 i = 0; i < n_tgt_chunk; ++i)
    {
        Ti dst = tgt_chunk[i];
        for (int64 local_g = 0; local_g < ngs; ++local_g)
        {
            const auto &group = all_groups[g_begin + local_g];
            Ti exc = is_alpha ? group.ax : group.bx;
            Ti src = dst ^ exc;
            auto it = old_idx_map.find(src);
            if (it == old_idx_map.end())
                continue;

            int src_blk_idx = it->second.second;
            int64 p = pos[local_g * num_blocks + src_blk_idx]++;
            result.dst_idxs[p] = static_cast<int>(tgt_global_offset + i);
            result.src_idxs[p] = it->second.first;
            result.src_blk_idxs[p] = src_blk_idx;

            Tv phase_tmp[2] = {};
            if (exc == 0)
                phase_tmp[0] = Tv(1);
            else if (is_alpha)
                precompute_phase_select<Ti, Tv>(src, group.unique_zas, group.num_za, group.wa, phase_tmp, 1, group.rank);
            else
                precompute_phase_select<Ti, Tv>(src, group.unique_zbs, group.num_zb, group.wb, phase_tmp, 1, group.rank);

            result.phase0[p] = phase_tmp[0];
            if (group.rank >= 2)
                result.phase1[p] = phase_tmp[1];
        }
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
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups,
    const Tv *src_psi,
    const BlockDesc<Ti> *src_blocks,
    Tv E_var, Tv eps,
    std::vector<std::pair<Ti, Ti>> &out_p1,
    std::vector<std::pair<Ti, Ti>> &out_p3)
{
    Tv eps_sq = eps * eps;
    Tv E_var_sq = E_var * E_var;
    constexpr int64 group_chunk_size = 512;
    constexpr int64 target_chunk_size = 4096;
    const int64 num_group_chunks = ((int64)all_groups.size() + group_chunk_size - 1) / group_chunk_size;
    const int64 num_a_chunks = (n_new_α + target_chunk_size - 1) / target_chunk_size;
    const ankerl::unordered_dense::set<Ti> new_a_set(new_α, new_α + n_new_α);

    auto run_beta_side = [&](const Ti *beta, int64 n_beta,
                             std::vector<std::pair<Ti, Ti>> &out) {
        for (int64 a_chunk = 0; a_chunk < num_a_chunks; ++a_chunk)
        {
            const int64 a_begin = a_chunk * target_chunk_size;
            const int64 a_end = std::min<int64>(a_begin + target_chunk_size, n_new_α);
            const int64 a_count = a_end - a_begin;
            std::vector<SelectPhaseCache<Tv>> phase_chunks;
            phase_chunks.reserve(num_group_chunks);
            for (int64 g_begin = 0; g_begin < (int64)all_groups.size(); g_begin += group_chunk_size)
            {
                int64 g_end = std::min<int64>(g_begin + group_chunk_size, (int64)all_groups.size());
                phase_chunks.push_back(precompute_select_phase_cache_chunk<Ti, Tv>(
                    new_α + a_begin, a_count, new_a_set, old_a_idx_map, all_groups, g_begin, g_end, true));
            }

            for (int64 b_begin = 0; b_begin < n_beta; b_begin += target_chunk_size)
            {
                const int64 b_end = std::min<int64>(b_begin + target_chunk_size, n_beta);
                const int64 b_count = b_end - b_begin;
                std::vector<ForwardShared<Tv>> shared_chunks;
                shared_chunks.reserve(num_group_chunks);
                for (int64 g_begin = 0; g_begin < (int64)all_groups.size(); g_begin += group_chunk_size)
                {
                    int64 g_end = std::min<int64>(g_begin + group_chunk_size, (int64)all_groups.size());
                    shared_chunks.push_back(precompute_shared_chunk<Ti, Tv>(beta + b_begin, b_count, b_begin,
                                                                            g_begin, g_end, old_b_idx_map, all_groups, false));
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

                        for (int64 chunk_id = 0; chunk_id < (int64)phase_chunks.size(); ++chunk_id)
                        {
                            const int64 g_begin = chunk_id * group_chunk_size;
                            const auto &phase_cache = phase_chunks[chunk_id];
                            const auto &shared = shared_chunks[chunk_id];
                            for (int local_g : phase_cache.links[local_ia])
                            {
                                const int64 ig = g_begin + local_g;
                                const auto &group = all_groups[ig];
                                const int64 phase_idx = phase_cache.index(local_ia, local_g);
                                int src_ia = phase_cache.src_idxs[phase_idx];
                                int src_a_blk_idx = phase_cache.src_blk_idxs[phase_idx];

                                const auto &blk = src_blocks[src_a_blk_idx];
                                const int64 row_base = blk.offset + src_ia * blk.num_b;
                                auto [off, end] = shared.range(local_g, src_a_blk_idx);
                                for (int64 j = off; j < end; ++j)
                                {
                                    int beta_idx = shared.dst_idxs[j];
                                    int src_ib = shared.src_idxs[j];
                                    Tv coeff = phase_cache.phase0[phase_idx] * shared.phase0[j];
                                    if (group.rank >= 2)
                                        coeff += phase_cache.phase1[phase_idx] * shared.phase1[j];
                                    accum[beta_idx - b_begin] += src_psi[row_base + src_ib] * coeff;
                                }
                            }
                        }

                        for (int64 local_b = 0; local_b < b_count; ++local_b)
                        {
                            Tv v = accum[local_b];
                            if (v != Tv{} && v * v > E_var_sq * eps_sq)
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
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups,
    const Tv *src_psi,
    const BlockDesc<Ti> *src_blocks,
    Tv E_var, Tv eps,
    std::vector<std::pair<Ti, Ti>> &out_p2)
{
    Tv eps_sq = eps * eps;
    Tv E_var_sq = E_var * E_var;
    constexpr int64 group_chunk_size = 512;
    constexpr int64 target_chunk_size = 4096;
    const int64 num_group_chunks = ((int64)all_groups.size() + group_chunk_size - 1) / group_chunk_size;
    const int64 num_b_chunks = (n_new_β + target_chunk_size - 1) / target_chunk_size;
    const ankerl::unordered_dense::set<Ti> new_b_set(new_β, new_β + n_new_β);

    for (int64 b_chunk = 0; b_chunk < num_b_chunks; ++b_chunk)
    {
        const int64 b_begin = b_chunk * target_chunk_size;
        const int64 b_end = std::min<int64>(b_begin + target_chunk_size, n_new_β);
        const int64 b_count = b_end - b_begin;
        std::vector<SelectPhaseCache<Tv>> phase_chunks;
        phase_chunks.reserve(num_group_chunks);
        for (int64 g_begin = 0; g_begin < (int64)all_groups.size(); g_begin += group_chunk_size)
        {
            int64 g_end = std::min<int64>(g_begin + group_chunk_size, (int64)all_groups.size());
            phase_chunks.push_back(precompute_select_phase_cache_chunk<Ti, Tv>(
                new_β + b_begin, b_count, new_b_set, old_b_idx_map, all_groups, g_begin, g_end, false));
        }

        for (int64 a_begin = 0; a_begin < n_old_α; a_begin += target_chunk_size)
        {
            const int64 a_end = std::min<int64>(a_begin + target_chunk_size, n_old_α);
            const int64 a_count = a_end - a_begin;
            std::vector<ForwardShared<Tv>> shared_chunks;
            shared_chunks.reserve(num_group_chunks);
            for (int64 g_begin = 0; g_begin < (int64)all_groups.size(); g_begin += group_chunk_size)
            {
                int64 g_end = std::min<int64>(g_begin + group_chunk_size, (int64)all_groups.size());
                shared_chunks.push_back(precompute_shared_chunk<Ti, Tv>(old_α + a_begin, a_count, a_begin,
                                                                        g_begin, g_end, old_a_idx_map, all_groups, true));
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

                    for (int64 chunk_id = 0; chunk_id < (int64)phase_chunks.size(); ++chunk_id)
                    {
                        const int64 g_begin = chunk_id * group_chunk_size;
                        const auto &phase_cache = phase_chunks[chunk_id];
                        const auto &shared = shared_chunks[chunk_id];
                        for (int local_g : phase_cache.links[local_ib])
                        {
                            const int64 ig = g_begin + local_g;
                            const auto &group = all_groups[ig];
                            const int64 phase_idx = phase_cache.index(local_ib, local_g);
                            int src_ib = phase_cache.src_idxs[phase_idx];
                            int src_b_blk_idx = phase_cache.src_blk_idxs[phase_idx];

                            const auto &blk = src_blocks[src_b_blk_idx];
                            const int64 col_or_row_base = blk.offset + src_ib;
                            auto [off, end] = shared.range(local_g, src_b_blk_idx);
                            for (int64 j = off; j < end; ++j)
                            {
                                int old_ia = shared.dst_idxs[j];
                                int src_ia = shared.src_idxs[j];
                                Tv coeff = shared.phase0[j] * phase_cache.phase0[phase_idx];
                                if (group.rank >= 2)
                                    coeff += shared.phase1[j] * phase_cache.phase1[phase_idx];
                                accum[old_ia - a_begin] += src_psi[col_or_row_base + src_ia * blk.num_b] * coeff;
                            }
                        }
                    }

                    for (int64 local_a = 0; local_a < a_count; ++local_a)
                    {
                        Tv v = accum[local_a];
                        if (v != Tv{} && v * v > E_var_sq * eps_sq)
                            thread_p2.emplace_back(old_α[a_begin + local_a], dst_b);
                    }
                }

#pragma omp critical
                out_p2.insert(out_p2.end(), std::make_move_iterator(thread_p2.begin()), std::make_move_iterator(thread_p2.end()));
            }
        }
    }
}
