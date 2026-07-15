#pragma once
#include "sci_select_test.hpp"
#include <algorithm>
#include <limits>

template <typename Tv>
struct ForwardShared
{
    int64 ngs;
    int64 num_blocks;
    std::vector<int64> offsets;       // [ngs+1], group totals retained for diagnostics
    std::vector<int64> block_offsets;       // [ngs * num_blocks + 1], combined group totals for diagnostics
    std::vector<int64> rank1_block_offsets; // [ngs * num_blocks + 1], rank-1 entries by (group, src block)
    std::vector<int64> rank2_block_offsets; // [ngs * num_blocks + 1], rank-2 entries by (group, src block)
    std::vector<int> dst_idxs;              // flat: global b index into target array; rank-1 entries precede rank-2 entries
    std::vector<int> src_idxs;              // flat: local b index within source block
    std::vector<int> src_blk_idxs;          // flat: source block index
    std::vector<Tv> phase0;                 // flat: first phase component for every entry
    std::vector<Tv> phase1;                 // flat: second phase component for rank-2 entries only

    std::pair<int64, int64> rank1_range(int64 group, int64 src_blk_idx) const
    {
        if (group < 0 || group >= ngs || src_blk_idx < 0 || src_blk_idx >= num_blocks)
            return {0, 0};
        int64 key = group * num_blocks + src_blk_idx;
        return {rank1_block_offsets[key], rank1_block_offsets[key + 1]};
    }

    std::pair<int64, int64> rank2_range(int64 group, int64 src_blk_idx) const
    {
        if (group < 0 || group >= ngs || src_blk_idx < 0 || src_blk_idx >= num_blocks)
            return {0, 0};
        int64 key = group * num_blocks + src_blk_idx;
        return {rank2_block_offsets[key], rank2_block_offsets[key + 1]};
    }

    int64 rank2_phase_idx(int64 entry_idx) const
    {
        return entry_idx - rank2_block_offsets[0];
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
static ForwardShared<Tv> precompute_shared(
    const Ti *tgt_strs, int64 n_tgt,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_idx_map,
    const std::vector<std::vector<int>> &link,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    bool is_alpha,
    bool sort_buckets_by_src_idx = false)
{
    int64 ngs = (int64)groups.size();
    ForwardShared<Tv> result;
    result.ngs = ngs;
    result.offsets.assign(ngs + 1, 0);

    int64 num_blocks = 0;
    for (const auto &kv : old_idx_map)
        num_blocks = std::max<int64>(num_blocks, (int64)kv.second.second + 1);
    result.num_blocks = num_blocks;

    const int64 num_buckets = ngs * num_blocks;
    std::vector<int64> rank1_cnts(num_buckets);
    std::vector<int64> rank2_cnts(num_buckets);
    for (int64 i = 0; i < n_tgt; ++i)
    {
        Ti dst = tgt_strs[i];
        for (int g : link[i])
        {
            Ti exc = is_alpha ? groups[g].ax : groups[g].bx;
            Ti src = dst ^ exc;
            auto it = old_idx_map.find(src);
            if (it == old_idx_map.end())
                continue;
            int64 key = (int64)g * num_blocks + it->second.second;
            if (groups[g].rank >= 2)
                ++rank2_cnts[key];
            else
                ++rank1_cnts[key];
        }
    }

    result.rank1_block_offsets.assign(num_buckets + 1, 0);
    for (int64 b = 0; b < num_buckets; ++b)
        result.rank1_block_offsets[b + 1] = result.rank1_block_offsets[b] + rank1_cnts[b];

    const int64 rank1_total = result.rank1_block_offsets[num_buckets];
    result.rank2_block_offsets.assign(num_buckets + 1, rank1_total);
    for (int64 b = 0; b < num_buckets; ++b)
        result.rank2_block_offsets[b + 1] = result.rank2_block_offsets[b] + rank2_cnts[b];

    result.block_offsets.assign(num_buckets + 1, 0);
    for (int64 b = 0; b < num_buckets; ++b)
        result.block_offsets[b + 1] = result.block_offsets[b] + rank1_cnts[b] + rank2_cnts[b];
    for (int64 g = 0; g < ngs; ++g)
        result.offsets[g + 1] = result.block_offsets[(g + 1) * num_blocks];

    const int64 total = result.rank2_block_offsets[num_buckets];
    const int64 rank2_total = total - rank1_total;
    result.dst_idxs.assign(total, 0);
    result.src_idxs.assign(total, 0);
    result.src_blk_idxs.assign(total, 0);
    result.phase0.assign(total, Tv{});
    result.phase1.assign(rank2_total, Tv{});

    std::vector<int64> rank1_pos = result.rank1_block_offsets;
    std::vector<int64> rank2_pos = result.rank2_block_offsets;

    for (int64 i = 0; i < n_tgt; ++i)
    {
        Ti dst = tgt_strs[i];
        for (int g : link[i])
        {
            Ti exc = is_alpha ? groups[g].ax : groups[g].bx;
            Ti src = dst ^ exc;
            auto it = old_idx_map.find(src);
            if (it == old_idx_map.end())
                continue;

            const bool is_rank2 = groups[g].rank >= 2;
            int src_blk_idx = it->second.second;
            int64 key = (int64)g * num_blocks + src_blk_idx;
            int64 p = is_rank2 ? rank2_pos[key]++ : rank1_pos[key]++;
            result.dst_idxs[p] = (int)i;
            result.src_idxs[p] = it->second.first;
            result.src_blk_idxs[p] = src_blk_idx;

            Tv phase[2] = {Tv{}, Tv{}};
            if (exc == 0)
                phase[0] = Tv(1);
            else if (is_alpha)
                precompute_phase_select<Ti, Tv>(src, groups[g].unique_zas, groups[g].num_za, groups[g].wa, phase, 1, groups[g].rank);
            else
                precompute_phase_select<Ti, Tv>(src, groups[g].unique_zbs, groups[g].num_zb, groups[g].wb, phase, 1, groups[g].rank);

            result.phase0[p] = phase[0];
            if (is_rank2)
                result.phase1[p - rank1_total] = phase[1];
        }
    }

    if (sort_buckets_by_src_idx)
    {
        struct SharedEntry { int dst_idx; int src_idx; int src_blk_idx; Tv phase0; Tv phase1; };
        std::vector<SharedEntry> bucket_entries;
        auto sort_range = [&](int64 off, int64 end, bool is_rank2) {
            if (end - off <= 1)
                return;
            bucket_entries.clear();
            bucket_entries.reserve(static_cast<size_t>(end - off));
            for (int64 p = off; p < end; ++p)
                bucket_entries.push_back({result.dst_idxs[p], result.src_idxs[p], result.src_blk_idxs[p], result.phase0[p], is_rank2 ? result.phase1[p - rank1_total] : Tv{}});
            std::stable_sort(bucket_entries.begin(), bucket_entries.end(), [](const SharedEntry &lhs, const SharedEntry &rhs) {
                if (lhs.src_idx != rhs.src_idx)
                    return lhs.src_idx < rhs.src_idx;
                return lhs.dst_idx < rhs.dst_idx;
            });
            for (int64 p = off; p < end; ++p)
            {
                const auto &entry = bucket_entries[static_cast<size_t>(p - off)];
                result.dst_idxs[p] = entry.dst_idx;
                result.src_idxs[p] = entry.src_idx;
                result.src_blk_idxs[p] = entry.src_blk_idx;
                result.phase0[p] = entry.phase0;
                if (is_rank2)
                    result.phase1[p - rank1_total] = entry.phase1;
            }
        };
        for (int64 bucket = 0; bucket < num_buckets; ++bucket)
        {
            sort_range(result.rank1_block_offsets[bucket], result.rank1_block_offsets[bucket + 1], false);
            sort_range(result.rank2_block_offsets[bucket], result.rank2_block_offsets[bucket + 1], true);
        }
    }

    return result;
}

template <typename Ti, typename Tv>
static void select_pass_a(
    const Ti *new_α, int64 n_new_α,
    const Ti *old_β, int64 n_old_β,
    const Ti *new_β, int64 n_new_β,
    const std::vector<std::vector<int>> &alink,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_a_idx_map,
    const ForwardShared<Tv> &shared_b_old,
    const ForwardShared<Tv> &shared_b_new,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups,
    const Tv *src_psi,
    const BlockDesc<Ti> *src_blocks,
    Tv E_var, Tv eps,
    std::vector<std::pair<Ti, Ti>> &out_p1,
    std::vector<std::pair<Ti, Ti>> &out_p3)
{
    Tv eps_sq = eps * eps;
    Tv E_var_sq = E_var * E_var;

#pragma omp parallel
    {
        std::vector<Tv> accum_old(n_old_β);
        std::vector<Tv> accum_new(n_new_β);
        std::vector<int> mark_old(n_old_β, 0);
        std::vector<int> mark_new(n_new_β, 0);
        std::vector<int> touched_old;
        std::vector<int> touched_new;
        int epoch_old = 1;
        int epoch_new = 1;
        std::vector<std::pair<Ti, Ti>> thread_p1, thread_p3;

#pragma omp for schedule(dynamic)
        for (int64 ia = 0; ia < n_new_α; ++ia)
        {
            Ti dst_a = new_α[ia];

            touched_old.clear();
            touched_new.clear();

            for (int ig : alink[ia])
            {
                const auto &group = all_groups[ig];
                Ti src_a = dst_a ^ group.ax;
                auto it = old_a_idx_map.find(src_a);
                if (it == old_a_idx_map.end())
                    continue;
                int src_ia = it->second.first;
                int src_a_blk_idx = it->second.second;

                // fresh alpha phase
                Tv pa[2] = {};
                precompute_phase_select<Ti, Tv>(
                    src_a, group.unique_zas, group.num_za, group.wa,
                    pa, 1, group.rank);
                if (group.rank == 1)
                    pa[1] = Tv{};

                const auto &blk = src_blocks[src_a_blk_idx];
                const int64 row_base = blk.offset + static_cast<int64>(src_ia) * blk.num_b;

                // ---- Part 1: old_β ----
                {
                    auto [r1_off, r1_end] = shared_b_old.rank1_range(ig, src_a_blk_idx);
                    auto [r2_off, r2_end] = shared_b_old.rank2_range(ig, src_a_blk_idx);
                    for (int64 j = r1_off; j < r1_end; ++j)
                    {
                        int old_ib = shared_b_old.dst_idxs[j];
                        if (mark_old[old_ib] != epoch_old)
                        {
                            mark_old[old_ib] = epoch_old;
                            accum_old[old_ib] = Tv{};
                            touched_old.push_back(old_ib);
                        }
                    }
                    for (int64 j = r2_off; j < r2_end; ++j)
                    {
                        int old_ib = shared_b_old.dst_idxs[j];
                        if (mark_old[old_ib] != epoch_old)
                        {
                            mark_old[old_ib] = epoch_old;
                            accum_old[old_ib] = Tv{};
                            touched_old.push_back(old_ib);
                        }
                    }

#pragma omp simd
                    for (int64 j = r1_off; j < r1_end; ++j)
                    {
                        int old_ib = shared_b_old.dst_idxs[j];
                        int src_ib = shared_b_old.src_idxs[j];
                        Tv coeff = pa[0] * shared_b_old.phase0[j];
                        int64 src_gid = row_base + src_ib;
                        accum_old[old_ib] += src_psi[src_gid] * coeff;
                    }
#pragma omp simd
                    for (int64 j = r2_off; j < r2_end; ++j)
                    {
                        int old_ib = shared_b_old.dst_idxs[j];
                        int src_ib = shared_b_old.src_idxs[j];
                        Tv coeff = pa[0] * shared_b_old.phase0[j] + pa[1] * shared_b_old.phase1[shared_b_old.rank2_phase_idx(j)];
                        int64 src_gid = row_base + src_ib;
                        accum_old[old_ib] += src_psi[src_gid] * coeff;
                    }
                }

                // ---- Part 3: new_β ----
                {
                    auto [r1_off, r1_end] = shared_b_new.rank1_range(ig, src_a_blk_idx);
                    auto [r2_off, r2_end] = shared_b_new.rank2_range(ig, src_a_blk_idx);
                    for (int64 j = r1_off; j < r1_end; ++j)
                    {
                        int new_ib = shared_b_new.dst_idxs[j];
                        if (mark_new[new_ib] != epoch_new)
                        {
                            mark_new[new_ib] = epoch_new;
                            accum_new[new_ib] = Tv{};
                            touched_new.push_back(new_ib);
                        }
                    }
                    for (int64 j = r2_off; j < r2_end; ++j)
                    {
                        int new_ib = shared_b_new.dst_idxs[j];
                        if (mark_new[new_ib] != epoch_new)
                        {
                            mark_new[new_ib] = epoch_new;
                            accum_new[new_ib] = Tv{};
                            touched_new.push_back(new_ib);
                        }
                    }

#pragma omp simd
                    for (int64 j = r1_off; j < r1_end; ++j)
                    {
                        int new_ib = shared_b_new.dst_idxs[j];
                        int src_ib = shared_b_new.src_idxs[j];
                        Tv coeff = pa[0] * shared_b_new.phase0[j];
                        int64 src_gid = row_base + src_ib;
                        accum_new[new_ib] += src_psi[src_gid] * coeff;
                    }
#pragma omp simd
                    for (int64 j = r2_off; j < r2_end; ++j)
                    {
                        int new_ib = shared_b_new.dst_idxs[j];
                        int src_ib = shared_b_new.src_idxs[j];
                        Tv coeff = pa[0] * shared_b_new.phase0[j] + pa[1] * shared_b_new.phase1[shared_b_new.rank2_phase_idx(j)];
                        int64 src_gid = row_base + src_ib;
                        accum_new[new_ib] += src_psi[src_gid] * coeff;
                    }
                }
            }

            // eps_check old_β → P1
            for (int ib : touched_old)
            {
                Tv v = accum_old[ib];
                if (v == Tv{})
                    continue;
                if (v * v > E_var_sq * eps_sq)
                    thread_p1.emplace_back(dst_a, old_β[ib]);
            }
            // eps_check new_β → P3
            for (int ib : touched_new)
            {
                Tv v = accum_new[ib];
                if (v == Tv{})
                    continue;
                if (v * v > E_var_sq * eps_sq)
                    thread_p3.emplace_back(dst_a, new_β[ib]);
            }

            if (epoch_old == std::numeric_limits<int>::max())
            {
                std::fill(mark_old.begin(), mark_old.end(), 0);
                epoch_old = 1;
            }
            else
                ++epoch_old;
            if (epoch_new == std::numeric_limits<int>::max())
            {
                std::fill(mark_new.begin(), mark_new.end(), 0);
                epoch_new = 1;
            }
            else
                ++epoch_new;
        }

#pragma omp critical
        {
            out_p1.insert(out_p1.end(),
                          std::make_move_iterator(thread_p1.begin()),
                          std::make_move_iterator(thread_p1.end()));
            out_p3.insert(out_p3.end(),
                          std::make_move_iterator(thread_p3.begin()),
                          std::make_move_iterator(thread_p3.end()));
        }
    }
}

template <typename Ti, typename Tv>
static void select_pass_b(
    const Ti *new_β, int64 n_new_β,
    const Ti *old_α, int64 n_old_α,
    const std::vector<std::vector<int>> &blink,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_b_idx_map,
    const ForwardShared<Tv> &shared_a_old,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups,
    const Tv *src_psi,
    const BlockDesc<Ti> *src_blocks,
    Tv E_var, Tv eps,
    std::vector<std::pair<Ti, Ti>> &out_p2)
{
    Tv eps_sq = eps * eps;
    Tv E_var_sq = E_var * E_var;

#pragma omp parallel
    {
        std::vector<Tv> accum_old(n_old_α);
        std::vector<int> mark_old(n_old_α, 0);
        std::vector<int> touched_old;
        int epoch_old = 1;
        std::vector<std::pair<Ti, Ti>> thread_p2;

#pragma omp for schedule(dynamic)
        for (int64 ib = 0; ib < n_new_β; ++ib)
        {
            Ti dst_b = new_β[ib];

            touched_old.clear();

            for (int ig : blink[ib])
            {
                const auto &group = all_groups[ig];
                Ti src_b = dst_b ^ group.bx;
                auto it = old_b_idx_map.find(src_b);
                if (it == old_b_idx_map.end())
                    continue;
                int src_ib = it->second.first;
                int src_b_blk_idx = it->second.second;

                // fresh beta phase
                Tv pb[2] = {};
                precompute_phase_select<Ti, Tv>(
                    src_b, group.unique_zbs, group.num_zb, group.wb,
                    pb, 1, group.rank);
                if (group.rank == 1)
                    pb[1] = Tv{};

                const auto &blk = src_blocks[src_b_blk_idx];
                const int64 col_or_row_base = blk.offset + src_ib;

                // ---- Part 2: old_α ----
                auto [r1_off, r1_end] = shared_a_old.rank1_range(ig, src_b_blk_idx);
                auto [r2_off, r2_end] = shared_a_old.rank2_range(ig, src_b_blk_idx);
                for (int64 j = r1_off; j < r1_end; ++j)
                {
                    int old_ia = shared_a_old.dst_idxs[j];
                    if (mark_old[old_ia] != epoch_old)
                    {
                        mark_old[old_ia] = epoch_old;
                        accum_old[old_ia] = Tv{};
                        touched_old.push_back(old_ia);
                    }
                }
                for (int64 j = r2_off; j < r2_end; ++j)
                {
                    int old_ia = shared_a_old.dst_idxs[j];
                    if (mark_old[old_ia] != epoch_old)
                    {
                        mark_old[old_ia] = epoch_old;
                        accum_old[old_ia] = Tv{};
                        touched_old.push_back(old_ia);
                    }
                }

                // The scalar mark pass above keeps touched_old updates ordered and
                // separate from the arithmetic loop. For one group/bucket, old_ia is
                // unique because dst -> (dst ^ excitation) is one-to-one, so the
                // scatter accumulation below has no loop-carried dependency and can
                // be safely vectorized.
#pragma omp simd
                for (int64 j = r1_off; j < r1_end; ++j)
                {
                    int old_ia = shared_a_old.dst_idxs[j];
                    int src_ia = shared_a_old.src_idxs[j];
                    Tv coeff = shared_a_old.phase0[j] * pb[0];
                    int64 src_gid = col_or_row_base + static_cast<int64>(src_ia) * blk.num_b;
                    accum_old[old_ia] += src_psi[src_gid] * coeff;
                }
#pragma omp simd
                for (int64 j = r2_off; j < r2_end; ++j)
                {
                    int old_ia = shared_a_old.dst_idxs[j];
                    int src_ia = shared_a_old.src_idxs[j];
                    Tv coeff = shared_a_old.phase0[j] * pb[0] + shared_a_old.phase1[shared_a_old.rank2_phase_idx(j)] * pb[1];
                    int64 src_gid = col_or_row_base + static_cast<int64>(src_ia) * blk.num_b;
                    accum_old[old_ia] += src_psi[src_gid] * coeff;
                }
            }

            // eps_check old_α → P2
            for (int ia : touched_old)
            {
                Tv v = accum_old[ia];
                if (v == Tv{})
                    continue;
                if (v * v > E_var_sq * eps_sq)
                    thread_p2.emplace_back(old_α[ia], dst_b);
            }

            if (epoch_old == std::numeric_limits<int>::max())
            {
                std::fill(mark_old.begin(), mark_old.end(), 0);
                epoch_old = 1;
            }
            else
                ++epoch_old;
        }

#pragma omp critical
        {
            out_p2.insert(out_p2.end(),
                          std::make_move_iterator(thread_p2.begin()),
                          std::make_move_iterator(thread_p2.end()));
        }
    }
}
