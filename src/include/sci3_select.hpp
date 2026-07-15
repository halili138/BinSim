#pragma once
#include "sci_select_test.hpp"
#include <limits>

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
    std::vector<Tv> phases;           // flat: 2 doubles per entry (rank-1 padded)

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
static ForwardShared<Tv> precompute_shared(
    const Ti *tgt_strs, int64 n_tgt,
    const ankerl::unordered_dense::map<Ti, std::pair<int, int>> &old_idx_map,
    const std::vector<std::vector<int>> &link,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    bool is_alpha)
{
    int64 ngs = (int64)groups.size();
    ForwardShared<Tv> result;
    result.ngs = ngs;
    result.offsets.assign(ngs + 1, 0);

    int64 num_blocks = 0;
    for (const auto &kv : old_idx_map)
        num_blocks = std::max<int64>(num_blocks, (int64)kv.second.second + 1);
    result.num_blocks = num_blocks;

    // count pass: bucket entries by (group, source block) so lookups can select
    // exactly the compatible block range during the select passes.
    const int64 num_buckets = ngs * num_blocks;
    std::vector<int64> cnts(num_buckets);
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
            ++cnts[(int64)g * num_blocks + it->second.second];
        }
    }

    result.block_offsets.assign(num_buckets + 1, 0);
    for (int64 b = 0; b < num_buckets; ++b)
        result.block_offsets[b + 1] = result.block_offsets[b] + cnts[b];
    for (int64 g = 0; g < ngs; ++g)
        result.offsets[g + 1] = result.block_offsets[(g + 1) * num_blocks];

    int64 total = result.block_offsets[num_buckets];
    result.dst_idxs.assign(total, 0);
    result.src_idxs.assign(total, 0);
    result.src_blk_idxs.assign(total, 0);
    result.phases.assign(total * 2, Tv{});

    // fill pass
    std::vector<int64> pos = result.block_offsets;

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

            int src_blk_idx = it->second.second;
            int64 p = pos[(int64)g * num_blocks + src_blk_idx]++;
            result.dst_idxs[p] = (int)i;
            result.src_idxs[p] = it->second.first;
            result.src_blk_idxs[p] = src_blk_idx;

            Tv *phase_ptr = result.phases.data() + p * 2;
            if (exc == 0)
            {
                // identity: first component = 1, rest = 0
                phase_ptr[0] = Tv(1);
                phase_ptr[1] = Tv{};
            }
            else if (is_alpha)
            {
                precompute_phase_select<Ti, Tv>(
                    src, groups[g].unique_zas, groups[g].num_za, groups[g].wa,
                    phase_ptr, 1, groups[g].rank);
                if (groups[g].rank == 1)
                    phase_ptr[1] = Tv{};
            }
            else
            {
                precompute_phase_select<Ti, Tv>(
                    src, groups[g].unique_zbs, groups[g].num_zb, groups[g].wb,
                    phase_ptr, 1, groups[g].rank);
                if (groups[g].rank == 1)
                    phase_ptr[1] = Tv{};
            }
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

                // ---- Part 1: old_β ----
                {
                    auto [off, end] = shared_b_old.range(ig, src_a_blk_idx);
                    for (int64 j = off; j < end; ++j)
                    {
                        int old_ib = shared_b_old.dst_idxs[j];
                        int src_ib = shared_b_old.src_idxs[j];
                        const Tv *pb = shared_b_old.phases.data() + j * 2;

                        Tv coeff = pa[0] * pb[0];
                        if (group.rank >= 2)
                            coeff += pa[1] * pb[1];

                        const auto &blk = src_blocks[src_a_blk_idx];
                        int64 src_gid = blk.offset + (int64)src_ia * blk.num_b + src_ib;

                        if (mark_old[old_ib] != epoch_old)
                        {
                            mark_old[old_ib] = epoch_old;
                            accum_old[old_ib] = Tv{};
                            touched_old.push_back(old_ib);
                        }
                        accum_old[old_ib] += src_psi[src_gid] * coeff;
                    }
                }

                // ---- Part 3: new_β ----
                {
                    auto [off, end] = shared_b_new.range(ig, src_a_blk_idx);
                    for (int64 j = off; j < end; ++j)
                    {
                        int new_ib = shared_b_new.dst_idxs[j];
                        int src_ib = shared_b_new.src_idxs[j];
                        const Tv *pb = shared_b_new.phases.data() + j * 2;

                        Tv coeff = pa[0] * pb[0];
                        if (group.rank >= 2)
                            coeff += pa[1] * pb[1];

                        const auto &blk = src_blocks[src_a_blk_idx];
                        int64 src_gid = blk.offset + (int64)src_ia * blk.num_b + src_ib;

                        if (mark_new[new_ib] != epoch_new)
                        {
                            mark_new[new_ib] = epoch_new;
                            accum_new[new_ib] = Tv{};
                            touched_new.push_back(new_ib);
                        }
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

                // ---- Part 2: old_α ----
                auto [off, end] = shared_a_old.range(ig, src_b_blk_idx);
                for (int64 j = off; j < end; ++j)
                {
                    int old_ia = shared_a_old.dst_idxs[j];
                    int src_ia = shared_a_old.src_idxs[j];
                    const Tv *pa = shared_a_old.phases.data() + j * 2;

                    Tv coeff = pa[0] * pb[0];
                    if (group.rank >= 2)
                        coeff += pa[1] * pb[1];

                    const auto &blk = src_blocks[src_b_blk_idx];
                    int64 src_gid = blk.offset + (int64)src_ia * blk.num_b + src_ib;

                    if (mark_old[old_ia] != epoch_old)
                    {
                        mark_old[old_ia] = epoch_old;
                        accum_old[old_ia] = Tv{};
                        touched_old.push_back(old_ia);
                    }
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
