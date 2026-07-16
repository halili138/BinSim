#pragma once
#include "sci_utils.hpp"

template <typename Tv>
struct ForwardSharedNosym
{
    int64 ngs;
    std::vector<int64> offsets; // [ngs+1]
    std::vector<int> dst_idxs;  // flat: global b index into target array
    std::vector<int> src_idxs;  // flat: local b index
    std::vector<Tv> phase0;     // flat
    std::vector<Tv> phase1;     // flat (rank-1 unused)

    std::pair<int64, int64> range(int64 group) const
    {
        assert(group >= 0 && group < ngs);
        return {offsets[group], offsets[group + 1]};
    }
};

template <typename Ti, bool IsAlpha>
static ankerl::unordered_dense::map<Ti, int> build_idx_map_nosym(const SciBasisManagerNosym<Ti> *basis)
{
    ankerl::unordered_dense::map<Ti, int> idx_map;
    const Ti *strs;
    int64 n;
    if constexpr (IsAlpha)
    {
        strs = basis->all_astrs;
        n = basis->num_a;
    }
    else
    {
        strs = basis->all_bstrs;
        n = basis->num_b;
    }
    for (int64 i = 0; i < n; ++i)
        idx_map[strs[i]] = (int)i;
    return idx_map;
}

template <typename Ti, typename Tv, bool IsAlpha>
static std::vector<std::vector<int>> build_old2new_link_chunk_nosym(
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
                link[i].push_back((int)local_g);
        }
    }
    return link;
}

template <typename Ti, typename Tv, bool IsAlpha>
static ForwardSharedNosym<Tv> precompute_shared_chunk_nosym(
    const Ti *tgt_chunk, int64 n_tgt_chunk, int64 tgt_global_offset,
    int64 g_begin, int64 g_end,
    const ankerl::unordered_dense::map<Ti, int> &old_idx_map,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups)
{
    int64 ngs = std::max<int64>(0, g_end - g_begin);
    ForwardSharedNosym<Tv> result;
    result.ngs = ngs;
    result.offsets.assign(ngs + 1, 0);

    std::vector<int64> cnts(ngs);

    struct Hit
    {
        int64 local_g;
        int dst_idx;
        int src_idx;
        Tv phase0;
        Tv phase1;
    };
    std::vector<Hit> hits;

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

            Tv ph[2] = {};
            if (exc == 0)
            {
                ph[0] = Tv(1);
            }
            else if constexpr (IsAlpha)
            {
                precompute_phase_select<Ti, Tv>(src, group.unique_zas, group.num_za, group.wa, ph, 1, group.rank);
            }
            else
            {
                precompute_phase_select<Ti, Tv>(src, group.unique_zbs, group.num_zb, group.wb, ph, 1, group.rank);
            }

            hits.push_back({local_g,
                            (int)(tgt_global_offset + i),
                            it->second,
                            ph[0],
                            group.rank >= 2 ? ph[1] : Tv{}});
            ++cnts[local_g];
        }
    }

    for (int64 g = 0; g < ngs; ++g)
        result.offsets[g + 1] = result.offsets[g] + cnts[g];

    int64 total = result.offsets[ngs];
    result.dst_idxs.assign(total, 0);
    result.src_idxs.assign(total, 0);
    result.phase0.assign(total, Tv{});
    result.phase1.assign(total, Tv{});

    std::vector<int64> pos = result.offsets;
    for (const auto &hit : hits)
    {
        int64 p = pos[hit.local_g]++;
        result.dst_idxs[p] = hit.dst_idx;
        result.src_idxs[p] = hit.src_idx;
        result.phase0[p] = hit.phase0;
        result.phase1[p] = hit.phase1;
    }
    return result;
}

template <typename Ti, typename Tv>
static void select_pass_a_nosym(
    const Ti *new_α, int64 n_new_α,
    const Ti *old_β, int64 n_old_β,
    const Ti *new_β, int64 n_new_β,
    const ankerl::unordered_dense::map<Ti, int> &old_a_idx_map,
    const ankerl::unordered_dense::map<Ti, int> &old_b_idx_map,
    int64 num_b,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups,
    const Tv *src_psi,
    const Tv *pa_diag, const Tv *pb_diag_old, const Tv *pb_diag_new, int diag_rank,
    Tv E_var, Tv eps,
    std::vector<std::pair<Ti, Ti>> &out_p1,
    std::vector<std::pair<Ti, Ti>> &out_p3)
{
    const int64 num_group_chunks = ((int64)all_groups.size() + GROUP_CHUNK_SIZE - 1) / GROUP_CHUNK_SIZE;
    const int64 num_a_chunks = (n_new_α + TARGET_CHUNK_SIZE - 1) / TARGET_CHUNK_SIZE;
    const ankerl::unordered_dense::set<Ti> new_a_set(new_α, new_α + n_new_α);

    auto run_beta_side = [&](const Ti *beta, int64 n_beta, const Tv *pb_diag,
                             std::vector<std::pair<Ti, Ti>> &out)
    {
        for (int64 b_begin = 0; b_begin < n_beta; b_begin += TARGET_CHUNK_SIZE)
        {
            const int64 b_end = std::min<int64>(b_begin + TARGET_CHUNK_SIZE, n_beta);
            const int64 b_count = b_end - b_begin;
            std::vector<ForwardSharedNosym<Tv>> shared_chunks;
            shared_chunks.reserve(num_group_chunks);
            for (int64 g_begin = 0; g_begin < (int64)all_groups.size(); g_begin += GROUP_CHUNK_SIZE)
            {
                int64 g_end = std::min<int64>(g_begin + GROUP_CHUNK_SIZE, (int64)all_groups.size());
                shared_chunks.push_back(precompute_shared_chunk_nosym<Ti, Tv, false>(
                    beta + b_begin, b_count, b_begin, g_begin, g_end, old_b_idx_map, all_groups));
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
                    link_chunks.push_back(build_old2new_link_chunk_nosym<Ti, Tv, true>(
                        new_α + a_begin, a_count, new_a_set, all_groups, g_begin, g_end));
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
                                int src_ia = it->second;

                                Tv pa[2] = {};
                                precompute_phase_select<Ti, Tv>(src_a, group.unique_zas, group.num_za, group.wa, pa, 1, group.rank);
                                if (group.rank == 1)
                                    pa[1] = Tv{};

                                const int64 row_base = (int64)src_ia * num_b;
                                auto [off, end] = shared.range(local_g);
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
                    out.insert(out.end(), std::make_move_iterator(thread_out.begin()),
                               std::make_move_iterator(thread_out.end()));
                }
            }
        }
    };

    run_beta_side(old_β, n_old_β, pb_diag_old, out_p1);
    run_beta_side(new_β, n_new_β, pb_diag_new, out_p3);
}

template <typename Ti, typename Tv>
static void select_pass_b_nosym(
    const Ti *new_β, int64 n_new_β,
    const Ti *old_α, int64 n_old_α,
    const ankerl::unordered_dense::map<Ti, int> &old_b_idx_map,
    const ankerl::unordered_dense::map<Ti, int> &old_a_idx_map,
    int64 num_b,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &all_groups,
    const Tv *src_psi,
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
        std::vector<ForwardSharedNosym<Tv>> shared_chunks;
        shared_chunks.reserve(num_group_chunks);
        for (int64 g_begin = 0; g_begin < (int64)all_groups.size(); g_begin += GROUP_CHUNK_SIZE)
        {
            int64 g_end = std::min<int64>(g_begin + GROUP_CHUNK_SIZE, (int64)all_groups.size());
            shared_chunks.push_back(precompute_shared_chunk_nosym<Ti, Tv, true>(
                old_α + a_begin, a_count, a_begin, g_begin, g_end, old_a_idx_map, all_groups));
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
                link_chunks.push_back(build_old2new_link_chunk_nosym<Ti, Tv, false>(
                    new_β + b_begin, b_count, new_b_set, all_groups, g_begin, g_end));
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
                            int src_ib = it->second;

                            Tv pb[2] = {};
                            precompute_phase_select<Ti, Tv>(src_b, group.unique_zbs, group.num_zb, group.wb, pb, 1, group.rank);
                            if (group.rank == 1)
                                pb[1] = Tv{};

                            auto [off, end] = shared.range(local_g);
                            for (int64 j = off; j < end; ++j)
                            {
                                int old_ia = shared.dst_idxs[j];
                                int src_ia = shared.src_idxs[j];
                                Tv coeff = shared.phase0[j] * pb[0];
                                if (group.rank >= 2)
                                    coeff += shared.phase1[j] * pb[1];
                                accum[old_ia - a_begin] += src_psi[src_ia * num_b + src_ib] * coeff;
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
                out_p2.insert(out_p2.end(), std::make_move_iterator(thread_p2.begin()),
                              std::make_move_iterator(thread_p2.end()));
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static void gather_diag_nosym(
    const Ti *astrs, int num_a, const Ti *bstrs, int num_b,
    const ankerl::unordered_dense::map<Ti, int> &a_idx_map,
    const ankerl::unordered_dense::map<Ti, int> &b_idx_map,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups,
    const Tv *src_vec, Tv *dst_vec)
{
    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;
    const int shift = num_b * MAX_RANK;

    std::vector<int> src_a_idxs(num_a), src_b_idxs(num_b);
    for (int a = 0; a < num_a; ++a)
    {
        auto it = a_idx_map.find(astrs[a]);
        src_a_idxs[a] = (it != a_idx_map.end()) ? it->second : -1;
    }
    for (int b = 0; b < num_b; ++b)
    {
        auto it = b_idx_map.find(bstrs[b]);
        src_b_idxs[b] = (it != b_idx_map.end()) ? it->second : -1;
    }

    std::vector<Tv> phase_b(BATCH_SIZE * shift);

#pragma omp parallel
    for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
    {
        const int64 cur = std::min<int64>(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
        for (int64 bi = 0; bi < cur; ++bi)
        {
            const auto &g = groups[batch_start + bi];
            Tv *pb0 = phase_b.data() + bi * shift;
            for (int b = 0; b < num_b; ++b)
                precompute_phase<Rank, Ti, Tv>(bstrs[b], g.unique_zbs, g.num_zb, g.wb, pb0 + b, num_b, g.rank);
        }

#pragma omp for schedule(dynamic)
        for (int a = 0; a < num_a; ++a)
        {
            if (src_a_idxs[a] == -1)
                continue;
            const Tv *sa = src_vec + (int64)src_a_idxs[a] * num_b;
            Tv *da = dst_vec + (int64)a * num_b;
            for (int64 bi = 0; bi < cur; ++bi)
            {
                const auto &g = groups[batch_start + bi];
                Tv pa[MAX_RANK] = {};
                precompute_phase<Rank, Ti, Tv>(astrs[a], g.unique_zas, g.num_za, g.wa, pa, 1, g.rank);
                const Tv *pb = phase_b.data() + bi * shift;
                const int rank = g.rank;
#pragma omp simd
                for (int b = 0; b < num_b; ++b)
                {
                    if (src_b_idxs[b] == -1)
                        continue;
                    Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, num_b, rank);
                    hvec_update<Tv>(sa + src_b_idxs[b], da + b, vt);
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static void gather_pure_a_nosym(
    const Ti *astrs, int num_a, const Ti *bstrs, int num_b,
    const ankerl::unordered_dense::map<Ti, int> &a_idx_map,
    const ankerl::unordered_dense::map<Ti, int> &b_idx_map,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups,
    const Tv *src_vec, Tv *dst_vec)
{
    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;
    const int shift = num_b * MAX_RANK;

    std::vector<int> src_b_idxs(num_b);
    for (int b = 0; b < num_b; ++b)
    {
        auto it = b_idx_map.find(bstrs[b]);
        src_b_idxs[b] = (it != b_idx_map.end()) ? it->second : -1;
    }

    std::vector<Tv> phase_b(BATCH_SIZE * shift);
    std::vector<int> src_a_idxs(BATCH_SIZE * num_a);
    std::vector<Tv> phase_a(BATCH_SIZE * num_a * MAX_RANK);

#pragma omp parallel
    for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
    {
        const int64 cur = std::min<int64>(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
        for (int64 bi = 0; bi < cur; ++bi)
        {
            const auto &g = groups[batch_start + bi];
            Tv *pb0 = phase_b.data() + bi * shift;
            for (int b = 0; b < num_b; ++b)
                precompute_phase<Rank, Ti, Tv>(bstrs[b], g.unique_zbs, g.num_zb, g.wb, pb0 + b, num_b, g.rank);

            int *sa_ptr = src_a_idxs.data() + bi * num_a;
            Tv *pa_ptr = phase_a.data() + bi * num_a * MAX_RANK;
            for (int a = 0; a < num_a; ++a)
            {
                Ti sa = astrs[a] ^ g.ax;
                auto it = a_idx_map.find(sa);
                if (it != a_idx_map.end())
                {
                    sa_ptr[a] = it->second;
                    precompute_phase<Rank, Ti, Tv>(sa, g.unique_zas, g.num_za, g.wa, pa_ptr + a * MAX_RANK, 1, g.rank);
                }
                else
                    sa_ptr[a] = -1;
            }
        }

#pragma omp for schedule(dynamic)
        for (int a = 0; a < num_a; ++a)
        {
            Tv *da = dst_vec + (int64)a * num_b;
            for (int64 bi = 0; bi < cur; ++bi)
            {
                int src_a_idx = src_a_idxs[bi * num_a + a];
                if (src_a_idx == -1)
                    continue;
                const auto &g = groups[batch_start + bi];
                const Tv *pa = phase_a.data() + bi * num_a * MAX_RANK + a * MAX_RANK;
                const Tv *pb = phase_b.data() + bi * shift;
                const Tv *sa = src_vec + (int64)src_a_idx * num_b;
                const int rank = g.rank;
#pragma omp simd
                for (int b = 0; b < num_b; ++b)
                {
                    if (src_b_idxs[b] == -1)
                        continue;
                    Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, num_b, rank);
                    hvec_update<Tv>(sa + src_b_idxs[b], da + b, vt);
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static void gather_pure_b_nosym(
    const Ti *astrs, int num_a, const Ti *bstrs, int num_b,
    const ankerl::unordered_dense::map<Ti, int> &a_idx_map,
    const ankerl::unordered_dense::map<Ti, int> &b_idx_map,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups,
    const Tv *src_vec, Tv *dst_vec)
{
    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;
    const int shift = num_b * MAX_RANK;

    std::vector<int> src_a_idxs(num_a);
    for (int a = 0; a < num_a; ++a)
    {
        auto it = a_idx_map.find(astrs[a]);
        src_a_idxs[a] = (it != a_idx_map.end()) ? it->second : -1;
    }

    std::vector<int> src_b_idxs_v(BATCH_SIZE * num_b), dst_b_idxs(BATCH_SIZE * num_b);
    std::vector<Tv> batch_phase(BATCH_SIZE * shift);
    std::vector<int> valid_b_counts(BATCH_SIZE);

#pragma omp parallel
    for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
    {
        const int64 cur = std::min<int64>(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
        for (int64 bi = 0; bi < cur; ++bi)
        {
            const auto &g = groups[batch_start + bi];
            Tv *pb0 = batch_phase.data() + bi * shift;
            int *sb_ptr = src_b_idxs_v.data() + bi * num_b;
            int *db_ptr = dst_b_idxs.data() + bi * num_b;
            int count = 0;
            for (int b = 0; b < num_b; ++b)
            {
                Ti sb = bstrs[b] ^ g.bx;
                auto it = b_idx_map.find(sb);
                if (it == b_idx_map.end())
                    continue;
                sb_ptr[count] = it->second;
                db_ptr[count] = b;
                precompute_phase<Rank, Ti, Tv>(sb, g.unique_zbs, g.num_zb, g.wb, pb0 + count, num_b, g.rank);
                count++;
            }
            valid_b_counts[bi] = count;
        }

#pragma omp for schedule(dynamic)
        for (int a = 0; a < num_a; ++a)
        {
            if (src_a_idxs[a] == -1)
                continue;
            Tv *da = dst_vec + (int64)a * num_b;
            for (int64 bi = 0; bi < cur; ++bi)
            {
                int vc = valid_b_counts[bi];
                if (vc == 0)
                    continue;
                const auto &g = groups[batch_start + bi];
                Tv pa[MAX_RANK] = {};
                precompute_phase<Rank, Ti, Tv>(astrs[a], g.unique_zas, g.num_za, g.wa, pa, 1, g.rank);
                const Tv *pb = batch_phase.data() + bi * shift;
                const int *si = src_b_idxs_v.data() + bi * num_b;
                const int *di = dst_b_idxs.data() + bi * num_b;
                const Tv *sa = src_vec + (int64)src_a_idxs[a] * num_b;
                const int rank = g.rank;
#pragma omp simd
                for (int b = 0; b < vc; ++b)
                {
                    Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, num_b, rank);
                    hvec_update<Tv>(sa + si[b], da + di[b], vt);
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static void gather_mixed_nosym(
    const Ti *astrs, int num_a, const Ti *bstrs, int num_b,
    const ankerl::unordered_dense::map<Ti, int> &a_idx_map,
    const ankerl::unordered_dense::map<Ti, int> &b_idx_map,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups,
    const Tv *src_vec, Tv *dst_vec)
{
    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE1 : (Rank == 2 ? BATCH_SIZE2 : BATCH_SIZE3);
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;
    const int shift = num_b * MAX_RANK;

    std::vector<int> src_b_idxs_v(BATCH_SIZE * num_b), dst_b_idxs(BATCH_SIZE * num_b);
    std::vector<Tv> batch_phase(BATCH_SIZE * shift);
    std::vector<int> valid_b_counts(BATCH_SIZE);
    std::vector<int> src_a_idxs(BATCH_SIZE * num_a);
    std::vector<Tv> phase_a(BATCH_SIZE * num_a * MAX_RANK);

#pragma omp parallel
    for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
    {
        const int64 cur = std::min<int64>(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
        for (int64 bi = 0; bi < cur; ++bi)
        {
            const auto &g = groups[batch_start + bi];
            Tv *pb0 = batch_phase.data() + bi * shift;
            int *sb_ptr = src_b_idxs_v.data() + bi * num_b;
            int *db_ptr = dst_b_idxs.data() + bi * num_b;
            int count = 0;
            for (int b = 0; b < num_b; ++b)
            {
                Ti sb = bstrs[b] ^ g.bx;
                auto it = b_idx_map.find(sb);
                if (it == b_idx_map.end())
                    continue;
                sb_ptr[count] = it->second;
                db_ptr[count] = b;
                precompute_phase<Rank, Ti, Tv>(sb, g.unique_zbs, g.num_zb, g.wb, pb0 + count, num_b, g.rank);
                count++;
            }
            valid_b_counts[bi] = count;

            int *sa_ptr = src_a_idxs.data() + bi * num_a;
            Tv *pa_ptr = phase_a.data() + bi * num_a * MAX_RANK;
            for (int a = 0; a < num_a; ++a)
            {
                Ti sa = astrs[a] ^ g.ax;
                auto it = a_idx_map.find(sa);
                if (it != a_idx_map.end())
                {
                    sa_ptr[a] = it->second;
                    precompute_phase<Rank, Ti, Tv>(sa, g.unique_zas, g.num_za, g.wa, pa_ptr + a * MAX_RANK, 1, g.rank);
                }
                else
                    sa_ptr[a] = -1;
            }
        }

#pragma omp for schedule(dynamic)
        for (int a = 0; a < num_a; ++a)
        {
            Tv *da = dst_vec + (int64)a * num_b;
            for (int64 bi = 0; bi < cur; ++bi)
            {
                int vc = valid_b_counts[bi];
                if (vc == 0)
                    continue;
                int src_a_idx = src_a_idxs[bi * num_a + a];
                if (src_a_idx == -1)
                    continue;
                const auto &g = groups[batch_start + bi];
                const Tv *pa = phase_a.data() + bi * num_a * MAX_RANK + a * MAX_RANK;
                const Tv *pb = batch_phase.data() + bi * shift;
                const int *si = src_b_idxs_v.data() + bi * num_b;
                const int *di = dst_b_idxs.data() + bi * num_b;
                const Tv *sa = src_vec + (int64)src_a_idx * num_b;
                const int rank = g.rank;
#pragma omp simd
                for (int b = 0; b < vc; ++b)
                {
                    Tv vt = compute_coeff<Rank, Tv>(b, pa, pb, num_b, rank);
                    hvec_update<Tv>(sa + si[b], da + di[b], vt);
                }
            }
        }
    }
}

template <int TypeCode, typename Ti, typename Tv>
static void dispatch_contract_chunks_nosym(
    const Ti *astrs, int num_a, const Ti *bstrs, int num_b,
    const ankerl::unordered_dense::map<Ti, int> &a_idx_map,
    const ankerl::unordered_dense::map<Ti, int> &b_idx_map,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const Tv *src_vec, Tv *dst_vec)
{
    const int64 total = groups.size();
    if (total == 0)
        return;
    const auto *gp = groups.data();

    int64 start = 0;
    while (start < total)
    {
        int cur_rank = gp[start].rank;
        int dr = (cur_rank == 1 || cur_rank == 2) ? cur_rank : 0;
        int64 end = start + 1;
        while (end < total)
        {
            int nr = gp[end].rank;
            if (((nr == 1 || nr == 2) ? nr : 0) != dr)
                break;
            end++;
        }
        int64 sz = end - start;
        const auto *cp = gp + start;

        auto call = [&](auto tag)
        {
            constexpr int R = tag.value;
            if constexpr (TypeCode == 0)
                gather_diag_nosym<R, Ti, Tv>(astrs, num_a, bstrs, num_b, a_idx_map, b_idx_map, cp, sz, src_vec, dst_vec);
            else if constexpr (TypeCode == 1)
                gather_pure_a_nosym<R, Ti, Tv>(astrs, num_a, bstrs, num_b, a_idx_map, b_idx_map, cp, sz, src_vec, dst_vec);
            else if constexpr (TypeCode == 2)
                gather_pure_b_nosym<R, Ti, Tv>(astrs, num_a, bstrs, num_b, a_idx_map, b_idx_map, cp, sz, src_vec, dst_vec);
            else
                gather_mixed_nosym<R, Ti, Tv>(astrs, num_a, bstrs, num_b, a_idx_map, b_idx_map, cp, sz, src_vec, dst_vec);
        };

        if (dr == 1)
            call(std::integral_constant<int, 1>{});
        else if (dr == 2)
            call(std::integral_constant<int, 2>{});
        else
            call(std::integral_constant<int, 0>{});
        start = end;
    }
}

template <typename Ti, typename Tv>
static void contract_hvec_sci_nosym(
    const SciBasisManagerNosym<Ti> *basis,
    const Network_OTF<Ti, Tv> *net,
    const Tv *src, Tv *dst)
{
    const int na = (int)basis->num_a, nb = (int)basis->num_b;
    std::fill_n(dst, basis->dim, Tv{});

    dispatch_contract_chunks_nosym<0, Ti, Tv>(basis->all_astrs, na, basis->all_bstrs, nb, basis->a_idx_map, basis->b_idx_map, net->diag_groups, src, dst);
    dispatch_contract_chunks_nosym<1, Ti, Tv>(basis->all_astrs, na, basis->all_bstrs, nb, basis->a_idx_map, basis->b_idx_map, net->pure_a_groups, src, dst);
    dispatch_contract_chunks_nosym<2, Ti, Tv>(basis->all_astrs, na, basis->all_bstrs, nb, basis->a_idx_map, basis->b_idx_map, net->pure_b_groups, src, dst);
    dispatch_contract_chunks_nosym<3, Ti, Tv>(basis->all_astrs, na, basis->all_bstrs, nb, basis->a_idx_map, basis->b_idx_map, net->mixed_groups, src, dst);
}
