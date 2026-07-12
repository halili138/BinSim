#pragma once
#include "sci_basis.hpp"
#include "sci_select.hpp"

template <typename Ti, typename Tv>
static int precompute_diag_phases(
    const Ti *astrs, int64 num_a, const Ti *bstrs, int64 num_b,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &diag_groups,
    std::vector<Tv> &a_diag_phase, std::vector<Tv> &b_diag_phase)
{
    int total_rank = 0;
    for (const auto &g : diag_groups)
        total_rank += g.rank;
    if (total_rank == 0)
        return 0;

    a_diag_phase.resize(num_a * total_rank);
    b_diag_phase.resize(num_b * total_rank);

    int offset = 0;
    for (const auto &group : diag_groups)
    {
        const int rank = group.rank;
#pragma omp parallel for schedule(static)
        for (int a = 0; a < num_a; ++a)
        {
            precompute_phase<0, Ti, Tv>(
                astrs[a], group.unique_zas, group.num_za, group.wa,
                a_diag_phase.data() + a * total_rank + offset, 1, rank);
        }
#pragma omp parallel for schedule(static)
        for (int b = 0; b < num_b; ++b)
        {
            precompute_phase<0, Ti, Tv>(
                bstrs[b], group.unique_zbs, group.num_zb, group.wb,
                b_diag_phase.data() + b * total_rank + offset, 1, rank);
        }
        offset += rank;
    }
    return total_rank;
}

template <typename Tv>
FORCE_INLINE Tv dot_phase(const Tv *a, const Tv *b, int n)
{
    Tv r = Tv{};
    for (int i = 0; i < n; ++i)
        r += a[i] * b[i];
    return r;
}

template <typename Ti, typename Tv>
FORCE_INLINE void precompute_phase_select(
    Ti str, const Ti *zas, int nza, const Tv *wa, Tv *dst, int stride, int rank)
{
    if (rank == 1)
    {
        Tv v = Tv{};
        for (int i = 0; i < nza; ++i)
        {
            Tv phase = (std::popcount(str & zas[i]) & 1) ? Tv(-1) : Tv(1);
            v += wa[i] * phase;
        }
        dst[0] = v;
    }
    else
    {
        Tv v0 = Tv{}, v1 = Tv{};
        for (int i = 0; i < nza; ++i)
        {
            Tv phase = (std::popcount(str & zas[i]) & 1) ? Tv(-1) : Tv(1);
            v0 += wa[i]         * phase;
            v1 += wa[nza + i]   * phase;
        }
        dst[0]      = v0;
        dst[stride] = v1;
    }
}

template <typename Tv>
FORCE_INLINE Tv compute_coeff_select(int b, const Tv *pa, const Tv *pb, int stride, int rank)
{
    Tv vt = pa[0] * pb[b];
    if (rank >= 2)
        vt += pa[1] * pb[b + stride];
    return vt;
}


template <typename Tv>
struct SelectAGroupCache
{
    int64 num_groups = 0;
    int64 num_a = 0;
    std::vector<int> src_a_idxs;
    std::vector<int> src_block_idxs;
    std::vector<Tv> phases;
    std::vector<int64> a_exts;
    std::vector<unsigned char> a_is_new;
};

template <typename Ti, typename Tv>
static inline SelectAGroupCache<Tv> precompute_select_a_group_cache(
    const BlockDesc<Ti> &a_desc,
    const SciBasisManager<Ti> *src_basis,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups,
    int64 a_start,
    const BlockDesc<Ti> &full_block,
    const Ti *tgt_all_astrs,
    const bool *is_new_a)
{
    SelectAGroupCache<Tv> cache;
    cache.num_groups = num_groups;
    cache.num_a = a_desc.num_a;

    const int tgt_num_a = (int)a_desc.num_a;
    const int64 num_irreps = src_basis->num_irreps;

    cache.src_a_idxs.assign(num_groups * tgt_num_a, -1);
    cache.src_block_idxs.assign(num_groups, -1);
    cache.phases.assign(num_groups * tgt_num_a * 2, Tv{});
    cache.a_exts.resize(tgt_num_a);
    cache.a_is_new.assign(tgt_num_a, 0);

#pragma omp parallel for schedule(static)
    for (int a = 0; a < tgt_num_a; ++a)
    {
        const int64 a_global = a_start + a;
        const int64 a_ext = (full_block.astrs + a_global) - tgt_all_astrs;
        cache.a_exts[a] = a_ext;
        cache.a_is_new[a] = (a_ext >= 0) && is_new_a[a_ext];
    }

#pragma omp parallel for schedule(dynamic)
    for (int64 group_idx = 0; group_idx < num_groups; ++group_idx)
    {
        const SVDGroup_OTF<Ti, Tv> &group = groups[group_idx];
        const int64 h = (a_desc.asym ^ group.asym) * num_irreps
                      + (a_desc.bsym ^ group.bsym);
        const int64 sidx = src_basis->block_map[h];
        cache.src_block_idxs[group_idx] = (int)sidx;

        if (sidx == -1)
            continue;

        int *src_a_ptr = cache.src_a_idxs.data() + group_idx * tgt_num_a;
        Tv *phase_ptr = cache.phases.data() + group_idx * tgt_num_a * 2;

        for (int a = 0; a < tgt_num_a; ++a)
        {
            const Ti src_a_str = a_desc.astrs[a] ^ group.ax;
            auto it_a = src_basis->a_idx_map.find(src_a_str);
            if (it_a == src_basis->a_idx_map.end())
                continue;

            src_a_ptr[a] = it_a->second;
            precompute_phase_select<Ti, Tv>(src_a_str, group.unique_zas,
                                            group.num_za, group.wa,
                                            phase_ptr + a * 2, 1, group.rank);
        }
    }

    return cache;
}

template <typename Ti, typename Tv>
static inline void gather_select_instant_rank(
    const BlockDesc<Ti> &tile_desc,
    const SciBasisManager<Ti> *src_basis,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups,
    const SelectAGroupCache<Tv> &a_cache,
    const Tv *src_vec,
    int64 a_start, int64 b_start,
    const BlockDesc<Ti> &full_block,
    const Ti *tgt_all_bstrs,
    const bool *is_new_b,
    const Tv *a_diag_phase, const Tv *b_diag_phase,
    int total_diag_rank,
    Tv variational_energy, double eps,
    bool wants_new_a, bool wants_new_b,
    bool *selected_a, bool *selected_b)
{
    const int tgt_num_a = (int)tile_desc.num_a;
    const int tgt_num_b = (int)tile_desc.num_b;
    const int shift = tgt_num_b * 2;
    const int64 total_ngs = num_groups;

    std::vector<int> src_b_idxs_v(total_ngs * tgt_num_b);
    std::vector<int> dst_b_idxs(total_ngs * tgt_num_b);
    std::vector<Tv> batch_phase(total_ngs * shift);
    std::vector<int> valid_b_counts(total_ngs);

#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (int64 group_idx = 0; group_idx < total_ngs; ++group_idx)
        {
            const SVDGroup_OTF<Ti, Tv> &group = groups[group_idx];
            const int64 sidx = a_cache.src_block_idxs[group_idx];

            if (sidx == -1)
            {
                valid_b_counts[group_idx] = 0;
                continue;
            }

            Tv *pb0 = batch_phase.data() + group_idx * shift;
            int *sb_ptr = src_b_idxs_v.data() + group_idx * tgt_num_b;
            int *db_ptr = dst_b_idxs.data() + group_idx * tgt_num_b;

            int count = 0;
            for (int i = 0; i < tgt_num_b; ++i)
            {
                const Ti src_b_str = tile_desc.bstrs[i] ^ group.bx;
                auto it = src_basis->b_idx_map.find(src_b_str);
                if (it == src_basis->b_idx_map.end())
                    continue;

                sb_ptr[count] = it->second;
                db_ptr[count] = i;
                precompute_phase_select<Ti, Tv>(src_b_str, group.unique_zbs,
                                                group.num_zb, group.wb,
                                                pb0 + count, tgt_num_b, group.rank);
                ++count;
            }
            valid_b_counts[group_idx] = count;
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < tgt_num_a; ++a)
    {
        const int64 a_global = a_start + a;
        const int64 a_ext = a_cache.a_exts[a];
        const bool a_is_new = a_cache.a_is_new[a];
        const Tv *a_phase = a_diag_phase
            ? a_diag_phase + a_global * total_diag_rank : nullptr;

        alignas(64) Tv accum_b[264] = {};

        for (int64 group_idx = 0; group_idx < total_ngs; ++group_idx)
        {
            const int valid_count = valid_b_counts[group_idx];
            if (valid_count == 0)
                continue;

            const auto &group = groups[group_idx];
            const int src_a_idx = a_cache.src_a_idxs[group_idx * tgt_num_a + a];
            if (src_a_idx == -1)
                continue;

            const int64 src_block_idx = a_cache.src_block_idxs[group_idx];
            const BlockDesc<Ti> &src_blk = src_basis->blocks[src_block_idx];
            const Tv *sa = src_vec + src_blk.offset + (int64)src_a_idx * src_blk.num_b;

            const Tv *pa = a_cache.phases.data()
                + (group_idx * tgt_num_a + a) * 2;

            const Tv *pb = batch_phase.data() + group_idx * shift;
            const int *si = src_b_idxs_v.data() + group_idx * tgt_num_b;
            const int *di = dst_b_idxs.data() + group_idx * tgt_num_b;

#pragma omp simd
            for (int b = 0; b < valid_count; ++b)
            {
                accum_b[di[b]] += sa[si[b]]
                    * compute_coeff_select<Tv>(b, pa, pb, tgt_num_b, group.rank);
            }
        }

        for (int b_local = 0; b_local < tgt_num_b; ++b_local)
        {
            if (accum_b[b_local] == Tv{})
                continue;

            const int64 b_global = b_start + b_local;
            const int64 b_ext = (full_block.bstrs + b_global) - tgt_all_bstrs;
            const bool b_is_new = (b_ext >= 0) && is_new_b[b_ext];

            if (a_is_new != wants_new_a || b_is_new != wants_new_b)
                continue;

            Tv haa = Tv{};
            if (total_diag_rank > 0 && a_phase)
            {
                const Tv *b_phase = b_diag_phase + b_global * total_diag_rank;
                haa = dot_phase(a_phase, b_phase, total_diag_rank);
            }

            if (sci_eps_check(accum_b[b_local], haa, variational_energy, eps))
            {
                if (a_ext >= 0)
                    selected_a[a_ext] = true;
                if (b_ext >= 0)
                    selected_b[b_ext] = true;
            }
        }
    }
}

template <typename Ti, typename Tv>
static inline void dispatch_select_instant_groups(
    const BlockDesc<Ti> &tile_desc,
    const SciBasisManager<Ti> *src_basis,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const SelectAGroupCache<Tv> &a_cache,
    const Tv *src_vec,
    int64 a_start, int64 b_start,
    const BlockDesc<Ti> &full_block,
    const Ti *tgt_all_bstrs,
    const bool *is_new_b,
    const Tv *a_diag_phase, const Tv *b_diag_phase,
    int total_diag_rank,
    Tv variational_energy, double eps,
    bool wants_new_a, bool wants_new_b,
    bool *selected_a, bool *selected_b)
{
    const int64 total_ngs = groups.size();
    if (total_ngs == 0)
        return;

    gather_select_instant_rank<Ti, Tv>(
        tile_desc, src_basis, groups.data(), total_ngs, a_cache, src_vec,
        a_start, b_start, full_block, tgt_all_bstrs,
        is_new_b, a_diag_phase, b_diag_phase, total_diag_rank,
        variational_energy, eps, wants_new_a, wants_new_b,
        selected_a, selected_b);
}

template <typename Ti, typename Tv>
static inline void dispatch_select_mixed_instant_groups(
    const BlockDesc<Ti> &tile_desc,
    const SciBasisManager<Ti> *src_basis,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const SelectAGroupCache<Tv> &a_cache,
    const Tv *src_vec,
    int64 a_start, int64 b_start,
    const BlockDesc<Ti> &full_block,
    const Ti *tgt_all_bstrs,
    const bool *is_new_b,
    const Tv *a_diag_phase, const Tv *b_diag_phase,
    int total_diag_rank,
    Tv variational_energy, double eps,
    bool *selected_a, bool *selected_b)
{
    const int tgt_num_a = (int)tile_desc.num_a;
    const int tgt_num_b = (int)tile_desc.num_b;
    const int shift = tgt_num_b * 2;
    const int64 total_ngs = groups.size();
    if (total_ngs == 0)
        return;

    std::vector<int64> b_exts(tgt_num_b);
    std::vector<unsigned char> b_is_new_cache(tgt_num_b);

#pragma omp parallel for schedule(static)
    for (int b = 0; b < tgt_num_b; ++b)
    {
        const int64 b_global = b_start + b;
        const int64 b_ext = (full_block.bstrs + b_global) - tgt_all_bstrs;
        b_exts[b] = b_ext;
        b_is_new_cache[b] = (b_ext >= 0) && is_new_b[b_ext];
    }

    std::vector<int> src_b_idxs_v(total_ngs * tgt_num_b);
    std::vector<int> dst_b_idxs(total_ngs * tgt_num_b);
    std::vector<Tv> batch_phase(total_ngs * shift);
    std::vector<int> valid_b_counts(total_ngs);

#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (int64 group_idx = 0; group_idx < total_ngs; ++group_idx)
        {
            const SVDGroup_OTF<Ti, Tv> &group = groups[group_idx];
            const int64 sidx = a_cache.src_block_idxs[group_idx];

            if (sidx == -1)
            {
                valid_b_counts[group_idx] = 0;
                continue;
            }

            Tv *pb0 = batch_phase.data() + group_idx * shift;
            int *sb_ptr = src_b_idxs_v.data() + group_idx * tgt_num_b;
            int *db_ptr = dst_b_idxs.data() + group_idx * tgt_num_b;

            int count = 0;
            for (int i = 0; i < tgt_num_b; ++i)
            {
                const Ti src_b_str = tile_desc.bstrs[i] ^ group.bx;
                auto it = src_basis->b_idx_map.find(src_b_str);
                if (it == src_basis->b_idx_map.end())
                    continue;

                sb_ptr[count] = it->second;
                db_ptr[count] = i;
                precompute_phase_select<Ti, Tv>(src_b_str, group.unique_zbs,
                                                group.num_zb, group.wb,
                                                pb0 + count, tgt_num_b, group.rank);
                ++count;
            }
            valid_b_counts[group_idx] = count;
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < tgt_num_a; ++a)
    {
        const int64 a_global = a_start + a;
        const int64 a_ext = a_cache.a_exts[a];
        const bool a_is_new = a_cache.a_is_new[a];
        const Tv *a_phase = a_diag_phase
            ? a_diag_phase + a_global * total_diag_rank : nullptr;

        alignas(64) Tv accum_new_a_old_b[264] = {};
        alignas(64) Tv accum_old_a_new_b[264] = {};
        alignas(64) Tv accum_new_a_new_b[264] = {};

        for (int64 group_idx = 0; group_idx < total_ngs; ++group_idx)
        {
            const int valid_count = valid_b_counts[group_idx];
            if (valid_count == 0)
                continue;

            const auto &group = groups[group_idx];
            const int src_a_idx = a_cache.src_a_idxs[group_idx * tgt_num_a + a];
            if (src_a_idx == -1)
                continue;

            const int64 src_block_idx = a_cache.src_block_idxs[group_idx];
            const BlockDesc<Ti> &src_blk = src_basis->blocks[src_block_idx];
            const Tv *sa = src_vec + src_blk.offset + (int64)src_a_idx * src_blk.num_b;

            const Tv *pa = a_cache.phases.data()
                + (group_idx * tgt_num_a + a) * 2;

            const Tv *pb = batch_phase.data() + group_idx * shift;
            const int *si = src_b_idxs_v.data() + group_idx * tgt_num_b;
            const int *di = dst_b_idxs.data() + group_idx * tgt_num_b;

#pragma omp simd
            for (int b = 0; b < valid_count; ++b)
            {
                const int b_local = di[b];
                const bool b_is_new = b_is_new_cache[b_local];
                Tv *accum_b = nullptr;
                if (a_is_new && !b_is_new)
                    accum_b = accum_new_a_old_b;
                else if (!a_is_new && b_is_new)
                    accum_b = accum_old_a_new_b;
                else if (a_is_new && b_is_new)
                    accum_b = accum_new_a_new_b;
                else
                    continue;

                accum_b[b_local] += sa[si[b]]
                    * compute_coeff_select<Tv>(b, pa, pb, tgt_num_b, group.rank);
            }
        }

        for (int b_local = 0; b_local < tgt_num_b; ++b_local)
        {
            Tv vt = Tv{};
            if (a_is_new && !b_is_new_cache[b_local])
                vt = accum_new_a_old_b[b_local];
            else if (!a_is_new && b_is_new_cache[b_local])
                vt = accum_old_a_new_b[b_local];
            else if (a_is_new && b_is_new_cache[b_local])
                vt = accum_new_a_new_b[b_local];
            else
                continue;

            if (vt == Tv{})
                continue;

            const int64 b_global = b_start + b_local;
            const int64 b_ext = b_exts[b_local];

            Tv haa = Tv{};
            if (total_diag_rank > 0 && a_phase)
            {
                const Tv *b_phase = b_diag_phase + b_global * total_diag_rank;
                haa = dot_phase(a_phase, b_phase, total_diag_rank);
            }

            if (sci_eps_check(vt, haa, variational_energy, eps))
            {
                if (a_ext >= 0)
                    selected_a[a_ext] = true;
                if (b_ext >= 0)
                    selected_b[b_ext] = true;
            }
        }
    }
}

template <typename Ti, typename Tv>
void sci_select_external_block(
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisManager<Ti> *src_basis,
    const Network_OTF<Ti, Tv> *net,
    const bool *is_new_a,
    const bool *is_new_b,
    int64 block_idx,
    const Tv *src_vec,
    Tv variational_energy,
    int a_chunk_size,
    int b_chunk_size,
    double eps,
    bool *selected_a,
    bool *selected_b)
{
    const BlockDesc<Ti> &full_block = tgt_basis->blocks[block_idx];
    const int64 num_a_total = full_block.num_a;
    const int64 num_b_total = full_block.num_b;

    std::vector<Tv> a_diag_phase, b_diag_phase;
    int total_diag_rank = precompute_diag_phases<Ti, Tv>(
        full_block.astrs, num_a_total, full_block.bstrs, num_b_total,
        net->diag_groups, a_diag_phase, b_diag_phase);

    for (int64 a_start = 0; a_start < num_a_total; a_start += a_chunk_size)
    {
        const int64 cur_num_a = std::min<int64>(a_chunk_size, num_a_total - a_start);

        BlockDesc<Ti> a_desc = full_block;
        a_desc.astrs = full_block.astrs + a_start;
        a_desc.num_a = cur_num_a;
        a_desc.offset = 0;

        const auto part1_a_cache = precompute_select_a_group_cache<Ti, Tv>(
            a_desc, src_basis, net->pure_a_groups.data(), net->pure_a_groups.size(),
            a_start, full_block, tgt_basis->all_astrs, is_new_a);
        const auto part2_a_cache = precompute_select_a_group_cache<Ti, Tv>(
            a_desc, src_basis, net->pure_b_groups.data(), net->pure_b_groups.size(),
            a_start, full_block, tgt_basis->all_astrs, is_new_a);
        const auto mixed_a_cache = precompute_select_a_group_cache<Ti, Tv>(
            a_desc, src_basis, net->mixed_groups.data(), net->mixed_groups.size(),
            a_start, full_block, tgt_basis->all_astrs, is_new_a);

        for (int64 b_start = 0; b_start < num_b_total; b_start += b_chunk_size)
        {
            const int64 cur_num_b = std::min<int64>(b_chunk_size, num_b_total - b_start);

            BlockDesc<Ti> tile_desc = full_block;
            tile_desc.astrs = full_block.astrs + a_start;
            tile_desc.bstrs = full_block.bstrs + b_start;
            tile_desc.num_a = cur_num_a;
            tile_desc.num_b = cur_num_b;
            tile_desc.offset = 0;

            dispatch_select_instant_groups<Ti, Tv>(
                tile_desc, src_basis, net->pure_a_groups, part1_a_cache, src_vec,
                a_start, b_start, full_block,
                tgt_basis->all_bstrs,
                is_new_b,
                a_diag_phase.data(), b_diag_phase.data(), total_diag_rank,
                variational_energy, eps, true, false,
                selected_a, selected_b);

            dispatch_select_instant_groups<Ti, Tv>(
                tile_desc, src_basis, net->pure_b_groups, part2_a_cache, src_vec,
                a_start, b_start, full_block,
                tgt_basis->all_bstrs,
                is_new_b,
                a_diag_phase.data(), b_diag_phase.data(), total_diag_rank,
                variational_energy, eps, false, true,
                selected_a, selected_b);

            dispatch_select_mixed_instant_groups<Ti, Tv>(
                tile_desc, src_basis, net->mixed_groups, mixed_a_cache, src_vec,
                a_start, b_start, full_block,
                tgt_basis->all_bstrs,
                is_new_b,
                a_diag_phase.data(), b_diag_phase.data(), total_diag_rank,
                variational_energy, eps,
                selected_a, selected_b);
        }
    }
}
