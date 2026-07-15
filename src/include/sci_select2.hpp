#pragma once
#include "sci_select.hpp"

namespace sci_select2_detail
{

template <typename Tv>
FORCE_INLINE Tv coeff_runtime(const Tv *pa, const Tv *pb, int rank)
{
    Tv vt = {};
    for (int r = 0; r < rank; ++r)
        vt += pa[r] * pb[r];
    return vt;
}

template <typename Ti, typename Tv>
FORCE_INLINE void phase_runtime(Ti str, const Ti *zs, int num_zs, const Tv *w, int rank, Tv *out)
{
    precompute_phase<0, Ti, Tv>(str, zs, num_zs, w, out, 1, rank);
}

template <typename Ti, typename Tv>
FORCE_INLINE int64 source_pos(
    const SciBasisManager<Ti> *src_basis,
    Ti src_a, Ti src_b,
    int &src_ia, int &src_ib)
{
    auto ita = src_basis->a_idx_map.find(src_a);
    if (ita == src_basis->a_idx_map.end())
        return -1;
    auto itb = src_basis->b_idx_map.find(src_b);
    if (itb == src_basis->b_idx_map.end())
        return -1;

    const int64 asym = get_string_sym(src_a, src_basis->orbsym);
    const int64 bsym = get_string_sym(src_b, src_basis->orbsym);
    if (asym >= src_basis->num_irreps || bsym >= src_basis->num_irreps)
        return -1;

    const int64 blk_idx = src_basis->block_map[asym * src_basis->num_irreps + bsym];
    if (blk_idx == -1)
        return -1;

    src_ia = ita->second;
    src_ib = itb->second;
    const BlockDesc<Ti> &blk = src_basis->blocks[blk_idx];
    return blk.offset + (int64)src_ia * blk.num_b + src_ib;
}

template <typename Ti>
FORCE_INLINE bool is_new_alpha(const SciBasisManager<Ti> *basis, const bool *is_new_a, const Ti *astr_ptr)
{
    const int64 idx = astr_ptr - basis->all_astrs;
    return is_new_a[idx];
}

template <typename Ti>
FORCE_INLINE bool is_new_beta(const SciBasisManager<Ti> *basis, const bool *is_new_b, const Ti *bstr_ptr)
{
    const int64 idx = bstr_ptr - basis->all_bstrs;
    return is_new_b[idx];
}

template <typename Ti, typename Tv, typename EmitFn>
static inline void select_pass_a_groups(
    const BlockDesc<Ti> &blk,
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisManager<Ti> *src_basis,
    const bool *is_new_a,
    const bool *is_new_b,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const Tv *src_vec,
    const Tv *candidate_diags,
    Tv variational_energy,
    double eps,
    int a_chunk_size,
    int g_chunk_size,
    int b_chunk_size,
    EmitFn &&emit)
{
    const int64 num_a = blk.num_a;
    const int64 num_b = blk.num_b;
    const int64 num_groups = (int64)groups.size();
    if (num_groups == 0)
        return;

    for (int64 a_start = 0; a_start < num_a; a_start += a_chunk_size)
    {
        const int64 a_end = std::min<int64>(a_start + a_chunk_size, num_a);
        std::vector<int64> new_a_locs;
        new_a_locs.reserve(a_end - a_start);
        for (int64 a = a_start; a < a_end; ++a)
            if (is_new_alpha(tgt_basis, is_new_a, blk.astrs + a))
                new_a_locs.push_back(a);
        if (new_a_locs.empty())
            continue;

        std::vector<Tv> accum((size_t)new_a_locs.size() * (size_t)num_b, Tv{});

        for (int64 g_start = 0; g_start < num_groups; g_start += g_chunk_size)
        {
            const int64 g_end = std::min<int64>(g_start + g_chunk_size, num_groups);
            for (int64 b_start = 0; b_start < num_b; b_start += b_chunk_size)
            {
                const int64 b_end = std::min<int64>(b_start + b_chunk_size, num_b);
                for (int64 ia_local = 0; ia_local < (int64)new_a_locs.size(); ++ia_local)
                {
                    const int64 a = new_a_locs[ia_local];
                    const Ti dst_a = blk.astrs[a];
                    Tv *row = accum.data() + ia_local * num_b;

                    for (int64 ig = g_start; ig < g_end; ++ig)
                    {
                        const auto &group = groups[ig];
                        const Ti src_a = dst_a ^ group.ax;
                        if (src_basis->a_idx_map.find(src_a) == src_basis->a_idx_map.end())
                            continue;

                        std::vector<Tv> pa(group.rank);
                        phase_runtime(src_a, group.unique_zas, group.num_za, group.wa, group.rank, pa.data());

                        for (int64 b = b_start; b < b_end; ++b)
                        {
                            const Ti dst_b = blk.bstrs[b];
                            const Ti src_b = dst_b ^ group.bx;
                            int src_ia = -1, src_ib = -1;
                            const int64 src_gid = source_pos<Ti, Tv>(src_basis, src_a, src_b, src_ia, src_ib);
                            if (src_gid == -1)
                                continue;

                            std::vector<Tv> pb(group.rank);
                            phase_runtime(src_b, group.unique_zbs, group.num_zb, group.wb, group.rank, pb.data());
                            row[b] += src_vec[src_gid] * coeff_runtime(pa.data(), pb.data(), group.rank);
                        }
                    }
                }
            }
        }

        for (int64 ia_local = 0; ia_local < (int64)new_a_locs.size(); ++ia_local)
        {
            const int64 a = new_a_locs[ia_local];
            const Tv *row = accum.data() + ia_local * num_b;
            for (int64 b = 0; b < num_b; ++b)
            {
                if (row[b] == Tv{})
                    continue;
                const int64 diag_idx = blk.offset + a * blk.num_b + b;
                if (sci_eps_check(row[b], candidate_diags[diag_idx], variational_energy, eps))
                    emit(blk.astrs[a], blk.bstrs[b], row[b]);
            }
        }
    }
}

template <typename Ti, typename Tv, typename EmitFn>
static inline void select_pass_b_groups(
    const BlockDesc<Ti> &blk,
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisManager<Ti> *src_basis,
    const bool *is_new_a,
    const bool *is_new_b,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const Tv *src_vec,
    const Tv *candidate_diags,
    Tv variational_energy,
    double eps,
    int b_chunk_size,
    int g_chunk_size,
    int a_chunk_size,
    EmitFn &&emit)
{
    const int64 num_a = blk.num_a;
    const int64 num_b = blk.num_b;
    const int64 num_groups = (int64)groups.size();
    if (num_groups == 0)
        return;

    for (int64 b_start = 0; b_start < num_b; b_start += b_chunk_size)
    {
        const int64 b_end = std::min<int64>(b_start + b_chunk_size, num_b);
        std::vector<int64> new_b_locs;
        new_b_locs.reserve(b_end - b_start);
        for (int64 b = b_start; b < b_end; ++b)
            if (is_new_beta(tgt_basis, is_new_b, blk.bstrs + b))
                new_b_locs.push_back(b);
        if (new_b_locs.empty())
            continue;

        std::vector<Tv> accum((size_t)new_b_locs.size() * (size_t)num_a, Tv{});

        for (int64 g_start = 0; g_start < num_groups; g_start += g_chunk_size)
        {
            const int64 g_end = std::min<int64>(g_start + g_chunk_size, num_groups);
            for (int64 a_start = 0; a_start < num_a; a_start += a_chunk_size)
            {
                const int64 a_end = std::min<int64>(a_start + a_chunk_size, num_a);
                for (int64 ib_local = 0; ib_local < (int64)new_b_locs.size(); ++ib_local)
                {
                    const int64 b = new_b_locs[ib_local];
                    const Ti dst_b = blk.bstrs[b];
                    Tv *row = accum.data() + ib_local * num_a;

                    for (int64 ig = g_start; ig < g_end; ++ig)
                    {
                        const auto &group = groups[ig];
                        const Ti src_b = dst_b ^ group.bx;
                        if (src_basis->b_idx_map.find(src_b) == src_basis->b_idx_map.end())
                            continue;

                        std::vector<Tv> pb(group.rank);
                        phase_runtime(src_b, group.unique_zbs, group.num_zb, group.wb, group.rank, pb.data());

                        for (int64 a = a_start; a < a_end; ++a)
                        {
                            if (is_new_alpha(tgt_basis, is_new_a, blk.astrs + a))
                                continue; // new_a x new_b is emitted by pass A.

                            const Ti dst_a = blk.astrs[a];
                            const Ti src_a = dst_a ^ group.ax;
                            int src_ia = -1, src_ib = -1;
                            const int64 src_gid = source_pos<Ti, Tv>(src_basis, src_a, src_b, src_ia, src_ib);
                            if (src_gid == -1)
                                continue;

                            std::vector<Tv> pa(group.rank);
                            phase_runtime(src_a, group.unique_zas, group.num_za, group.wa, group.rank, pa.data());
                            row[a] += src_vec[src_gid] * coeff_runtime(pa.data(), pb.data(), group.rank);
                        }
                    }
                }
            }
        }

        for (int64 ib_local = 0; ib_local < (int64)new_b_locs.size(); ++ib_local)
        {
            const int64 b = new_b_locs[ib_local];
            const Tv *row = accum.data() + ib_local * num_a;
            for (int64 a = 0; a < num_a; ++a)
            {
                if (is_new_alpha(tgt_basis, is_new_a, blk.astrs + a))
                    continue;
                if (row[a] == Tv{})
                    continue;
                const int64 diag_idx = blk.offset + a * blk.num_b + b;
                if (sci_eps_check(row[a], candidate_diags[diag_idx], variational_energy, eps))
                    emit(blk.astrs[a], blk.bstrs[b], row[a]);
            }
        }
    }
}

} // namespace sci_select2_detail

template <typename Ti, typename Tv>
int64 sci_select_external_block2(
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisManager<Ti> *src_basis,
    const Network_OTF<Ti, Tv> *net,
    const bool *is_new_a,
    const bool *is_new_b,
    int64 block_idx,
    const Tv *src_vec,
    const Tv *candidate_diags,
    Tv variational_energy,
    int a_chunk_size,
    int g_chunk_size,
    int b_chunk_size,
    double eps,
    BufferedEntry<Ti, Tv> *out_entries,
    int64 max_entries)
{
    if (block_idx < 0 || block_idx >= tgt_basis->num_blocks || max_entries <= 0)
        return 0;

    a_chunk_size = std::max(1, a_chunk_size);
    g_chunk_size = std::max(1, g_chunk_size);
    b_chunk_size = std::max(1, b_chunk_size);

    const BlockDesc<Ti> &blk = tgt_basis->blocks[block_idx];
    int64 out_count = 0;
    auto emit = [&](Ti astr, Ti bstr, Tv val)
    {
        if (out_count < max_entries)
            out_entries[out_count++] = {astr, bstr, val};
    };

    auto pass_a = [&](const std::vector<SVDGroup_OTF<Ti, Tv>> &groups)
    {
        if (out_count >= max_entries)
            return;
        sci_select2_detail::select_pass_a_groups(
            blk, tgt_basis, src_basis, is_new_a, is_new_b, groups,
            src_vec, candidate_diags, variational_energy, eps,
            a_chunk_size, g_chunk_size, b_chunk_size, emit);
    };
    auto pass_b = [&](const std::vector<SVDGroup_OTF<Ti, Tv>> &groups)
    {
        if (out_count >= max_entries)
            return;
        sci_select2_detail::select_pass_b_groups(
            blk, tgt_basis, src_basis, is_new_a, is_new_b, groups,
            src_vec, candidate_diags, variational_energy, eps,
            b_chunk_size, g_chunk_size, a_chunk_size, emit);
    };

    pass_a(net->pure_a_groups);
    pass_a(net->pure_b_groups);
    pass_a(net->mixed_groups);

    pass_b(net->pure_a_groups);
    pass_b(net->pure_b_groups);
    pass_b(net->mixed_groups);

    return out_count;
}
