#pragma once
#include "sci_common.hpp"
#include "sci_links.hpp"
#include "sci_hvec.hpp"
#include <cmath>
#include <limits>

template <typename Tv>
FORCE_INLINE auto sqnorm(const Tv &v)
{
    if constexpr (std::is_arithmetic_v<Tv>)
        return v * v;
    else
        return v.real() * v.real() + v.imag() * v.imag();
}

template <typename Ti>
static inline void append_block_frontier_targets(
    const ankerl::unordered_dense::map<Ti, SpinLinkCSR<Ti>> &frontiers_by_mask,
    const Ti *block_strings,
    const Ti *all_strings,
    int64 num_block_strings,
    std::vector<Ti> &strings,
    std::vector<int64> &block_idxs)
{
    if (num_block_strings == 0)
        return;

    const int64 block_begin = block_strings - all_strings;
    const int64 block_end = block_begin + num_block_strings;
    std::vector<unsigned char> seen(num_block_strings, 0);

    for (const auto &kv : frontiers_by_mask)
    {
        const SpinLinkCSR<Ti> &frontier = kv.second;
        for (int dst_global_idx : frontier.colidx)
        {
            if (dst_global_idx < block_begin || dst_global_idx >= block_end)
                continue;
            const int64 local_idx = (int64)dst_global_idx - block_begin;
            if (seen[local_idx])
                continue;
            seen[local_idx] = 1;
            strings.push_back(block_strings[local_idx]);
            block_idxs.push_back(local_idx);
        }
    }
}

template <typename Ti>
static inline int64 count_spin_link_entries(
    const ankerl::unordered_dense::map<Ti, SpinLinkCSR<Ti>> &links_by_mask)
{
    int64 total = 0;
    for (const auto &kv : links_by_mask)
        total += (int64)kv.second.colidx.size();
    return total;
}

template <typename Ti, typename Tv>
struct LinkBucket
{
    const SpinLinkCSR<Ti> *alpha_links = nullptr;
    const SpinLinkCSR<Ti> *beta_links = nullptr;
    const SVDGroup_OTF<Ti, Tv> *group = nullptr;
};

struct SourceStringInfo
{
    int64 sym = -1;
    int64 local = -1;
    bool valid = false;
};

template <typename Ti, typename Tv>
struct NetworkSCI
{
    struct Bucket
    {
        Ti ax, bx;
        const SVDGroup_OTF<Ti, Tv> *group;
    };
    std::vector<Bucket> buckets;
};

template <typename Ti, typename Tv>
NetworkSCI<Ti, Tv> build_network_sci_from_otf(const Network_OTF<Ti, Tv> &net)
{
    NetworkSCI<Ti, Tv> sci;

    auto group_from_original_idx = [&](int64 original_idx) -> const SVDGroup_OTF<Ti, Tv> *
    {
        if (original_idx < 0 || original_idx >= net.num_groups)
            return nullptr;
        const int64 sorted_idx = net.sorted_idxs[original_idx];
        if (sorted_idx < 0)
            return nullptr;
        switch (net.excit_types[original_idx])
        {
        case 0:
            return sorted_idx < (int64)net.diag_groups.size() ? &net.diag_groups[sorted_idx] : nullptr;
        case 1:
            return sorted_idx < (int64)net.pure_a_groups.size() ? &net.pure_a_groups[sorted_idx] : nullptr;
        case 2:
            return sorted_idx < (int64)net.pure_b_groups.size() ? &net.pure_b_groups[sorted_idx] : nullptr;
        case 3:
            return sorted_idx < (int64)net.mixed_groups.size() ? &net.mixed_groups[sorted_idx] : nullptr;
        default:
            return nullptr;
        }
    };

    for (const GroupAxBxKey<Ti> &key : net.group_index.unique_ax_bx_pairs)
    {
        const auto groups_it = net.group_index.groups_by_ax_bx.find(key);
        if (groups_it == net.group_index.groups_by_ax_bx.end())
            continue;

        for (int64 original_idx : groups_it->second)
        {
            const SVDGroup_OTF<Ti, Tv> *group = group_from_original_idx(original_idx);
            if (group != nullptr)
                sci.buckets.push_back({key.ax, key.bx, group});
        }
    }

    return sci;
}

template <typename Ti>
struct ExternalLinkSelectContext
{
    std::vector<Ti> unique_axs;
    std::vector<Ti> unique_bxs;
    SpinLinksByMask<Ti> full_links;
    SpinLinksByMask<Ti> new_frontiers;
    int64 alpha_link_entries = 0;
    int64 beta_link_entries = 0;
};

template <typename Ti, typename Tv>
ExternalLinkSelectContext<Ti> build_external_link_select_context(
    const SciBasisManager<Ti> *src_basis,
    const SciBasisManager<Ti> *tgt_basis,
    const Network_OTF<Ti, Tv> *net,
    const bool *is_new_a,
    const bool *is_new_b,
    const Ti *unique_axs,
    int64 num_unique_axs,
    const Ti *unique_bxs,
    int64 num_unique_bxs)
{
    ExternalLinkSelectContext<Ti> ctx;
    ctx.unique_axs.assign(unique_axs, unique_axs + num_unique_axs);
    ctx.unique_bxs.assign(unique_bxs, unique_bxs + num_unique_bxs);
    ctx.full_links = build_spin_links_by_mask(src_basis, tgt_basis, ctx.unique_axs, ctx.unique_bxs);
    ctx.new_frontiers = build_new_frontier_links_by_mask(ctx.full_links, is_new_a, is_new_b);
    ctx.alpha_link_entries = count_spin_link_entries(ctx.full_links.alpha_links_by_ax);
    ctx.beta_link_entries = count_spin_link_entries(ctx.full_links.beta_links_by_bx);
    return ctx;
}

template <typename Ti, typename Tv>
int64 sci_hvec_select_for_block_bitstr(
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisManager<Ti> *src_basis,
    const Network_OTF<Ti, Tv> *net,
    int64 block_idx,
    const Tv *src_vec,
    const Tv *candidate_diags,
    Tv variational_energy,
    int chunk_size,
    double eps,
    BufferedEntry<Ti, Tv> *out_entries,
    int64 max_entries)
{
    const BlockDesc<Ti> &full_block = tgt_basis->blocks[block_idx];
    const int64 num_a_total = full_block.num_a;
    const int64 num_b = full_block.num_b;

    int64 out_count = 0;

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

        for (int a = 0; a < cur_num_a && out_count < max_entries; ++a)
        {
            const int64 a_global = a_start + a;
            const Tv *row = chunk_acc + a * num_b;

            for (int b = 0; b < num_b && out_count < max_entries; ++b)
            {
                if (row[b] == Tv{})
                    continue;

                const Ti candidate_astr = full_block.astrs[a_global];
                const Ti candidate_bstr = full_block.bstrs[b];
                const int64 src_block_idx =
                    (full_block.asym < src_basis->num_irreps && full_block.bsym < src_basis->num_irreps)
                        ? src_basis->block_map[full_block.asym * src_basis->num_irreps + full_block.bsym]
                        : -1;
                const auto src_a_it = src_basis->a_idx_map.find(candidate_astr);
                const auto src_b_it = src_basis->b_idx_map.find(candidate_bstr);
                const bool candidate_in_src_basis =
                    src_a_it != src_basis->a_idx_map.end() && src_a_it->second != -1 &&
                    src_b_it != src_basis->b_idx_map.end() && src_b_it->second != -1 &&
                    src_block_idx != -1;

                const Tv haa = candidate_diags[full_block.offset + a_global * num_b + b];
                const Tv denom = variational_energy - haa;
                const double denom_norm = std::sqrt(sqnorm(denom));
                if (denom_norm == 0.0)
                    continue;
                const Tv selection_amplitude = row[b] / denom;
                const double selection_norm = std::sqrt(sqnorm(selection_amplitude));
                const bool passes_eps = selection_norm > eps;

                if (!passes_eps)
                    continue;
                out_entries[out_count++] = {candidate_astr, candidate_bstr, row[b]};
            }
        }

        delete[] chunk_acc;
    }

    return out_count;
}

template <typename Ti, typename Tv>
int64 sci_hvec_select_external_block_bitstr(
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisManager<Ti> *src_basis,
    const Network_OTF<Ti, Tv> *net,
    const bool *is_new_a,
    const bool *is_new_b,
    int64 block_idx,
    const Tv *src_vec,
    const Tv *candidate_diags,
    Tv variational_energy,
    int chunk_size,
    double eps,
    BufferedEntry<Ti, Tv> *out_entries,
    int64 max_entries)
{
    const BlockDesc<Ti> &full_block = tgt_basis->blocks[block_idx];
    const int64 num_a_total = full_block.num_a;
    const int64 num_b = full_block.num_b;
    int64 out_count = 0;

    for (int64 a_start = 0; a_start < num_a_total; a_start += chunk_size)
    {
        const int64 a_end = std::min(a_start + (int64)chunk_size, num_a_total);
        const int64 cur_num_a = a_end - a_start;

        BlockDesc<Ti> chunk_desc = full_block;
        chunk_desc.astrs = full_block.astrs + a_start;
        chunk_desc.num_a = cur_num_a;
        chunk_desc.offset = 0;

        std::vector<Tv> chunk_acc((size_t)cur_num_a * (size_t)num_b, Tv{});
        contract_hvec_sci_for_desc(chunk_desc, src_basis, net, src_vec, chunk_acc.data());

        for (int64 a = 0; a < cur_num_a && out_count < max_entries; ++a)
        {
            const int64 a_global = a_start + a;
            const int64 a_external_idx = (full_block.astrs + a_global) - tgt_basis->all_astrs;
            const bool new_a = is_new_a[a_external_idx];
            const Tv *row = chunk_acc.data() + a * num_b;
            for (int64 b = 0; b < num_b && out_count < max_entries; ++b)
            {
                const int64 b_external_idx = (full_block.bstrs + b) - tgt_basis->all_bstrs;
                if (!new_a && !is_new_b[b_external_idx])
                    continue;
                if (row[b] == Tv{})
                    continue;

                const Tv haa = candidate_diags[full_block.offset + a_global * num_b + b];
                const Tv denom = variational_energy - haa;
                const double denom_norm = std::sqrt(sqnorm(denom));
                if (denom_norm == 0.0)
                    continue;
                const Tv selection_amplitude = row[b] / denom;
                const double selection_norm = std::sqrt(sqnorm(selection_amplitude));
                if (selection_norm <= eps)
                    continue;

                out_entries[out_count++] = {full_block.astrs[a_global], full_block.bstrs[b], row[b]};
            }
        }
    }
    return out_count;
}

template <typename Ti, typename Tv>
int64 sci_hvec_select_external_link_block_bitstr(
    const ExternalLinkSelectContext<Ti> *ctx,
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisManager<Ti> *src_basis,
    const Network_OTF<Ti, Tv> *net,
    const bool *is_new_a,
    const bool *is_new_b,
    int64 block_idx,
    const Tv *src_vec,
    const Tv *candidate_diags,
    Tv variational_energy,
    int chunk_size,
    double eps,
    BufferedEntry<Ti, Tv> *out_entries,
    int64 max_entries,
    const std::vector<LinkBucket<Ti, Tv>> &alpha_new_buckets,
    const std::vector<LinkBucket<Ti, Tv>> &beta_new_buckets,
    const std::vector<SourceStringInfo> &src_a_info,
    const std::vector<SourceStringInfo> &src_b_info,
    int64 num_src_astrs_total,
    int64 num_src_bstrs_total)
{
    const BlockDesc<Ti> &full_block = tgt_basis->blocks[block_idx];
    const int64 num_a_total = full_block.num_a;
    const int64 num_b_total = full_block.num_b;

    int64 block_new_a_count = 0;
    int64 block_new_b_count = 0;
    for (int64 a = 0; a < num_a_total; ++a)
    {
        const int64 a_external_idx = (full_block.astrs + a) - tgt_basis->all_astrs;
        if (is_new_a[a_external_idx])
            ++block_new_a_count;
    }
    for (int64 b = 0; b < num_b_total; ++b)
    {
        const int64 b_external_idx = (full_block.bstrs + b) - tgt_basis->all_bstrs;
        if (is_new_b[b_external_idx])
            ++block_new_b_count;
    }
    const int64 full_candidate_count_heuristic = num_a_total * num_b_total;
    const int64 external_candidate_count_heuristic = block_new_a_count * num_b_total + (num_a_total - block_new_a_count) * block_new_b_count;
    constexpr int64 max_link_entries = 50000000;
    constexpr double dense_threshold = 0.85;
    const double external_candidate_ratio = full_candidate_count_heuristic == 0
                                                ? 0.0
                                                : (double)external_candidate_count_heuristic / (double)full_candidate_count_heuristic;

    if (external_candidate_ratio > dense_threshold)
    {
        return sci_hvec_select_external_block_bitstr(
            tgt_basis, src_basis, net, is_new_a, is_new_b, block_idx, src_vec,
            candidate_diags, variational_energy, chunk_size, eps, out_entries, max_entries);
    }

    if (ctx->alpha_link_entries + ctx->beta_link_entries > max_link_entries)
    {
        return sci_hvec_select_external_block_bitstr(
            tgt_basis, src_basis, net, is_new_a, is_new_b, block_idx, src_vec,
            candidate_diags, variational_energy, chunk_size, eps, out_entries, max_entries);
    }

    std::vector<Ti> new_astrs;
    std::vector<int64> new_a_idxs;
    std::vector<Ti> old_astrs;
    std::vector<int64> old_a_idxs;
    std::vector<Ti> new_bstrs;
    std::vector<int64> new_b_idxs;
    new_astrs.reserve(num_a_total);
    new_a_idxs.reserve(num_a_total);
    old_astrs.reserve(num_a_total);
    old_a_idxs.reserve(num_a_total);
    new_bstrs.reserve(num_b_total);
    new_b_idxs.reserve(num_b_total);

    append_block_frontier_targets(ctx->new_frontiers.alpha_links_by_ax,
                                  full_block.astrs, tgt_basis->all_astrs, num_a_total,
                                  new_astrs, new_a_idxs);
    append_block_frontier_targets(ctx->new_frontiers.beta_links_by_bx,
                                  full_block.bstrs, tgt_basis->all_bstrs, num_b_total,
                                  new_bstrs, new_b_idxs);

    // The beta-new pass is A_old x B_new.  A_old is exactly the source alpha
    // strings in this symmetry, so build it from src_basis instead of scanning
    // every target alpha string.  This also makes A_new x B_new absent from the
    // beta-new pass and leaves those determinants to the alpha-new pass.
    if (full_block.asym < src_basis->num_irreps)
    {
        const Ti *src_astrs = src_basis->astrs_vec[full_block.asym];
        const int64 num_src_astrs = src_basis->num_astrs[full_block.asym];
        for (int64 i = 0; i < num_src_astrs; ++i)
        {
            auto it = tgt_basis->a_idx_map.find(src_astrs[i]);
            if (it == tgt_basis->a_idx_map.end())
                continue;
            old_astrs.push_back(src_astrs[i]);
            old_a_idxs.push_back(it->second);
        }
    }

    int64 out_count = 0;

    const int64 target_a_begin = full_block.astrs - tgt_basis->all_astrs;
    const int64 target_a_end = target_a_begin + num_a_total;
    const int64 target_b_begin = full_block.bstrs - tgt_basis->all_bstrs;
    const int64 target_b_end = target_b_begin + num_b_total;

    std::vector<Tv> dst_acc((size_t)num_a_total * (size_t)num_b_total, Tv{});

#pragma omp parallel
    {
        std::vector<Tv> phase_a_buf((size_t)num_src_astrs_total * RANK3);
        std::vector<Tv> phase_b_buf((size_t)num_src_bstrs_total * RANK3);

        struct DstAInfo
        {
            int a_full;
            int64 src_a_global;
        };
        struct DstBInfo
        {
            int b_full;
            int64 src_b_global;
        };
        std::vector<DstAInfo> dst_a_list;
        std::vector<DstBInfo> dst_b_list;
        dst_a_list.reserve((size_t)num_a_total);
        dst_b_list.reserve((size_t)num_b_total);

        auto process_bucket = [&]<int Rank>(const LinkBucket<Ti, Tv> &bucket, bool skip_new_alpha_targets, Tv *dst_ptr)
        {
            constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;
            const auto &alpha_links = *bucket.alpha_links;
            const auto &beta_links = *bucket.beta_links;
            const auto &group = *bucket.group;
            const int64 num_irreps = src_basis->num_irreps;
            const int64 *block_map = src_basis->block_map;

            for (int64 i = 0; i < num_src_astrs_total; ++i)
                precompute_phase<Rank, Ti, Tv>(src_basis->all_astrs[i], group.unique_zas, group.num_za,
                                               group.wa, phase_a_buf.data() + (size_t)i * MAX_RANK, 1, group.rank);
            for (int64 j = 0; j < num_src_bstrs_total; ++j)
                precompute_phase<Rank, Ti, Tv>(src_basis->all_bstrs[j], group.unique_zbs, group.num_zb,
                                               group.wb, phase_b_buf.data() + (size_t)j * MAX_RANK, 1, group.rank);

            dst_a_list.clear();
            for (int dst_global : alpha_links.colidx)
            {
                if (dst_global < target_a_begin || dst_global >= target_a_end)
                    continue;
                if (skip_new_alpha_targets && is_new_a[dst_global])
                    continue;
                int a_full = dst_global - (int)target_a_begin;
                Ti src_a = full_block.astrs[a_full] ^ group.ax;
                auto it = src_basis->a_idx_map.find(src_a);
                if (it == src_basis->a_idx_map.end())
                    continue;
                int64 src_a_sym = full_block.asym ^ group.asym;
                if (src_a_sym >= num_irreps)
                    continue;
                int64 src_a_global = (src_basis->astrs_vec[src_a_sym] - src_basis->all_astrs) + (int64)it->second;
                if (src_a_global >= num_src_astrs_total || !src_a_info[src_a_global].valid)
                    continue;
                dst_a_list.push_back({a_full, src_a_global});
            }

            dst_b_list.clear();
            for (int dst_global : beta_links.colidx)
            {
                if (dst_global < target_b_begin || dst_global >= target_b_end)
                    continue;
                int b_full = dst_global - (int)target_b_begin;
                Ti src_b = full_block.bstrs[b_full] ^ group.bx;
                auto it = src_basis->b_idx_map.find(src_b);
                if (it == src_basis->b_idx_map.end())
                    continue;
                int64 src_b_sym = full_block.bsym ^ group.bsym;
                if (src_b_sym >= num_irreps)
                    continue;
                int64 src_b_global = (src_basis->bstrs_vec[src_b_sym] - src_basis->all_bstrs) + (int64)it->second;
                if (src_b_global >= num_src_bstrs_total || !src_b_info[src_b_global].valid)
                    continue;
                dst_b_list.push_back({b_full, src_b_global});
            }

            if (dst_a_list.empty() || dst_b_list.empty())
                return;

#pragma omp for schedule(dynamic)
            for (int ai = 0; ai < (int)dst_a_list.size(); ++ai)
            {
                int a_full = dst_a_list[ai].a_full;
                int64 src_a_global = dst_a_list[ai].src_a_global;
                const SourceStringInfo &a_info = src_a_info[src_a_global];
                const Tv *pa = phase_a_buf.data() + (size_t)src_a_global * MAX_RANK;
                Tv *da = dst_ptr + (int64)a_full * num_b_total;

                for (int bi = 0; bi < (int)dst_b_list.size(); ++bi)
                {
                    int b_full = dst_b_list[bi].b_full;
                    int64 src_b_global = dst_b_list[bi].src_b_global;
                    const SourceStringInfo &b_info = src_b_info[src_b_global];
                    const Tv *pb = phase_b_buf.data() + (size_t)src_b_global * MAX_RANK;

                    int64 src_block_idx = block_map[a_info.sym * num_irreps + b_info.sym];
                    if (src_block_idx == -1)
                        continue;
                    const BlockDesc<Ti> &src_block = src_basis->blocks[src_block_idx];
                    const Tv src_amp = src_vec[src_block.offset + a_info.local * src_block.num_b + b_info.local];
                    if (src_amp == Tv{})
                        continue;

                    da[b_full] += src_amp * compute_coeff<Rank, Tv>(0, pa, pb, 1, group.rank);
                }
            }
        };

        for (const auto &b : alpha_new_buckets)
        {
            int r = b.group->rank;
            int dr = (r == 1 || r == 2) ? r : 0;
            switch (dr)
            {
            case 1:
                process_bucket.template operator()<1>(b, false, dst_acc.data());
                break;
            case 2:
                process_bucket.template operator()<2>(b, false, dst_acc.data());
                break;
            default:
                process_bucket.template operator()<0>(b, false, dst_acc.data());
                break;
            }
        }
        for (const auto &b : beta_new_buckets)
        {
            int r = b.group->rank;
            int dr = (r == 1 || r == 2) ? r : 0;
            switch (dr)
            {
            case 1:
                process_bucket.template operator()<1>(b, true, dst_acc.data());
                break;
            case 2:
                process_bucket.template operator()<2>(b, true, dst_acc.data());
                break;
            default:
                process_bucket.template operator()<0>(b, true, dst_acc.data());
                break;
            }
        }
    }

    auto emit_if_selected = [&](int64 a, int64 b, Tv acc)
    {
        if (out_count >= max_entries || acc == Tv{})
            return;
        int64 target_global = full_block.offset + a * num_b_total + b;
        Tv haa = candidate_diags[target_global];
        Tv denom = variational_energy - haa;
        double denom_norm = std::sqrt(sqnorm(denom));
        if (denom_norm == 0.0)
            return;
        Tv selection_amplitude = acc / denom;
        double selection_norm = std::sqrt(sqnorm(selection_amplitude));
        if (selection_norm <= eps)
            return;
        out_entries[out_count++] = {full_block.astrs[a], full_block.bstrs[b], acc};
    };

    for (int64 a = 0; a < num_a_total && out_count < max_entries; ++a)
    {
        int64 a_global = target_a_begin + a;
        bool new_a = is_new_a[a_global];
        Tv *row = dst_acc.data() + a * num_b_total;
        for (int64 b = 0; b < num_b_total && out_count < max_entries; ++b)
        {
            int64 b_global = target_b_begin + b;
            if (!new_a && !is_new_b[b_global])
                continue;
            emit_if_selected(a, b, row[b]);
        }
    }

    return out_count;
}

template <typename Ti, typename Tv>
int64 sci_hvec_select_external_link_all_blocks_bitstr(
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisManager<Ti> *src_basis,
    const Network_OTF<Ti, Tv> *net,
    const NetworkSCI<Ti, Tv> *net_sci,
    const bool *is_new_a,
    const bool *is_new_b,
    const Ti *unique_axs,
    int64 num_unique_axs,
    const Ti *unique_bxs,
    int64 num_unique_bxs,
    const Tv *src_vec,
    const Tv *candidate_diags,
    Tv variational_energy,
    int chunk_size,
    double eps,
    BufferedEntry<Ti, Tv> *out_entries,
    int64 max_entries)
{
    auto ctx = build_external_link_select_context(
        src_basis, tgt_basis, net, is_new_a, is_new_b,
        unique_axs, num_unique_axs, unique_bxs, num_unique_bxs);

    std::vector<LinkBucket<Ti, Tv>> alpha_new_buckets;
    std::vector<LinkBucket<Ti, Tv>> beta_new_buckets;
    alpha_new_buckets.reserve(net_sci->buckets.size());
    beta_new_buckets.reserve(net_sci->buckets.size());

    for (const auto &bkt : net_sci->buckets)
    {
        const auto alpha_new_it = ctx.new_frontiers.alpha_links_by_ax.find(bkt.ax);
        const auto alpha_full_it = ctx.full_links.alpha_links_by_ax.find(bkt.ax);
        const auto beta_new_it = ctx.new_frontiers.beta_links_by_bx.find(bkt.bx);
        const auto beta_full_it = ctx.full_links.beta_links_by_bx.find(bkt.bx);

        if (alpha_new_it != ctx.new_frontiers.alpha_links_by_ax.end() &&
            beta_full_it != ctx.full_links.beta_links_by_bx.end())
            alpha_new_buckets.push_back({&alpha_new_it->second, &beta_full_it->second, bkt.group});

        if (alpha_full_it != ctx.full_links.alpha_links_by_ax.end() &&
            beta_new_it != ctx.new_frontiers.beta_links_by_bx.end())
            beta_new_buckets.push_back({&alpha_full_it->second, &beta_new_it->second, bkt.group});
    }

    int64 num_src_astrs_total = 0;
    int64 num_src_bstrs_total = 0;
    for (int64 sym = 0; sym < src_basis->num_irreps; ++sym)
    {
        num_src_astrs_total += src_basis->num_astrs[sym];
        num_src_bstrs_total += src_basis->num_bstrs[sym];
    }

    std::vector<SourceStringInfo> src_a_info(num_src_astrs_total);
    for (int64 src_a_global = 0; src_a_global < num_src_astrs_total; ++src_a_global)
    {
        SourceStringInfo &info = src_a_info[src_a_global];
        info.sym = get_string_sym(src_basis->all_astrs[src_a_global], src_basis->orbsym);
        if (info.sym >= src_basis->num_irreps)
            continue;
        info.local = src_a_global - (src_basis->astrs_vec[info.sym] - src_basis->all_astrs);
        info.valid = info.local >= 0 && info.local < src_basis->num_astrs[info.sym];
    }

    std::vector<SourceStringInfo> src_b_info(num_src_bstrs_total);
    for (int64 src_b_global = 0; src_b_global < num_src_bstrs_total; ++src_b_global)
    {
        SourceStringInfo &info = src_b_info[src_b_global];
        info.sym = get_string_sym(src_basis->all_bstrs[src_b_global], src_basis->orbsym);
        if (info.sym >= src_basis->num_irreps)
            continue;
        info.local = src_b_global - (src_basis->bstrs_vec[info.sym] - src_basis->all_bstrs);
        info.valid = info.local >= 0 && info.local < src_basis->num_bstrs[info.sym];
    }

    int64 out_count = 0;
    for (int64 blk = 0; blk < tgt_basis->num_blocks && out_count < max_entries; ++blk)
    {
        out_count += sci_hvec_select_external_link_block_bitstr(
            &ctx, tgt_basis, src_basis, net, is_new_a, is_new_b, blk, src_vec,
            candidate_diags, variational_energy, chunk_size, eps,
            out_entries + out_count, max_entries - out_count,
            alpha_new_buckets, beta_new_buckets,
            src_a_info, src_b_info, num_src_astrs_total, num_src_bstrs_total);
    }
    return out_count;
}
