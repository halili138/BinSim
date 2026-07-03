#pragma once
#include "sci_common.hpp"
#include "sci_links.hpp"
#include "sci_hvec.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <unordered_map>

template <typename Tv>
FORCE_INLINE auto sqnorm(const Tv &v)
{
    if constexpr (std::is_arithmetic_v<Tv>) return v * v;
    else return v.real() * v.real() + v.imag() * v.imag();
}


template <typename Ti>
static inline void append_block_frontier_targets(
    const std::unordered_map<Ti, SpinLinkCSR<Ti>> &frontiers_by_mask,
    const Ti *block_strings,
    const Ti *all_strings,
    int64 num_block_strings,
    std::vector<Ti> &strings,
    std::vector<int64> &block_idxs)
{
    if (num_block_strings == 0) return;

    const int64 block_begin = block_strings - all_strings;
    const int64 block_end = block_begin + num_block_strings;
    std::vector<unsigned char> seen(num_block_strings, 0);

    for (const auto &kv : frontiers_by_mask)
    {
        const SpinLinkCSR<Ti> &frontier = kv.second;
        for (int dst_global_idx : frontier.colidx)
        {
            if (dst_global_idx < block_begin || dst_global_idx >= block_end) continue;
            const int64 local_idx = (int64)dst_global_idx - block_begin;
            if (seen[local_idx]) continue;
            seen[local_idx] = 1;
            strings.push_back(block_strings[local_idx]);
            block_idxs.push_back(local_idx);
        }
    }
}


template <typename Ti>
static inline int64 count_spin_link_entries(
    const std::unordered_map<Ti, SpinLinkCSR<Ti>> &links_by_mask)
{
    int64 total = 0;
    for (const auto &kv : links_by_mask)
        total += (int64)kv.second.colidx.size();
    return total;
}



template <typename Ti, typename Tv>
struct ExternalLinkSelectContext
{
    struct GroupPtrRange
    {
        int64 begin = 0;
        int64 end = 0;
    };

    struct LinkBucket
    {
        const SpinLinkCSR<Ti> *alpha_links = nullptr;
        const SpinLinkCSR<Ti> *beta_links = nullptr;
        GroupPtrRange groups;
    };

    std::vector<Ti> unique_axs;
    std::vector<Ti> unique_bxs;
    std::vector<const SVDGroup_OTF<Ti, Tv> *> group_ptrs;
    std::vector<LinkBucket> alpha_new_buckets;
    std::vector<LinkBucket> beta_new_buckets;
    SpinLinksByMask<Ti> full_links;
    SpinLinksByMask<Ti> new_frontiers;
    int64 alpha_link_entries = 0;
    int64 beta_link_entries = 0;
    int64 alpha_new_link_entries = 0;
    int64 beta_new_link_entries = 0;
    int64 num_group_buckets = 0;
    double link_build_time = 0.0;
};

template <typename Ti, typename Tv>
ExternalLinkSelectContext<Ti, Tv> build_external_link_select_context(
    const SciBasisManager<Ti> *src_basis,
    const SciBasisManager<Ti> *tgt_basis,
    const Network_OTF<Ti, Tv> *net,
    const bool *is_new_a,
    const bool *is_new_b,
    const Ti *unique_axs,
    int64 num_unique_axs,
    const Ti *unique_bxs,
    int64 num_unique_bxs,
    const Ti *bucket_axs,
    const Ti *bucket_bxs,
    const int64 *bucket_offsets,
    const int64 *bucket_group_ids,
    int64 num_buckets)
{
    ExternalLinkSelectContext<Ti, Tv> ctx;
    ctx.unique_axs.assign(unique_axs, unique_axs + num_unique_axs);
    ctx.unique_bxs.assign(unique_bxs, unique_bxs + num_unique_bxs);
    const auto t0 = std::chrono::steady_clock::now();
    ctx.full_links = build_spin_links_by_mask(src_basis, tgt_basis, ctx.unique_axs, ctx.unique_bxs);
    ctx.new_frontiers = build_new_frontier_links_by_mask(ctx.full_links, is_new_a, is_new_b);
    const auto t1 = std::chrono::steady_clock::now();
    ctx.link_build_time = std::chrono::duration<double>(t1 - t0).count();
    ctx.alpha_link_entries = count_spin_link_entries(ctx.full_links.alpha_links_by_ax);
    ctx.beta_link_entries = count_spin_link_entries(ctx.full_links.beta_links_by_bx);
    ctx.alpha_new_link_entries = count_spin_link_entries(ctx.new_frontiers.alpha_links_by_ax);
    ctx.beta_new_link_entries = count_spin_link_entries(ctx.new_frontiers.beta_links_by_bx);

    auto group_from_original_idx = [&](int64 original_idx) -> const SVDGroup_OTF<Ti, Tv> *
    {
        if (original_idx < 0 || original_idx >= net->num_groups) return nullptr;
        const int64 sorted_idx = net->sorted_idxs[original_idx];
        if (sorted_idx < 0) return nullptr;
        switch (net->excit_types[original_idx])
        {
        case 0:
            return sorted_idx < (int64)net->diag_groups.size() ? &net->diag_groups[sorted_idx] : nullptr;
        case 1:
            return sorted_idx < (int64)net->pure_a_groups.size() ? &net->pure_a_groups[sorted_idx] : nullptr;
        case 2:
            return sorted_idx < (int64)net->pure_b_groups.size() ? &net->pure_b_groups[sorted_idx] : nullptr;
        case 3:
            return sorted_idx < (int64)net->mixed_groups.size() ? &net->mixed_groups[sorted_idx] : nullptr;
        default:
            return nullptr;
        }
    };

    auto append_bucket = [&](const GroupAxBxKey<Ti> &key, const int64 *ids_begin, const int64 *ids_end)
    {
        const int64 group_begin = (int64)ctx.group_ptrs.size();
        for (const int64 *id = ids_begin; id != ids_end; ++id)
        {
            const SVDGroup_OTF<Ti, Tv> *group = group_from_original_idx(*id);
            if (group != nullptr) ctx.group_ptrs.push_back(group);
        }
        const typename ExternalLinkSelectContext<Ti, Tv>::GroupPtrRange group_range{group_begin, (int64)ctx.group_ptrs.size()};
        if (group_range.begin == group_range.end) return;
        const auto alpha_new_it = ctx.new_frontiers.alpha_links_by_ax.find(key.ax);
        const auto alpha_full_it = ctx.full_links.alpha_links_by_ax.find(key.ax);
        const auto beta_new_it = ctx.new_frontiers.beta_links_by_bx.find(key.bx);
        const auto beta_full_it = ctx.full_links.beta_links_by_bx.find(key.bx);

        if (alpha_new_it != ctx.new_frontiers.alpha_links_by_ax.end() &&
            beta_full_it != ctx.full_links.beta_links_by_bx.end())
            ctx.alpha_new_buckets.push_back({&alpha_new_it->second, &beta_full_it->second, group_range});

        if (alpha_full_it != ctx.full_links.alpha_links_by_ax.end() &&
            beta_new_it != ctx.new_frontiers.beta_links_by_bx.end())
            ctx.beta_new_buckets.push_back({&alpha_full_it->second, &beta_new_it->second, group_range});
    };

    ctx.num_group_buckets = num_buckets;
    ctx.group_ptrs.reserve(bucket_offsets[num_buckets]);
    ctx.alpha_new_buckets.reserve(num_buckets);
    ctx.beta_new_buckets.reserve(num_buckets);
    for (int64 i = 0; i < num_buckets; ++i)
        append_bucket(GroupAxBxKey<Ti>{bucket_axs[i], bucket_bxs[i]},
                      bucket_group_ids + bucket_offsets[i],
                      bucket_group_ids + bucket_offsets[i + 1]);
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

    const bool debug_external_selection = std::getenv("BINSIM_SCI_BITSTR_DEBUG_SELECTION") != nullptr;
    double max_external_hpsi_abs = 0.0;
    double max_external_selection_abs = 0.0;
    int64 external_candidates_scanned = 0;
    int64 external_candidates_passing_eps = 0;

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
                if (row[b] == Tv{}) continue;

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
                if (denom_norm == 0.0) continue;
                const Tv selection_amplitude = row[b] / denom;
                const double selection_norm = std::sqrt(sqnorm(selection_amplitude));
                const bool passes_eps = selection_norm > eps;

                if (!candidate_in_src_basis)
                {
                    ++external_candidates_scanned;
                    max_external_hpsi_abs = std::max(max_external_hpsi_abs, std::sqrt(sqnorm(row[b])));
                    max_external_selection_abs = std::max(max_external_selection_abs, selection_norm);
                    if (passes_eps) ++external_candidates_passing_eps;
                }

                if (!passes_eps) continue;
                out_entries[out_count++] = {candidate_astr, candidate_bstr, row[b]};
            }
        }

        delete[] chunk_acc;
    }

    if (debug_external_selection)
    {
        std::fprintf(stderr,
                     "[sci_bitstr debug] block=%lld external_scanned=%lld "
                     "external_pass_eps=%lld max_external_abs_Hpsi=%.17g "
                     "max_external_abs_Hpsi_over_E_minus_Haa=%.17g out_count=%lld\n",
                     (long long)block_idx,
                     (long long)external_candidates_scanned,
                     (long long)external_candidates_passing_eps,
                     max_external_hpsi_abs,
                     max_external_selection_abs,
                     (long long)out_count);
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
                if (!new_a && !is_new_b[b_external_idx]) continue;
                if (row[b] == Tv{}) continue;

                const Tv haa = candidate_diags[full_block.offset + a_global * num_b + b];
                const Tv denom = variational_energy - haa;
                const double denom_norm = std::sqrt(sqnorm(denom));
                if (denom_norm == 0.0) continue;
                const Tv selection_amplitude = row[b] / denom;
                const double selection_norm = std::sqrt(sqnorm(selection_amplitude));
                if (selection_norm <= eps) continue;

                out_entries[out_count++] = {full_block.astrs[a_global], full_block.bstrs[b], row[b]};
            }
        }
    }
    return out_count;
}

template <typename Ti, typename Tv>
int64 sci_hvec_select_external_link_block_bitstr(
    const ExternalLinkSelectContext<Ti, Tv> *ctx,
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
    const int64 num_b_total = full_block.num_b;

    int64 block_new_a_count = 0;
    int64 block_new_b_count = 0;
    for (int64 a = 0; a < num_a_total; ++a)
    {
        const int64 a_external_idx = (full_block.astrs + a) - tgt_basis->all_astrs;
        if (is_new_a[a_external_idx]) ++block_new_a_count;
    }
    for (int64 b = 0; b < num_b_total; ++b)
    {
        const int64 b_external_idx = (full_block.bstrs + b) - tgt_basis->all_bstrs;
        if (is_new_b[b_external_idx]) ++block_new_b_count;
    }
    const int64 full_candidate_count_heuristic = num_a_total * num_b_total;
    const int64 external_candidate_count_heuristic =
        block_new_a_count * num_b_total + (num_a_total - block_new_a_count) * block_new_b_count;
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

    const bool print_perf = std::getenv("BINSIM_SCI_BITSTR_PRINT_SELECT_PERF") != nullptr ||
                            std::getenv("BINSIM_SCI_BITSTR_PRINT_LINK_SELECT_PERF") != nullptr;
    const bool check_mask_scan = std::getenv("BINSIM_SCI_BITSTR_CHECK_EXTERNAL_SELECT") != nullptr ||
                                 std::getenv("BINSIM_SCI_BITSTR_CHECK_LINK_SELECT") != nullptr;
    const auto t0 = std::chrono::steady_clock::now();

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
            if (it == tgt_basis->a_idx_map.end()) continue;
            old_astrs.push_back(src_astrs[i]);
            old_a_idxs.push_back(it->second);
        }
    }

    int64 out_count = 0;
    int64 external_candidate_count = 0;
    int64 generated_target_edges = 0;

    struct AccumPair
    {
        int64 target_global;
        Tv hpsi;
    };

    struct DenseAccumState
    {
        std::vector<Tv> values;
        std::vector<unsigned char> touched;
        std::vector<int64> touched_targets;
    };

    struct SourceStringInfo
    {
        int64 sym = -1;
        int64 local = -1;
        bool valid = false;
    };

    using LinkBucket = typename ExternalLinkSelectContext<Ti, Tv>::LinkBucket;

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
        if (info.sym >= src_basis->num_irreps) continue;
        info.local = src_a_global - (src_basis->astrs_vec[info.sym] - src_basis->all_astrs);
        info.valid = info.local >= 0 && info.local < src_basis->num_astrs[info.sym];
    }

    std::vector<SourceStringInfo> src_b_info(num_src_bstrs_total);
    for (int64 src_b_global = 0; src_b_global < num_src_bstrs_total; ++src_b_global)
    {
        SourceStringInfo &info = src_b_info[src_b_global];
        info.sym = get_string_sym(src_basis->all_bstrs[src_b_global], src_basis->orbsym);
        if (info.sym >= src_basis->num_irreps) continue;
        info.local = src_b_global - (src_basis->bstrs_vec[info.sym] - src_basis->all_bstrs);
        info.valid = info.local >= 0 && info.local < src_basis->num_bstrs[info.sym];
    }

    const int64 target_a_begin = full_block.astrs - tgt_basis->all_astrs;
    const int64 target_a_end = target_a_begin + num_a_total;
    const int64 target_b_begin = full_block.bstrs - tgt_basis->all_bstrs;
    const int64 target_b_end = target_b_begin + num_b_total;

    const int64 reachable_target_count = (int64)new_astrs.size() * num_b_total
                                       + (int64)old_astrs.size() * (int64)new_bstrs.size();
    constexpr size_t max_dense_accumulator_bytes = (size_t)512 * 1024 * 1024;
    const bool dense_accumulator_entry_count_fits = num_a_total >= 0 && num_b_total >= 0 &&
        (num_b_total == 0 || (size_t)num_a_total <= std::numeric_limits<size_t>::max() / (size_t)num_b_total);
    const size_t dense_accumulator_entries = dense_accumulator_entry_count_fits
        ? (size_t)num_a_total * (size_t)num_b_total
        : std::numeric_limits<size_t>::max();
    const bool use_dense_accumulator = dense_accumulator_entries <= max_dense_accumulator_bytes / sizeof(Tv);

    DenseAccumState dense_acc;
    std::vector<AccumPair> sparse_acc_pairs;
    if (use_dense_accumulator)
    {
        dense_acc.values.assign(dense_accumulator_entries, Tv{});
        dense_acc.touched.assign(dense_accumulator_entries, 0);
        dense_acc.touched_targets.reserve((size_t)reachable_target_count);
    }
    else
    {
        sparse_acc_pairs.reserve((size_t)std::min<int64>(reachable_target_count, max_link_entries));
    }

    auto append_accumulated_hpsi = [&](int64 target_global, Tv hpsi)
    {
        if (hpsi == Tv{}) return;
        if (use_dense_accumulator)
        {
            const int64 local_target = target_global - full_block.offset;
            Tv &acc = dense_acc.values[(size_t)local_target];
            if (!dense_acc.touched[(size_t)local_target])
            {
                dense_acc.touched[(size_t)local_target] = 1;
                dense_acc.touched_targets.push_back(target_global);
            }
            acc += hpsi;
        }
        else
        {
            sparse_acc_pairs.push_back({target_global, hpsi});
        }
    };

    auto accumulate_for_buckets = [&](const std::vector<LinkBucket> &buckets, bool skip_new_alpha_targets)
    {
        constexpr int max_rank = RANK3;
        for (const LinkBucket &bucket : buckets)
        {
            const SpinLinkCSR<Ti> &alpha_links = *bucket.alpha_links;
            const SpinLinkCSR<Ti> &beta_links = *bucket.beta_links;
            const int64 num_src_a = alpha_links.rowptr.empty() ? 0 : (int64)alpha_links.rowptr.size() - 1;
            const int64 num_src_b = beta_links.rowptr.empty() ? 0 : (int64)beta_links.rowptr.size() - 1;
            const int64 group_count = bucket.groups.end - bucket.groups.begin;
            std::vector<Tv> phase_a((size_t)group_count * max_rank);
            std::vector<Tv> phase_b((size_t)group_count * max_rank);

            for (int64 src_a_global = 0; src_a_global < num_src_a; ++src_a_global)
            {
                if (src_a_global >= (int64)src_a_info.size() || !src_a_info[src_a_global].valid) continue;
                const SourceStringInfo &a_info = src_a_info[src_a_global];
                const Ti src_astr = src_basis->all_astrs[src_a_global];
                for (int64 pa = alpha_links.rowptr[src_a_global]; pa < alpha_links.rowptr[src_a_global + 1]; ++pa)
                {
                    const int64 dst_a_global = alpha_links.colidx[pa];
                    if (dst_a_global < target_a_begin || dst_a_global >= target_a_end) continue;
                    if (skip_new_alpha_targets && is_new_a[dst_a_global]) continue;
                    const int64 a_full = dst_a_global - target_a_begin;
                    const Ti dst_astr = tgt_basis->all_astrs[dst_a_global];

                    for (int64 local_group_idx = 0; local_group_idx < group_count; ++local_group_idx)
                    {
                        const SVDGroup_OTF<Ti, Tv> &group = *ctx->group_ptrs[bucket.groups.begin + local_group_idx];
                        const Ti phase_astr = (group.ax == Ti{}) ? dst_astr : src_astr;
                        precompute_phase<0, Ti, Tv>(phase_astr, group.unique_zas, group.num_za,
                                                    group.wa, phase_a.data() + (size_t)local_group_idx * max_rank,
                                                    1, group.rank);
                    }

                    for (int64 src_b_global = 0; src_b_global < num_src_b; ++src_b_global)
                    {
                        if (src_b_global >= (int64)src_b_info.size() || !src_b_info[src_b_global].valid) continue;
                        const SourceStringInfo &b_info = src_b_info[src_b_global];
                        const int64 src_block_idx = src_basis->block_map[a_info.sym * src_basis->num_irreps + b_info.sym];
                        if (src_block_idx == -1) continue;
                        const BlockDesc<Ti> &src_block = src_basis->blocks[src_block_idx];
                        const Tv src_amp = src_vec[src_block.offset + a_info.local * src_block.num_b + b_info.local];
                        if (src_amp == Tv{}) continue;
                        const Ti src_bstr = src_basis->all_bstrs[src_b_global];

                        for (int64 pb = beta_links.rowptr[src_b_global]; pb < beta_links.rowptr[src_b_global + 1]; ++pb)
                        {
                            ++generated_target_edges;
                            const int64 dst_b_global = beta_links.colidx[pb];
                            if (dst_b_global < target_b_begin || dst_b_global >= target_b_end) continue;
                            const int64 b_full = dst_b_global - target_b_begin;
                            const Ti dst_bstr = tgt_basis->all_bstrs[dst_b_global];

                            Tv hpsi = {};
                            for (int64 local_group_idx = 0; local_group_idx < group_count; ++local_group_idx)
                            {
                                const SVDGroup_OTF<Ti, Tv> &group = *ctx->group_ptrs[bucket.groups.begin + local_group_idx];
                                const Ti phase_bstr = (group.bx == Ti{}) ? dst_bstr : src_bstr;
                                Tv *pb_phase = phase_b.data() + (size_t)local_group_idx * max_rank;
                                precompute_phase<0, Ti, Tv>(phase_bstr, group.unique_zbs, group.num_zb,
                                                            group.wb, pb_phase, 1, group.rank);
                                hpsi += src_amp * compute_group_coeff_from_phases(
                                    phase_a.data() + (size_t)local_group_idx * max_rank,
                                    pb_phase, group.rank);
                            }
                            append_accumulated_hpsi(full_block.offset + a_full * num_b_total + b_full, hpsi);
                        }
                    }
                }
            }
        }
    };

    // External space = (A_new x B_all) union (A_old x B_new).  This assigns
    // A_new x B_new only to the first subspace, avoiding duplicate selection.
    // Contributions are accumulated by full target determinant before applying
    // the selection threshold so all source determinants and SVD groups that
    // reach the same external determinant contribute to the final Hψ value.
    const auto t_accumulate0 = std::chrono::steady_clock::now();
    external_candidate_count += (int64)new_astrs.size() * num_b_total;
    accumulate_for_buckets(ctx->alpha_new_buckets, false);
    external_candidate_count += (int64)old_astrs.size() * (int64)new_bstrs.size();
    accumulate_for_buckets(ctx->beta_new_buckets, true);
    const auto t_accumulate1 = std::chrono::steady_clock::now();

    const auto t_threshold0 = std::chrono::steady_clock::now();
    int64 unique_accum_target_count = 0;
    auto emit_if_selected = [&](int64 target_global, const Tv &acc)
    {
        if (out_count >= max_entries || acc == Tv{}) return;
        const Tv haa = candidate_diags[target_global];
        const Tv denom = variational_energy - haa;
        const double denom_norm = std::sqrt(sqnorm(denom));
        if (denom_norm == 0.0) return;
        const Tv selection_amplitude = acc / denom;
        const double selection_norm = std::sqrt(sqnorm(selection_amplitude));
        if (selection_norm <= eps) return;

        const int64 local_target = target_global - full_block.offset;
        const int64 a_full = local_target / num_b_total;
        const int64 b_full = local_target - a_full * num_b_total;
        out_entries[out_count++] = {full_block.astrs[a_full], full_block.bstrs[b_full], acc};
    };

    if (use_dense_accumulator)
    {
        unique_accum_target_count = (int64)dense_acc.touched_targets.size();
        for (int64 target_global : dense_acc.touched_targets)
        {
            if (out_count >= max_entries) break;
            emit_if_selected(target_global, dense_acc.values[(size_t)(target_global - full_block.offset)]);
        }
    }
    else
    {
        std::sort(sparse_acc_pairs.begin(), sparse_acc_pairs.end(),
                  [](const AccumPair &lhs, const AccumPair &rhs)
                  {
                      return lhs.target_global < rhs.target_global;
                  });
        for (int64 i = 0; i < (int64)sparse_acc_pairs.size() && out_count < max_entries;)
        {
            const int64 target_global = sparse_acc_pairs[i].target_global;
            ++unique_accum_target_count;
            Tv acc = {};
            do
            {
                acc += sparse_acc_pairs[i].hpsi;
                ++i;
            } while (i < (int64)sparse_acc_pairs.size() && sparse_acc_pairs[i].target_global == target_global);
            emit_if_selected(target_global, acc);
        }
    }

    const auto t_threshold1 = std::chrono::steady_clock::now();

    if (check_mask_scan && out_count < max_entries)
    {
        std::vector<std::pair<Ti, Ti>> block_selected;
        block_selected.reserve(out_count);
        for (int64 i = 0; i < out_count; ++i)
            block_selected.emplace_back(out_entries[i].astr, out_entries[i].bstr);
        std::sort(block_selected.begin(), block_selected.end());

        std::vector<std::pair<Ti, Ti>> mask_selected;
        for (int64 a_start = 0; a_start < num_a_total; a_start += chunk_size)
        {
            const int64 a_end = std::min(a_start + (int64)chunk_size, num_a_total);
            const int64 cur_num_a = a_end - a_start;

            BlockDesc<Ti> chunk_desc = full_block;
            chunk_desc.astrs = full_block.astrs + a_start;
            chunk_desc.num_a = cur_num_a;
            chunk_desc.offset = 0;

            std::vector<Tv> chunk_acc(cur_num_a * num_b_total, Tv{});
            contract_hvec_sci_for_desc(chunk_desc, src_basis, net, src_vec, chunk_acc.data());

            for (int64 a = 0; a < cur_num_a; ++a)
            {
                const int64 a_global = a_start + a;
                const int64 a_external_idx = (full_block.astrs + a_global) - tgt_basis->all_astrs;
                const bool new_a = is_new_a[a_external_idx];
                const Tv *row = chunk_acc.data() + a * num_b_total;
                for (int64 b = 0; b < num_b_total; ++b)
                {
                    const int64 b_external_idx = (full_block.bstrs + b) - tgt_basis->all_bstrs;
                    if (!new_a && !is_new_b[b_external_idx]) continue;
                    if (row[b] == Tv{}) continue;

                    const Tv haa = candidate_diags[full_block.offset + a_global * num_b_total + b];
                    const Tv denom = variational_energy - haa;
                    const double denom_norm = std::sqrt(sqnorm(denom));
                    if (denom_norm == 0.0) continue;
                    const Tv selection_amplitude = row[b] / denom;
                    const double selection_norm = std::sqrt(sqnorm(selection_amplitude));
                    if (selection_norm <= eps) continue;

                    mask_selected.emplace_back(full_block.astrs[a_global], full_block.bstrs[b]);
                }
            }
        }
        std::sort(mask_selected.begin(), mask_selected.end());
        if (block_selected != mask_selected)
        {
            std::fprintf(stderr,
                         "[sci_bitstr link check] external block select mismatch: "
                         "true_external_selected=%lld mask_scan_selected=%lld\n",
                         (long long)block_selected.size(),
                         (long long)mask_selected.size());
        }
        else
        {
            std::fprintf(stderr,
                         "[sci_bitstr link check] external block select matches mask-scan selected set: %lld entries\n",
                         (long long)block_selected.size());
        }
    }

    if (print_perf)
    {
        const auto t1 = std::chrono::steady_clock::now();
        const double link_select_time = std::chrono::duration<double>(t1 - t0).count();
        const double link_build_time = ctx->link_build_time;
        const double accumulate_time = std::chrono::duration<double>(t_accumulate1 - t_accumulate0).count();
        const double threshold_time = std::chrono::duration<double>(t_threshold1 - t_threshold0).count();
        const int64 full_candidate_count = num_a_total * num_b_total;
        std::fprintf(stderr,
                     "[sci_bitstr link perf] "
                     "num_unique_ax=%lld num_unique_bx=%lld num_ax_bx_buckets=%lld "
                     "alpha_link_entries=%lld beta_link_entries=%lld "
                     "alpha_new_link_entries=%lld beta_new_link_entries=%lld "
                     "generated_target_edges=%lld unique_accum_targets=%lld selected_count=%lld "
                     "link_build_time=%.9f link_select_time=%.9f accumulate_time=%.9f threshold_time=%.9f "
                     "full_target_select_time=unmeasured external_block_select_time=unmeasured "
                     "link_frontier_select_time=%.9f full_candidate_count=%lld "
                     "external_block_candidate_count=%lld link_generated_edge_count=%lld "
                     "space_model=link_csr_plus_unique_targets_no_Nstr_times_ngroups\n",
                     (long long)ctx->unique_axs.size(),
                     (long long)ctx->unique_bxs.size(),
                     (long long)ctx->num_group_buckets,
                     (long long)ctx->alpha_link_entries,
                     (long long)ctx->beta_link_entries,
                     (long long)ctx->alpha_new_link_entries,
                     (long long)ctx->beta_new_link_entries,
                     (long long)generated_target_edges,
                     (long long)unique_accum_target_count,
                     (long long)out_count,
                     link_build_time,
                     link_select_time,
                     accumulate_time,
                     threshold_time,
                     link_select_time,
                     (long long)full_candidate_count,
                     (long long)external_candidate_count,
                     (long long)generated_target_edges);
    }

    return out_count;
}



template <typename Ti, typename Tv>
int64 sci_hvec_select_external_bitstr(
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
    return sci_hvec_select_external_block_bitstr(
        tgt_basis, src_basis, net, is_new_a, is_new_b, block_idx, src_vec,
        candidate_diags, variational_energy, chunk_size, eps, out_entries, max_entries);
}
