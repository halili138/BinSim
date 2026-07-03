#pragma once
#include "sci_common.hpp"
#include "sci_links.hpp"
#include "sci_hvec.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>

template <typename Tv>
FORCE_INLINE auto sqnorm(const Tv &v)
{
    if constexpr (std::is_arithmetic_v<Tv>) return v * v;
    else return v.real() * v.real() + v.imag() * v.imag();
}


template <typename Ti, typename Tv>
static inline void collect_unique_spin_masks(
    const Network_OTF<Ti, Tv> *net,
    std::vector<Ti> &unique_axs,
    std::vector<Ti> &unique_bxs)
{
    std::unordered_set<Ti> ax_seen;
    std::unordered_set<Ti> bx_seen;
    auto add_groups = [&](const std::vector<SVDGroup_OTF<Ti, Tv>> &groups)
    {
        for (const auto &group : groups)
        {
            if (ax_seen.insert(group.ax).second) unique_axs.push_back(group.ax);
            if (bx_seen.insert(group.bx).second) unique_bxs.push_back(group.bx);
        }
    };
    add_groups(net->diag_groups);
    add_groups(net->pure_a_groups);
    add_groups(net->pure_b_groups);
    add_groups(net->mixed_groups);
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

    std::vector<Ti> unique_axs;
    std::vector<Ti> unique_bxs;
    collect_unique_spin_masks(net, unique_axs, unique_bxs);
    const auto t_link_build0 = std::chrono::steady_clock::now();
    SpinLinksByMask<Ti> full_links = build_spin_links_by_mask(src_basis, tgt_basis, unique_axs, unique_bxs);
    SpinLinksByMask<Ti> new_frontiers = build_new_frontier_links_by_mask(full_links, is_new_a, is_new_b);
    const auto t_link_build1 = std::chrono::steady_clock::now();

    const int64 alpha_link_entries = count_spin_link_entries(full_links.alpha_links_by_ax);
    const int64 beta_link_entries = count_spin_link_entries(full_links.beta_links_by_bx);
    const int64 alpha_new_link_entries = count_spin_link_entries(new_frontiers.alpha_links_by_ax);
    const int64 beta_new_link_entries = count_spin_link_entries(new_frontiers.beta_links_by_bx);
    if (alpha_link_entries + beta_link_entries > max_link_entries)
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

    append_block_frontier_targets(new_frontiers.alpha_links_by_ax,
                                  full_block.astrs, tgt_basis->all_astrs, num_a_total,
                                  new_astrs, new_a_idxs);
    append_block_frontier_targets(new_frontiers.beta_links_by_bx,
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
    int64 contraction_candidate_count = 0;
    int64 generated_target_edges = 0;

    struct AccumTarget
    {
        int64 target_global;
        int64 a_full;
        int64 b_full;
    };

    std::unordered_map<int64, Tv> target_acc;
    std::vector<AccumTarget> target_order;

    auto accumulate_target = [&](int64 a_full, int64 b_full, Tv hpsi)
    {
        if (hpsi == Tv{}) return;

        const int64 target_global = full_block.offset + a_full * num_b_total + b_full;
        auto inserted = target_acc.emplace(target_global, Tv{});
        if (inserted.second)
            target_order.push_back({target_global, a_full, b_full});
        inserted.first->second += hpsi;
    };

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

    auto accumulate_alpha_new_frontier = [&]()
    {
        const int64 target_a_begin = full_block.astrs - tgt_basis->all_astrs;
        const int64 target_a_end = target_a_begin + num_a_total;
        const int64 target_b_begin = full_block.bstrs - tgt_basis->all_bstrs;
        const int64 target_b_end = target_b_begin + num_b_total;

        for (const GroupAxBxKey<Ti> &key : net->group_index.unique_ax_bx_pairs)
        {
            const auto alpha_it = new_frontiers.alpha_links_by_ax.find(key.ax);
            const auto beta_it = full_links.beta_links_by_bx.find(key.bx);
            const auto groups_it = net->group_index.groups_by_ax_bx.find(key);
            if (alpha_it == new_frontiers.alpha_links_by_ax.end() ||
                beta_it == full_links.beta_links_by_bx.end() ||
                groups_it == net->group_index.groups_by_ax_bx.end())
                continue;

            const SpinLinkCSR<Ti> &alpha_links = alpha_it->second;
            const SpinLinkCSR<Ti> &beta_links = beta_it->second;
            const std::vector<int64> &group_original_idxs = groups_it->second;

            const int64 num_src_a = alpha_links.rowptr.empty() ? 0 : (int64)alpha_links.rowptr.size() - 1;
            const int64 num_src_b = beta_links.rowptr.empty() ? 0 : (int64)beta_links.rowptr.size() - 1;
            for (int64 src_a_global = 0; src_a_global < num_src_a; ++src_a_global)
            {
                const int64 src_a_sym = get_string_sym(src_basis->all_astrs[src_a_global], src_basis->orbsym);
                if (src_a_sym >= src_basis->num_irreps) continue;
                const int64 src_a_local = src_a_global - (src_basis->astrs_vec[src_a_sym] - src_basis->all_astrs);
                if (src_a_local < 0 || src_a_local >= src_basis->num_astrs[src_a_sym]) continue;

                for (int64 pa = alpha_links.rowptr[src_a_global]; pa < alpha_links.rowptr[src_a_global + 1]; ++pa)
                {
                    const int64 dst_a_global = alpha_links.colidx[pa];
                    if (dst_a_global < target_a_begin || dst_a_global >= target_a_end) continue;
                    const int64 a_full = dst_a_global - target_a_begin;

                    for (int64 src_b_global = 0; src_b_global < num_src_b; ++src_b_global)
                    {
                        const int64 src_b_sym = get_string_sym(src_basis->all_bstrs[src_b_global], src_basis->orbsym);
                        if (src_b_sym >= src_basis->num_irreps) continue;
                        const int64 src_block_idx = src_basis->block_map[src_a_sym * src_basis->num_irreps + src_b_sym];
                        if (src_block_idx == -1) continue;
                        const BlockDesc<Ti> &src_block = src_basis->blocks[src_block_idx];
                        const int64 src_b_local = src_b_global - (src_basis->bstrs_vec[src_b_sym] - src_basis->all_bstrs);
                        if (src_b_local < 0 || src_b_local >= src_basis->num_bstrs[src_b_sym]) continue;
                        const Tv src_amp = src_vec[src_block.offset + src_a_local * src_block.num_b + src_b_local];
                        if (src_amp == Tv{}) continue;

                        for (int64 pb = beta_links.rowptr[src_b_global]; pb < beta_links.rowptr[src_b_global + 1]; ++pb)
                        {
                            ++generated_target_edges;
                            const int64 dst_b_global = beta_links.colidx[pb];
                            if (dst_b_global < target_b_begin || dst_b_global >= target_b_end) continue;
                            const int64 b_full = dst_b_global - target_b_begin;

                            Tv hpsi = {};
                            for (int64 original_idx : group_original_idxs)
                            {
                                const SVDGroup_OTF<Ti, Tv> *group = group_from_original_idx(original_idx);
                                if (group == nullptr) continue;

                                hpsi += src_amp * compute_group_coeff_for_pair(
                                    *group,
                                    src_basis->all_astrs[src_a_global],
                                    src_basis->all_bstrs[src_b_global],
                                    tgt_basis->all_astrs[dst_a_global],
                                    tgt_basis->all_bstrs[dst_b_global]);
                            }
                            accumulate_target(a_full, b_full, hpsi);
                        }
                    }
                }
            }
        }
    };

    auto accumulate_beta_new_frontier = [&]()
    {
        const int64 target_a_begin = full_block.astrs - tgt_basis->all_astrs;
        const int64 target_a_end = target_a_begin + num_a_total;
        const int64 target_b_begin = full_block.bstrs - tgt_basis->all_bstrs;
        const int64 target_b_end = target_b_begin + num_b_total;

        for (const GroupAxBxKey<Ti> &key : net->group_index.unique_ax_bx_pairs)
        {
            const auto alpha_it = full_links.alpha_links_by_ax.find(key.ax);
            const auto beta_it = new_frontiers.beta_links_by_bx.find(key.bx);
            const auto groups_it = net->group_index.groups_by_ax_bx.find(key);
            if (alpha_it == full_links.alpha_links_by_ax.end() ||
                beta_it == new_frontiers.beta_links_by_bx.end() ||
                groups_it == net->group_index.groups_by_ax_bx.end())
                continue;

            const SpinLinkCSR<Ti> &alpha_links = alpha_it->second;
            const SpinLinkCSR<Ti> &beta_links = beta_it->second;
            const std::vector<int64> &group_original_idxs = groups_it->second;

            const int64 num_src_a = alpha_links.rowptr.empty() ? 0 : (int64)alpha_links.rowptr.size() - 1;
            const int64 num_src_b = beta_links.rowptr.empty() ? 0 : (int64)beta_links.rowptr.size() - 1;
            for (int64 src_a_global = 0; src_a_global < num_src_a; ++src_a_global)
            {
                const int64 src_a_sym = get_string_sym(src_basis->all_astrs[src_a_global], src_basis->orbsym);
                if (src_a_sym >= src_basis->num_irreps) continue;
                const int64 src_a_local = src_a_global - (src_basis->astrs_vec[src_a_sym] - src_basis->all_astrs);
                if (src_a_local < 0 || src_a_local >= src_basis->num_astrs[src_a_sym]) continue;

                for (int64 pa = alpha_links.rowptr[src_a_global]; pa < alpha_links.rowptr[src_a_global + 1]; ++pa)
                {
                    const int64 dst_a_global = alpha_links.colidx[pa];
                    if (dst_a_global < target_a_begin || dst_a_global >= target_a_end) continue;
                    if (is_new_a[dst_a_global]) continue;
                    const int64 a_full = dst_a_global - target_a_begin;

                    for (int64 src_b_global = 0; src_b_global < num_src_b; ++src_b_global)
                    {
                        const int64 src_b_sym = get_string_sym(src_basis->all_bstrs[src_b_global], src_basis->orbsym);
                        if (src_b_sym >= src_basis->num_irreps) continue;
                        const int64 src_block_idx = src_basis->block_map[src_a_sym * src_basis->num_irreps + src_b_sym];
                        if (src_block_idx == -1) continue;
                        const BlockDesc<Ti> &src_block = src_basis->blocks[src_block_idx];
                        const int64 src_b_local = src_b_global - (src_basis->bstrs_vec[src_b_sym] - src_basis->all_bstrs);
                        if (src_b_local < 0 || src_b_local >= src_basis->num_bstrs[src_b_sym]) continue;
                        const Tv src_amp = src_vec[src_block.offset + src_a_local * src_block.num_b + src_b_local];
                        if (src_amp == Tv{}) continue;

                        for (int64 pb = beta_links.rowptr[src_b_global]; pb < beta_links.rowptr[src_b_global + 1]; ++pb)
                        {
                            ++generated_target_edges;
                            const int64 dst_b_global = beta_links.colidx[pb];
                            if (dst_b_global < target_b_begin || dst_b_global >= target_b_end) continue;
                            const int64 b_full = dst_b_global - target_b_begin;

                            Tv hpsi = {};
                            for (int64 original_idx : group_original_idxs)
                            {
                                const SVDGroup_OTF<Ti, Tv> *group = group_from_original_idx(original_idx);
                                if (group == nullptr) continue;

                                hpsi += src_amp * compute_group_coeff_for_pair(
                                    *group,
                                    src_basis->all_astrs[src_a_global],
                                    src_basis->all_bstrs[src_b_global],
                                    tgt_basis->all_astrs[dst_a_global],
                                    tgt_basis->all_bstrs[dst_b_global]);
                            }
                            accumulate_target(a_full, b_full, hpsi);
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
    std::vector<int64> all_b_idxs(num_b_total);
    for (int64 b = 0; b < num_b_total; ++b) all_b_idxs[b] = b;

    const int64 reachable_target_count = (int64)new_astrs.size() * num_b_total
                                       + (int64)old_astrs.size() * (int64)new_bstrs.size();
    target_acc.reserve((size_t)reachable_target_count);
    target_order.reserve((size_t)reachable_target_count);

    // Alpha-new pass: traverse each (ax,bx) link-frontier bucket, keep only
    // alpha links whose dst_a is new, use the full beta source->target link,
    // and accumulate all contributions to A_new x B_all.  This includes
    // A_new x B_new, so the beta-new pass below is restricted to A_old x B_new.
    const auto t_accumulate0 = std::chrono::steady_clock::now();
    external_candidate_count += (int64)new_astrs.size() * num_b_total;
    accumulate_alpha_new_frontier();
    external_candidate_count += (int64)old_astrs.size() * (int64)new_bstrs.size();
    accumulate_beta_new_frontier();
    const auto t_accumulate1 = std::chrono::steady_clock::now();

    const auto t_threshold0 = std::chrono::steady_clock::now();
    for (const AccumTarget &target : target_order)
    {
        if (out_count >= max_entries) break;
        const auto acc_it = target_acc.find(target.target_global);
        if (acc_it == target_acc.end() || acc_it->second == Tv{}) continue;

        const Tv haa = candidate_diags[target.target_global];
        const Tv denom = variational_energy - haa;
        const double denom_norm = std::sqrt(sqnorm(denom));
        if (denom_norm == 0.0) continue;
        const Tv selection_amplitude = acc_it->second / denom;
        const double selection_norm = std::sqrt(sqnorm(selection_amplitude));
        if (selection_norm <= eps) continue;

        out_entries[out_count++] = {full_block.astrs[target.a_full],
                                    full_block.bstrs[target.b_full],
                                    acc_it->second};
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
        const double link_build_time = std::chrono::duration<double>(t_link_build1 - t_link_build0).count();
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
                     (long long)unique_axs.size(),
                     (long long)unique_bxs.size(),
                     (long long)net->group_index.unique_ax_bx_pairs.size(),
                     (long long)alpha_link_entries,
                     (long long)beta_link_entries,
                     (long long)alpha_new_link_entries,
                     (long long)beta_new_link_entries,
                     (long long)generated_target_edges,
                     (long long)target_acc.size(),
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
