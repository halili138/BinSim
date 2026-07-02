#pragma once
#include "sci_common.hpp"
#include "sci_hvec.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

template <typename Tv>
FORCE_INLINE auto sqnorm(const Tv &v)
{
    if constexpr (std::is_arithmetic_v<Tv>) return v * v;
    else return v.real() * v.real() + v.imag() * v.imag();
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

    const bool print_perf = std::getenv("BINSIM_SCI_BITSTR_PRINT_SELECT_PERF") != nullptr;
    const bool check_mask_scan = std::getenv("BINSIM_SCI_BITSTR_CHECK_EXTERNAL_SELECT") != nullptr;
    const auto t0 = std::chrono::steady_clock::now();

    std::vector<Ti> new_astrs;
    std::vector<int64> new_a_idxs;
    std::vector<Ti> old_astrs;
    std::vector<int64> old_a_idxs;
    new_astrs.reserve(num_a_total);
    new_a_idxs.reserve(num_a_total);
    old_astrs.reserve(num_a_total);
    old_a_idxs.reserve(num_a_total);

    for (int64 a = 0; a < num_a_total; ++a)
    {
        const int64 a_external_idx = (full_block.astrs + a) - tgt_basis->all_astrs;
        if (is_new_a[a_external_idx])
        {
            new_astrs.push_back(full_block.astrs[a]);
            new_a_idxs.push_back(a);
        }
        else
        {
            old_astrs.push_back(full_block.astrs[a]);
            old_a_idxs.push_back(a);
        }
    }

    std::vector<Ti> new_bstrs;
    std::vector<int64> new_b_idxs;
    new_bstrs.reserve(num_b_total);
    new_b_idxs.reserve(num_b_total);
    for (int64 b = 0; b < num_b_total; ++b)
    {
        const int64 b_external_idx = (full_block.bstrs + b) - tgt_basis->all_bstrs;
        if (is_new_b[b_external_idx])
        {
            new_bstrs.push_back(full_block.bstrs[b]);
            new_b_idxs.push_back(b);
        }
    }

    int64 out_count = 0;
    int64 external_candidate_count = 0;
    int64 contraction_candidate_count = 0;

    auto select_subblock = [&](const Ti *astrs, const int64 *a_idxs, int64 num_a,
                               const Ti *bstrs, const int64 *b_idxs, int64 num_b)
    {
        if (num_a == 0 || num_b == 0 || out_count >= max_entries) return;
        external_candidate_count += num_a * num_b;

        BlockDesc<Ti> subblock = full_block;
        subblock.bstrs = bstrs;
        subblock.num_b = num_b;
        subblock.offset = 0;

        for (int64 a_start = 0; a_start < num_a && out_count < max_entries; a_start += chunk_size)
        {
            const int64 a_end = std::min(a_start + (int64)chunk_size, num_a);
            const int64 cur_num_a = a_end - a_start;

            subblock.astrs = astrs + a_start;
            subblock.num_a = cur_num_a;

            std::vector<Tv> chunk_acc(cur_num_a * num_b, Tv{});
            contract_hvec_sci_for_desc(subblock, src_basis, net, src_vec, chunk_acc.data());
            contraction_candidate_count += cur_num_a * num_b;

            for (int64 a = 0; a < cur_num_a && out_count < max_entries; ++a)
            {
                const int64 a_full = a_idxs[a_start + a];
                const Tv *row = chunk_acc.data() + a * num_b;
                for (int64 b = 0; b < num_b && out_count < max_entries; ++b)
                {
                    if (row[b] == Tv{}) continue;

                    const int64 b_full = b_idxs[b];
                    const Tv haa = candidate_diags[full_block.offset + a_full * num_b_total + b_full];
                    const Tv denom = variational_energy - haa;
                    const double denom_norm = std::sqrt(sqnorm(denom));
                    if (denom_norm == 0.0) continue;
                    const Tv selection_amplitude = row[b] / denom;
                    const double selection_norm = std::sqrt(sqnorm(selection_amplitude));
                    if (selection_norm <= eps) continue;

                    out_entries[out_count++] = {astrs[a_start + a], bstrs[b], row[b]};
                }
            }
        }
    };

    // External space = (A_new x B_all) union (A_old x B_new).  This assigns
    // A_new x B_new only to the first subspace, avoiding duplicate selection.
    std::vector<int64> all_b_idxs(num_b_total);
    for (int64 b = 0; b < num_b_total; ++b) all_b_idxs[b] = b;

    select_subblock(new_astrs.data(), new_a_idxs.data(), (int64)new_astrs.size(),
                    full_block.bstrs, all_b_idxs.data(), num_b_total);
    select_subblock(old_astrs.data(), old_a_idxs.data(), (int64)old_astrs.size(),
                    new_bstrs.data(), new_b_idxs.data(), (int64)new_bstrs.size());

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
                         "[sci_bitstr check] external block select mismatch: "
                         "true_external_selected=%lld mask_scan_selected=%lld\n",
                         (long long)block_selected.size(),
                         (long long)mask_selected.size());
        }
        else
        {
            std::fprintf(stderr,
                         "[sci_bitstr check] external block select matches mask-scan selected set: %lld entries\n",
                         (long long)block_selected.size());
        }
    }

    if (print_perf)
    {
        const auto t1 = std::chrono::steady_clock::now();
        const double true_external_time = std::chrono::duration<double>(t1 - t0).count();
        std::fprintf(stderr,
                     "[sci_bitstr perf] full_target_select_time=unmeasured "
                     "mask_scan_external_select_time=unmeasured "
                     "true_external_block_select_time=%.9f "
                     "external_candidate_count=%lld contraction_candidate_count=%lld\n",
                     true_external_time,
                     (long long)external_candidate_count,
                     (long long)contraction_candidate_count);
    }

    return out_count;
}
