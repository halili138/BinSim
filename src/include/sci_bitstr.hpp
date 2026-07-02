#pragma once
#include "sci_common.hpp"
#include "sci_hvec.hpp"
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
