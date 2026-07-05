#pragma once

#include "sci_common.hpp"
#include "bitintegers.hpp"
#include <vector>

template <typename Ti>
struct SpinLinkCSR
{
    Ti mask = {};
    std::vector<int64> rowptr;
    std::vector<int> colidx;
};

template <typename Ti>
struct SpinLinksByMask
{
    ankerl::unordered_dense::map<Ti, SpinLinkCSR<Ti>> alpha_links_by_ax;
    ankerl::unordered_dense::map<Ti, SpinLinkCSR<Ti>> beta_links_by_bx;

    void clear()
    {
        alpha_links_by_ax.clear();
        beta_links_by_bx.clear();
    }
};

template <typename Ti>
FORCE_INLINE int sci_link_popcnt(Ti value)
{
    return popcnt(value);
}

template <typename Ti>
int64 sci_num_alpha_strings(const SciBasisManager<Ti> *basis)
{
    int64 n = 0;
    for (int64 sym = 0; sym < basis->num_irreps; ++sym)
        n += basis->num_astrs[sym];
    return n;
}

template <typename Ti>
int64 sci_num_beta_strings(const SciBasisManager<Ti> *basis)
{
    int64 n = 0;
    for (int64 sym = 0; sym < basis->num_irreps; ++sym)
        n += basis->num_bstrs[sym];
    return n;
}

template <typename Ti>
SpinLinkCSR<Ti> build_alpha_spin_link_csr(
    const SciBasisManager<Ti> *src_basis,
    const SciBasisManager<Ti> *tgt_basis,
    Ti ax)
{
    const int64 num_tgt = sci_num_alpha_strings(tgt_basis);
    SpinLinkCSR<Ti> link;
    link.mask = ax;
    link.rowptr.resize(num_tgt + 1, 0);
    link.colidx.reserve(num_tgt);

    if (ax == Ti{})
    {
        for (int64 dst_global = 0; dst_global < num_tgt; ++dst_global)
        {
            link.rowptr[dst_global] = (int64)link.colidx.size();
            const Ti dst_a = tgt_basis->all_astrs[dst_global];
            const int64 src_sym = get_string_sym(dst_a, src_basis->orbsym);
            if (src_sym >= src_basis->num_irreps)
                continue;
            const auto src_it = src_basis->a_idx_map.find(dst_a);
            if (src_it == src_basis->a_idx_map.end())
                continue;
            const int64 src_global = (src_basis->astrs_vec[src_sym] - src_basis->all_astrs) + src_it->second;
            link.colidx.push_back((int)src_global);
        }
        link.rowptr[num_tgt] = (int64)link.colidx.size();
        return link;
    }

    for (int64 dst_global = 0; dst_global < num_tgt; ++dst_global)
    {
        link.rowptr[dst_global] = (int64)link.colidx.size();
        const Ti dst_a = tgt_basis->all_astrs[dst_global];
        const Ti src_a = dst_a ^ ax;
        if (sci_link_popcnt(src_a) != sci_link_popcnt(dst_a))
            continue;
        const int64 src_sym = get_string_sym(src_a, src_basis->orbsym);
        if (src_sym >= src_basis->num_irreps)
            continue;
        const auto src_it = src_basis->a_idx_map.find(src_a);
        if (src_it == src_basis->a_idx_map.end())
            continue;
        const int64 src_global = (src_basis->astrs_vec[src_sym] - src_basis->all_astrs) + src_it->second;
        link.colidx.push_back((int)src_global);
    }
    link.rowptr[num_tgt] = (int64)link.colidx.size();
    return link;
}

template <typename Ti>
SpinLinkCSR<Ti> build_beta_spin_link_csr(
    const SciBasisManager<Ti> *src_basis,
    const SciBasisManager<Ti> *tgt_basis,
    Ti bx)
{
    const int64 num_tgt = sci_num_beta_strings(tgt_basis);
    SpinLinkCSR<Ti> link;
    link.mask = bx;
    link.rowptr.resize(num_tgt + 1, 0);
    link.colidx.reserve(num_tgt);

    if (bx == Ti{})
    {
        for (int64 dst_global = 0; dst_global < num_tgt; ++dst_global)
        {
            link.rowptr[dst_global] = (int64)link.colidx.size();
            const Ti dst_b = tgt_basis->all_bstrs[dst_global];
            const int64 src_sym = get_string_sym(dst_b, src_basis->orbsym);
            if (src_sym >= src_basis->num_irreps)
                continue;
            const auto src_it = src_basis->b_idx_map.find(dst_b);
            if (src_it == src_basis->b_idx_map.end())
                continue;
            const int64 src_global = (src_basis->bstrs_vec[src_sym] - src_basis->all_bstrs) + src_it->second;
            link.colidx.push_back((int)src_global);
        }
        link.rowptr[num_tgt] = (int64)link.colidx.size();
        return link;
    }

    for (int64 dst_global = 0; dst_global < num_tgt; ++dst_global)
    {
        link.rowptr[dst_global] = (int64)link.colidx.size();
        const Ti dst_b = tgt_basis->all_bstrs[dst_global];
        const Ti src_b = dst_b ^ bx;
        if (sci_link_popcnt(src_b) != sci_link_popcnt(dst_b))
            continue;
        const int64 src_sym = get_string_sym(src_b, src_basis->orbsym);
        if (src_sym >= src_basis->num_irreps)
            continue;
        const auto src_it = src_basis->b_idx_map.find(src_b);
        if (src_it == src_basis->b_idx_map.end())
            continue;
        const int64 src_global = (src_basis->bstrs_vec[src_sym] - src_basis->all_bstrs) + src_it->second;
        link.colidx.push_back((int)src_global);
    }
    link.rowptr[num_tgt] = (int64)link.colidx.size();
    return link;
}

template <typename Ti>
SpinLinkCSR<Ti> filter_spin_link_csr_to_new_targets(
    const SpinLinkCSR<Ti> &full_link,
    const bool *is_new_target)
{
    SpinLinkCSR<Ti> frontier;
    frontier.mask = full_link.mask;
    const int64 num_tgt = full_link.rowptr.empty() ? 0 : (int64)full_link.rowptr.size() - 1;
    frontier.rowptr.resize(num_tgt + 1, 0);
    frontier.colidx.reserve(full_link.colidx.size());

    for (int64 dst_idx = 0; dst_idx < num_tgt; ++dst_idx)
    {
        frontier.rowptr[dst_idx] = (int64)frontier.colidx.size();
        if (is_new_target[dst_idx])
        {
            for (int64 p = full_link.rowptr[dst_idx]; p < full_link.rowptr[dst_idx + 1]; ++p)
                frontier.colidx.push_back(full_link.colidx[p]);
        }
    }
    frontier.rowptr[num_tgt] = (int64)frontier.colidx.size();
    return frontier;
}

template <typename Ti>
SpinLinksByMask<Ti> build_new_frontier_links_by_mask(
    const SpinLinksByMask<Ti> &full_links,
    const bool *is_new_a,
    const bool *is_new_b)
{
    SpinLinksByMask<Ti> frontiers;
    frontiers.alpha_links_by_ax.reserve(full_links.alpha_links_by_ax.size());
    frontiers.beta_links_by_bx.reserve(full_links.beta_links_by_bx.size());

    for (const auto &kv : full_links.alpha_links_by_ax)
        frontiers.alpha_links_by_ax.emplace(kv.first, filter_spin_link_csr_to_new_targets(kv.second, is_new_a));
    for (const auto &kv : full_links.beta_links_by_bx)
        frontiers.beta_links_by_bx.emplace(kv.first, filter_spin_link_csr_to_new_targets(kv.second, is_new_b));

    return frontiers;
}

template <typename Ti>
SpinLinksByMask<Ti> build_spin_links_by_mask(
    const SciBasisManager<Ti> *src_basis,
    const SciBasisManager<Ti> *tgt_basis,
    const std::vector<Ti> &unique_axs,
    const std::vector<Ti> &unique_bxs)
{
    SpinLinksByMask<Ti> links;
    links.alpha_links_by_ax.reserve(unique_axs.size());
    links.beta_links_by_bx.reserve(unique_bxs.size());

    for (Ti ax : unique_axs)
        links.alpha_links_by_ax.emplace(ax, build_alpha_spin_link_csr(src_basis, tgt_basis, ax));
    for (Ti bx : unique_bxs)
        links.beta_links_by_bx.emplace(bx, build_beta_spin_link_csr(src_basis, tgt_basis, bx));

    return links;
}
