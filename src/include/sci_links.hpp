#pragma once

#include "sci_common.hpp"
#include "bitintegers.hpp"
#include <unordered_map>
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
    std::unordered_map<Ti, SpinLinkCSR<Ti>> alpha_links_by_ax;
    std::unordered_map<Ti, SpinLinkCSR<Ti>> beta_links_by_bx;

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
    const int64 num_src = sci_num_alpha_strings(src_basis);
    SpinLinkCSR<Ti> link;
    link.mask = ax;
    link.rowptr.resize(num_src + 1, 0);
    link.colidx.reserve(num_src);

    if (ax == Ti{})
    {
        for (int64 src_a_idx = 0; src_a_idx < num_src; ++src_a_idx)
        {
            link.rowptr[src_a_idx] = (int64)link.colidx.size();
            link.colidx.push_back((int)src_a_idx);
        }
        link.rowptr[num_src] = (int64)link.colidx.size();
        return link;
    }

    for (int64 src_a_idx = 0; src_a_idx < num_src; ++src_a_idx)
    {
        link.rowptr[src_a_idx] = (int64)link.colidx.size();
        const Ti src_a = src_basis->all_astrs[src_a_idx];
        const Ti dst_a = src_a ^ ax;
        if (sci_link_popcnt(dst_a) != sci_link_popcnt(src_a))
            continue;
        const int64 dst_sym = get_string_sym(dst_a, tgt_basis->orbsym);
        if (dst_sym >= tgt_basis->num_irreps)
            continue;
        const auto dst_it = tgt_basis->a_idx_map.find(dst_a);
        if (dst_it == tgt_basis->a_idx_map.end())
            continue;
        link.colidx.push_back(dst_it->second);
    }
    link.rowptr[num_src] = (int64)link.colidx.size();
    return link;
}

template <typename Ti>
SpinLinkCSR<Ti> build_beta_spin_link_csr(
    const SciBasisManager<Ti> *src_basis,
    const SciBasisManager<Ti> *tgt_basis,
    Ti bx)
{
    const int64 num_src = sci_num_beta_strings(src_basis);
    SpinLinkCSR<Ti> link;
    link.mask = bx;
    link.rowptr.resize(num_src + 1, 0);
    link.colidx.reserve(num_src);

    if (bx == Ti{})
    {
        for (int64 src_b_idx = 0; src_b_idx < num_src; ++src_b_idx)
        {
            link.rowptr[src_b_idx] = (int64)link.colidx.size();
            link.colidx.push_back((int)src_b_idx);
        }
        link.rowptr[num_src] = (int64)link.colidx.size();
        return link;
    }

    for (int64 src_b_idx = 0; src_b_idx < num_src; ++src_b_idx)
    {
        link.rowptr[src_b_idx] = (int64)link.colidx.size();
        const Ti src_b = src_basis->all_bstrs[src_b_idx];
        const Ti dst_b = src_b ^ bx;
        if (sci_link_popcnt(dst_b) != sci_link_popcnt(src_b))
            continue;
        const int64 dst_sym = get_string_sym(dst_b, tgt_basis->orbsym);
        if (dst_sym >= tgt_basis->num_irreps)
            continue;
        const auto dst_it = tgt_basis->b_idx_map.find(dst_b);
        if (dst_it == tgt_basis->b_idx_map.end())
            continue;
        link.colidx.push_back(dst_it->second);
    }
    link.rowptr[num_src] = (int64)link.colidx.size();
    return link;
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
