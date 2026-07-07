#pragma once
#include "sci_common.hpp"
#include "sci_links.hpp"
#include "sci_hvec.hpp"
#include "otf.hpp"
#include <cmath>

template <typename Tv>
FORCE_INLINE auto sqnorm(const Tv &v)
{
    if constexpr (std::is_arithmetic_v<Tv>)
        return v * v;
    else
        return v.real() * v.real() + v.imag() * v.imag();
}

template <typename Tv>
FORCE_INLINE bool sci_eps_check(Tv acc, Tv haa, Tv e_var, double eps)
{
    if (acc == Tv{})
        return false;
    Tv denom = e_var - haa;
    double dn_sq = sqnorm(denom);
    if (dn_sq == 0.0)
        return false;
    return sqnorm(acc) / dn_sq > eps * eps;
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

template <typename Ti>
static inline int excitation_type_code(Ti ax, Ti bx)
{
    if (ax != Ti{} && bx == Ti{})
        return 1;
    if (ax == Ti{} && bx != Ti{})
        return 2;
    return 3;
}

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
static inline int64 count_spin_link_entries(const ankerl::unordered_dense::map<Ti, SpinLinkCSR<Ti>> &links_by_mask)
{
    int64 total = 0;
    for (const auto &kv : links_by_mask)
        total += (int64)kv.second.colidx.size();
    return total;
}

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

template <typename Ti>
static inline void build_src_info_vec(
    const SciBasisManager<Ti> *src_basis,
    int64 num_src_astrs_total,
    int64 num_src_bstrs_total,
    std::vector<SourceStringInfo> &src_a_info,
    std::vector<SourceStringInfo> &src_b_info)
{
    src_a_info.resize(num_src_astrs_total);
    for (int64 i = 0; i < num_src_astrs_total; ++i)
    {
        SourceStringInfo &info = src_a_info[i];
        info.sym = get_string_sym(src_basis->all_astrs[i], src_basis->orbsym);
        if (info.sym >= src_basis->num_irreps)
            continue;
        info.local = i - (src_basis->astrs_vec[info.sym] - src_basis->all_astrs);
        info.valid = info.local >= 0 && info.local < src_basis->num_astrs[info.sym];
    }

    src_b_info.resize(num_src_bstrs_total);
    for (int64 j = 0; j < num_src_bstrs_total; ++j)
    {
        SourceStringInfo &info = src_b_info[j];
        info.sym = get_string_sym(src_basis->all_bstrs[j], src_basis->orbsym);
        if (info.sym >= src_basis->num_irreps)
            continue;
        info.local = j - (src_basis->bstrs_vec[info.sym] - src_basis->all_bstrs);
        info.valid = info.local >= 0 && info.local < src_basis->num_bstrs[info.sym];
    }
}

struct DstInfo
{
    int full;
    int64 src_global;
};

template <int Rank, typename Ti, typename Tv>
static inline void sci_link_pure_a_batched_impl(
    const LinkBucket<Ti, Tv> *buckets, 
    int64 num_buckets,
    const SciBasisManager<Ti> *src_basis,
    const SourceStringInfo *src_a_info, 
    const SourceStringInfo *src_b_info,
    int64 num_src_astrs_total, 
    int64 num_src_bstrs_total,
    int64 tgt_a_begin, 
    int64 tgt_b_begin,
    int64 num_a, 
    int64 num_b, 
    int64 tgt_asym, 
    int64 tgt_bsym,
    const bool *is_new_a, 
    bool skip_new_alpha,
    const Tv *src_vec, 
    Tv *dst_acc)
{
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;
    const int64 num_irreps = src_basis->num_irreps;
    const int64 *block_map = src_basis->block_map;

#pragma omp parallel
    {
        std::vector<Tv> phase_a_buf((size_t)num_src_astrs_total * MAX_RANK);
        std::vector<Tv> phase_b_buf((size_t)num_src_bstrs_total * MAX_RANK);

        std::vector<DstInfo> dst_a_list;
        std::vector<DstInfo> dst_b_list;
        dst_a_list.reserve((size_t)num_a);
        dst_b_list.reserve((size_t)num_b);

        for (int64 bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx)
        {
            const LinkBucket<Ti, Tv> &bucket = buckets[bucket_idx];
            const auto &alpha_links = *bucket.alpha_links;
            const auto &beta_links = *bucket.beta_links;
            const auto &group = *bucket.group;

            for (int64 i = 0; i < num_src_astrs_total; ++i)
                precompute_phase<Rank, Ti, Tv>(src_basis->all_astrs[i], group.unique_zas, group.num_za,
                                               group.wa, phase_a_buf.data() + (size_t)i * MAX_RANK, 1, group.rank);
            for (int64 j = 0; j < num_src_bstrs_total; ++j)
                precompute_phase<Rank, Ti, Tv>(src_basis->all_bstrs[j], group.unique_zbs, group.num_zb,
                                               group.wb, phase_b_buf.data() + (size_t)j * MAX_RANK, 1, group.rank);

            dst_a_list.clear();
            for (int64 dst_global = tgt_a_begin; dst_global < tgt_a_begin + num_a; ++dst_global)
            {
                if (skip_new_alpha && is_new_a[dst_global])
                    continue;
                for (int64 p = alpha_links.rowptr[dst_global]; p < alpha_links.rowptr[dst_global + 1]; ++p)
                {
                    int64 src_a_global = alpha_links.colidx[p];
                    if (src_a_global >= num_src_astrs_total || !src_a_info[src_a_global].valid)
                        continue;
                    dst_a_list.push_back({(int)(dst_global - tgt_a_begin), src_a_global});
                    break;
                }
            }

            dst_b_list.clear();
            for (int64 dst_global = tgt_b_begin; dst_global < tgt_b_begin + num_b; ++dst_global)
            {
                for (int64 p = beta_links.rowptr[dst_global]; p < beta_links.rowptr[dst_global + 1]; ++p)
                {
                    int64 src_b_global = beta_links.colidx[p];
                    if (src_b_global >= num_src_bstrs_total || !src_b_info[src_b_global].valid)
                        continue;
                    dst_b_list.push_back({(int)(dst_global - tgt_b_begin), src_b_global});
                    break;
                }
            }

            if (dst_a_list.empty() || dst_b_list.empty())
                continue;

#pragma omp for schedule(dynamic)
            for (int ai = 0; ai < (int)dst_a_list.size(); ++ai)
            {
                int a_full = dst_a_list[ai].full;
                int64 src_a_global = dst_a_list[ai].src_global;
                const SourceStringInfo &a_info = src_a_info[src_a_global];
                const Tv *pa = phase_a_buf.data() + (size_t)src_a_global * MAX_RANK;
                Tv *da = dst_acc + (int64)a_full * num_b;

                for (int bi = 0; bi < (int)dst_b_list.size(); ++bi)
                {
                    int b_full = dst_b_list[bi].full;
                    int64 src_b_global = dst_b_list[bi].src_global;
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
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline void sci_link_pure_b_batched_impl(
    const LinkBucket<Ti, Tv> *buckets, 
    int64 num_buckets,
    const SciBasisManager<Ti> *src_basis,
    const SourceStringInfo *src_a_info, 
    const SourceStringInfo *src_b_info,
    int64 num_src_astrs_total, 
    int64 num_src_bstrs_total,
    int64 tgt_a_begin, 
    int64 tgt_b_begin,
    int64 num_a, 
    int64 num_b, 
    int64 tgt_asym, 
    int64 tgt_bsym,
    const bool *is_new_a, 
    const Tv *src_vec, 
    Tv *dst_acc)
{
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;
    const int64 num_irreps = src_basis->num_irreps;
    const int64 *block_map = src_basis->block_map;

#pragma omp parallel
    {
        std::vector<Tv> phase_a_buf((size_t)num_src_astrs_total * MAX_RANK);
        std::vector<Tv> phase_b_buf((size_t)num_src_bstrs_total * MAX_RANK);

        std::vector<DstInfo> dst_a_list;
        std::vector<DstInfo> dst_b_list;
        dst_a_list.reserve((size_t)num_a);
        dst_b_list.reserve((size_t)num_b);

        for (int64 bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx)
        {
            const LinkBucket<Ti, Tv> &bucket = buckets[bucket_idx];
            const auto &alpha_links = *bucket.alpha_links;
            const auto &beta_links = *bucket.beta_links;
            const auto &group = *bucket.group;

            for (int64 i = 0; i < num_src_astrs_total; ++i)
                precompute_phase<Rank, Ti, Tv>(src_basis->all_astrs[i], group.unique_zas, group.num_za,
                                               group.wa, phase_a_buf.data() + (size_t)i * MAX_RANK, 1, group.rank);
            for (int64 j = 0; j < num_src_bstrs_total; ++j)
                precompute_phase<Rank, Ti, Tv>(src_basis->all_bstrs[j], group.unique_zbs, group.num_zb,
                                               group.wb, phase_b_buf.data() + (size_t)j * MAX_RANK, 1, group.rank);

            dst_a_list.clear();
            for (int64 dst_global = tgt_a_begin; dst_global < tgt_a_begin + num_a; ++dst_global)
            {
                for (int64 p = alpha_links.rowptr[dst_global]; p < alpha_links.rowptr[dst_global + 1]; ++p)
                {
                    int64 src_a_global = alpha_links.colidx[p];
                    if (src_a_global >= num_src_astrs_total || !src_a_info[src_a_global].valid)
                        continue;
                    dst_a_list.push_back({(int)(dst_global - tgt_a_begin), src_a_global});
                    break;
                }
            }

            dst_b_list.clear();
            for (int64 dst_global = tgt_b_begin; dst_global < tgt_b_begin + num_b; ++dst_global)
            {
                for (int64 p = beta_links.rowptr[dst_global]; p < beta_links.rowptr[dst_global + 1]; ++p)
                {
                    int64 src_b_global = beta_links.colidx[p];
                    if (src_b_global >= num_src_bstrs_total || !src_b_info[src_b_global].valid)
                        continue;
                    dst_b_list.push_back({(int)(dst_global - tgt_b_begin), src_b_global});
                    break;
                }
            }

            if (dst_a_list.empty() || dst_b_list.empty())
                continue;

#pragma omp for schedule(dynamic)
            for (int ai = 0; ai < (int)dst_a_list.size(); ++ai)
            {
                int a_full = dst_a_list[ai].full;
                int64 src_a_global = dst_a_list[ai].src_global;
                const SourceStringInfo &a_info = src_a_info[src_a_global];
                const Tv *pa = phase_a_buf.data() + (size_t)src_a_global * MAX_RANK;
                Tv *da = dst_acc + (int64)a_full * num_b;

                for (int bi = 0; bi < (int)dst_b_list.size(); ++bi)
                {
                    int b_full = dst_b_list[bi].full;
                    int64 src_b_global = dst_b_list[bi].src_global;
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
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static inline void sci_link_mixed_batched_impl(
    const LinkBucket<Ti, Tv> *buckets, 
    int64 num_buckets,
    const SciBasisManager<Ti> *src_basis,
    const SourceStringInfo *src_a_info, 
    const SourceStringInfo *src_b_info,
    int64 num_src_astrs_total, 
    int64 num_src_bstrs_total,
    int64 tgt_a_begin, 
    int64 tgt_b_begin,
    int64 num_a, 
    int64 num_b, 
    int64 tgt_asym, 
    int64 tgt_bsym,
    const bool *is_new_a, 
    bool skip_new_alpha,
    const Tv *src_vec, 
    Tv *dst_acc)
{
    constexpr int MAX_RANK = (Rank == 0) ? RANK3 : Rank;
    const int64 num_irreps = src_basis->num_irreps;
    const int64 *block_map = src_basis->block_map;

#pragma omp parallel
    {
        std::vector<Tv> phase_a_buf((size_t)num_src_astrs_total * MAX_RANK);
        std::vector<Tv> phase_b_buf((size_t)num_src_bstrs_total * MAX_RANK);

        std::vector<DstInfo> dst_a_list;
        std::vector<DstInfo> dst_b_list;
        dst_a_list.reserve((size_t)num_a);
        dst_b_list.reserve((size_t)num_b);

        for (int64 bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx)
        {
            const LinkBucket<Ti, Tv> &bucket = buckets[bucket_idx];
            const auto &alpha_links = *bucket.alpha_links;
            const auto &beta_links = *bucket.beta_links;
            const auto &group = *bucket.group;

            for (int64 i = 0; i < num_src_astrs_total; ++i)
                precompute_phase<Rank, Ti, Tv>(src_basis->all_astrs[i], group.unique_zas, group.num_za,
                                               group.wa, phase_a_buf.data() + (size_t)i * MAX_RANK, 1, group.rank);
            for (int64 j = 0; j < num_src_bstrs_total; ++j)
                precompute_phase<Rank, Ti, Tv>(src_basis->all_bstrs[j], group.unique_zbs, group.num_zb,
                                               group.wb, phase_b_buf.data() + (size_t)j * MAX_RANK, 1, group.rank);

            dst_a_list.clear();
            for (int64 dst_global = tgt_a_begin; dst_global < tgt_a_begin + num_a; ++dst_global)
            {
                if (skip_new_alpha && is_new_a[dst_global])
                    continue;
                for (int64 p = alpha_links.rowptr[dst_global]; p < alpha_links.rowptr[dst_global + 1]; ++p)
                {
                    int64 src_a_global = alpha_links.colidx[p];
                    if (src_a_global >= num_src_astrs_total || !src_a_info[src_a_global].valid)
                        continue;
                    dst_a_list.push_back({(int)(dst_global - tgt_a_begin), src_a_global});
                    break;
                }
            }

            dst_b_list.clear();
            for (int64 dst_global = tgt_b_begin; dst_global < tgt_b_begin + num_b; ++dst_global)
            {
                for (int64 p = beta_links.rowptr[dst_global]; p < beta_links.rowptr[dst_global + 1]; ++p)
                {
                    int64 src_b_global = beta_links.colidx[p];
                    if (src_b_global >= num_src_bstrs_total || !src_b_info[src_b_global].valid)
                        continue;
                    dst_b_list.push_back({(int)(dst_global - tgt_b_begin), src_b_global});
                    break;
                }
            }

            if (dst_a_list.empty() || dst_b_list.empty())
                continue;

#pragma omp for schedule(dynamic)
            for (int ai = 0; ai < (int)dst_a_list.size(); ++ai)
            {
                int a_full = dst_a_list[ai].full;
                int64 src_a_global = dst_a_list[ai].src_global;
                const SourceStringInfo &a_info = src_a_info[src_a_global];
                const Tv *pa = phase_a_buf.data() + (size_t)src_a_global * MAX_RANK;
                Tv *da = dst_acc + (int64)a_full * num_b;

                for (int bi = 0; bi < (int)dst_b_list.size(); ++bi)
                {
                    int b_full = dst_b_list[bi].full;
                    int64 src_b_global = dst_b_list[bi].src_global;
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
        }
    }
}

template <int TypeCode, typename Ti, typename Tv>
static inline void dispatch_link_chunks_by_rank(
    const std::vector<LinkBucket<Ti, Tv>> &buckets,
    const SciBasisManager<Ti> *src_basis,
    const SourceStringInfo *src_a_info, 
    const SourceStringInfo *src_b_info,
    int64 num_src_astrs_total, 
    int64 num_src_bstrs_total,
    int64 tgt_a_begin, 
    int64 tgt_b_begin,
    int64 num_a, 
    int64 num_b, 
    int64 tgt_asym, 
    int64 tgt_bsym,
    const bool *is_new_a, 
    bool skip_new_alpha,
    const Tv *src_vec, 
    Tv *dst_acc)
{
    const int64 total_buckets = (int64)buckets.size();
    if (total_buckets == 0)
        return;

    int64 start = 0;
    while (start < total_buckets)
    {
        const int current_rank = buckets[start].group->rank;
        const int dispatch_rank = (current_rank == 1 || current_rank == 2) ? current_rank : 0;

        int64 end = start + 1;
        while (end < total_buckets)
        {
            const int next_rank = buckets[end].group->rank;
            const int next_dispatch_rank = (next_rank == 1 || next_rank == 2) ? next_rank : 0;
            if (next_dispatch_rank != dispatch_rank)
                break;
            end++;
        }

        const int64 chunk_size = end - start;
        const LinkBucket<Ti, Tv> *chunk_ptr = buckets.data() + start;

        if constexpr (TypeCode == 1)
        {
            switch (dispatch_rank)
            {
            case 1:
                sci_link_pure_a_batched_impl<1>(chunk_ptr, chunk_size, src_basis, src_a_info, src_b_info,
                                                num_src_astrs_total, num_src_bstrs_total,
                                                tgt_a_begin, tgt_b_begin, num_a, num_b, tgt_asym, tgt_bsym,
                                                is_new_a, skip_new_alpha, src_vec, dst_acc);
                break;
            case 2:
                sci_link_pure_a_batched_impl<2>(chunk_ptr, chunk_size, src_basis, src_a_info, src_b_info,
                                                num_src_astrs_total, num_src_bstrs_total,
                                                tgt_a_begin, tgt_b_begin, num_a, num_b, tgt_asym, tgt_bsym,
                                                is_new_a, skip_new_alpha, src_vec, dst_acc);
                break;
            default:
                sci_link_pure_a_batched_impl<0>(chunk_ptr, chunk_size, src_basis, src_a_info, src_b_info,
                                                num_src_astrs_total, num_src_bstrs_total,
                                                tgt_a_begin, tgt_b_begin, num_a, num_b, tgt_asym, tgt_bsym,
                                                is_new_a, skip_new_alpha, src_vec, dst_acc);
                break;
            }
        }
        else if constexpr (TypeCode == 2)
        {
            switch (dispatch_rank)
            {
            case 1:
                sci_link_pure_b_batched_impl<1>(chunk_ptr, chunk_size, src_basis, src_a_info, src_b_info,
                                                num_src_astrs_total, num_src_bstrs_total,
                                                tgt_a_begin, tgt_b_begin, num_a, num_b, tgt_asym, tgt_bsym,
                                                is_new_a, src_vec, dst_acc);
                break;
            case 2:
                sci_link_pure_b_batched_impl<2>(chunk_ptr, chunk_size, src_basis, src_a_info, src_b_info,
                                                num_src_astrs_total, num_src_bstrs_total,
                                                tgt_a_begin, tgt_b_begin, num_a, num_b, tgt_asym, tgt_bsym,
                                                is_new_a, src_vec, dst_acc);
                break;
            default:
                sci_link_pure_b_batched_impl<0>(chunk_ptr, chunk_size, src_basis, src_a_info, src_b_info,
                                                num_src_astrs_total, num_src_bstrs_total,
                                                tgt_a_begin, tgt_b_begin, num_a, num_b, tgt_asym, tgt_bsym,
                                                is_new_a, src_vec, dst_acc);
                break;
            }
        }
        else if constexpr (TypeCode == 3)
        {
            switch (dispatch_rank)
            {
            case 1:
                sci_link_mixed_batched_impl<1>(chunk_ptr, chunk_size, src_basis, src_a_info, src_b_info,
                                               num_src_astrs_total, num_src_bstrs_total,
                                               tgt_a_begin, tgt_b_begin, num_a, num_b, tgt_asym, tgt_bsym,
                                               is_new_a, skip_new_alpha, src_vec, dst_acc);
                break;
            case 2:
                sci_link_mixed_batched_impl<2>(chunk_ptr, chunk_size, src_basis, src_a_info, src_b_info,
                                               num_src_astrs_total, num_src_bstrs_total,
                                               tgt_a_begin, tgt_b_begin, num_a, num_b, tgt_asym, tgt_bsym,
                                               is_new_a, skip_new_alpha, src_vec, dst_acc);
                break;
            default:
                sci_link_mixed_batched_impl<0>(chunk_ptr, chunk_size, src_basis, src_a_info, src_b_info,
                                               num_src_astrs_total, num_src_bstrs_total,
                                               tgt_a_begin, tgt_b_begin, num_a, num_b, tgt_asym, tgt_bsym,
                                               is_new_a, skip_new_alpha, src_vec, dst_acc);
                break;
            }
        }
        start = end;
    }
}

template <typename Ti, typename Tv>
int64 sci_select_external_block(
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
        dispatch_chunks_for_block<1>(chunk_desc, src_basis, net->pure_a_groups, src_vec, chunk_acc.data());
        dispatch_chunks_for_block<2>(chunk_desc, src_basis, net->pure_b_groups, src_vec, chunk_acc.data());
        dispatch_chunks_for_block<3>(chunk_desc, src_basis, net->mixed_groups, src_vec, chunk_acc.data());

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

                if (sci_eps_check(row[b], candidate_diags[full_block.offset + a_global * num_b + b],
                                  variational_energy, eps))
                {
                    out_entries[out_count++] = {full_block.astrs[a_global], full_block.bstrs[b], row[b]};
                }
            }
        }
    }
    return out_count;
}

template <typename Ti, typename Tv>
int64 sci_select_external_link_block(
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
    const std::vector<LinkBucket<Ti, Tv>> &alpha_1,
    const std::vector<LinkBucket<Ti, Tv>> &alpha_2,
    const std::vector<LinkBucket<Ti, Tv>> &alpha_3,
    const std::vector<LinkBucket<Ti, Tv>> &beta_1,
    const std::vector<LinkBucket<Ti, Tv>> &beta_2,
    const std::vector<LinkBucket<Ti, Tv>> &beta_3,
    const std::vector<SourceStringInfo> &src_a_info,
    const std::vector<SourceStringInfo> &src_b_info,
    int64 num_src_astrs_total,
    int64 num_src_bstrs_total)
{
    const BlockDesc<Ti> &full_block = tgt_basis->blocks[block_idx];
    const int64 num_a_total = full_block.num_a;
    const int64 num_b_total = full_block.num_b;

    const int64 full_candidate_count = num_a_total * num_b_total;

    int64 block_new_a_count = 0;
    int64 block_new_b_count = 0;
    for (int64 a = 0; a < num_a_total; ++a)
    {
        const int64 a_ext = (full_block.astrs + a) - tgt_basis->all_astrs;
        if (is_new_a[a_ext])
            ++block_new_a_count;
    }
    for (int64 b = 0; b < num_b_total; ++b)
    {
        const int64 b_ext = (full_block.bstrs + b) - tgt_basis->all_bstrs;
        if (is_new_b[b_ext])
            ++block_new_b_count;
    }

    const int64 external_candidate_count = block_new_a_count * num_b_total + (num_a_total - block_new_a_count) * block_new_b_count;
    constexpr double dense_threshold = 0.85;
    const double external_ratio = full_candidate_count == 0
                                      ? 0.0
                                      : (double)external_candidate_count / (double)full_candidate_count;

    if (external_ratio > dense_threshold)
    {
        return sci_select_external_block(
            tgt_basis, src_basis, net, is_new_a, is_new_b, block_idx, src_vec,
            candidate_diags, variational_energy, chunk_size, eps, out_entries, max_entries);
    }

    constexpr int64 max_link_entries = 1 << 26;
    if (ctx->alpha_link_entries + ctx->beta_link_entries > max_link_entries)
    {
        return sci_select_external_block(
            tgt_basis, src_basis, net, is_new_a, is_new_b, block_idx, src_vec,
            candidate_diags, variational_energy, chunk_size, eps, out_entries, max_entries);
    }

    const int64 tgt_a_begin = full_block.astrs - tgt_basis->all_astrs;
    const int64 tgt_b_begin = full_block.bstrs - tgt_basis->all_bstrs;

    std::vector<Tv> dst_acc((size_t)num_a_total * (size_t)num_b_total, Tv{});

    dispatch_link_chunks_by_rank<1>(alpha_1, src_basis, src_a_info.data(), src_b_info.data(),
                                    num_src_astrs_total, num_src_bstrs_total,
                                    tgt_a_begin, tgt_b_begin, num_a_total, num_b_total,
                                    full_block.asym, full_block.bsym,
                                    is_new_a, false, src_vec, dst_acc.data());

    dispatch_link_chunks_by_rank<3>(alpha_3, src_basis, src_a_info.data(), src_b_info.data(),
                                    num_src_astrs_total, num_src_bstrs_total,
                                    tgt_a_begin, tgt_b_begin, num_a_total, num_b_total,
                                    full_block.asym, full_block.bsym,
                                    is_new_a, false, src_vec, dst_acc.data());

    dispatch_link_chunks_by_rank<2>(beta_2, src_basis, src_a_info.data(), src_b_info.data(),
                                    num_src_astrs_total, num_src_bstrs_total,
                                    tgt_a_begin, tgt_b_begin, num_a_total, num_b_total,
                                    full_block.asym, full_block.bsym,
                                    is_new_a, true, src_vec, dst_acc.data());

    dispatch_link_chunks_by_rank<3>(beta_3, src_basis, src_a_info.data(), src_b_info.data(),
                                    num_src_astrs_total, num_src_bstrs_total,
                                    tgt_a_begin, tgt_b_begin, num_a_total, num_b_total,
                                    full_block.asym, full_block.bsym,
                                    is_new_a, true, src_vec, dst_acc.data());

    int64 out_count = 0;
    for (int64 a = 0; a < num_a_total && out_count < max_entries; ++a)
    {
        int64 a_global = tgt_a_begin + a;
        bool new_a = is_new_a[a_global];
        Tv *row = dst_acc.data() + a * num_b_total;

        for (int64 b = 0; b < num_b_total && out_count < max_entries; ++b)
        {
            int64 b_global = tgt_b_begin + b;
            if (!new_a && !is_new_b[b_global])
                continue;

            if (sci_eps_check(row[b], candidate_diags[full_block.offset + a * num_b_total + b],
                              variational_energy, eps))
            {
                out_entries[out_count++] = {full_block.astrs[a], full_block.bstrs[b], row[b]};
            }
        }
    }
    return out_count;
}

template <typename Ti, typename Tv>
int64 sci_select_external_link_all_blocks(
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

    std::vector<LinkBucket<Ti, Tv>> alpha_new[4];
    std::vector<LinkBucket<Ti, Tv>> beta_new[4];

    for (const auto &bkt : net_sci->buckets)
    {
        int tc = excitation_type_code<Ti>(bkt.ax, bkt.bx);

        const auto alpha_new_it = ctx.new_frontiers.alpha_links_by_ax.find(bkt.ax);
        const auto alpha_full_it = ctx.full_links.alpha_links_by_ax.find(bkt.ax);
        const auto beta_new_it = ctx.new_frontiers.beta_links_by_bx.find(bkt.bx);
        const auto beta_full_it = ctx.full_links.beta_links_by_bx.find(bkt.bx);

        if (alpha_new_it != ctx.new_frontiers.alpha_links_by_ax.end() &&
            beta_full_it != ctx.full_links.beta_links_by_bx.end())
            alpha_new[tc].push_back({&alpha_new_it->second, &beta_full_it->second, bkt.group});

        if (alpha_full_it != ctx.full_links.alpha_links_by_ax.end() &&
            beta_new_it != ctx.new_frontiers.beta_links_by_bx.end())
            beta_new[tc].push_back({&alpha_full_it->second, &beta_new_it->second, bkt.group});
    }

    auto rank_cmp = [](const LinkBucket<Ti, Tv> &a, const LinkBucket<Ti, Tv> &b)
    { return a.group->rank < b.group->rank; };
    for (int tc = 1; tc <= 3; ++tc)
    {
        std::sort(alpha_new[tc].begin(), alpha_new[tc].end(), rank_cmp);
        std::sort(beta_new[tc].begin(), beta_new[tc].end(), rank_cmp);
    }

    int64 num_src_astrs_total = 0;
    int64 num_src_bstrs_total = 0;
    for (int64 sym = 0; sym < src_basis->num_irreps; ++sym)
    {
        num_src_astrs_total += src_basis->num_astrs[sym];
        num_src_bstrs_total += src_basis->num_bstrs[sym];
    }

    std::vector<SourceStringInfo> src_a_info, src_b_info;
    build_src_info_vec(src_basis, num_src_astrs_total, num_src_bstrs_total, src_a_info, src_b_info);

    int64 out_count = 0;
    for (int64 blk = 0; blk < tgt_basis->num_blocks && out_count < max_entries; ++blk)
    {
        out_count += sci_select_external_link_block(
            &ctx, tgt_basis, src_basis, net, is_new_a, is_new_b, blk, src_vec,
            candidate_diags, variational_energy, chunk_size, eps,
            out_entries + out_count, max_entries - out_count,
            alpha_new[1], alpha_new[2], alpha_new[3],
            beta_new[1], beta_new[2], beta_new[3],
            src_a_info, src_b_info, num_src_astrs_total, num_src_bstrs_total);
    }

    return out_count;
}
