#pragma once
#include "otf.hpp"

template <int Rank, int MemLay, typename Tv>
FORCE_INLINE Tv compute_coeff_grad(int b, int rank, Tv pa0, Tv pa1, const Tv *pan, const Tv *pb, int max_b_count)
{
    Tv vt = {};
    if constexpr (Rank == 1)
    {
        vt = pa0 * pb[b];
    }
    else if constexpr (Rank == 2)
    {
        vt = pa0 * pb[b] + pa1 * pb[b + max_b_count];
    }
    else
    {
        if constexpr (MemLay == 1)
        {
            const Tv *pbn = pb + b * rank;
            for (int r = 0; r < rank; ++r)
            {
                vt += pan[r] * pbn[r];
            }
        }
        else
        {
            for (int r = 0; r < rank; ++r)
            {
                vt += pan[r] * pb[b + r * max_b_count];
            }
        }
    }

    return vt;
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void grad_contract_diag_batched_impl(
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const std::vector<PoolGradJob<Ti, Tv>> &batch_groups,
    const double *thetas,
    const Tv *lp,
    const Tv *rp,
    Tv *grads)
{
    const int64 num_groups = batch_groups.size();
    if (num_groups == 0)
        return;

    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    
    int64 batch_size = 0;
    int max_a_count = 0, max_b_count = 0, max_rank = 0;
    get_upper_batched<Rank, Ti, Tv>(blocks, batch_groups, num_blocks, batch_size, max_a_count, max_b_count, max_rank);

    std::vector<Tv> phase_b(batch_size * max_b_count * max_rank);

#pragma omp parallel
    {
        std::vector<Tv> thread_grads(num_groups, Tv{});
        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += batch_size)
            {
                const int64 cur_batch_size = std::min(batch_size, num_groups - batch_start);
#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> *group = batch_groups[batch_start + batch_idx].group;
                    Tv *pb0 = phase_b.data() + batch_idx * max_b_count * max_rank;
                    compute_phases<Rank, 1, Ti, Tv>(block.bstrs, block.num_b, group->unique_zbs, group->num_zb, group->wb, pb0, max_b_count, group->rank);
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < block.num_a; ++a)
                {
                    const Ti str_a = block.astrs[a];
                    const Tv *la = lp + block.offset + a * block.num_b;
                    const Tv *ra = rp + block.offset + a * block.num_b;
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int64 g = batch_start + batch_idx;
                        const PoolGradJob<Ti, Tv> &item = batch_groups[g];
                        const int rank = item.group->rank;
                        const double theta = thetas[item.original_idx];

                        Tv pa0 = {}, pa1 = {}, pan[64] = {};
                        compute_a_phase<Rank, Ti, Tv>(str_a, item.group->unique_zas, item.group->num_za, item.group->wa, rank, pa0, pa1, pan);
                        const Tv *pb = phase_b.data() + batch_idx * max_b_count * max_rank;

                        Tv local_res = {};
#pragma omp simd reduction(+ : local_res)
                        for (int b = 0; b < block.num_b; ++b)
                        {
                            const Tv vt = compute_coeff_grad<Rank, 1, Tv>(b, rank, pa0, pa1, pan, pb, max_b_count);
                            const Tv du = fast_diag_grad<Tv>(vt, theta);
                            local_res += math_conj(la[b] * du) * ra[b];
                        }
                        thread_grads[g] += local_res;
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int64 g = 0; g < num_groups; ++g)
                grads[batch_groups[g].original_idx] += thread_grads[g];
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void grad_contract_pure_a_batched_impl(
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const std::vector<PoolGradJob<Ti, Tv>> &batch_groups,
    const double *thetas,
    const Tv *lp,
    const Tv *rp,
    Tv *grads)
{
    const int64 num_groups = batch_groups.size();
    if (num_groups == 0)
        return;

    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 batch_size = 0;
    int max_a_count = 0, max_b_count = 0, max_rank = 0;
    get_upper_batched<Rank, Ti, Tv>(blocks, batch_groups, num_blocks, batch_size, max_a_count, max_b_count, max_rank);

    std::vector<Tv> phase_b(batch_size * max_b_count * max_rank);
    std::vector<int> src_block_idxs(batch_size);

#pragma omp parallel
    {
        std::vector<Tv> thread_grads(num_groups, Tv{});
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += batch_size)
            {
                const int64 cur_batch_size = std::min(batch_size, num_groups - batch_start);
#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const PoolGradJob<Ti, Tv> &item = batch_groups[batch_start + batch_idx];
                    const SVDGroup_OTF<Ti, Tv> *group = item.group;
                    const int64 axsym = get_string_sym(group->ax, orbsym);
                    const int64 bid = (dst_block.asym ^ axsym) * num_irreps + dst_block.bsym;
                    const int64 src_block_idx = block_map[bid];
                    src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                    {
                        continue;
                    }

                    Tv *pb0 = phase_b.data() + batch_idx * max_b_count * max_rank;
                    compute_phases<Rank, 1, Ti, Tv>(dst_block.bstrs, dst_block.num_b, group->unique_zbs, group->num_zb, group->wb, pb0, max_b_count, group->rank);
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti dst_str_a = dst_block.astrs[a];
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int src_block_idx = src_block_idxs[batch_idx];

                        if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                            continue;

                        const int64 g = batch_start + batch_idx;
                        const PoolGradJob<Ti, Tv> &item = batch_groups[g];
                        const SVDGroup_OTF<Ti, Tv> *group = item.group;

                        const Ti src_str_a = dst_str_a ^ group->ax;
                        const int src_a_idx = idx_map.a_idx_map[src_str_a];

                        if (src_a_idx == -1 || (src_block_idx == dst_block_idx && src_a_idx < a))
                            continue;

                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const int rank = group->rank;
                        const double theta = thetas[item.original_idx];
                        const double cd = -std::sin(theta);
                        const double co = std::cos(theta);

                        Tv pa0 = {}, pa1 = {}, pan[64] = {};
                        compute_a_phase<Rank, Ti, Tv>(src_str_a, group->unique_zas, group->num_za, group->wa, rank, pa0, pa1, pan);
                        const Tv *pb = phase_b.data() + batch_idx * max_b_count * max_rank;

                        const int64 sa = src_block.offset + src_a_idx * src_block.num_b;
                        const int64 da = dst_block.offset + a * dst_block.num_b;

                        Tv local_res = {};
#pragma omp simd reduction(+ : local_res)
                        for (int b = 0; b < dst_block.num_b; ++b)
                        {
                            const Tv vt = compute_coeff_grad<Rank, 1, Tv>(b, rank, pa0, pa1, pan, pb, max_b_count);
                            const int64 si = sa + b;
                            const int64 di = da + b;
                            grad_update<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
                        }
                        thread_grads[g] += local_res;
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int64 g = 0; g < num_groups; ++g)
                grads[batch_groups[g].original_idx] += thread_grads[g];
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void grad_contract_pure_b_batched_impl(
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const std::vector<PoolGradJob<Ti, Tv>> &batch_groups,
    const double *thetas,
    const Tv *lp,
    const Tv *rp,
    Tv *grads)
{
    const int64 num_groups = batch_groups.size();
    if (num_groups == 0)
        return;

    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 batch_size = 0;
    int max_a_count = 0, max_b_count = 0, max_rank = 0;
    get_upper_batched<Rank, Ti, Tv>(blocks, batch_groups, num_blocks, batch_size, max_a_count, max_b_count, max_rank);

    std::vector<int> src_b_idxs(batch_size * max_b_count);
    std::vector<int> dst_b_idxs(batch_size * max_b_count);
    std::vector<Tv> phase_b(batch_size * max_b_count * max_rank);
    std::vector<int> valid_b_counts(batch_size);
    std::vector<int> src_block_idxs(batch_size);

#pragma omp parallel
    {
        std::vector<Tv> thread_grads(num_groups, Tv{});
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += batch_size)
            {
                const int64 cur_batch_size = std::min(batch_size, num_groups - batch_start);
#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const PoolGradJob<Ti, Tv> &item = batch_groups[batch_start + batch_idx];
                    const SVDGroup_OTF<Ti, Tv> *group = item.group;

                    const int64 bxsym = get_string_sym(group->bx, orbsym);
                    const int64 bid = dst_block.asym * num_irreps + (dst_block.bsym ^ bxsym);
                    const int64 src_block_idx = block_map[bid];

                    src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                    {
                        valid_b_counts[batch_idx] = 0;
                        continue;
                    }

                    const bool is_same_block = (src_block_idx == dst_block_idx);

                    Tv *pb0 = phase_b.data() + batch_idx * max_b_count * max_rank;
                    int *sb_ptr = src_b_idxs.data() + batch_idx * max_b_count;
                    int *db_ptr = dst_b_idxs.data() + batch_idx * max_b_count;

                    valid_b_counts[batch_idx] = compute_phases_symm<Rank, 0, Ti, Tv>(
                        group->bx, idx_map.b_idx_map, dst_block.bstrs, dst_block.num_b,
                        group->unique_zbs, group->num_zb, group->wb, pb0, max_b_count, group->rank,
                        sb_ptr, db_ptr, is_same_block, true);
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti str_a = dst_block.astrs[a];
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_nb = valid_b_counts[batch_idx];

                        if (valid_nb == 0)
                            continue;

                        const int src_block_idx = src_block_idxs[batch_idx];
                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const int64 g = batch_start + batch_idx;
                        const PoolGradJob<Ti, Tv> &item = batch_groups[g];
                        const SVDGroup_OTF<Ti, Tv> *group = item.group;
                        const int rank = group->rank;
                        const double theta = thetas[item.original_idx];
                        const double cd = -std::sin(theta);
                        const double co = std::cos(theta);

                        Tv pa0 = {}, pa1 = {}, pan[64] = {};
                        compute_a_phase<Rank, Ti, Tv>(str_a, group->unique_zas, group->num_za, group->wa, rank, pa0, pa1, pan);
                        const Tv *pb0 = phase_b.data() + batch_idx * max_b_count * max_rank;

                        const int *sb_ptr = src_b_idxs.data() + batch_idx * max_b_count;
                        const int *db_ptr = dst_b_idxs.data() + batch_idx * max_b_count;
                        const int64 sa = src_block.offset + a * src_block.num_b;
                        const int64 da = dst_block.offset + a * dst_block.num_b;

                        Tv local_res = {};
#pragma omp simd reduction(+ : local_res)
                        for (int b = 0; b < valid_nb; ++b)
                        {
                            const Tv vt = compute_coeff_grad<Rank, 0, Tv>(b, rank, pa0, pa1, pan, pb0, max_b_count);
                            const int64 si = sa + sb_ptr[b];
                            const int64 di = da + db_ptr[b];
                            grad_update<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
                        }
                        thread_grads[g] += local_res;
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int64 g = 0; g < num_groups; ++g)
            {
                grads[batch_groups[g].original_idx] += thread_grads[g];
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void grad_contract_mixed_batched_impl(
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const std::vector<PoolGradJob<Ti, Tv>> &batch_groups,
    const double *thetas,
    const Tv *lp,
    const Tv *rp,
    Tv *grads)
{
    const int64 num_groups = batch_groups.size();
    if (num_groups == 0)
        return;

    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 batch_size = 0;
    int max_a_count = 0, max_b_count = 0, max_rank = 0;
    get_upper_batched<Rank, Ti, Tv>(blocks, batch_groups, num_blocks, batch_size, max_a_count, max_b_count, max_rank);

    std::vector<int> src_b_idxs(batch_size * max_b_count);
    std::vector<int> dst_b_idxs(batch_size * max_b_count);
    std::vector<Tv> phase_b(batch_size * max_b_count * max_rank);
    std::vector<int> valid_b_counts(batch_size);
    std::vector<int> src_block_idxs(batch_size);

#pragma omp parallel
    {
        std::vector<Tv> thread_grads(num_groups, Tv{});
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += batch_size)
            {
                const int64 cur_batch_size = std::min(batch_size, num_groups - batch_start);
#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const PoolGradJob<Ti, Tv> &item = batch_groups[batch_start + batch_idx];
                    const SVDGroup_OTF<Ti, Tv> *group = item.group;
                    const int64 axsym = get_string_sym(group->ax, orbsym);
                    const int64 bxsym = get_string_sym(group->bx, orbsym);
                    const int64 bid = (dst_block.asym ^ axsym) * num_irreps + (dst_block.bsym ^ bxsym);
                    const int64 src_block_idx = block_map[bid];
                    src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                    {
                        valid_b_counts[batch_idx] = 0;
                        continue;
                    }

                    Tv *pb0 = phase_b.data() + batch_idx * max_b_count * max_rank;
                    int *sb_ptr = src_b_idxs.data() + batch_idx * max_b_count;
                    int *db_ptr = dst_b_idxs.data() + batch_idx * max_b_count;

                    valid_b_counts[batch_idx] = compute_phases_symm<Rank, 0, Ti, Tv>(
                        group->bx, idx_map.b_idx_map, dst_block.bstrs, dst_block.num_b,
                        group->unique_zbs, group->num_zb, group->wb, pb0, max_b_count, group->rank,
                        sb_ptr, db_ptr, false, false);
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti dst_str_a = dst_block.astrs[a];
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_nb = valid_b_counts[batch_idx];

                        if (valid_nb == 0)
                            continue;

                        const int src_block_idx = src_block_idxs[batch_idx];
                        if (src_block_idx == -1 || src_block_idx < dst_block_idx)
                            continue;

                        const int64 g = batch_start + batch_idx;
                        const PoolGradJob<Ti, Tv> &item = batch_groups[g];
                        const SVDGroup_OTF<Ti, Tv> *group = item.group;
                        const Ti src_str_a = dst_str_a ^ group->ax;
                        const int src_a_idx = idx_map.a_idx_map[src_str_a];

                        // Mixed 特有的上三角过滤：因为 src_a_idx 绝不会等于 a，所以这里完美解决了对角块的去重
                        if (src_a_idx == -1 || (src_block_idx == dst_block_idx && src_a_idx < a))
                            continue;

                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const int rank = group->rank;
                        const double theta = thetas[item.original_idx];
                        const double cd = -std::sin(theta);
                        const double co = std::cos(theta);

                        Tv pa0 = {}, pa1 = {}, pan[64] = {};
                        compute_a_phase<Rank, Ti, Tv>(src_str_a, group->unique_zas, group->num_za, group->wa, rank, pa0, pa1, pan);
                        const Tv *pb = phase_b.data() + batch_idx * max_b_count * max_rank;

                        const int *sb_ptr = src_b_idxs.data() + batch_idx * max_b_count;
                        const int *db_ptr = dst_b_idxs.data() + batch_idx * max_b_count;
                        const int64 sa = src_block.offset + src_a_idx * src_block.num_b;
                        const int64 da = dst_block.offset + a * dst_block.num_b;

                        Tv local_res = {};
#pragma omp simd reduction(+ : local_res)
                        for (int b = 0; b < valid_nb; ++b)
                        {
                            const Tv vt = compute_coeff_grad<Rank, 0, Tv>(b, rank, pa0, pa1, pan, pb, max_b_count);
                            const int64 si = sa + sb_ptr[b];
                            const int64 di = da + db_ptr[b];
                            grad_update<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
                        }
                        thread_grads[g] += local_res;
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int64 g = 0; g < num_groups; ++g)
                grads[batch_groups[g].original_idx] += thread_grads[g];
        }
    }
}

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_grad_chunks_by_rank(
    const BasisManager<Ti> *basis,
    const IndexMap &map,
    const std::vector<PoolGradJob<Ti, Tv>> &groups,
    const double *thetas,
    const Tv *lp,
    const Tv *rp,
    Tv *grads)
{
    const int64 total_ngs = groups.size();
    if (total_ngs == 0)
        return;

    int64 start = 0;
    while (start < total_ngs)
    {
        const int dispatch_rank = groups[start].group->rank;
        const int eff_rank = (dispatch_rank == 1 || dispatch_rank == 2) ? dispatch_rank : 0;

        int64 end = start + 1;
        while (end < total_ngs)
        {
            const int next_rank = groups[end].group->rank;
            const int next_eff_rank = (next_rank == 1 || next_rank == 2) ? next_rank : 0;
            if (next_eff_rank != eff_rank)
                break;
            end++;
        }

        std::vector<PoolGradJob<Ti, Tv>> chunk(groups.begin() + start, groups.begin() + end);

        if constexpr (TypeCode == 0)
        {
            if (eff_rank == 1)
                grad_contract_diag_batched_impl<1>(basis, map, chunk, thetas, lp, rp, grads);
            else if (eff_rank == 2)
                grad_contract_diag_batched_impl<2>(basis, map, chunk, thetas, lp, rp, grads);
            else
                grad_contract_diag_batched_impl<0>(basis, map, chunk, thetas, lp, rp, grads);
        }
        else if constexpr (TypeCode == 1)
        {
            if (eff_rank == 1)
                grad_contract_pure_a_batched_impl<1>(basis, map, chunk, thetas, lp, rp, grads);
            else if (eff_rank == 2)
                grad_contract_pure_a_batched_impl<2>(basis, map, chunk, thetas, lp, rp, grads);
            else
                grad_contract_pure_a_batched_impl<0>(basis, map, chunk, thetas, lp, rp, grads);
        }
        else if constexpr (TypeCode == 2)
        {
            if (eff_rank == 1)
                grad_contract_pure_b_batched_impl<1>(basis, map, chunk, thetas, lp, rp, grads);
            else if (eff_rank == 2)
                grad_contract_pure_b_batched_impl<2>(basis, map, chunk, thetas, lp, rp, grads);
            else
                grad_contract_pure_b_batched_impl<0>(basis, map, chunk, thetas, lp, rp, grads);
        }
        else if constexpr (TypeCode == 3)
        {
            if (eff_rank == 1)
                grad_contract_mixed_batched_impl<1>(basis, map, chunk, thetas, lp, rp, grads);
            else if (eff_rank == 2)
                grad_contract_mixed_batched_impl<2>(basis, map, chunk, thetas, lp, rp, grads);
            else
                grad_contract_mixed_batched_impl<0>(basis, map, chunk, thetas, lp, rp, grads);
        }
        start = end;
    }
}

template <typename Ti, typename Tv>
void grad_pool_network_batched_otf(
    const BasisManager<Ti> *__restrict__ basis,
    const Network_OTF<Ti, Tv> *__restrict__ net,
    const double *__restrict__ thetas,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp,
    Tv *__restrict__ grads)
{
    std::fill(grads, grads + net->num_groups, Tv{});

    dispatch_grad_chunks_by_rank<0>(basis, net->map, net->pool_diag_jobs, thetas, lp, rp, grads);
    dispatch_grad_chunks_by_rank<1>(basis, net->map, net->pool_pure_a_jobs, thetas, lp, rp, grads);
    dispatch_grad_chunks_by_rank<2>(basis, net->map, net->pool_pure_b_jobs, thetas, lp, rp, grads);
    dispatch_grad_chunks_by_rank<3>(basis, net->map, net->pool_mixed_jobs, thetas, lp, rp, grads);
}
