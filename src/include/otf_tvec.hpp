#pragma once
#include "otf.hpp"

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void tvec_contract_diag_otf_impl(
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const Tv *src_vec,
    Tv *dst_vec)
{
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

#pragma omp parallel
    {
        std::vector<Tv> local_a_phase(max_a_count * rank);
        std::vector<Tv> local_b_phase(max_b_count * rank);

        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            const int a_count = block.num_a;
            const int b_count = block.num_b;
            Tv *pa0 = local_a_phase.data();
            Tv *pb0 = local_b_phase.data();

            compute_phases<Rank, 0, Ti, Tv>(
                block.astrs, a_count, zas, num_za, wa0, pa0, max_a_count, rank);

            compute_phases<Rank, 0, Ti, Tv>(
                block.bstrs, b_count, zbs, num_zb, wb0, pb0, max_b_count, rank);

            const Tv *pa = local_a_phase.data();
            const Tv *pb = local_b_phase.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < a_count; ++a)
            {
                for (int b = 0; b < b_count; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const int64 i = block.offset + (int64)a * b_count + b;
                    dst_vec[i] = src_vec[i] * vt;
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void tvec_contract_pure_a_otf_impl(
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const Tv *src_vec,
    Tv *dst_vec)
{
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

#pragma omp parallel
    {
        std::vector<int> src_a(max_a_count);
        std::vector<int> dst_a(max_a_count);
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int64 axsym = get_string_sym(group.ax, orbsym);
            const int64 bid = (dst_block.asym ^ axsym) * num_irreps + dst_block.bsym;
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

            int valid_na = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.ax, idx_map.a_idx_map,
                dst_block.astrs, dst_block.num_a, zas, num_za,
                wa0, phase_a.data(), max_a_count, rank,
                src_a.data(), dst_a.data(), is_same_block, true);

            if (valid_na == 0)
                continue;

            compute_phases<Rank, 0, Ti, Tv>(
                dst_block.bstrs, dst_block.num_b, zbs, num_zb,
                wb0, phase_b.data(), max_b_count, rank);

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < dst_block.num_b; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const int64 si = src_block.offset + src_a[a] * src_block.num_b + b;
                    const int64 di = dst_block.offset + dst_a[a] * dst_block.num_b + b;
                    tvec_update<Tv>(src_vec + si, src_vec + di, dst_vec + si, dst_vec + di, vt);
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void tvec_contract_pure_b_otf_impl(
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const Tv *src_vec,
    Tv *dst_vec)
{
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

#pragma omp parallel
    {
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<int> src_b(max_b_count);
        std::vector<int> dst_b(max_b_count);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int64 bxsym = get_string_sym(group.bx, orbsym);
            const int64 bid = dst_block.asym * num_irreps + (dst_block.bsym ^ bxsym);
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

            int valid_nb = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.bx, idx_map.b_idx_map,
                dst_block.bstrs, dst_block.num_b, zbs, num_zb,
                wb0, phase_b.data(), max_b_count, rank,
                src_b.data(), dst_b.data(), is_same_block, true);

            if (valid_nb == 0)
                continue;

            compute_phases<Rank, 0, Ti, Tv>(
                dst_block.astrs, dst_block.num_a, zas, num_za,
                wa0, phase_a.data(), max_a_count, rank);

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < dst_block.num_a; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const int64 si = src_block.offset + a * src_block.num_b + src_b[b];
                    const int64 di = dst_block.offset + a * dst_block.num_b + dst_b[b];
                    tvec_update<Tv>(src_vec + si, src_vec + di, dst_vec + si, dst_vec + di, vt);
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void tvec_contract_mixed_otf_impl(
    const BasisManager<Ti> *basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> &group,
    const Tv *src_vec,
    Tv *dst_vec)
{
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int64 max_a_count = 0;
    int64 max_b_count = 0;
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

#pragma omp parallel
    {
        std::vector<int> src_a(max_a_count);
        std::vector<int> dst_a(max_a_count);
        std::vector<Tv> phase_a(max_a_count * rank);
        std::vector<int> src_b(max_b_count);
        std::vector<int> dst_b(max_b_count);
        std::vector<Tv> phase_b(max_b_count * rank);

        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int64 axsym = get_string_sym(group.ax, orbsym);
            const int64 bxsym = get_string_sym(group.bx, orbsym);
            const int64 bid = (dst_block.asym ^ axsym) * num_irreps + (dst_block.bsym ^ bxsym);
            const int64 src_block_idx = block_map[bid];

            if (src_block_idx == -1)
                continue;

            if (src_block_idx < dst_block_idx)
                continue;

            const BlockDesc<Ti> &src_block = blocks[src_block_idx];
            const bool is_same_block = (src_block_idx == dst_block_idx);

            int valid_na = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.ax, idx_map.a_idx_map,
                dst_block.astrs, dst_block.num_a, zas, num_za,
                wa0, phase_a.data(), max_a_count, rank,
                src_a.data(), dst_a.data(), is_same_block, true);

            if (valid_na == 0)
                continue;

            int valid_nb = compute_phases_symm<Rank, 0, Ti, Tv>(
                group.bx, idx_map.b_idx_map,
                dst_block.bstrs, dst_block.num_b, zbs, num_zb,
                wb0, phase_b.data(), max_b_count, rank,
                src_b.data(), dst_b.data(), is_same_block, false);

            if (valid_nb == 0)
                continue;

            const Tv *pa = phase_a.data();
            const Tv *pb = phase_b.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < valid_na; ++a)
            {
                for (int b = 0; b < valid_nb; ++b)
                {
                    const Tv vt = compute_coeff<Rank, 0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const int64 si = src_block.offset + src_a[a] * src_block.num_b + src_b[b];
                    const int64 di = dst_block.offset + dst_a[a] * dst_block.num_b + dst_b[b];
                    tvec_update<Tv>(src_vec + si, src_vec + di, dst_vec + si, dst_vec + di, vt);
                }
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void tvec_svd_network_otf(
    const BasisManager<Ti> *basis,
    const Network_OTF<Ti, Tv> *__restrict__ net,
    const int64 idx,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const uint8 type = net->excit_types[idx];
    const SVDGroup_OTF<Ti, Tv> &group = net->flat_groups[idx];
    const int rank = group.rank;

#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
    {
        dst_vec[i] = {};
    }

    switch (type)
    {
    case 0: // Diag
        if (rank == 1)
            tvec_contract_diag_otf_impl<1>(basis, net->map, group, src_vec, dst_vec);
        else if (rank == 2)
            tvec_contract_diag_otf_impl<2>(basis, net->map, group, src_vec, dst_vec);
        else
            tvec_contract_diag_otf_impl<0>(basis, net->map, group, src_vec, dst_vec);
        break;
    case 1: // Pure A
        if (rank == 1)
            tvec_contract_pure_a_otf_impl<1>(basis, net->map, group, src_vec, dst_vec);
        else if (rank == 2)
            tvec_contract_pure_a_otf_impl<2>(basis, net->map, group, src_vec, dst_vec);
        else
            tvec_contract_pure_a_otf_impl<0>(basis, net->map, group, src_vec, dst_vec);
        break;
    case 2: // Pure B
        if (rank == 1)
            tvec_contract_pure_b_otf_impl<1>(basis, net->map, group, src_vec, dst_vec);
        else if (rank == 2)
            tvec_contract_pure_b_otf_impl<2>(basis, net->map, group, src_vec, dst_vec);
        else
            tvec_contract_pure_b_otf_impl<0>(basis, net->map, group, src_vec, dst_vec);
        break;
    case 3: // Mixed
        if (rank == 1)
            tvec_contract_mixed_otf_impl<1>(basis, net->map, group, src_vec, dst_vec);
        else if (rank == 2)
            tvec_contract_mixed_otf_impl<2>(basis, net->map, group, src_vec, dst_vec);
        else
            tvec_contract_mixed_otf_impl<0>(basis, net->map, group, src_vec, dst_vec);
        break;
    default:
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in expm_svd" << std::endl;
        break;
    }
}
