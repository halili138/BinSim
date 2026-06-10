#pragma once
#include "cuda_utils.cuh"
#include "cuda_basis.cuh"
#include "cuda_otf.cuh"

template <int Rank, typename Ti, typename Tv>
__global__ void expm_diag_kernel_2d(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos, double theta,
    Tv *__restrict__ vec)
{
    const int db = blockIdx.x * blockDim.x + threadIdx.x;
    const int da = blockIdx.y * blockDim.y + threadIdx.y;

    if (da >= basis.max_a_count || db >= basis.max_b_count)
        return;

    const int num_zas = groups.num_zas[pos];
    const int num_zbs = groups.num_zbs[pos];
    const int rank = groups.ranks[pos];
    const Ti *zas = groups.flat_zas + groups.za_start[pos];
    const Ti *zbs = groups.flat_zbs + groups.zb_start[pos];
    const Tv *wa = groups.flat_wa + groups.wa_start[pos];
    const Tv *wb = groups.flat_wb + groups.wb_start[pos];

    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
    Tv pa[STACK_SIZE] = {};
    Tv pb[STACK_SIZE] = {};

    for (int dst_bid = 0; dst_bid < basis.num_blocks; ++dst_bid)
    {
        if (da >= basis.block_num_a[dst_bid] || db >= basis.block_num_b[dst_bid])
            continue;

        const Ti dst_str_a = basis.astrs_flat[basis.astrs_start[dst_bid] + da];
        const Ti dst_str_b = basis.bstrs_flat[basis.bstrs_start[dst_bid] + db];

        compute_phase_dev<Rank, Ti, Tv>(dst_str_a, zas, num_zas, wa, pa, 1, rank);
        compute_phase_dev<Rank, Ti, Tv>(dst_str_b, zbs, num_zbs, wb, pb, 1, rank);

        const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, rank);

        const int64 di = basis.block_offsets[dst_bid] + (int64)da * basis.block_num_b[dst_bid] + db;

        const Tv u = fast_diag_exp<Tv>(vt, theta);
        vec[di] *= u;
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void expm_mixed_kernel_2d(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos, double cd, double co,
    Tv *__restrict__ vec)
{
    const int db = blockIdx.x * blockDim.x + threadIdx.x;
    const int da = blockIdx.y * blockDim.y + threadIdx.y;

    if (da >= basis.max_a_count || db >= basis.max_b_count)
        return;

    const Ti ax = groups.axs[pos];
    const Ti bx = groups.bxs[pos];
    const int group_asym = groups.asyms[pos];
    const int group_bsym = groups.bsyms[pos];
    const int num_zas = groups.num_zas[pos];
    const int num_zbs = groups.num_zbs[pos];
    const int rank = groups.ranks[pos];
    const Ti *zas = groups.flat_zas + groups.za_start[pos];
    const Ti *zbs = groups.flat_zbs + groups.zb_start[pos];
    const Tv *wa = groups.flat_wa + groups.wa_start[pos];
    const Tv *wb = groups.flat_wb + groups.wb_start[pos];

    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
    Tv pa[STACK_SIZE] = {};
    Tv pb[STACK_SIZE] = {};

    for (int dst_bid = 0; dst_bid < basis.num_blocks; ++dst_bid)
    {
        if (da >= basis.block_num_a[dst_bid] || db >= basis.block_num_b[dst_bid])
            continue;

        const int dst_asym = basis.block_asym[dst_bid];
        const int dst_bsym = basis.block_bsym[dst_bid];
        const int h = (dst_asym ^ group_asym) * basis.num_irreps + (dst_bsym ^ group_bsym);
        const int src_bid = basis.block_map[h];

        if (src_bid == -1 || src_bid < dst_bid)
            continue;

        const bool is_same_block = (src_bid == dst_bid);

        const Ti dst_str_a = basis.astrs_flat[basis.astrs_start[dst_bid] + da];
        const Ti dst_str_b = basis.bstrs_flat[basis.bstrs_start[dst_bid] + db];

        const Ti src_str_a = dst_str_a ^ ax;
        const Ti src_str_b = dst_str_b ^ bx;

        const int sa = basis.astr2idx[src_str_a];
        const int sb = basis.bstr2idx[src_str_b];

        if (sa == -1 || sb == -1 || (is_same_block && sa < da))
            continue;

        compute_phase_dev<Rank, Ti, Tv>(src_str_a, zas, num_zas, wa, pa, 1, rank);
        compute_phase_dev<Rank, Ti, Tv>(src_str_b, zbs, num_zbs, wb, pb, 1, rank);

        const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, rank);

        const int64 si = basis.block_offsets[src_bid] + (int64)sa * basis.block_num_b[src_bid] + sb;
        const int64 di = basis.block_offsets[dst_bid] + (int64)da * basis.block_num_b[dst_bid] + db;

        expm_update<Tv>(vec + si, vec + di, vt, cd, co);
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void expm_pure_a_kernel_2d(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos, double cd, double co,
    Tv *__restrict__ vec)
{
    const int db = blockIdx.x * blockDim.x + threadIdx.x;
    const int da = blockIdx.y * blockDim.y + threadIdx.y;

    if (da >= basis.max_a_count || db >= basis.max_b_count)
        return;

    const Ti ax = groups.axs[pos];
    const int group_asym = groups.asyms[pos];
    const int num_zas = groups.num_zas[pos];
    const int num_zbs = groups.num_zbs[pos];
    const int rank = groups.ranks[pos];
    const Ti *zas = groups.flat_zas + groups.za_start[pos];
    const Ti *zbs = groups.flat_zbs + groups.zb_start[pos];
    const Tv *wa = groups.flat_wa + groups.wa_start[pos];
    const Tv *wb = groups.flat_wb + groups.wb_start[pos];

    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
    Tv pa[STACK_SIZE] = {};
    Tv pb[STACK_SIZE] = {};

    for (int dst_bid = 0; dst_bid < basis.num_blocks; ++dst_bid)
    {
        if (da >= basis.block_num_a[dst_bid] || db >= basis.block_num_b[dst_bid])
            continue;

        const int dst_asym = basis.block_asym[dst_bid];
        const int dst_bsym = basis.block_bsym[dst_bid];
        const int h = (dst_asym ^ group_asym) * basis.num_irreps + dst_bsym;
        const int src_bid = basis.block_map[h];

        if (src_bid == -1 || src_bid < dst_bid)
            continue;

        const bool is_same_block = (src_bid == dst_bid);

        const Ti dst_str_a = basis.astrs_flat[basis.astrs_start[dst_bid] + da];
        const Ti dst_str_b = basis.bstrs_flat[basis.bstrs_start[dst_bid] + db];

        const Ti src_str_a = dst_str_a ^ ax;
        const int sa = basis.astr2idx[src_str_a];
        const int sb = db;

        if (sa == -1 || (is_same_block && sa < da))
            continue;

        compute_phase_dev<Rank, Ti, Tv>(src_str_a, zas, num_zas, wa, pa, 1, rank);
        compute_phase_dev<Rank, Ti, Tv>(dst_str_b, zbs, num_zbs, wb, pb, 1, rank);

        const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, rank);

        const int64 si = basis.block_offsets[src_bid] + (int64)sa * basis.block_num_b[src_bid] + sb;
        const int64 di = basis.block_offsets[dst_bid] + (int64)da * basis.block_num_b[dst_bid] + db;

        expm_update<Tv>(vec + si, vec + di, vt, cd, co);
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void expm_pure_b_kernel_2d(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos, double cd, double co,
    Tv *__restrict__ vec)
{
    const int db = blockIdx.x * blockDim.x + threadIdx.x;
    const int da = blockIdx.y * blockDim.y + threadIdx.y;

    if (da >= basis.max_a_count || db >= basis.max_b_count)
        return;

    const Ti bx = groups.bxs[pos];
    const int group_bsym = groups.bsyms[pos];
    const int num_zas = groups.num_zas[pos];
    const int num_zbs = groups.num_zbs[pos];
    const int rank = groups.ranks[pos];
    const Ti *zas = groups.flat_zas + groups.za_start[pos];
    const Ti *zbs = groups.flat_zbs + groups.zb_start[pos];
    const Tv *wa = groups.flat_wa + groups.wa_start[pos];
    const Tv *wb = groups.flat_wb + groups.wb_start[pos];

    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
    Tv pa[STACK_SIZE] = {};
    Tv pb[STACK_SIZE] = {};

    for (int dst_bid = 0; dst_bid < basis.num_blocks; ++dst_bid)
    {
        if (da >= basis.block_num_a[dst_bid] || db >= basis.block_num_b[dst_bid])
            continue;

        const int dst_asym = basis.block_asym[dst_bid];
        const int dst_bsym = basis.block_bsym[dst_bid];
        const int h = dst_asym * basis.num_irreps + (dst_bsym ^ group_bsym);
        const int src_bid = basis.block_map[h];

        if (src_bid == -1 || src_bid < dst_bid)
            continue;

        const bool is_same_block = (src_bid == dst_bid);

        const Ti dst_str_a = basis.astrs_flat[basis.astrs_start[dst_bid] + da];
        const Ti dst_str_b = basis.bstrs_flat[basis.bstrs_start[dst_bid] + db];

        const Ti src_str_b = dst_str_b ^ bx;
        const int sa = da;
        const int sb = basis.bstr2idx[src_str_b];

        if (sb == -1 || (is_same_block && sb < db))
            continue;

        compute_phase_dev<Rank, Ti, Tv>(dst_str_a, zas, num_zas, wa, pa, 1, rank);
        compute_phase_dev<Rank, Ti, Tv>(src_str_b, zbs, num_zbs, wb, pb, 1, rank);

        const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, rank);

        const int64 si = basis.block_offsets[src_bid] + (int64)sa * basis.block_num_b[src_bid] + sb;
        const int64 di = basis.block_offsets[dst_bid] + (int64)da * basis.block_num_b[dst_bid] + db;

        expm_update<Tv>(vec + si, vec + di, vt, cd, co);
    }
}

template <typename Ti, typename Tv>
void expm_svd_network_otf_gpu(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    int64 idx, double theta,
    Tv *__restrict__ dev_vec)
{
    const uint8 type = net.host_excit_types[idx];
    const int64 pos = net.host_sorted_idxs[idx];

    BasisSliceDev<Ti> basis_slice;
    basis_slice.num_blocks = basis.num_blocks;
    basis_slice.num_irreps = basis.num_irreps;
    basis_slice.max_a_count = basis.max_a_count;
    basis_slice.max_b_count = basis.max_b_count;
    basis_slice.dim = basis.dim;
    basis_slice.block_offsets = basis.block_offsets;
    basis_slice.block_num_a = basis.block_num_a;
    basis_slice.block_num_b = basis.block_num_b;
    basis_slice.block_asym = basis.block_asym;
    basis_slice.block_bsym = basis.block_bsym;
    basis_slice.astrs_flat = basis.astrs_flat;
    basis_slice.bstrs_flat = basis.bstrs_flat;
    basis_slice.astrs_start = basis.astrs_start;
    basis_slice.bstrs_start = basis.bstrs_start;
    basis_slice.block_map = basis.block_map;
    basis_slice.astr2idx = basis.astr2idx;
    basis_slice.bstr2idx = basis.bstr2idx;

    const double cd = std::cos(theta) - 1.0;
    const double co = std::sin(theta);

    dim3 block(32, 16);
    dim3 grid((basis.max_b_count + block.x - 1) / block.x,
              (basis.max_a_count + block.y - 1) / block.y);

    switch (type)
    {
    case 0:
    {
        if constexpr (std::is_arithmetic_v<Tv>)
        {
            break;
        }
        else
        {
            GroupsSliceDev<Ti, Tv> slice = make_groups_slice(net.diag_groups);
            int rank = net.diag_groups.host_ranks[pos];
            if (rank == 1)
                expm_diag_kernel_2d<1, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, theta, dev_vec);
            else if (rank == 2)
                expm_diag_kernel_2d<2, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, theta, dev_vec);
            else
                expm_diag_kernel_2d<0, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, theta, dev_vec);
            break;
        }
    }
    case 1:
    {
        const GroupsSliceDev<Ti, Tv> slice = make_groups_slice(net.pure_a_groups);
        int rank = net.pure_a_groups.host_ranks[pos];
        if (rank == 1)
            expm_pure_a_kernel_2d<1, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, cd, co, dev_vec);
        else if (rank == 2)
            expm_pure_a_kernel_2d<2, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, cd, co, dev_vec);
        else
            expm_pure_a_kernel_2d<0, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, cd, co, dev_vec);
        break;
    }
    case 2:
    {
        const GroupsSliceDev<Ti, Tv> slice = make_groups_slice(net.pure_b_groups);
        int rank = net.pure_b_groups.host_ranks[pos];
        if (rank == 1)
            expm_pure_b_kernel_2d<1, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, cd, co, dev_vec);
        else if (rank == 2)
            expm_pure_b_kernel_2d<2, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, cd, co, dev_vec);
        else
            expm_pure_b_kernel_2d<0, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, cd, co, dev_vec);
        break;
    }
    case 3:
    {
        const GroupsSliceDev<Ti, Tv> slice = make_groups_slice(net.mixed_groups);
        int rank = net.mixed_groups.host_ranks[pos];
        if (rank == 1)
            expm_mixed_kernel_2d<1, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, cd, co, dev_vec);
        else if (rank == 2)
            expm_mixed_kernel_2d<2, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, cd, co, dev_vec);
        else
            expm_mixed_kernel_2d<0, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, cd, co, dev_vec);
        break;
    }
    default:
        break;
    }
}
