#pragma once
#include <algorithm>
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

        const Tv u = fast_diag_exp_dev<Tv>(vt, theta);
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

        expm_update_dev<Tv>(vec + si, vec + di, vt, cd, co);
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

        expm_update_dev<Tv>(vec + si, vec + di, vt, cd, co);
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

        expm_update_dev<Tv>(vec + si, vec + di, vt, cd, co);
    }
}



inline constexpr int SINGLE_EXPM_BETA_REG_TILE = 8;

template <int Rank, typename Ti, typename Tv>
__device__ __forceinline__ Tv single_expm_coeff_from_reg_pb(
    const Tv *__restrict__ pa,
    const Tv *__restrict__ pb,
    int rank,
    int b_offset)
{
    if constexpr (Rank == 1)
        return pa[0] * pb[b_offset];
    else if constexpr (Rank == 2)
        return pa[0] * pb[b_offset] + pa[1] * pb[SINGLE_EXPM_BETA_REG_TILE + b_offset];
    else
    {
        Tv vt = {};
        for (int r = 0; r < rank; ++r)
            vt += pa[r] * pb[r * SINGLE_EXPM_BETA_REG_TILE + b_offset];
        return vt;
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv>
__global__ void expm_single_group_regtile_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos,
    double theta,
    Tv *__restrict__ vec)
{
    const int bid = blockIdx.x;
    const int a_tile_idx = blockIdx.y;
    const int b_tile_idx = blockIdx.z;
    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int a = a_tile_idx * blockDim.x + threadIdx.x;
    if (a >= n_a)
        return;

    const int b_start = b_tile_idx * SINGLE_EXPM_BETA_REG_TILE;
    const int cur_b = min(SINGLE_EXPM_BETA_REG_TILE, n_b - b_start);
    if (cur_b <= 0)
        return;

    const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
    const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];
    const Ti dst_astr = astrs[a];
    const int rank = groups.ranks[pos];
    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
    constexpr int PB_SIZE = Rank == 1 ? SINGLE_EXPM_BETA_REG_TILE : (Rank == 2 ? 2 * SINGLE_EXPM_BETA_REG_TILE : 128 * SINGLE_EXPM_BETA_REG_TILE);
    Tv pa[STACK_SIZE] = {};
    Tv pb[PB_SIZE] = {};

    if constexpr (TypeCode == 0)
    {
        if constexpr (std::is_arithmetic_v<Tv>)
            return;
        else
        {
            compute_phase_dev<Rank, Ti, Tv>(dst_astr, groups.flat_zas + groups.za_start[pos], groups.num_zas[pos], groups.flat_wa + groups.wa_start[pos], pa, 1, rank);
            for (int b_offset = 0; b_offset < cur_b; ++b_offset)
            {
                compute_phase_dev<Rank, Ti, Tv>(bstrs[b_start + b_offset], groups.flat_zbs + groups.zb_start[pos], groups.num_zbs[pos], groups.flat_wb + groups.wb_start[pos], pb + b_offset, SINGLE_EXPM_BETA_REG_TILE, rank);
                const Tv vt = single_expm_coeff_from_reg_pb<Rank, Ti, Tv>(pa, pb, rank, b_offset);
                const int64 di = basis.block_offsets[bid] + (int64)a * n_b + b_start + b_offset;
                vec[di] *= fast_diag_exp_dev<Tv>(vt, theta);
            }
        }
    }
    else
    {
        const double cd = std::cos(theta) - 1.0;
        const double co = std::sin(theta);
        int src_bid = -1;
        Ti src_astr = dst_astr;
        if constexpr (TypeCode == 1 || TypeCode == 3)
            src_astr ^= groups.axs[pos];

        if constexpr (TypeCode == 1)
        {
            const int h = (basis.block_asym[bid] ^ groups.asyms[pos]) * basis.num_irreps + basis.block_bsym[bid];
            src_bid = basis.block_map[h];
        }
        else if constexpr (TypeCode == 2)
        {
            const int h = basis.block_asym[bid] * basis.num_irreps + (basis.block_bsym[bid] ^ groups.bsyms[pos]);
            src_bid = basis.block_map[h];
        }
        else
        {
            const int h = (basis.block_asym[bid] ^ groups.asyms[pos]) * basis.num_irreps + (basis.block_bsym[bid] ^ groups.bsyms[pos]);
            src_bid = basis.block_map[h];
        }

        if (src_bid == -1 || src_bid < bid)
            return;

        const int sa = (TypeCode == 2) ? a : basis.astr2idx[src_astr];
        if (sa == -1 || (src_bid == bid && sa < a))
            return;

        compute_phase_dev<Rank, Ti, Tv>(src_astr, groups.flat_zas + groups.za_start[pos], groups.num_zas[pos], groups.flat_wa + groups.wa_start[pos], pa, 1, rank);
        for (int b_offset = 0; b_offset < cur_b; ++b_offset)
        {
            Ti src_bstr = bstrs[b_start + b_offset];
            if constexpr (TypeCode == 2 || TypeCode == 3)
                src_bstr ^= groups.bxs[pos];
            const int sb = (TypeCode == 1) ? (b_start + b_offset) : basis.bstr2idx[src_bstr];
            if (sb == -1 || (src_bid == bid && sa == a && sb < b_start + b_offset))
                continue;

            compute_phase_dev<Rank, Ti, Tv>(src_bstr, groups.flat_zbs + groups.zb_start[pos], groups.num_zbs[pos], groups.flat_wb + groups.wb_start[pos], pb + b_offset, SINGLE_EXPM_BETA_REG_TILE, rank);
            const Tv vt = single_expm_coeff_from_reg_pb<Rank, Ti, Tv>(pa, pb, rank, b_offset);
            const int src_n_b = basis.block_num_b[src_bid];
            const int64 si = basis.block_offsets[src_bid] + (int64)sa * src_n_b + sb;
            const int64 di = basis.block_offsets[bid] + (int64)a * n_b + b_start + b_offset;
            expm_update_dev<Tv>(vec + si, vec + di, vt, cd, co);
        }
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv>
__global__ void expm_single_group_sharedtile_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos,
    double theta,
    Tv *__restrict__ vec)
{
    const int bid = blockIdx.x;
    const int task_idx = blockIdx.y;
    constexpr int SHARED_MEM_SIZE = Rank == 1 ? TILE_B : (Rank == 2 ? TILE_B * 2 : TILE_B * KERNEL_MAX_RANK);
    __shared__ Tv sh_pb[SHARED_MEM_SIZE];
    __shared__ int sh_sb[TILE_B];

    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
    const int num_a_tiles = (n_a + blockDim.x - 1) / blockDim.x;
    if (task_idx >= num_a_tiles * num_b_tiles)
        return;
    const int b_tile_idx = task_idx % num_b_tiles;
    const int a_tile_idx = task_idx / num_b_tiles;
    const int b_start = b_tile_idx * TILE_B;
    const int cur_b = min(TILE_B, n_b - b_start);
    const int a = a_tile_idx * blockDim.x + threadIdx.x;
    const bool valid_a = a < n_a;
    const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
    const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];
    const int rank = groups.ranks[pos];

    int src_bid = bid;
    if constexpr (TypeCode != 0)
    {
        int h;
        if constexpr (TypeCode == 1)
            h = (basis.block_asym[bid] ^ groups.asyms[pos]) * basis.num_irreps + basis.block_bsym[bid];
        else if constexpr (TypeCode == 2)
            h = basis.block_asym[bid] * basis.num_irreps + (basis.block_bsym[bid] ^ groups.bsyms[pos]);
        else
            h = (basis.block_asym[bid] ^ groups.asyms[pos]) * basis.num_irreps + (basis.block_bsym[bid] ^ groups.bsyms[pos]);
        src_bid = basis.block_map[h];
    }

    for (int b_offset = threadIdx.x; b_offset < cur_b; b_offset += blockDim.x)
    {
        Ti src_bstr = bstrs[b_start + b_offset];
        int sb = b_start + b_offset;
        if constexpr (TypeCode == 2 || TypeCode == 3)
        {
            src_bstr ^= groups.bxs[pos];
            sb = basis.bstr2idx[src_bstr];
        }
        sh_sb[b_offset] = sb;
        if constexpr (TypeCode == 0)
            compute_phase_dev<Rank, Ti, Tv>(src_bstr, groups.flat_zbs + groups.zb_start[pos], groups.num_zbs[pos], groups.flat_wb + groups.wb_start[pos], sh_pb + b_offset, TILE_B, rank);
        else if (src_bid != -1 && src_bid >= bid && sb != -1)
            compute_phase_dev<Rank, Ti, Tv>(src_bstr, groups.flat_zbs + groups.zb_start[pos], groups.num_zbs[pos], groups.flat_wb + groups.wb_start[pos], sh_pb + b_offset, TILE_B, rank);
    }
    __syncthreads();

    if (!valid_a)
        return;

    const Ti dst_astr = astrs[a];
    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
    Tv pa[STACK_SIZE] = {};

    if constexpr (TypeCode == 0)
    {
        if constexpr (std::is_arithmetic_v<Tv>)
            return;
        else
        {
            compute_phase_dev<Rank, Ti, Tv>(dst_astr, groups.flat_zas + groups.za_start[pos], groups.num_zas[pos], groups.flat_wa + groups.wa_start[pos], pa, 1, rank);
            for (int b_offset = 0; b_offset < cur_b; ++b_offset)
            {
                const Tv vt = compute_coeff_dev<Rank, Tv>(pa, sh_pb, TILE_B, rank, b_offset);
                const int64 di = basis.block_offsets[bid] + (int64)a * n_b + b_start + b_offset;
                vec[di] *= fast_diag_exp_dev<Tv>(vt, theta);
            }
        }
    }
    else
    {
        if (src_bid == -1 || src_bid < bid)
            return;
        Ti src_astr = dst_astr;
        if constexpr (TypeCode == 1 || TypeCode == 3)
            src_astr ^= groups.axs[pos];
        const int sa = (TypeCode == 2) ? a : basis.astr2idx[src_astr];
        if (sa == -1 || (src_bid == bid && sa < a))
            return;

        compute_phase_dev<Rank, Ti, Tv>(src_astr, groups.flat_zas + groups.za_start[pos], groups.num_zas[pos], groups.flat_wa + groups.wa_start[pos], pa, 1, rank);
        const double cd = std::cos(theta) - 1.0;
        const double co = std::sin(theta);
        const int src_n_b = basis.block_num_b[src_bid];
        const int64 src_row = basis.block_offsets[src_bid] + (int64)sa * src_n_b;
        const int64 dst_row = basis.block_offsets[bid] + (int64)a * n_b;
        for (int b_offset = 0; b_offset < cur_b; ++b_offset)
        {
            const int sb = sh_sb[b_offset];
            if (sb == -1 || (src_bid == bid && sa == a && sb < b_start + b_offset))
                continue;
            const Tv vt = compute_coeff_dev<Rank, Tv>(pa, sh_pb, TILE_B, rank, b_offset);
            expm_update_dev<Tv>(vec + src_row + sb, vec + dst_row + b_start + b_offset, vt, cd, co);
        }
    }
}

template <int TypeCode, typename Ti, typename Tv, bool UseSharedTile>
static inline void launch_single_group_expm_tile(
    const BasisViewDev<Ti> &basis,
    const GroupsSliceDev<Ti, Tv> &groups,
    int64 pos,
    double theta,
    Tv *__restrict__ dev_vec,
    int host_rank)
{
    const int rank = host_rank;
    const int block_size = 256;
    const BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    if constexpr (UseSharedTile)
    {
        int max_tasks = 0;
        if ((int)basis.host_block_num_a.size() == basis.num_blocks &&
            (int)basis.host_block_num_b.size() == basis.num_blocks)
        {
            for (int bid = 0; bid < basis.num_blocks; ++bid)
            {
                const int num_a_tiles = (basis.host_block_num_a[bid] + block_size - 1) / block_size;
                const int num_b_tiles = (basis.host_block_num_b[bid] + TILE_B - 1) / TILE_B;
                max_tasks = std::max(max_tasks, num_a_tiles * num_b_tiles);
            }
        }
        else
        {
            const int max_a_tiles = (basis_slice.max_a_count + block_size - 1) / block_size;
            const int max_b_tiles = (basis_slice.max_b_count + TILE_B - 1) / TILE_B;
            max_tasks = max_a_tiles * max_b_tiles;
        }
        if (max_tasks == 0)
            return;
        dim3 grid(basis_slice.num_blocks, max_tasks);
        if (rank == 1)
            expm_single_group_sharedtile_kernel<1, TypeCode, Ti, Tv><<<grid, block_size>>>(basis_slice, groups, pos, theta, dev_vec);
        else if (rank == 2)
            expm_single_group_sharedtile_kernel<2, TypeCode, Ti, Tv><<<grid, block_size>>>(basis_slice, groups, pos, theta, dev_vec);
        else
            expm_single_group_sharedtile_kernel<0, TypeCode, Ti, Tv><<<grid, block_size>>>(basis_slice, groups, pos, theta, dev_vec);
    }
    else
    {
        const int max_a_tiles = (basis_slice.max_a_count + block_size - 1) / block_size;
        const int max_b_tiles = (basis_slice.max_b_count + SINGLE_EXPM_BETA_REG_TILE - 1) / SINGLE_EXPM_BETA_REG_TILE;
        dim3 grid(basis_slice.num_blocks, max_a_tiles, max_b_tiles);
        if (rank == 1)
            expm_single_group_regtile_kernel<1, TypeCode, Ti, Tv><<<grid, block_size>>>(basis_slice, groups, pos, theta, dev_vec);
        else if (rank == 2)
            expm_single_group_regtile_kernel<2, TypeCode, Ti, Tv><<<grid, block_size>>>(basis_slice, groups, pos, theta, dev_vec);
        else
            expm_single_group_regtile_kernel<0, TypeCode, Ti, Tv><<<grid, block_size>>>(basis_slice, groups, pos, theta, dev_vec);
    }
}

template <typename Ti, typename Tv, bool UseSharedTile>
void expm_svd_network_otf_gpu_tile(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    int64 idx,
    double theta,
    Tv *__restrict__ dev_vec)
{
    const uint8 type = net.host_excit_types[idx];
    const int64 pos = net.host_sorted_idxs[idx];
    switch (type)
    {
    case 0:
        launch_single_group_expm_tile<0, Ti, Tv, UseSharedTile>(basis, make_groups_slice(net.diag_groups), pos, theta, dev_vec, net.diag_groups.host_ranks[pos]);
        break;
    case 1:
        launch_single_group_expm_tile<1, Ti, Tv, UseSharedTile>(basis, make_groups_slice(net.pure_a_groups), pos, theta, dev_vec, net.pure_a_groups.host_ranks[pos]);
        break;
    case 2:
        launch_single_group_expm_tile<2, Ti, Tv, UseSharedTile>(basis, make_groups_slice(net.pure_b_groups), pos, theta, dev_vec, net.pure_b_groups.host_ranks[pos]);
        break;
    case 3:
        launch_single_group_expm_tile<3, Ti, Tv, UseSharedTile>(basis, make_groups_slice(net.mixed_groups), pos, theta, dev_vec, net.mixed_groups.host_ranks[pos]);
        break;
    default:
        break;
    }
}


template <typename Ti, typename Tv>
void expm_svd_network_otf_gpu(
    const BasisSliceDev<Ti> &basis_slice,
    const NetworkDev<Ti, Tv> &net,
    int64 idx, double theta,
    Tv *__restrict__ dev_vec)
{
    const uint8 type = net.host_excit_types[idx];
    const int64 pos = net.host_sorted_idxs[idx];


    const double cd = std::cos(theta) - 1.0;
    const double co = std::sin(theta);

    dim3 block(32, 16);
    dim3 grid((basis_slice.max_b_count + block.x - 1) / block.x,
              (basis_slice.max_a_count + block.y - 1) / block.y);

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


template <typename Ti, typename Tv>
void expm_svd_network_otf_gpu(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    int64 idx, double theta,
    Tv *__restrict__ dev_vec)
{
    BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    expm_svd_network_otf_gpu<Ti, Tv>(basis_slice, net, idx, theta, dev_vec);
}
