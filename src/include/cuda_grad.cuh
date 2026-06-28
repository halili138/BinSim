#pragma once
#include <algorithm>
#include "cuda_utils.cuh"
#include "cuda_basis.cuh"
#include "cuda_otf.cuh"

template <typename Tv>
__device__ __forceinline__ void grad_block_reduce_atomic_add(Tv local_res, Tv *__restrict__ d_res)
{
    local_res = warp_reduce_sum(local_res);
    __shared__ Tv shared_sums[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0)
        shared_sums[warp] = local_res;
    __syncthreads();

    if (warp == 0)
    {
        const int num_warps = (blockDim.x + 31) >> 5;
        local_res = (lane < num_warps) ? shared_sums[lane] : Tv{};
        local_res = warp_reduce_sum(local_res);
        if (lane == 0)
            atomicAdd_Tv(d_res, local_res);
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv>
__global__ void grad_single_group_sharedtile_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos,
    int task_offset,
    double theta,
    double cd,
    double co,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp,
    Tv *__restrict__ d_res)
{
    const int bid = blockIdx.x;
    const int task_idx = task_offset + blockIdx.y;
    constexpr int SHARED_MEM_SIZE = Rank == 1 ? TILE_B : (Rank == 2 ? TILE_B * 2 : TILE_B * KERNEL_MAX_RANK);

    __shared__ Tv sh_pb[SHARED_MEM_SIZE];
    __shared__ int sh_sb[TILE_B];

    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
    const int num_a_tiles = (n_a + blockDim.x - 1) / blockDim.x;

    Tv local_res = {};
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

    if (valid_a)
    {
        const Ti dst_astr = astrs[a];
        constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
        Tv pa[STACK_SIZE] = {};

        if constexpr (TypeCode == 0)
        {
            compute_phase_dev<Rank, Ti, Tv>(dst_astr, groups.flat_zas + groups.za_start[pos], groups.num_zas[pos], groups.flat_wa + groups.wa_start[pos], pa, 1, rank);
            for (int b_offset = 0; b_offset < cur_b; ++b_offset)
            {
                const Tv vt = compute_coeff_dev<Rank, Tv>(pa, sh_pb, TILE_B, rank, b_offset);
                const int64 di = basis.block_offsets[bid] + (int64)a * n_b + b_start + b_offset;
                local_res += dev_conj(lp[di] * fast_diag_grad_dev<Tv>(vt, theta)) * rp[di];
            }
        }
        else if (src_bid != -1 && src_bid >= bid)
        {
            Ti src_astr = dst_astr;
            if constexpr (TypeCode == 1 || TypeCode == 3)
                src_astr ^= groups.axs[pos];
            const int sa = (TypeCode == 2) ? a : basis.astr2idx[src_astr];
            if (sa != -1 && !(src_bid == bid && sa < a))
            {
                compute_phase_dev<Rank, Ti, Tv>(src_astr, groups.flat_zas + groups.za_start[pos], groups.num_zas[pos], groups.flat_wa + groups.wa_start[pos], pa, 1, rank);
                const int src_n_b = basis.block_num_b[src_bid];
                const int64 src_row = basis.block_offsets[src_bid] + (int64)sa * src_n_b;
                const int64 dst_row = basis.block_offsets[bid] + (int64)a * n_b;
                for (int b_offset = 0; b_offset < cur_b; ++b_offset)
                {
                    const int sb = sh_sb[b_offset];
                    if (sb == -1 || (src_bid == bid && sa == a && sb < b_start + b_offset))
                        continue;
                    const Tv vt = compute_coeff_dev<Rank, Tv>(pa, sh_pb, TILE_B, rank, b_offset);
                    grad_update_dev<Tv>(local_res, lp + src_row + sb, lp + dst_row + b_start + b_offset, rp + src_row + sb, rp + dst_row + b_start + b_offset, vt, cd, co);
                }
            }
        }
    }

    grad_block_reduce_atomic_add(local_res, d_res);
}

template <int TypeCode, typename Ti, typename Tv>
static inline void launch_single_group_grad_tile(
    const BasisSliceDev<Ti> &basis_slice,
    const GroupsSliceDev<Ti, Tv> &groups,
    int64 pos,
    double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp,
    Tv *__restrict__ d_res,
    int host_rank,
    int max_tasks)
{
    if constexpr (TypeCode == 0 && std::is_arithmetic_v<Tv>)
        return;

    const int block_size = 256;
    if (max_tasks <= 0)
        return;
    const double cd = -std::sin(theta);
    const double co = std::cos(theta);
    constexpr int max_grid_y = 65535;
    for (int task_offset = 0; task_offset < max_tasks; task_offset += max_grid_y)
    {
        const int launch_tasks = std::min(max_grid_y, max_tasks - task_offset);
        dim3 grid(basis_slice.num_blocks, launch_tasks);
        if (host_rank == 1)
            grad_single_group_sharedtile_kernel<1, TypeCode, Ti, Tv><<<grid, block_size>>>(basis_slice, groups, pos, task_offset, theta, cd, co, lp, rp, d_res);
        else if (host_rank == 2)
            grad_single_group_sharedtile_kernel<2, TypeCode, Ti, Tv><<<grid, block_size>>>(basis_slice, groups, pos, task_offset, theta, cd, co, lp, rp, d_res);
        else
            grad_single_group_sharedtile_kernel<0, TypeCode, Ti, Tv><<<grid, block_size>>>(basis_slice, groups, pos, task_offset, theta, cd, co, lp, rp, d_res);
    }
}

template <typename Ti, typename Tv>
Tv grad_svd_network_otf_gpu(
    const BasisSliceDev<Ti> &basis_slice,
    const NetworkDev<Ti, Tv> &net,
    int64 idx, double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp,
    int max_tasks)
{
    const uint8 type = net.host_excit_types[idx];
    const int64 pos = net.host_sorted_idxs[idx];

    Tv h_res = {};
    Tv *d_res = nullptr;
    cudaMalloc(&d_res, sizeof(Tv));
    cudaMemset(d_res, 0, sizeof(Tv));

    switch (type)
    {
    case 0:
        if constexpr (!std::is_arithmetic_v<Tv>)
            launch_single_group_grad_tile<0, Ti, Tv>(basis_slice, make_groups_slice(net.diag_groups), pos, theta, lp, rp, d_res, net.diag_groups.host_ranks[pos], max_tasks);
        break;
    case 1:
        launch_single_group_grad_tile<1, Ti, Tv>(basis_slice, make_groups_slice(net.pure_a_groups), pos, theta, lp, rp, d_res, net.pure_a_groups.host_ranks[pos], max_tasks);
        break;
    case 2:
        launch_single_group_grad_tile<2, Ti, Tv>(basis_slice, make_groups_slice(net.pure_b_groups), pos, theta, lp, rp, d_res, net.pure_b_groups.host_ranks[pos], max_tasks);
        break;
    case 3:
        launch_single_group_grad_tile<3, Ti, Tv>(basis_slice, make_groups_slice(net.mixed_groups), pos, theta, lp, rp, d_res, net.mixed_groups.host_ranks[pos], max_tasks);
        break;
    default:
        break;
    }

    cudaMemcpy(&h_res, d_res, sizeof(Tv), cudaMemcpyDeviceToHost);
    cudaFree(d_res);
    return h_res;
}

template <typename Ti, typename Tv>
Tv grad_svd_network_otf_gpu(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    int64 idx, double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    const int block_size = 256;
    int max_tasks = 0;
    for (int bid = 0; bid < basis.num_blocks; ++bid)
    {
        const int num_a_tiles = (basis.host_block_num_a[bid] + block_size - 1) / block_size;
        const int num_b_tiles = (basis.host_block_num_b[bid] + TILE_B - 1) / TILE_B;
        max_tasks = std::max(max_tasks, num_a_tiles * num_b_tiles);
    }
    return grad_svd_network_otf_gpu<Ti, Tv>(basis_slice, net, idx, theta, lp, rp, max_tasks);
}
