#pragma once
#include "cuda_utils.cuh"
#include "cuda_basis.cuh"
#include "cuda_otf.cuh"

template <int Rank, typename Ti, typename Tv>
__global__ void backgrad_diag_kernel_2d(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos, double theta,
    Tv *__restrict__ lp,
    Tv *__restrict__ rp,
    Tv *__restrict__ d_res)
{
    const int db = blockIdx.x * blockDim.x + threadIdx.x;
    const int da = blockIdx.y * blockDim.y + threadIdx.y;

    Tv local_res = {};

    if (da < basis.max_a_count && db < basis.max_b_count)
    {
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

            const Tv u = fast_diag_exp_dev<Tv>(vt, -theta);
            const Tv du = fast_diag_grad_dev<Tv>(vt, theta);

            lp[di] *= u;
            local_res += dev_conj(lp[di] * du) * rp[di];
            rp[di] *= u;
        }
    }

    local_res = warp_reduce_sum(local_res);

    __shared__ Tv shared_sums[16];
    if (threadIdx.x == 0)
        shared_sums[threadIdx.y] = local_res;

    __syncthreads();

    if (threadIdx.y == 0)
    {
        local_res = (threadIdx.x < 16) ? shared_sums[threadIdx.x] : Tv{};
        local_res = warp_reduce_sum(local_res);

        if (threadIdx.x == 0)
        {
            atomicAdd_Tv(d_res, local_res);
        }
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void backgrad_mixed_kernel_2d(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos, double ecd, double eco, double gcd, double gco,
    Tv *__restrict__ lp,
    Tv *__restrict__ rp,
    Tv *__restrict__ d_res)
{
    const int db = blockIdx.x * blockDim.x + threadIdx.x;
    const int da = blockIdx.y * blockDim.y + threadIdx.y;

    Tv local_res = {};

    if (da < basis.max_a_count && db < basis.max_b_count)
    {
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

            backgrad_update_dev<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, ecd, eco, gcd, gco);
        }
    }

    // -----------------------------------------------------------------
    // 【终极归约区】：用 1 纳秒的时间，把这 256 个线程的 local_res 捏成 1 个
    // -----------------------------------------------------------------
    // 1. Warp 内部 32 线程快速规约
    local_res = warp_reduce_sum(local_res);

    // 2. 将每个 Warp 的 0 号线程结果存入共享内存（Block 为 32x8，刚好 8 个 Warp）
    __shared__ Tv shared_sums[16];
    if (threadIdx.x == 0)
        shared_sums[threadIdx.y] = local_res;

    __syncthreads(); // 等待 8 个 Warp 都写完

    // 3. 让 0 号 Warp 再做一次归约，把 8 个结果合而为一
    if (threadIdx.y == 0)
    {
        local_res = (threadIdx.x < 16) ? shared_sums[threadIdx.x] : Tv{};
        local_res = warp_reduce_sum(local_res);

        // 4. Block 的总代表（线程 0）去敲全局内存的门
        if (threadIdx.x == 0)
        {
            atomicAdd_Tv(d_res, local_res);
        }
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void backgrad_pure_a_kernel_2d(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos, double ecd, double eco, double gcd, double gco,
    Tv *__restrict__ lp,
    Tv *__restrict__ rp,
    Tv *__restrict__ d_res)
{
    const int db = blockIdx.x * blockDim.x + threadIdx.x;
    const int da = blockIdx.y * blockDim.y + threadIdx.y;

    Tv local_res = {};

    if (da < basis.max_a_count && db < basis.max_b_count)
    {
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

            backgrad_update_dev<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, ecd, eco, gcd, gco);
        }
    }

    local_res = warp_reduce_sum(local_res);

    __shared__ Tv shared_sums[16];
    if (threadIdx.x == 0)
        shared_sums[threadIdx.y] = local_res;

    __syncthreads();

    if (threadIdx.y == 0)
    {
        local_res = (threadIdx.x < 16) ? shared_sums[threadIdx.x] : Tv{};
        local_res = warp_reduce_sum(local_res);

        if (threadIdx.x == 0)
        {
            atomicAdd_Tv(d_res, local_res);
        }
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void backgrad_pure_b_kernel_2d(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos, double ecd, double eco, double gcd, double gco,
    Tv *__restrict__ lp,
    Tv *__restrict__ rp,
    Tv *__restrict__ d_res)
{
    const int db = blockIdx.x * blockDim.x + threadIdx.x;
    const int da = blockIdx.y * blockDim.y + threadIdx.y;

    Tv local_res = {};

    if (da < basis.max_a_count && db < basis.max_b_count)
    {
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

            backgrad_update_dev<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, ecd, eco, gcd, gco);
        }
    }

    local_res = warp_reduce_sum(local_res);

    __shared__ Tv shared_sums[16];
    if (threadIdx.x == 0)
        shared_sums[threadIdx.y] = local_res;

    __syncthreads();

    if (threadIdx.y == 0)
    {
        local_res = (threadIdx.x < 16) ? shared_sums[threadIdx.x] : Tv{};
        local_res = warp_reduce_sum(local_res);

        if (threadIdx.x == 0)
        {
            atomicAdd_Tv(d_res, local_res);
        }
    }
}

template <typename Ti, typename Tv>
Tv backgrad_svd_network_otf_gpu(
    const BasisSliceDev<Ti> &basis_slice,
    const NetworkDev<Ti, Tv> &net,
    int64 idx, double theta,
    Tv *__restrict__ lp, 
    Tv *__restrict__ rp)
{
    const uint8 type = net.host_excit_types[idx];
    const int64 pos = net.host_sorted_idxs[idx];

    Tv h_res = {};
    Tv *d_res = nullptr;
    cudaMalloc(&d_res, sizeof(Tv));
    cudaMemset(d_res, 0, sizeof(Tv));


    const double ecd = std::cos(theta) - 1.0;
    const double eco = -std::sin(theta);
    const double gcd = -std::sin(theta);
    const double gco = std::cos(theta);

    dim3 block(32, 16);
    dim3 grid((basis_slice.max_b_count + block.x - 1) / block.x,
              (basis_slice.max_a_count + block.y - 1) / block.y);

    switch (type)
    {
    case 0: // Diag
    {
        if constexpr (std::is_arithmetic_v<Tv>)
        {
            cudaFree(d_res);
            return h_res;
        }
        else
        {
            GroupsSliceDev<Ti, Tv> slice = make_groups_slice(net.diag_groups);
            int rank = net.diag_groups.host_ranks[pos];
            if (rank == 1)
                backgrad_diag_kernel_2d<1, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, theta, lp, rp, d_res);
            else if (rank == 2)
                backgrad_diag_kernel_2d<2, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, theta, lp, rp, d_res);
            else
                backgrad_diag_kernel_2d<0, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, theta, lp, rp, d_res);
            break;
        }
    }
    case 1:
    {
        const GroupsSliceDev<Ti, Tv> slice = make_groups_slice(net.pure_a_groups);
        int rank = net.pure_a_groups.host_ranks[pos];
        if (rank == 1)
            backgrad_pure_a_kernel_2d<1, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, ecd, eco, gcd, gco, lp, rp, d_res);
        else if (rank == 2)
            backgrad_pure_a_kernel_2d<2, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, ecd, eco, gcd, gco, lp, rp, d_res);
        else
            backgrad_pure_a_kernel_2d<0, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, ecd, eco, gcd, gco, lp, rp, d_res);
        break;
    }
    case 2:
    {
        const GroupsSliceDev<Ti, Tv> slice = make_groups_slice(net.pure_b_groups);
        int rank = net.pure_b_groups.host_ranks[pos];
        if (rank == 1)
            backgrad_pure_b_kernel_2d<1, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, ecd, eco, gcd, gco, lp, rp, d_res);
        else if (rank == 2)
            backgrad_pure_b_kernel_2d<2, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, ecd, eco, gcd, gco, lp, rp, d_res);
        else
            backgrad_pure_b_kernel_2d<0, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, ecd, eco, gcd, gco, lp, rp, d_res);
        break;
    }
    case 3:
    {
        const GroupsSliceDev<Ti, Tv> slice = make_groups_slice(net.mixed_groups);
        int rank = net.mixed_groups.host_ranks[pos];
        if (rank == 1)
            backgrad_mixed_kernel_2d<1, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, ecd, eco, gcd, gco, lp, rp, d_res);
        else if (rank == 2)
            backgrad_mixed_kernel_2d<2, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, ecd, eco, gcd, gco, lp, rp, d_res);
        else
            backgrad_mixed_kernel_2d<0, Ti, Tv><<<grid, block>>>(basis_slice, slice, pos, ecd, eco, gcd, gco, lp, rp, d_res);
        break;
    }
    default:
        break;
    }

    cudaMemcpy(&h_res, d_res, sizeof(Tv), cudaMemcpyDeviceToHost);
    cudaFree(d_res);

    return h_res;
}


template <typename Ti, typename Tv>
Tv backgrad_svd_network_otf_gpu(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    int64 idx, double theta,
    Tv *__restrict__ lp,
    Tv *__restrict__ rp)
{
    BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    return backgrad_svd_network_otf_gpu<Ti, Tv>(basis_slice, net, idx, theta, lp, rp);
}
