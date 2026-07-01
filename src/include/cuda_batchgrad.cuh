#pragma once
#include "cuda_hvec.cuh"

template <typename Tv>
__device__ __forceinline__ void atomic_add_grad(Tv *__restrict__ grads, int original_idx, Tv val)
{
    atomicAdd_Tv(grads + original_idx, val);
}

template <int Rank, typename Ti, typename Tv>
__global__ void batchgrad_diag_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    const double *__restrict__ thetas,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp,
    Tv *__restrict__ grads)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
        return;
    }
    else
    {
        const int bid = basis.target_bids ? basis.target_bids[blockIdx.x] : blockIdx.x;
        const int total_groups = groups.num_groups;

        constexpr int SHARED_MEM_SIZE = BATCH_GROUP_SHARED_MEM<Rank>;
        constexpr int BATCH_SIZE = BATCH_GROUP_SIZE<Rank>;

        __shared__ Tv sh_pb[SHARED_MEM_SIZE];

        const int num_chunks = (total_groups + BATCH_SIZE - 1) / BATCH_SIZE;
        const int n_a = basis.block_num_a[bid];
        const int n_b = basis.block_num_b[bid];
        const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
        const int num_a_tiles = (n_a + TILE_A - 1) / TILE_A;
        const int total_tiles = num_b_tiles * num_a_tiles;
        const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
        const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];
        const Tv *lp_bid = lp + basis.block_offsets[bid];
        const Tv *rp_bid = rp + basis.block_offsets[bid];

        for (int task_idx = blockIdx.y; task_idx < total_tiles; task_idx += gridDim.y)
        {
            const int b_tile_idx = task_idx % num_b_tiles;
            const int a_tile_idx = task_idx / num_b_tiles;
            const int b_tile_start = b_tile_idx * TILE_B;
            const int current_tile_b = min(TILE_B, n_b - b_tile_start);
            const Ti *bstrs_tile_start = bstrs + b_tile_start;
            const int a = a_tile_idx * TILE_A + threadIdx.x;
            const bool valid_a = a < min(n_a, a_tile_idx * TILE_A + TILE_A);
            const Ti astr = valid_a ? astrs[a] : 0;
            const Tv *la = valid_a ? (lp_bid + (int64)a * n_b) : nullptr;
            const Tv *ra = valid_a ? (rp_bid + (int64)a * n_b) : nullptr;

            for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
            {
                const int chunk_start_g = chunk_idx * BATCH_SIZE;
                const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);
                const int total_sh_elements = current_chunk_groups * current_tile_b;

                for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
                {
                    const int g_offset = sh_idx / current_tile_b;
                    const int b_offset = sh_idx % current_tile_b;
                    const int g = chunk_start_g + g_offset;
                    Tv *sh_pb_ptr = sh_pb + (g_offset * TILE_B + b_offset);
                    compute_phase_dev<Rank, Ti, Tv>(
                        bstrs_tile_start[b_offset], groups.flat_zbs + groups.zb_start[g], groups.num_zbs[g],
                        groups.flat_wb + groups.wb_start[g], sh_pb_ptr, BATCH_SIZE * TILE_B, groups.ranks[g]);
                }
                __syncthreads();

                if (valid_a)
                {
                    for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                    {
                        const int g = chunk_start_g + g_offset;
                        const int rank = groups.ranks[g];
                        constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
                        Tv pa[STACK_SIZE] = {};
                        compute_phase_dev<Rank, Ti, Tv>(
                            astr, groups.flat_zas + groups.za_start[g], groups.num_zas[g], groups.flat_wa + groups.wa_start[g], pa, 1, rank);

                        const Tv *pb = sh_pb + (g_offset * TILE_B);
                        const int original_idx = groups.original_idx[g];
                        const double theta = thetas[original_idx];
                        Tv local_res = {};
                        for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                        {
                            const int b = b_tile_start + b_offset;
                            const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                            const Tv du = fast_diag_grad_dev<Tv>(vt, theta);
                            local_res += dev_conj(la[b] * du) * ra[b];
                        }
                        atomic_add_grad(grads, original_idx, local_res);
                    }
                }
                __syncthreads();
            }
        }
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv>
__global__ void batchgrad_offdiag_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    const double *__restrict__ thetas,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp,
    Tv *__restrict__ grads)
{
    const int bid = basis.target_bids ? basis.target_bids[blockIdx.x] : blockIdx.x;
    const int total_groups = groups.num_groups;
    const int *a_idx_map = basis.astr2idx;
    const int *b_idx_map = basis.bstr2idx;

    constexpr int SHARED_MEM_SIZE = BATCH_GROUP_SHARED_MEM<Rank>;
    constexpr int BATCH_SIZE = BATCH_GROUP_SIZE<Rank>;
    constexpr int IDX_MEM_SIZE = (TypeCode == 1) ? 1 : BATCH_GROUP_IDX_MEM<Rank>;

    __shared__ Tv sh_pb[SHARED_MEM_SIZE];
    __shared__ int sh_sb[IDX_MEM_SIZE];
    __shared__ int sh_src_bid[BATCH_SIZE];
    __shared__ int sh_valid[BATCH_SIZE];

    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int asym = basis.block_asym[bid];
    const int bsym = basis.block_bsym[bid];
    const int nirp = basis.num_irreps;
    const int num_chunks = (total_groups + BATCH_SIZE - 1) / BATCH_SIZE;
    const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
    const int num_a_tiles = (n_a + TILE_A - 1) / TILE_A;
    const int total_tiles = num_b_tiles * num_a_tiles;
    const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
    const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];

    for (int task_idx = blockIdx.y; task_idx < total_tiles; task_idx += gridDim.y)
    {
        const int b_tile_idx = task_idx % num_b_tiles;
        const int a_tile_idx = task_idx / num_b_tiles;
        const int b_tile_start = b_tile_idx * TILE_B;
        const int current_tile_b = min(TILE_B, n_b - b_tile_start);
        const Ti *bstrs_tile_start = bstrs + b_tile_start;
        const int a = a_tile_idx * TILE_A + threadIdx.x;
        const bool valid_a = a < min(n_a, a_tile_idx * TILE_A + TILE_A);
        const Ti dst_astr = valid_a ? astrs[a] : 0;

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);

            for (int g_offset = threadIdx.x; g_offset < current_chunk_groups; g_offset += blockDim.x)
            {
                const int g = chunk_start_g + g_offset;
                const int h = compute_sym_hash<TypeCode>(asym, bsym, groups.asyms[g], groups.bsyms[g], nirp);

                const int src_bid = basis.block_map[h];
                sh_src_bid[g_offset] = src_bid;
                sh_valid[g_offset] = (src_bid != -1 && src_bid >= bid) ? 1 : 0;
            }
            __syncthreads();

            const int total_sh_elements = current_chunk_groups * current_tile_b;
            for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
            {
                const int g_offset = sh_idx / current_tile_b;
                const int b_offset = sh_idx % current_tile_b;
                const int sh_flat_offset = g_offset * TILE_B + b_offset;
                if constexpr (TypeCode != 1)
                    sh_sb[sh_flat_offset] = -1;
                if (sh_valid[g_offset] == 0)
                    continue;

                const int g = chunk_start_g + g_offset;
                Ti src_bstr = bstrs_tile_start[b_offset];
                if constexpr (TypeCode == 2 || TypeCode == 3)
                    src_bstr ^= groups.bxs[g];

                if constexpr (TypeCode != 1)
                {
                    const int sb = b_idx_map[src_bstr];
                    sh_sb[sh_flat_offset] = sb;
                    if (sb == -1)
                        continue;
                }

                compute_phase_dev<Rank, Ti, Tv>(
                    src_bstr, groups.flat_zbs + groups.zb_start[g], groups.num_zbs[g], groups.flat_wb + groups.wb_start[g],
                    sh_pb + sh_flat_offset, BATCH_SIZE * TILE_B, groups.ranks[g]);
            }
            __syncthreads();

            if (valid_a)
            {
                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    if (sh_valid[g_offset] == 0)
                        continue;

                    const int g = chunk_start_g + g_offset;
                    const int src_bid = sh_src_bid[g_offset];
                    Ti src_astr = dst_astr;
                    if constexpr (TypeCode == 1 || TypeCode == 3)
                        src_astr ^= groups.axs[g];
                    const int sa = (TypeCode == 2) ? a : a_idx_map[src_astr];
                    if (sa == -1 || (src_bid == bid && sa < a))
                        continue;

                    const int rank = groups.ranks[g];
                    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
                    Tv pa[STACK_SIZE] = {};
                    compute_phase_dev<Rank, Ti, Tv>(
                        src_astr, groups.flat_zas + groups.za_start[g], groups.num_zas[g], groups.flat_wa + groups.wa_start[g], pa, 1, rank);

                    const int src_n_b = basis.block_num_b[src_bid];
                    const int64 src_row = basis.block_offsets[src_bid] + (int64)sa * src_n_b;
                    const int64 dst_row = basis.block_offsets[bid] + (int64)a * n_b;
                    const Tv *pb = sh_pb + (g_offset * TILE_B);
                    const int original_idx = groups.original_idx[g];
                    const double theta = thetas[original_idx];
                    const double cd = -std::sin(theta);
                    const double co = std::cos(theta);

                    Tv local_res = {};
                    if constexpr (TypeCode == 1)
                    {
                        for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                        {
                            const int sb = b_tile_start + b_offset;
                            if (src_bid == bid && sa == a && sb < b_tile_start + b_offset)
                                continue;

                            const int64 si = src_row + sb;
                            const int64 di = dst_row + b_tile_start + b_offset;
                            const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                            grad_update_dev<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
                        }
                    }
                    else
                    {
                        const int *sb_tile = sh_sb + (g_offset * TILE_B);

                        for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                        {
                            const int sb = sb_tile[b_offset];
                            if (sb == -1 || (src_bid == bid && sa == a && sb < b_tile_start + b_offset))
                                continue;

                            const int64 si = src_row + sb;
                            const int64 di = dst_row + b_tile_start + b_offset;
                            const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                            grad_update_dev<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
                        }
                    }
                    atomic_add_grad(grads, original_idx, local_res);
                }
            }
            __syncthreads();
        }
    }
}

template <int TypeCode, typename Ti, typename Tv>
static inline void dispatch_batchgrad_chunks_by_rank_gpu(
    const BasisSliceDev<Ti> &basis_slice,
    int num_active_blocks,
    const GroupsViewDev<Ti, Tv> &groups,
    const double *__restrict__ thetas,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp,
    Tv *__restrict__ grads)
{
    const int64 total_ngs = groups.num_groups;
    if (total_ngs == 0)
        return;

    int64 start = 0;
    while (start < total_ngs)
    {
        const int current_rank = groups.host_ranks[start];
        const int dispatch_rank = (current_rank == 1 || current_rank == 2) ? current_rank : 0;
        int64 end = start + 1;
        while (end < total_ngs)
        {
            const int next_rank = groups.host_ranks[end];
            const int next_dispatch_rank = (next_rank == 1 || next_rank == 2) ? next_rank : 0;
            if (next_dispatch_rank != dispatch_rank)
                break;
            ++end;
        }

        GroupsSliceDev<Ti, Tv> slice;
        slice.num_groups = end - start;
        slice.axs = groups.axs + start;
        slice.bxs = groups.bxs + start;
        slice.asyms = groups.asyms + start;
        slice.bsyms = groups.bsyms + start;
        slice.ranks = groups.ranks + start;
        slice.num_zas = groups.num_zas + start;
        slice.num_zbs = groups.num_zbs + start;
        slice.za_start = groups.za_start + start;
        slice.zb_start = groups.zb_start + start;
        slice.wa_start = groups.wa_start + start;
        slice.wb_start = groups.wb_start + start;
        slice.flat_zas = groups.flat_zas;
        slice.flat_zbs = groups.flat_zbs;
        slice.flat_wa = groups.flat_wa;
        slice.flat_wb = groups.flat_wb;
        slice.original_idx = groups.original_idx + start;

        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        dim3 grid_size(num_active_blocks, num_sms * 4);
        constexpr int block_size = 256;

        if constexpr (TypeCode == 0)
        {
            if (dispatch_rank == 1)
                batchgrad_diag_kernel<1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
            else if (dispatch_rank == 2)
                batchgrad_diag_kernel<2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
            else
                batchgrad_diag_kernel<0, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
        }
        else if constexpr (TypeCode == 1)
        {
            if (dispatch_rank == 1)
                batchgrad_offdiag_kernel<1, 1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
            else if (dispatch_rank == 2)
                batchgrad_offdiag_kernel<2, 1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
            else
                batchgrad_offdiag_kernel<0, 1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
        }
        else if constexpr (TypeCode == 2)
        {
            if (dispatch_rank == 1)
                batchgrad_offdiag_kernel<1, 2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
            else if (dispatch_rank == 2)
                batchgrad_offdiag_kernel<2, 2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
            else
                batchgrad_offdiag_kernel<0, 2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
        }
        else if constexpr (TypeCode == 3)
        {
            if (dispatch_rank == 1)
                batchgrad_offdiag_kernel<1, 3, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
            else if (dispatch_rank == 2)
                batchgrad_offdiag_kernel<2, 3, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
            else
                batchgrad_offdiag_kernel<0, 3, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, thetas, lp, rp, grads);
        }

        start = end;
    }
}

template <typename Ti, typename Tv>
void cuda_batchgrad(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    const double *__restrict__ thetas,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp,
    Tv *__restrict__ grads)
{
    CUDA_CHECK(cudaMemset(grads, 0, net.host_sorted_idxs.size() * sizeof(Tv)));
    BasisSliceDev<Ti> slice = make_basis_slice(basis);

    dispatch_batchgrad_chunks_by_rank_gpu<0>(slice, basis.num_blocks, net.diag_groups, thetas, lp, rp, grads);
    dispatch_batchgrad_chunks_by_rank_gpu<1>(slice, basis.num_blocks, net.pure_a_groups, thetas, lp, rp, grads);
    dispatch_batchgrad_chunks_by_rank_gpu<2>(slice, basis.num_blocks, net.pure_b_groups, thetas, lp, rp, grads);
    dispatch_batchgrad_chunks_by_rank_gpu<3>(slice, basis.num_blocks, net.mixed_groups, thetas, lp, rp, grads);
}
