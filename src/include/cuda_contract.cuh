#pragma once
#include <algorithm>
#include <cmath>
#include "cuda_utils.cuh"
#include "cuda_basis.cuh"
#include "cuda_otf.cuh"

template <typename Tv>
__device__ __forceinline__ void cuda_block_reduce_atomic_add(Tv local_res, Tv *d_res)
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

template <typename Tv>
struct CudaExpmSingleGroupOp
{
    static constexpr bool SkipRealDiagonal = true;

    double theta;
    double cd;
    double co;
    Tv *vec;
    Tv *d_res;

    __device__ __forceinline__ void diag(Tv &, Tv vt, int64 di) const
    {
        vec[di] *= fast_diag_exp_dev<Tv>(vt, theta);
    }

    __device__ __forceinline__ void offdiag(Tv &, Tv vt, int64 si, int64 di) const
    {
        expm_update_dev<Tv>(vec + si, vec + di, vt, cd, co);
    }

    __device__ __forceinline__ void finish_block(Tv &) const {}
};

template <typename Tv>
struct CudaGradSingleGroupOp
{
    static constexpr bool SkipRealDiagonal = true;

    double theta;
    double cd;
    double co;
    const Tv *lp;
    const Tv *rp;
    Tv *d_res;

    __device__ __forceinline__ void diag(Tv &local_res, Tv vt, int64 di) const
    {
        local_res += dev_conj(lp[di] * fast_diag_grad_dev<Tv>(vt, theta)) * rp[di];
    }

    __device__ __forceinline__ void offdiag(Tv &local_res, Tv vt, int64 si, int64 di) const
    {
        grad_update_dev<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
    }

    __device__ __forceinline__ void finish_block(Tv &local_res) const
    {
        cuda_block_reduce_atomic_add(local_res, d_res);
    }
};

template <typename Tv>
struct CudaBackgradSingleGroupOp
{
    static constexpr bool SkipRealDiagonal = true;

    double theta;
    double ecd;
    double eco;
    double gcd;
    double gco;
    Tv *lp;
    Tv *rp;
    Tv *d_res;

    __device__ __forceinline__ void diag(Tv &local_res, Tv vt, int64 di) const
    {
        const Tv u = fast_diag_exp_dev<Tv>(vt, -theta);
        const Tv du = fast_diag_grad_dev<Tv>(vt, theta);
        lp[di] *= u;
        local_res += dev_conj(lp[di] * du) * rp[di];
        rp[di] *= u;
    }

    __device__ __forceinline__ void offdiag(Tv &local_res, Tv vt, int64 si, int64 di) const
    {
        backgrad_update_dev<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, ecd, eco, gcd, gco);
    }

    __device__ __forceinline__ void finish_block(Tv &local_res) const
    {
        cuda_block_reduce_atomic_add(local_res, d_res);
    }
};

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
__global__ void cuda_single_group_sharedtile_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos,
    Op op)
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
        constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : KERNEL_MAX_RANK);
        Tv pa[STACK_SIZE] = {};

        if constexpr (TypeCode == 0)
        {
            compute_phase_dev<Rank, Ti, Tv>(dst_astr, groups.flat_zas + groups.za_start[pos], groups.num_zas[pos], groups.flat_wa + groups.wa_start[pos], pa, 1, rank);
            for (int b_offset = 0; b_offset < cur_b; ++b_offset)
            {
                const Tv vt = compute_coeff_dev<Rank, Tv>(pa, sh_pb, TILE_B, rank, b_offset);
                const int64 di = basis.block_offsets[bid] + (int64)a * n_b + b_start + b_offset;
                op.diag(local_res, vt, di);
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
                    op.offdiag(local_res, vt, src_row + sb, dst_row + b_start + b_offset);
                }
            }
        }
    }

    op.finish_block(local_res);
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
static inline void launch_cuda_single_group_rank(
    const BasisSliceDev<Ti> &basis_slice,
    const GroupsSliceDev<Ti, Tv> &groups,
    int64 pos,
    Op op,
    int max_tasks)
{
    constexpr int block_size = 256;
    if (max_tasks <= 0)
        return;

    dim3 grid(basis_slice.num_blocks, max_tasks);
    cuda_single_group_sharedtile_kernel<Rank, TypeCode, Ti, Tv, Op><<<grid, block_size>>>(basis_slice, groups, pos, op);
}

template <int TypeCode, typename Ti, typename Tv, typename Op>
static inline void launch_cuda_single_group_by_rank(
    const BasisSliceDev<Ti> &basis_slice,
    const GroupsSliceDev<Ti, Tv> &groups,
    int64 pos,
    Op op,
    int host_rank,
    int max_tasks)
{
    if constexpr (TypeCode == 0 && Op::SkipRealDiagonal && std::is_arithmetic_v<Tv>)
        return;

    if (host_rank == 1)
        launch_cuda_single_group_rank<1, TypeCode, Ti, Tv>(basis_slice, groups, pos, op, max_tasks);
    else if (host_rank == 2)
        launch_cuda_single_group_rank<2, TypeCode, Ti, Tv>(basis_slice, groups, pos, op, max_tasks);
    else
        launch_cuda_single_group_rank<0, TypeCode, Ti, Tv>(basis_slice, groups, pos, op, max_tasks);
}

template <typename Ti, typename Tv, typename Launcher>
static inline void dispatch_cuda_network_group(
    const BasisSliceDev<Ti> &basis_slice,
    const NetworkDev<Ti, Tv> &net,
    int64 idx,
    int max_tasks,
    Launcher launcher)
{
    const uint8 type = net.host_excit_types[idx];
    const int64 pos = net.host_sorted_idxs[idx];

    switch (type)
    {
    case 0:
        launcher.template operator()<0>(basis_slice, net.diag_groups, pos, max_tasks);
        break;
    case 1:
        launcher.template operator()<1>(basis_slice, net.pure_a_groups, pos, max_tasks);
        break;
    case 2:
        launcher.template operator()<2>(basis_slice, net.pure_b_groups, pos, max_tasks);
        break;
    case 3:
        launcher.template operator()<3>(basis_slice, net.mixed_groups, pos, max_tasks);
        break;
    default:
        break;
    }
}

template <typename Ti>
static inline int cuda_single_group_max_tasks(const BasisViewDev<Ti> &basis)
{
    constexpr int block_size = 256;
    int max_tasks = 0;
    for (int bid = 0; bid < basis.num_blocks; ++bid)
    {
        const int num_a_tiles = (basis.host_block_num_a[bid] + block_size - 1) / block_size;
        const int num_b_tiles = (basis.host_block_num_b[bid] + TILE_B - 1) / TILE_B;
        max_tasks = std::max(max_tasks, num_a_tiles * num_b_tiles);
    }
    return max_tasks;
}

template <typename Tv>
struct CudaHVecMultiGroupOp
{
    static constexpr bool SkipRealDiagonal = false;
    static constexpr bool SkipLowerBlocks = false;
    static constexpr bool SkipSameBlockReverse = false;

    const Tv *src_vec;
    Tv *dst_vec;

    __device__ __forceinline__ void init_tile(Tv (&accum)[TILE_B]) const
    {
#pragma unroll
        for (int i = 0; i < TILE_B; ++i)
            accum[i] = {};
    }

    __device__ __forceinline__ void diag(Tv (&accum)[TILE_B], Tv &, Tv vt, int64 di, int, int b_offset) const
    {
        accum[b_offset] += __ldg(src_vec + di) * vt;
    }

    __device__ __forceinline__ void offdiag(Tv (&accum)[TILE_B], Tv &, Tv vt, int64 si, int64, int, int b_offset) const
    {
        accum[b_offset] += __ldg(src_vec + si) * vt;
    }

    __device__ __forceinline__ void finish_group(Tv &, int) const {}

    template <typename Ti>
    __device__ __forceinline__ void finish_tile(
        const BasisSliceDev<Ti> &basis, int64 dst_row, int64, int b_tile_start, int current_tile_b, bool valid_a,
        Tv (&accum)[TILE_B]) const
    {
        if (!valid_a)
            return;
        Tv *dst_row_ptr = dst_vec + dst_row;
        for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
            dst_row_ptr[b_tile_start + b_offset] += accum[b_offset];
    }
};

template <typename Tv>
struct CudaBatchGradMultiGroupOp
{
    static constexpr bool SkipRealDiagonal = true;
    static constexpr bool SkipLowerBlocks = true;
    static constexpr bool SkipSameBlockReverse = true;

    const double *thetas;
    const Tv *lp;
    const Tv *rp;
    Tv *grads;

    __device__ __forceinline__ void init_tile(Tv (&accum)[TILE_B]) const
    {
        (void)accum;
    }

    __device__ __forceinline__ void diag(Tv (&)[TILE_B], Tv &local_res, Tv vt, int64 di, int original_idx, int) const
    {
        const double theta = thetas[original_idx];
        const Tv du = fast_diag_grad_dev<Tv>(vt, theta);
        local_res += dev_conj(lp[di] * du) * rp[di];
    }

    __device__ __forceinline__ void offdiag(Tv (&)[TILE_B], Tv &local_res, Tv vt, int64 si, int64 di, int original_idx, int) const
    {
        const double theta = thetas[original_idx];
        grad_update_dev<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, -std::sin(theta), std::cos(theta));
    }

    __device__ __forceinline__ void finish_group(Tv &local_res, int original_idx) const
    {
        atomicAdd_Tv(grads + original_idx, local_res);
    }

    template <typename Ti>
    __device__ __forceinline__ void finish_tile(
        const BasisSliceDev<Ti> &, int64, int64, int, int, bool,
        Tv (&)[TILE_B]) const {}
};

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
__global__ void cuda_multi_group_tile_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    Op op)
{
    constexpr bool IsDiagonal = TypeCode == 0;
    constexpr bool UsesAExcitation = TypeCode == 1 || TypeCode == 3;
    constexpr bool UsesBExcitation = TypeCode == 2 || TypeCode == 3;
    constexpr int BATCH_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1
        : Rank == 2 ? BATCH_SIZE_SH2
                    : BATCH_SIZE_SH3;
    constexpr int SHARED_MEM_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1 * TILE_B
        : Rank == 2 ? BATCH_SIZE_SH2 * TILE_B * 2
                    : BATCH_SIZE_SH3 * TILE_B * KERNEL_MAX_RANK;
    constexpr int IDX_MEM_SIZE = UsesBExcitation ? BATCH_SIZE * TILE_B : 1;
    constexpr int GROUP_MEM_SIZE = IsDiagonal ? 1 : BATCH_SIZE;
    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : KERNEL_MAX_RANK);

    __shared__ Tv sh_pb[SHARED_MEM_SIZE];
    __shared__ int sh_sb[IDX_MEM_SIZE];
    __shared__ int sh_src_bid[GROUP_MEM_SIZE];
    __shared__ int sh_valid_group[GROUP_MEM_SIZE];

    const int bid = basis.target_bids ? basis.target_bids[blockIdx.x] : blockIdx.x;
    const int total_groups = groups.num_groups;
    const int num_chunks = (total_groups + BATCH_SIZE - 1) / BATCH_SIZE;
    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int asym = basis.block_asym[bid];
    const int bsym = basis.block_bsym[bid];
    const int nirp = basis.num_irreps;
    const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
    const int num_a_tiles = (n_a + TILE_A - 1) / TILE_A;
    const int total_tiles = num_b_tiles * num_a_tiles;
    const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
    const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];
    const int *a_idx_map = basis.astr2idx;
    const int *b_idx_map = basis.bstr2idx;
    const int64 dst_block_offset = basis.block_offsets[bid];

    for (int task_idx = blockIdx.y; task_idx < total_tiles; task_idx += gridDim.y)
    {
        const int b_tile_idx = task_idx % num_b_tiles;
        const int a_tile_idx = task_idx / num_b_tiles;
        const int b_tile_start = b_tile_idx * TILE_B;
        const int current_tile_b = min(TILE_B, n_b - b_tile_start);
        const Ti *bstrs_tile_start = bstrs + b_tile_start;
        const int a_tile_start = a_tile_idx * TILE_A;
        const int a_tile_end = min(n_a, a_tile_start + TILE_A);
        const int a = a_tile_start + threadIdx.x;
        const bool valid_a = a < a_tile_end;
        const Ti dst_astr = valid_a ? astrs[a] : 0;
        const int64 dst_row = dst_block_offset + (int64)a * n_b;

        Tv accum[TILE_B];
        op.init_tile(accum);

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);

            if constexpr (!IsDiagonal)
            {
                for (int g_offset = threadIdx.x; g_offset < current_chunk_groups; g_offset += blockDim.x)
                {
                    const int g = chunk_start_g + g_offset;
                    int h;
                    if constexpr (TypeCode == 1)
                        h = (asym ^ groups.asyms[g]) * nirp + bsym;
                    else if constexpr (TypeCode == 2)
                        h = asym * nirp + (bsym ^ groups.bsyms[g]);
                    else
                        h = (asym ^ groups.asyms[g]) * nirp + (bsym ^ groups.bsyms[g]);

                    const int src_bid = basis.block_map[h];
                    sh_src_bid[g_offset] = src_bid;
                    sh_valid_group[g_offset] = (src_bid != -1 && (!Op::SkipLowerBlocks || src_bid >= bid)) ? 1 : 0;
                }
                __syncthreads();
            }

            const int total_sh_elements = current_chunk_groups * current_tile_b;
            for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
            {
                const int g_offset = sh_idx / current_tile_b;
                const int b_offset = sh_idx % current_tile_b;
                const int sh_flat_offset = g_offset * TILE_B + b_offset;

                if constexpr (!IsDiagonal)
                {
                    if (sh_valid_group[g_offset] == 0)
                    {
                        if constexpr (UsesBExcitation)
                            sh_sb[sh_flat_offset] = -1;
                        continue;
                    }
                }

                const int g = chunk_start_g + g_offset;
                Ti src_bstr = bstrs_tile_start[b_offset];
                int sb = b_tile_start + b_offset;
                if constexpr (UsesBExcitation)
                {
                    src_bstr ^= groups.bxs[g];
                    sb = b_idx_map[src_bstr];
                    sh_sb[sh_flat_offset] = sb;
                    if (sb == -1)
                        continue;
                }

                compute_phase_dev<Rank, Ti, Tv>(
                    src_bstr, groups.flat_zbs + groups.zb_start[g], groups.num_zbs[g],
                    groups.flat_wb + groups.wb_start[g], sh_pb + sh_flat_offset, BATCH_SIZE * TILE_B, groups.ranks[g]);
            }
            __syncthreads();

            if (valid_a)
            {
                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    if constexpr (!IsDiagonal)
                    {
                        if (sh_valid_group[g_offset] == 0)
                            continue;
                    }

                    const int g = chunk_start_g + g_offset;
                    const int src_bid = IsDiagonal ? bid : sh_src_bid[g_offset];
                    Ti src_astr = dst_astr;
                    int sa = a;
                    if constexpr (UsesAExcitation)
                    {
                        src_astr ^= groups.axs[g];
                        sa = a_idx_map[src_astr];
                    }
                    if (sa == -1 || (Op::SkipSameBlockReverse && src_bid == bid && sa < a))
                        continue;

                    Tv pa[STACK_SIZE] = {};
                    const int rank = groups.ranks[g];
                    compute_phase_dev<Rank, Ti, Tv>(
                        src_astr, groups.flat_zas + groups.za_start[g], groups.num_zas[g],
                        groups.flat_wa + groups.wa_start[g], pa, 1, rank);

                    const int src_n_b = IsDiagonal ? n_b : basis.block_num_b[src_bid];
                    const int64 src_row = IsDiagonal ? dst_row : basis.block_offsets[src_bid] + (int64)sa * src_n_b;
                    const Tv *pb = sh_pb + (g_offset * TILE_B);
                    const int original_idx = groups.original_idx[g];
                    Tv local_res = {};

                    for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                    {
                        int sb = b_tile_start + b_offset;
                        if constexpr (UsesBExcitation)
                            sb = sh_sb[g_offset * TILE_B + b_offset];
                        if (sb == -1 || (Op::SkipSameBlockReverse && src_bid == bid && sa == a && sb < b_tile_start + b_offset))
                            continue;

                        const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                        const int64 si = src_row + sb;
                        const int64 di = dst_row + b_tile_start + b_offset;
                        if constexpr (IsDiagonal)
                            op.diag(accum, local_res, vt, di, original_idx, b_offset);
                        else
                            op.offdiag(accum, local_res, vt, si, di, original_idx, b_offset);
                    }
                    op.finish_group(local_res, original_idx);
                }
            }
            __syncthreads();
        }

        op.finish_tile(basis, dst_row, dst_block_offset, b_tile_start, current_tile_b, valid_a, accum);
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
static inline void launch_cuda_multi_group_rank(
    const BasisSliceDev<Ti> &basis_slice,
    const GroupsSliceDev<Ti, Tv> &groups,
    dim3 grid_size,
    int block_size,
    Op op)
{
    if constexpr (TypeCode == 0 && Op::SkipRealDiagonal && std::is_arithmetic_v<Tv>)
        return;
    cuda_multi_group_tile_kernel<Rank, TypeCode, Ti, Tv, Op><<<grid_size, block_size>>>(basis_slice, groups, op);
}

template <int TypeCode, typename Ti, typename Tv, typename Op>
static inline void dispatch_cuda_multi_group_chunks_by_rank(
    const BasisSliceDev<Ti> &basis_slice,
    int num_active_blocks,
    const GroupsViewDev<Ti, Tv> &groups,
    Op op)
{
    const int64 total_ngs = groups.num_groups;
    if (total_ngs == 0)
        return;

    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    constexpr int block_size = 256;
    const dim3 grid_size(num_active_blocks, num_sms * 4);

    int64 start = 0;
    while (start < total_ngs)
    {
        const int dispatch_rank = normalized_dispatch_rank(groups, start);
        const int64 end = next_rank_chunk_end(groups, start);
        const GroupsSliceDev<Ti, Tv> slice = make_groups_slice(groups, start, end - start);

        switch (dispatch_rank)
        {
        case 1:
            launch_cuda_multi_group_rank<1, TypeCode, Ti, Tv>(basis_slice, slice, grid_size, block_size, op);
            break;
        case 2:
            launch_cuda_multi_group_rank<2, TypeCode, Ti, Tv>(basis_slice, slice, grid_size, block_size, op);
            break;
        default:
            launch_cuda_multi_group_rank<0, TypeCode, Ti, Tv>(basis_slice, slice, grid_size, block_size, op);
            break;
        }

        start = end;
    }
}
