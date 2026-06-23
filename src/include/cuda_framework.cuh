#pragma once
#include "cuda_basis.cuh"
#include "cuda_otf.cuh"
#include "cuda_utils.cuh"

template <typename T, int Size>
struct StaticSharedStorage
{
    T data[Size];

    __device__ __forceinline__ T &operator[](int idx)
    {
        return data[idx];
    }

    __device__ __forceinline__ const T &operator[](int idx) const
    {
        return data[idx];
    }
};

template <typename T>
struct StaticSharedStorage<T, 0>
{
};

template <typename Tag, typename T, int Size, bool Enabled>
struct OptionalStaticSharedStorage
{
};

template <typename Tag, typename T, int Size>
struct OptionalStaticSharedStorage<Tag, T, Size, true>
{
    StaticSharedStorage<T, Size> data;
};

struct BExcitationStorageTag;
struct SourceBlockStorageTag;
struct SourceBlockNumBStorageTag;
struct SourceBlockOffsetStorageTag;
struct ValidGroupStorageTag;
struct GroupRankStorageTag;

template <typename Tv, int PhaseMemSize, int IdxMemSize, int GroupMemSize, bool UsesBExcitation, bool IsDiagonal>
struct MultiGroupTileSharedStorage
    : OptionalStaticSharedStorage<BExcitationStorageTag, int, IdxMemSize, UsesBExcitation>,
      OptionalStaticSharedStorage<SourceBlockStorageTag, int, GroupMemSize, !IsDiagonal>,
      OptionalStaticSharedStorage<SourceBlockNumBStorageTag, int, GroupMemSize, !IsDiagonal>,
      OptionalStaticSharedStorage<SourceBlockOffsetStorageTag, int64, GroupMemSize, !IsDiagonal>,
      OptionalStaticSharedStorage<ValidGroupStorageTag, int, GroupMemSize, !IsDiagonal>,
      OptionalStaticSharedStorage<GroupRankStorageTag, int, GroupMemSize, true>
{
    Tv sh_pb[PhaseMemSize];

    __device__ __forceinline__ StaticSharedStorage<int, IdxMemSize> &sh_sb()
    {
        return OptionalStaticSharedStorage<BExcitationStorageTag, int, IdxMemSize, UsesBExcitation>::data;
    }

    __device__ __forceinline__ StaticSharedStorage<int, GroupMemSize> &sh_src_bid()
    {
        return OptionalStaticSharedStorage<SourceBlockStorageTag, int, GroupMemSize, !IsDiagonal>::data;
    }

    __device__ __forceinline__ StaticSharedStorage<int, GroupMemSize> &sh_src_n_b()
    {
        return OptionalStaticSharedStorage<SourceBlockNumBStorageTag, int, GroupMemSize, !IsDiagonal>::data;
    }

    __device__ __forceinline__ StaticSharedStorage<int64, GroupMemSize> &sh_src_block_offset()
    {
        return OptionalStaticSharedStorage<SourceBlockOffsetStorageTag, int64, GroupMemSize, !IsDiagonal>::data;
    }

    __device__ __forceinline__ StaticSharedStorage<int, GroupMemSize> &sh_valid_group()
    {
        return OptionalStaticSharedStorage<ValidGroupStorageTag, int, GroupMemSize, !IsDiagonal>::data;
    }

    __device__ __forceinline__ StaticSharedStorage<int, GroupMemSize> &sh_rank()
    {
        return OptionalStaticSharedStorage<GroupRankStorageTag, int, GroupMemSize, true>::data;
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
    constexpr int IDX_MEM_SIZE = UsesBExcitation ? BATCH_SIZE * TILE_B : 0;
    constexpr int GROUP_MEM_SIZE = BATCH_SIZE;
    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : KERNEL_MAX_RANK);

    __shared__ MultiGroupTileSharedStorage<Tv, SHARED_MEM_SIZE, IDX_MEM_SIZE, GROUP_MEM_SIZE, UsesBExcitation, IsDiagonal> sh;

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
        const bool full_b_tile = current_tile_b == TILE_B;
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

            for (int g_offset = threadIdx.x; g_offset < current_chunk_groups; g_offset += blockDim.x)
            {
                const int g = chunk_start_g + g_offset;
                sh.sh_rank()[g_offset] = groups.ranks[g];

                if constexpr (!IsDiagonal)
                {
                    int h;
                    if constexpr (TypeCode == 1)
                        h = (asym ^ groups.asyms[g]) * nirp + bsym;
                    else if constexpr (TypeCode == 2)
                        h = asym * nirp + (bsym ^ groups.bsyms[g]);
                    else
                        h = (asym ^ groups.asyms[g]) * nirp + (bsym ^ groups.bsyms[g]);

                    const int src_bid = basis.block_map[h];
                    const int valid_group = (src_bid != -1 && (!Op::SkipLowerBlocks || src_bid >= bid)) ? 1 : 0;
                    sh.sh_src_bid()[g_offset] = src_bid;
                    sh.sh_valid_group()[g_offset] = valid_group;
                    if (valid_group)
                    {
                        sh.sh_src_n_b()[g_offset] = basis.block_num_b[src_bid];
                        sh.sh_src_block_offset()[g_offset] = basis.block_offsets[src_bid];
                    }
                }
            }
            __syncthreads();

            if (full_b_tile)
            {
                const int total_sh_elements = current_chunk_groups * TILE_B;
                for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
                {
                    const int g_offset = sh_idx / TILE_B;
                    const int b_offset = sh_idx % TILE_B;
                    const int sh_flat_offset = g_offset * TILE_B + b_offset;

                    if constexpr (!IsDiagonal)
                    {
                        if (sh.sh_valid_group()[g_offset] == 0)
                        {
                            if constexpr (UsesBExcitation)
                                sh.sh_sb()[sh_flat_offset] = -1;
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
                        sh.sh_sb()[sh_flat_offset] = sb;
                        if (sb == -1)
                            continue;
                    }

                    compute_phase_dev<Rank, Ti, Tv>(
                        src_bstr, groups.flat_zbs + groups.zb_start[g], groups.num_zbs[g],
                        groups.flat_wb + groups.wb_start[g], sh.sh_pb + sh_flat_offset, BATCH_SIZE * TILE_B, sh.sh_rank()[g_offset]);
                }
            }
            else
            {
                const int total_sh_elements = current_chunk_groups * current_tile_b;
                for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
                {
                    const int g_offset = sh_idx / current_tile_b;
                    const int b_offset = sh_idx % current_tile_b;
                    const int sh_flat_offset = g_offset * TILE_B + b_offset;

                    if constexpr (!IsDiagonal)
                    {
                        if (sh.sh_valid_group()[g_offset] == 0)
                        {
                            if constexpr (UsesBExcitation)
                                sh.sh_sb()[sh_flat_offset] = -1;
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
                        sh.sh_sb()[sh_flat_offset] = sb;
                        if (sb == -1)
                            continue;
                    }

                    compute_phase_dev<Rank, Ti, Tv>(
                        src_bstr, groups.flat_zbs + groups.zb_start[g], groups.num_zbs[g],
                        groups.flat_wb + groups.wb_start[g], sh.sh_pb + sh_flat_offset, BATCH_SIZE * TILE_B, sh.sh_rank()[g_offset]);
                }
            }
            __syncthreads();

            if (valid_a)
            {
                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    if constexpr (!IsDiagonal)
                    {
                        if (sh.sh_valid_group()[g_offset] == 0)
                            continue;
                    }

                    const int g = chunk_start_g + g_offset;
                    int src_bid = bid;
                    if constexpr (!IsDiagonal)
                        src_bid = sh.sh_src_bid()[g_offset];
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
                    const int rank = sh.sh_rank()[g_offset];
                    compute_phase_dev<Rank, Ti, Tv>(
                        src_astr, groups.flat_zas + groups.za_start[g], groups.num_zas[g],
                        groups.flat_wa + groups.wa_start[g], pa, 1, rank);

                    int src_n_b = n_b;
                    int64 src_row = dst_row;
                    if constexpr (!IsDiagonal)
                    {
                        src_n_b = sh.sh_src_n_b()[g_offset];
                        src_row = sh.sh_src_block_offset()[g_offset] + (int64)sa * src_n_b;
                    }
                    const Tv *pb = sh.sh_pb + (g_offset * TILE_B);
                    const int original_idx = groups.original_idx[g];
                    Tv local_res = {};

                    for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                    {
                        int sb = b_tile_start + b_offset;
                        if constexpr (UsesBExcitation)
                            sb = sh.sh_sb()[g_offset * TILE_B + b_offset];
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
