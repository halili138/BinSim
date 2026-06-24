#pragma once
#include "cuda_basis.cuh"
#include "cuda_otf.cuh"
#include "cuda_utils.cuh"

// Semantic names for CUDA excitation type codes. Keep TypeCode template
// parameters as int to avoid changing existing template instantiations.
inline constexpr int kDiagType = 0;
inline constexpr int kPureAType = 1;
inline constexpr int kPureBType = 2;
inline constexpr int kMixedType = 3;

template <int TypeCode>
__host__ __device__ constexpr bool uses_a_excitation()
{
    return TypeCode == kPureAType || TypeCode == kMixedType;
}

template <int TypeCode>
__host__ __device__ constexpr bool uses_b_excitation()
{
    return TypeCode == kPureBType || TypeCode == kMixedType;
}

template <int TypeCode>
__host__ __device__ constexpr bool is_diagonal_type()
{
    return TypeCode == kDiagType;
}


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

// Empty specialization used when a shared-storage field has size 0;
// this avoids declaring a non-standard zero-length array.
template <typename T>
struct StaticSharedStorage<T, 0>
{
};

// Optional static shared storage uses a distinct Tag plus the Enabled bool
// template parameter to control whether the named field exists in a storage type.
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
struct ValidGroupStorageTag;
struct GroupRankStorageTag;

struct CudaSingleGroupTask
{
    int bid;
    int task_idx;
};

struct CudaMultiGroupTask
{
    int active_block_idx;
    int tile_idx;
};

template <typename Tv, int PhaseMemSize, int IdxMemSize, bool UsesBExcitation>
struct SingleGroupSharedTileStorage
    : OptionalStaticSharedStorage<BExcitationStorageTag, int, IdxMemSize, UsesBExcitation>
{
    // sh_pb is laid out with a TILE_B stride for single-group phase tiles.
    Tv sh_pb[PhaseMemSize];

    // Only call this accessor on if constexpr (UsesBExcitation) paths,
    // where the optional B-excitation storage field is present.
    __device__ __forceinline__ StaticSharedStorage<int, IdxMemSize> &sh_sb()
    {
        return OptionalStaticSharedStorage<BExcitationStorageTag, int, IdxMemSize, UsesBExcitation>::data;
    }
};

template <typename Tv, int PhaseMemSize, int IdxMemSize, int GroupMemSize, bool UsesBExcitation, bool IsDiagonal>
struct MultiGroupTileSharedStorage
    : OptionalStaticSharedStorage<BExcitationStorageTag, int, IdxMemSize, UsesBExcitation>,
      OptionalStaticSharedStorage<SourceBlockStorageTag, int, GroupMemSize, !IsDiagonal>,
      OptionalStaticSharedStorage<ValidGroupStorageTag, int, GroupMemSize, !IsDiagonal>,
      OptionalStaticSharedStorage<GroupRankStorageTag, int, GroupMemSize, true>
{
    // sh_pb is laid out with a BATCH_SIZE * TILE_B stride for multi-group phase tiles.
    Tv sh_pb[PhaseMemSize];

    // Only call these accessors on the matching if constexpr paths where
    // their OptionalStaticSharedStorage fields are present.
    __device__ __forceinline__ StaticSharedStorage<int, IdxMemSize> &sh_sb()
    {
        return OptionalStaticSharedStorage<BExcitationStorageTag, int, IdxMemSize, UsesBExcitation>::data;
    }

    // Requires if constexpr (!IsDiagonal).
    __device__ __forceinline__ StaticSharedStorage<int, GroupMemSize> &sh_src_bid()
    {
        return OptionalStaticSharedStorage<SourceBlockStorageTag, int, GroupMemSize, !IsDiagonal>::data;
    }

    // Requires if constexpr (!IsDiagonal).
    __device__ __forceinline__ StaticSharedStorage<int, GroupMemSize> &sh_valid_group()
    {
        return OptionalStaticSharedStorage<ValidGroupStorageTag, int, GroupMemSize, !IsDiagonal>::data;
    }

    // Always present.
    __device__ __forceinline__ StaticSharedStorage<int, GroupMemSize> &sh_rank()
    {
        return OptionalStaticSharedStorage<GroupRankStorageTag, int, GroupMemSize, true>::data;
    }
};

__device__ __forceinline__ void cuda_single_group_decode_rect_task(int &bid, int &task_idx)
{
    bid = blockIdx.x;
    task_idx = blockIdx.y;
}

__device__ __forceinline__ void cuda_single_group_decode_compact_task(
    const CudaSingleGroupTask *__restrict__ tasks, int &bid, int &task_idx)
{
    const CudaSingleGroupTask task = tasks[blockIdx.x];
    bid = task.bid;
    task_idx = task.task_idx;
}

__device__ __forceinline__ void cuda_multi_group_decode_rect_task(int &active_block_idx, int &task_idx, int &task_stride)
{
    active_block_idx = blockIdx.x;
    task_idx = blockIdx.y;
    task_stride = gridDim.y;
}

__device__ __forceinline__ void cuda_multi_group_decode_compact_task(
    const CudaMultiGroupTask *__restrict__ tasks, int &active_block_idx, int &task_idx, int &task_stride)
{
    const CudaMultiGroupTask task = tasks[blockIdx.x];
    active_block_idx = task.active_block_idx;
    task_idx = task.tile_idx;
    task_stride = 0;
}


template <int TypeCode, typename Ti, typename Tv>
__device__ __forceinline__ int resolve_source_block(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int bid,
    int group_idx)
{
    if constexpr (is_diagonal_type<TypeCode>())
    {
        return bid;
    }
    else
    {
        int h;
        if constexpr (TypeCode == kPureAType)
            h = (basis.block_asym[bid] ^ groups.asyms[group_idx]) * basis.num_irreps + basis.block_bsym[bid];
        else if constexpr (TypeCode == kPureBType)
            h = basis.block_asym[bid] * basis.num_irreps + (basis.block_bsym[bid] ^ groups.bsyms[group_idx]);
        else
            h = (basis.block_asym[bid] ^ groups.asyms[group_idx]) * basis.num_irreps + (basis.block_bsym[bid] ^ groups.bsyms[group_idx]);
        return basis.block_map[h];
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Storage>
__device__ __forceinline__ void load_single_group_b_tile_phases(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    Storage &sh,
    int pos,
    int bid,
    int src_bid,
    const Ti *__restrict__ bstrs,
    int b_start,
    int cur_b)
{
    constexpr bool UsesBExcitation = uses_b_excitation<TypeCode>();
    const int rank = groups.ranks[pos];
    const auto *group_zbs = groups.flat_zbs + groups.zb_start[pos];
    const auto *group_wb = groups.flat_wb + groups.wb_start[pos];
    Ti bx = {};
    if constexpr (UsesBExcitation)
        bx = groups.bxs[pos];

    for (int b_offset = threadIdx.x; b_offset < cur_b; b_offset += blockDim.x)
    {
        Ti src_bstr = bstrs[b_start + b_offset];
        int sb = b_start + b_offset;
        if constexpr (UsesBExcitation)
        {
            src_bstr ^= bx;
            sb = basis.bstr2idx[src_bstr];
            sh.sh_sb()[b_offset] = sb;
        }
        if constexpr (is_diagonal_type<TypeCode>())
            compute_phase_dev<Rank, Ti, Tv>(
                src_bstr, group_zbs, groups.num_zbs[pos],
                group_wb, sh.sh_pb + b_offset, TILE_B, rank);
        else if (src_bid != -1 && src_bid >= bid && sb != -1)
            compute_phase_dev<Rank, Ti, Tv>(
                src_bstr, group_zbs, groups.num_zbs[pos],
                group_wb, sh.sh_pb + b_offset, TILE_B, rank);
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op, typename Storage>
__device__ __forceinline__ void process_single_group_a_row(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    Storage &sh,
    int pos,
    Op op,
    Tv &local_res,
    int bid,
    int src_bid,
    const Ti *__restrict__ astrs,
    int n_b,
    int a,
    int b_start,
    int cur_b)
{
    constexpr bool UsesAExcitation = uses_a_excitation<TypeCode>();
    constexpr bool UsesBExcitation = uses_b_excitation<TypeCode>();
    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : KERNEL_MAX_RANK);

    const Ti dst_astr = astrs[a];
    Tv pa[STACK_SIZE] = {};
    const int rank = groups.ranks[pos];

    if constexpr (is_diagonal_type<TypeCode>())
    {
        compute_phase_dev<Rank, Ti, Tv>(
            dst_astr, groups.flat_zas + groups.za_start[pos], groups.num_zas[pos],
            groups.flat_wa + groups.wa_start[pos], pa, 1, rank);

        for (int b_offset = 0; b_offset < cur_b; ++b_offset)
        {
            const Tv vt = compute_coeff_dev<Rank, Tv>(pa, sh.sh_pb, TILE_B, rank, b_offset);
            const int64 di = basis.block_offsets[bid] + (int64)a * n_b + b_start + b_offset;
            op.diag(local_res, vt, di);
        }
    }
    else if (src_bid != -1 && src_bid >= bid)
    {
        Ti src_astr = dst_astr;
        if constexpr (UsesAExcitation)
            src_astr ^= groups.axs[pos];
        const int sa = (TypeCode == kPureBType) ? a : basis.astr2idx[src_astr];
        if (sa != -1 && !(src_bid == bid && sa < a))
        {
            compute_phase_dev<Rank, Ti, Tv>(
                src_astr, groups.flat_zas + groups.za_start[pos], groups.num_zas[pos],
                groups.flat_wa + groups.wa_start[pos], pa, 1, rank);

            const int src_n_b = basis.block_num_b[src_bid];
            const int64 src_row = basis.block_offsets[src_bid] + (int64)sa * src_n_b;
            const int64 dst_row = basis.block_offsets[bid] + (int64)a * n_b;
            for (int b_offset = 0; b_offset < cur_b; ++b_offset)
            {
                int sb;
                if constexpr (UsesBExcitation)
                    sb = sh.sh_sb()[b_offset];
                else
                    sb = b_start + b_offset;
                if (sb == -1 || (src_bid == bid && sa == a && sb < b_start + b_offset))
                    continue;
                const Tv vt = compute_coeff_dev<Rank, Tv>(pa, sh.sh_pb, TILE_B, rank, b_offset);
                op.offdiag(local_res, vt, src_row + sb, dst_row + b_start + b_offset);
            }
        }
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Storage>
__device__ __forceinline__ void load_multi_group_b_tile_phases(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    Storage &sh,
    int chunk_start_g,
    int current_chunk_groups,
    const Ti *__restrict__ bstrs_tile_start,
    int b_tile_start,
    int current_tile_b,
    bool full_b_tile)
{
    constexpr bool IsDiagonal = is_diagonal_type<TypeCode>();
    constexpr bool UsesBExcitation = uses_b_excitation<TypeCode>();
    constexpr int BATCH_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1
        : Rank == 2 ? BATCH_SIZE_SH2
                    : BATCH_SIZE_SH3;
    const int b_stride = full_b_tile ? TILE_B : current_tile_b;
    const int total_sh_elements = current_chunk_groups * b_stride;
    for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
    {
        const int g_offset = sh_idx / b_stride;
        const int b_offset = sh_idx % b_stride;
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
            sb = basis.bstr2idx[src_bstr];
            sh.sh_sb()[sh_flat_offset] = sb;
            if (sb == -1)
                continue;
        }

        compute_phase_dev<Rank, Ti, Tv>(
            src_bstr, groups.flat_zbs + groups.zb_start[g], groups.num_zbs[g],
            groups.flat_wb + groups.wb_start[g], sh.sh_pb + sh_flat_offset, BATCH_SIZE * TILE_B, sh.sh_rank()[g_offset]);
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op, typename Storage>
__device__ __forceinline__ void process_multi_group_a_row(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    Storage &sh,
    Op op,
    Tv (&accum)[TILE_B],
    int bid,
    int chunk_start_g,
    int current_chunk_groups,
    int n_b,
    int a,
    Ti dst_astr,
    int64 dst_row,
    int b_tile_start,
    int current_tile_b)
{
    constexpr bool IsDiagonal = is_diagonal_type<TypeCode>();
    constexpr bool UsesAExcitation = uses_a_excitation<TypeCode>();
    constexpr bool UsesBExcitation = uses_b_excitation<TypeCode>();
    constexpr int BATCH_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1
        : Rank == 2 ? BATCH_SIZE_SH2
                    : BATCH_SIZE_SH3;
    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : KERNEL_MAX_RANK);

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
            sa = basis.astr2idx[src_astr];
        }
        if (sa == -1 || (Op::SkipSameBlockReverse && src_bid == bid && sa < a))
            continue;

        Tv pa[STACK_SIZE] = {};
        const int rank = sh.sh_rank()[g_offset];
        compute_phase_dev<Rank, Ti, Tv>(
            src_astr, groups.flat_zas + groups.za_start[g], groups.num_zas[g],
            groups.flat_wa + groups.wa_start[g], pa, 1, rank);

        const int src_n_b = IsDiagonal ? n_b : basis.block_num_b[src_bid];
        const int64 src_row = IsDiagonal ? dst_row : basis.block_offsets[src_bid] + (int64)sa * src_n_b;
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

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
__device__ __forceinline__ void cuda_single_group_sharedtile_impl(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos,
    Op op,
    int bid,
    int task_idx)
{
    constexpr bool UsesBExcitation = uses_b_excitation<TypeCode>();
    constexpr int SHARED_MEM_SIZE = Rank == 1 ? TILE_B : (Rank == 2 ? TILE_B * 2 : TILE_B * KERNEL_MAX_RANK);
    constexpr int IDX_MEM_SIZE = UsesBExcitation ? TILE_B : 0;

    __shared__ SingleGroupSharedTileStorage<Tv, SHARED_MEM_SIZE, IDX_MEM_SIZE, UsesBExcitation> sh;

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

    const int src_bid = resolve_source_block<TypeCode>(basis, groups, bid, pos);

    load_single_group_b_tile_phases<Rank, TypeCode, Ti, Tv>(
        basis, groups, sh, pos, bid, src_bid, bstrs, b_start, cur_b);
    __syncthreads();

    if (valid_a)
    {
        process_single_group_a_row<Rank, TypeCode, Ti, Tv>(
            basis, groups, sh, pos, op, local_res, bid, src_bid, astrs, n_b, a, b_start, cur_b);
    }

    op.finish_block(local_res);
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
__global__ void cuda_single_group_sharedtile_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos,
    Op op)
{
    int bid;
    int task_idx;
    cuda_single_group_decode_rect_task(bid, task_idx);
    cuda_single_group_sharedtile_impl<Rank, TypeCode, Ti, Tv, Op>(basis, groups, pos, op, bid, task_idx);
}

// Compact scheduler variant: each CUDA block corresponds to one valid (basis block, tile task)
// pair, so imbalanced layouts avoid launching rectangular-grid blocks that immediately return.
template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
__global__ void cuda_single_group_sharedtile_compact_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    int pos,
    Op op,
    const CudaSingleGroupTask *__restrict__ tasks)
{
    int bid;
    int task_idx;
    cuda_single_group_decode_compact_task(tasks, bid, task_idx);
    cuda_single_group_sharedtile_impl<Rank, TypeCode, Ti, Tv, Op>(basis, groups, pos, op, bid, task_idx);
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
__device__ __forceinline__ void cuda_multi_group_tile_impl(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    Op op,
    int active_block_idx,
    int first_task_idx,
    int task_stride)
{
    constexpr bool IsDiagonal = is_diagonal_type<TypeCode>();
    constexpr bool UsesBExcitation = uses_b_excitation<TypeCode>();
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

    __shared__ MultiGroupTileSharedStorage<Tv, SHARED_MEM_SIZE, IDX_MEM_SIZE, GROUP_MEM_SIZE, UsesBExcitation, IsDiagonal> sh;

    const int bid = basis.target_bids ? basis.target_bids[active_block_idx] : active_block_idx;
    const int total_groups = groups.num_groups;
    const int num_chunks = (total_groups + BATCH_SIZE - 1) / BATCH_SIZE;
    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
    const int num_a_tiles = (n_a + TILE_A - 1) / TILE_A;
    const int total_tiles = num_b_tiles * num_a_tiles;
    const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
    const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];
    const int64 dst_block_offset = basis.block_offsets[bid];

    for (int task_idx = first_task_idx; task_idx < total_tiles; task_idx += task_stride)
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
                    const int src_bid = resolve_source_block<TypeCode>(basis, groups, bid, g);
                    sh.sh_src_bid()[g_offset] = src_bid;
                    sh.sh_valid_group()[g_offset] = (src_bid != -1 && (!Op::SkipLowerBlocks || src_bid >= bid)) ? 1 : 0;
                }
            }
            __syncthreads();

            load_multi_group_b_tile_phases<Rank, TypeCode, Ti, Tv>(
                basis, groups, sh, chunk_start_g, current_chunk_groups,
                bstrs_tile_start, b_tile_start, current_tile_b, full_b_tile);
            __syncthreads();

            if (valid_a)
            {
                process_multi_group_a_row<Rank, TypeCode, Ti, Tv>(
                    basis, groups, sh, op, accum, bid, chunk_start_g, current_chunk_groups,
                    n_b, a, dst_astr, dst_row, b_tile_start, current_tile_b);
            }
            __syncthreads();
        }

        op.finish_tile(basis, dst_row, dst_block_offset, b_tile_start, current_tile_b, valid_a, accum);
        if (task_stride <= 0)
            break;
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
__global__ void cuda_multi_group_tile_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    Op op)
{
    int active_block_idx;
    int task_idx;
    int task_stride;
    cuda_multi_group_decode_rect_task(active_block_idx, task_idx, task_stride);
    cuda_multi_group_tile_impl<Rank, TypeCode, Ti, Tv, Op>(basis, groups, op, active_block_idx, task_idx, task_stride);
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
__global__ void cuda_multi_group_tile_compact_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    Op op,
    const CudaMultiGroupTask *__restrict__ tasks)
{
    int active_block_idx;
    int task_idx;
    int task_stride;
    cuda_multi_group_decode_compact_task(tasks, active_block_idx, task_idx, task_stride);
    cuda_multi_group_tile_impl<Rank, TypeCode, Ti, Tv, Op>(basis, groups, op, active_block_idx, task_idx, task_stride);
}
