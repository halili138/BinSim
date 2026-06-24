#pragma once
#include "cuda_schedule.cuh"
#include <limits>
#include <type_traits>

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
static inline void launch_cuda_single_group_rank(
    const BasisSliceDev<Ti> &basis_slice, const GroupsSliceDev<Ti, Tv> &groups, int64 pos, Op op, int max_tasks,
    const CudaSingleGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0)
{
    constexpr int block_size = kCudaBlockSize;
    if (max_tasks <= 0)
        return;

    if (compact_tasks && compact_num_tasks > 0 && compact_num_tasks <= std::numeric_limits<unsigned int>::max())
    {
        dim3 grid(static_cast<unsigned int>(compact_num_tasks));
        cuda_single_group_sharedtile_compact_kernel<Rank, TypeCode, Ti, Tv, Op><<<grid, block_size>>>(basis_slice, groups, pos, op, compact_tasks);
        return;
    }

    dim3 grid(basis_slice.num_blocks, max_tasks);
    cuda_single_group_sharedtile_kernel<Rank, TypeCode, Ti, Tv, Op><<<grid, block_size>>>(basis_slice, groups, pos, op);
}

template <int TypeCode, typename Ti, typename Tv, typename Op>
static inline void launch_cuda_single_group_by_rank(const BasisSliceDev<Ti> &basis_slice, const GroupsSliceDev<Ti, Tv> &groups, int64 pos, Op op, int host_rank, int max_tasks, const CudaSingleGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0)
{
    if constexpr (is_diagonal_type<TypeCode>() && Op::SkipRealDiagonal && std::is_arithmetic_v<Tv>)
        return;

    if (host_rank == 1)
        launch_cuda_single_group_rank<1, TypeCode, Ti, Tv>(basis_slice, groups, pos, op, max_tasks, compact_tasks, compact_num_tasks);
    else if (host_rank == 2)
        launch_cuda_single_group_rank<2, TypeCode, Ti, Tv>(basis_slice, groups, pos, op, max_tasks, compact_tasks, compact_num_tasks);
    else
        launch_cuda_single_group_rank<0, TypeCode, Ti, Tv>(basis_slice, groups, pos, op, max_tasks, compact_tasks, compact_num_tasks);
}

template <typename Ti, typename Tv, typename Launcher>
static inline void dispatch_cuda_network_group(
    const BasisSliceDev<Ti> &basis_slice, const NetworkDev<Ti, Tv> &net, int64 idx, int max_tasks, Launcher launcher,
    const CudaSingleGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0)
{
    const uint8 type = net.host_excit_types[idx];
    const int64 pos = net.host_sorted_idxs[idx];

    switch (type)
    {
    case kDiagType:
        launcher.template operator()<kDiagType>(basis_slice, net.diag_groups, pos, max_tasks, compact_tasks, compact_num_tasks);
        break;
    case kPureAType:
        launcher.template operator()<kPureAType>(basis_slice, net.pure_a_groups, pos, max_tasks, compact_tasks, compact_num_tasks);
        break;
    case kPureBType:
        launcher.template operator()<kPureBType>(basis_slice, net.pure_b_groups, pos, max_tasks, compact_tasks, compact_num_tasks);
        break;
    case kMixedType:
        launcher.template operator()<kMixedType>(basis_slice, net.mixed_groups, pos, max_tasks, compact_tasks, compact_num_tasks);
        break;
    default:
        break;
    }
}



template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
static inline void launch_cuda_multi_group_rank(
    const BasisSliceDev<Ti> &basis_slice, const GroupsSliceDev<Ti, Tv> &groups, dim3 grid_size, int block_size, Op op,
    const CudaMultiGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0)
{
    if constexpr (is_diagonal_type<TypeCode>() && Op::SkipRealDiagonal && std::is_arithmetic_v<Tv>)
        return;
    if (compact_tasks && compact_num_tasks > 0 && compact_num_tasks <= std::numeric_limits<unsigned int>::max())
    {
        dim3 compact_grid(static_cast<unsigned int>(compact_num_tasks));
        cuda_multi_group_tile_compact_kernel<Rank, TypeCode, Ti, Tv, Op><<<compact_grid, block_size>>>(basis_slice, groups, op, compact_tasks);
        return;
    }
    cuda_multi_group_tile_kernel<Rank, TypeCode, Ti, Tv, Op><<<grid_size, block_size>>>(basis_slice, groups, op);
}

template <int TypeCode, typename Ti, typename Tv, typename Op>
static inline void dispatch_cuda_multi_group_chunks_by_rank(
    const BasisSliceDev<Ti> &basis_slice, int num_active_blocks, const GroupsViewDev<Ti, Tv> &groups, Op op,
    const CudaMultiGroupSchedule *schedule = nullptr)
{
    const int64 total_ngs = groups.num_groups;
    if (total_ngs == 0)
        return;

    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    constexpr int block_size = kCudaBlockSize;
    const dim3 grid_size(num_active_blocks, num_sms * kBlocksPerSmForRectangularGrid);
    const CudaMultiGroupTask *compact_tasks = (schedule && schedule->compact_enabled()) ? schedule->dev_tasks.get() : nullptr;
    const int64 compact_num_tasks = compact_tasks ? schedule->compact_task_count() : 0;

    int64 start = 0;
    while (start < total_ngs)
    {
        const int dispatch_rank = normalized_dispatch_rank(groups, start);
        const int64 end = next_rank_chunk_end(groups, start);
        const GroupsSliceDev<Ti, Tv> slice = make_groups_slice(groups, start, end - start);

        switch (dispatch_rank)
        {
        case 1:
            launch_cuda_multi_group_rank<1, TypeCode, Ti, Tv>(basis_slice, slice, grid_size, block_size, op, compact_tasks, compact_num_tasks);
            break;
        case 2:
            launch_cuda_multi_group_rank<2, TypeCode, Ti, Tv>(basis_slice, slice, grid_size, block_size, op, compact_tasks, compact_num_tasks);
            break;
        default:
            launch_cuda_multi_group_rank<0, TypeCode, Ti, Tv>(basis_slice, slice, grid_size, block_size, op, compact_tasks, compact_num_tasks);
            break;
        }

        start = end;
    }
}

template <int TypeCode, typename Ti, typename Tv, typename Op>
static inline void dispatch_cuda_multi_group_chunks_by_rank(
    const BasisViewDev<Ti> &basis, const GroupsViewDev<Ti, Tv> &groups, Op op)
{
    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    CudaMultiGroupSchedule schedule = cuda_multi_group_schedule(basis, num_sms);
    const BasisSliceDev<Ti> slice = make_basis_slice(basis);
    dispatch_cuda_multi_group_chunks_by_rank<TypeCode>(slice, basis.num_blocks, groups, op, &schedule);
}

