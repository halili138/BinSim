#pragma once
#include "cuda_framework.cuh"
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

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


struct CudaSingleGroupSchedule
{
    int max_tasks = 0;
    int64 total_tasks = 0;
    int64 rectangular_tasks = 0;
    CudaSingleGroupTask *dev_tasks = nullptr;

    CudaSingleGroupSchedule() = default;
    CudaSingleGroupSchedule(const CudaSingleGroupSchedule &) = delete;
    CudaSingleGroupSchedule &operator=(const CudaSingleGroupSchedule &) = delete;

    CudaSingleGroupSchedule(CudaSingleGroupSchedule &&other) noexcept
    {
        *this = std::move(other);
    }

    CudaSingleGroupSchedule &operator=(CudaSingleGroupSchedule &&other) noexcept
    {
        if (this != &other)
        {
            clear();
            max_tasks = other.max_tasks;
            total_tasks = other.total_tasks;
            rectangular_tasks = other.rectangular_tasks;
            dev_tasks = other.dev_tasks;
            other.max_tasks = 0;
            other.total_tasks = 0;
            other.rectangular_tasks = 0;
            other.dev_tasks = nullptr;
        }
        return *this;
    }

    ~CudaSingleGroupSchedule() { clear(); }

    void clear()
    {
        if (dev_tasks)
        {
            cudaFree(dev_tasks);
            dev_tasks = nullptr;
        }
    }

    bool compact_enabled() const { return dev_tasks != nullptr && total_tasks > 0; }

    double early_return_rate() const
    {
        return rectangular_tasks == 0 ? 0.0 : double(rectangular_tasks - total_tasks) / double(rectangular_tasks);
    }
};

struct CudaMultiGroupSchedule
{
    int64 total_tiles = 0;
    int64 rectangular_tasks = 0;
    CudaMultiGroupTask *dev_tasks = nullptr;

    CudaMultiGroupSchedule() = default;
    CudaMultiGroupSchedule(const CudaMultiGroupSchedule &) = delete;
    CudaMultiGroupSchedule &operator=(const CudaMultiGroupSchedule &) = delete;

    CudaMultiGroupSchedule(CudaMultiGroupSchedule &&other) noexcept
    {
        *this = std::move(other);
    }

    CudaMultiGroupSchedule &operator=(CudaMultiGroupSchedule &&other) noexcept
    {
        if (this != &other)
        {
            clear();
            total_tiles = other.total_tiles;
            rectangular_tasks = other.rectangular_tasks;
            dev_tasks = other.dev_tasks;
            other.total_tiles = 0;
            other.rectangular_tasks = 0;
            other.dev_tasks = nullptr;
        }
        return *this;
    }

    ~CudaMultiGroupSchedule() { clear(); }

    void clear()
    {
        if (dev_tasks)
        {
            cudaFree(dev_tasks);
            dev_tasks = nullptr;
        }
    }

    bool compact_enabled() const { return dev_tasks != nullptr && total_tiles > 0; }
};

template <typename Ti>
static inline CudaMultiGroupSchedule cuda_multi_group_schedule(const BasisViewDev<Ti> &basis, int num_sms)
{
    CudaMultiGroupSchedule schedule;
    std::vector<CudaMultiGroupTask> host_tasks;

    for (int active_block_idx = 0; active_block_idx < basis.num_blocks; ++active_block_idx)
    {
        const int num_a_tiles = (basis.host_block_num_a[active_block_idx] + TILE_A - 1) / TILE_A;
        const int num_b_tiles = (basis.host_block_num_b[active_block_idx] + TILE_B - 1) / TILE_B;
        const int block_tiles = num_a_tiles * num_b_tiles;
        for (int tile_idx = 0; tile_idx < block_tiles; ++tile_idx)
            host_tasks.push_back(CudaMultiGroupTask{active_block_idx, tile_idx});
    }

    schedule.total_tiles = static_cast<int64>(host_tasks.size());
    schedule.rectangular_tasks = static_cast<int64>(basis.num_blocks) * num_sms * 4;

    const bool grid_fits = schedule.total_tiles <= std::numeric_limits<unsigned int>::max();
    const bool compact_is_smaller = schedule.total_tiles < schedule.rectangular_tasks;
    const bool high_idle = schedule.rectangular_tasks > 0 && schedule.total_tiles * 2 < schedule.rectangular_tasks;
    if (!host_tasks.empty() && grid_fits && compact_is_smaller && high_idle)
    {
        CUDA_CHECK(cudaMalloc(&schedule.dev_tasks, host_tasks.size() * sizeof(CudaMultiGroupTask)));
        CUDA_CHECK(cudaMemcpy(schedule.dev_tasks, host_tasks.data(), host_tasks.size() * sizeof(CudaMultiGroupTask), cudaMemcpyHostToDevice));
    }

    return schedule;
}

template <typename Ti>
static inline CudaSingleGroupSchedule cuda_single_group_schedule(const BasisViewDev<Ti> &basis)
{
    constexpr int block_size = 256;
    CudaSingleGroupSchedule schedule;
    std::vector<CudaSingleGroupTask> host_tasks;

    for (int bid = 0; bid < basis.num_blocks; ++bid)
    {
        const int num_a_tiles = (basis.host_block_num_a[bid] + block_size - 1) / block_size;
        const int num_b_tiles = (basis.host_block_num_b[bid] + TILE_B - 1) / TILE_B;
        const int block_tasks = num_a_tiles * num_b_tiles;
        schedule.max_tasks = std::max(schedule.max_tasks, block_tasks);
        for (int task_idx = 0; task_idx < block_tasks; ++task_idx)
            host_tasks.push_back(CudaSingleGroupTask{bid, task_idx});
    }

    schedule.total_tasks = static_cast<int64>(host_tasks.size());
    schedule.rectangular_tasks = static_cast<int64>(basis.num_blocks) * schedule.max_tasks;

    if (!host_tasks.empty() && schedule.total_tasks < schedule.rectangular_tasks)
    {
        CUDA_CHECK(cudaMalloc(&schedule.dev_tasks, host_tasks.size() * sizeof(CudaSingleGroupTask)));
        CUDA_CHECK(cudaMemcpy(schedule.dev_tasks, host_tasks.data(), host_tasks.size() * sizeof(CudaSingleGroupTask), cudaMemcpyHostToDevice));
    }

    return schedule;
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

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
static inline void launch_cuda_single_group_rank(
    const BasisSliceDev<Ti> &basis_slice, const GroupsSliceDev<Ti, Tv> &groups, int64 pos, Op op, int max_tasks,
    const CudaSingleGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0)
{
    constexpr int block_size = 256;
    if (max_tasks <= 0)
        return;

    if (compact_tasks && compact_num_tasks > 0 && compact_num_tasks <= std::numeric_limits<unsigned int>::max())
    {
        dim3 grid(static_cast<unsigned int>(compact_num_tasks));
        cuda_single_group_sharedtile_compact_kernel<Rank, TypeCode, Ti, Tv, Op><<<grid, block_size>>>(basis_slice, groups, pos, op, compact_tasks);
        return;
    }

    constexpr int MAX_GRID_Y = 65535;
    for (int task_offset = 0; task_offset < max_tasks; task_offset += MAX_GRID_Y)
    {
        const int tasks_this_launch = std::min(MAX_GRID_Y, max_tasks - task_offset);
        dim3 grid(basis_slice.num_blocks, tasks_this_launch);
        cuda_single_group_sharedtile_kernel<Rank, TypeCode, Ti, Tv, Op><<<grid, block_size>>>(basis_slice, groups, pos, op, task_offset);
    }
}

template <int TypeCode, typename Ti, typename Tv, typename Op>
static inline void launch_cuda_single_group_by_rank(const BasisSliceDev<Ti> &basis_slice, const GroupsSliceDev<Ti, Tv> &groups, int64 pos, Op op, int host_rank, int max_tasks, const CudaSingleGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0)
{
    if constexpr (TypeCode == 0 && Op::SkipRealDiagonal && std::is_arithmetic_v<Tv>)
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
    case 0:
        launcher.template operator()<0>(basis_slice, net.diag_groups, pos, max_tasks, compact_tasks, compact_num_tasks);
        break;
    case 1:
        launcher.template operator()<1>(basis_slice, net.pure_a_groups, pos, max_tasks, compact_tasks, compact_num_tasks);
        break;
    case 2:
        launcher.template operator()<2>(basis_slice, net.pure_b_groups, pos, max_tasks, compact_tasks, compact_num_tasks);
        break;
    case 3:
        launcher.template operator()<3>(basis_slice, net.mixed_groups, pos, max_tasks, compact_tasks, compact_num_tasks);
        break;
    default:
        break;
    }
}

template <typename Tv>
struct CudaExpmSingleGroupOp
{
    static constexpr bool SkipRealDiagonal = true;
    static constexpr bool UsesBlockResult = false;

    double theta;
    double cd;
    double co;
    Tv *vec;

    __device__ __forceinline__ void diag(Tv vt, int64 di) const
    {
        vec[di] *= fast_diag_exp_dev<Tv>(vt, theta);
    }

    __device__ __forceinline__ void offdiag(Tv vt, int64 si, int64 di) const
    {
        expm_update_dev<Tv>(vec + si, vec + di, vt, cd, co);
    }
};

template <typename Ti, typename Tv>
struct CudaExpmLauncher
{
    double theta;
    Tv *dev_vec;

    template <int TypeCode>
    void operator()(
        const BasisSliceDev<Ti> &basis_slice, const GroupsViewDev<Ti, Tv> &groups, int64 pos, int max_tasks,
        const CudaSingleGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0) const
    {
        CudaExpmSingleGroupOp<Tv> op{theta, std::cos(theta) - 1.0, std::sin(theta), dev_vec};
        launch_cuda_single_group_by_rank<TypeCode, Ti, Tv>(basis_slice, make_groups_slice(groups), pos, op, groups.host_ranks[pos], max_tasks, compact_tasks, compact_num_tasks);
    }
};

template <typename Ti, typename Tv>
void expm_svd_network_otf_gpu(
    const BasisSliceDev<Ti> &basis_slice, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, Tv *dev_vec, int max_tasks,
    const CudaSingleGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0)
{
    const CudaExpmLauncher<Ti, Tv> launcher{theta, dev_vec};
    dispatch_cuda_network_group<Ti, Tv>(basis_slice, net, idx, max_tasks, launcher, compact_tasks, compact_num_tasks);
}

template <typename Ti, typename Tv>
void expm_svd_network_otf_gpu(const BasisViewDev<Ti> &basis, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, Tv *dev_vec)
{
    const BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    CudaSingleGroupSchedule schedule = cuda_single_group_schedule(basis);
    expm_svd_network_otf_gpu<Ti, Tv>(basis_slice, net, idx, theta, dev_vec, schedule.max_tasks, schedule.dev_tasks, schedule.total_tasks);
}

template <typename Tv>
struct CudaGradSingleGroupOp
{
    static constexpr bool SkipRealDiagonal = true;
    static constexpr bool UsesBlockResult = true;

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

template <typename Ti, typename Tv>
struct CudaGradLauncher
{
    double theta;
    const Tv *lp;
    const Tv *rp;
    Tv *d_res;

    template <int TypeCode>
    void operator()(
        const BasisSliceDev<Ti> &basis_slice, const GroupsViewDev<Ti, Tv> &groups, int64 pos, int max_tasks,
        const CudaSingleGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0) const
    {
        CudaGradSingleGroupOp<Tv> op{theta, -std::sin(theta), std::cos(theta), lp, rp, d_res};
        launch_cuda_single_group_by_rank<TypeCode, Ti, Tv>(basis_slice, make_groups_slice(groups), pos, op, groups.host_ranks[pos], max_tasks, compact_tasks, compact_num_tasks);
    }
};

template <typename Ti, typename Tv>
Tv grad_svd_network_otf_gpu(const BasisSliceDev<Ti> &basis_slice, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, const Tv *lp, const Tv *rp, int max_tasks, const CudaSingleGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0)
{
    Tv h_res = {};
    Tv *d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(Tv)));
    CUDA_CHECK(cudaMemset(d_res, 0, sizeof(Tv)));

    const CudaGradLauncher<Ti, Tv> launcher{theta, lp, rp, d_res};
    dispatch_cuda_network_group<Ti, Tv>(basis_slice, net, idx, max_tasks, launcher, compact_tasks, compact_num_tasks);

    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(Tv), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    return h_res;
}

template <typename Ti, typename Tv>
Tv grad_svd_network_otf_gpu(const BasisViewDev<Ti> &basis, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, const Tv *lp, const Tv *rp)
{
    const BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    CudaSingleGroupSchedule schedule = cuda_single_group_schedule(basis);
    return grad_svd_network_otf_gpu<Ti, Tv>(basis_slice, net, idx, theta, lp, rp, schedule.max_tasks, schedule.dev_tasks, schedule.total_tasks);
}

template <typename Tv>
struct CudaBackgradSingleGroupOp
{
    static constexpr bool SkipRealDiagonal = true;
    static constexpr bool UsesBlockResult = true;

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

template <typename Ti, typename Tv>
struct CudaBackgradLauncher
{
    double theta;
    Tv *lp;
    Tv *rp;
    Tv *d_res;

    template <int TypeCode>
    void operator()(
        const BasisSliceDev<Ti> &basis_slice, const GroupsViewDev<Ti, Tv> &groups, int64 pos, int max_tasks,
        const CudaSingleGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0) const
    {
        CudaBackgradSingleGroupOp<Tv> op{theta, std::cos(theta) - 1.0, -std::sin(theta), -std::sin(theta), std::cos(theta), lp, rp, d_res};
        launch_cuda_single_group_by_rank<TypeCode, Ti, Tv>(basis_slice, make_groups_slice(groups), pos, op, groups.host_ranks[pos], max_tasks, compact_tasks, compact_num_tasks);
    }
};

template <typename Ti, typename Tv>
Tv backgrad_svd_network_otf_gpu(const BasisSliceDev<Ti> &basis_slice, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, Tv *lp, Tv *rp, int max_tasks, const CudaSingleGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0)
{
    Tv h_res = {};
    Tv *d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(Tv)));
    CUDA_CHECK(cudaMemset(d_res, 0, sizeof(Tv)));

    const CudaBackgradLauncher<Ti, Tv> launcher{theta, lp, rp, d_res};
    dispatch_cuda_network_group<Ti, Tv>(basis_slice, net, idx, max_tasks, launcher, compact_tasks, compact_num_tasks);

    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(Tv), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    return h_res;
}

template <typename Ti, typename Tv>
Tv backgrad_svd_network_otf_gpu(const BasisViewDev<Ti> &basis, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, Tv *lp, Tv *rp)
{
    const BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    CudaSingleGroupSchedule schedule = cuda_single_group_schedule(basis);
    return backgrad_svd_network_otf_gpu<Ti, Tv>(basis_slice, net, idx, theta, lp, rp, schedule.max_tasks, schedule.dev_tasks, schedule.total_tasks);
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
static inline void launch_cuda_multi_group_rank(
    const BasisSliceDev<Ti> &basis_slice, const GroupsSliceDev<Ti, Tv> &groups, dim3 grid_size, int block_size, Op op,
    const CudaMultiGroupTask *compact_tasks = nullptr, int64 compact_num_tasks = 0)
{
    if constexpr (TypeCode == 0 && Op::SkipRealDiagonal && std::is_arithmetic_v<Tv>)
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
    constexpr int block_size = 256;
    const dim3 grid_size(num_active_blocks, num_sms * 4);
    const CudaMultiGroupTask *compact_tasks = (schedule && schedule->compact_enabled()) ? schedule->dev_tasks : nullptr;
    const int64 compact_num_tasks = compact_tasks ? schedule->total_tiles : 0;

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

template <typename Tv>
struct CudaHVecMultiGroupOp
{
    static constexpr bool SkipRealDiagonal = false;
    static constexpr bool SkipLowerBlocks = false;
    static constexpr bool SkipSameBlockReverse = false;
    static constexpr bool UsesOriginalIdx = false;
    static constexpr bool UsesGroupResult = false;
    static constexpr bool UsesTileAccumulator = true;

    const Tv *src_vec;
    Tv *dst_vec;

    __device__ __forceinline__ void init_tile(Tv (&accum)[TILE_B]) const
    {
#pragma unroll
        for (int i = 0; i < TILE_B; ++i)
            accum[i] = {};
    }

    __device__ __forceinline__ void diag(Tv (&accum)[TILE_B], Tv vt, int64 di, int b_offset) const
    {
        accum[b_offset] += __ldg(src_vec + di) * vt;
    }

    __device__ __forceinline__ void offdiag(Tv (&accum)[TILE_B], Tv vt, int64 si, int64, int b_offset) const
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

template <typename Ti, typename Tv>
void cuda_hvec(const BasisViewDev<Ti> &basis, const NetworkDev<Ti, Tv> &net, const Tv *src_vec, Tv *dst_vec)
{
    CUDA_CHECK(cudaMemset(dst_vec, 0, basis.dim * sizeof(Tv)));
    const CudaHVecMultiGroupOp<Tv> op{src_vec, dst_vec};

    dispatch_cuda_multi_group_chunks_by_rank<0>(basis, net.diag_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<1>(basis, net.pure_a_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<2>(basis, net.pure_b_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<3>(basis, net.mixed_groups, op);
}

template <typename Tv>
struct CudaBatchGradMultiGroupOp
{
    static constexpr bool SkipRealDiagonal = true;
    static constexpr bool SkipLowerBlocks = true;
    static constexpr bool SkipSameBlockReverse = true;
    static constexpr bool UsesOriginalIdx = true;
    static constexpr bool UsesGroupResult = true;
    static constexpr bool UsesTileAccumulator = false;

    const double *thetas;
    const Tv *lp;
    const Tv *rp;
    Tv *grads;

    __device__ __forceinline__ void diag(Tv &local_res, Tv vt, int64 di, int original_idx) const
    {
        const double theta = thetas[original_idx];
        const Tv du = fast_diag_grad_dev<Tv>(vt, theta);
        local_res += dev_conj(lp[di] * du) * rp[di];
    }

    __device__ __forceinline__ void offdiag(Tv &local_res, Tv vt, int64 si, int64 di, int original_idx) const
    {
        const double theta = thetas[original_idx];
        grad_update_dev<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, -std::sin(theta), std::cos(theta));
    }

    __device__ __forceinline__ void finish_group(Tv &local_res, int original_idx) const
    {
        atomicAdd_Tv(grads + original_idx, local_res);
    }
};

template <typename Ti, typename Tv>
void cuda_batchgrad(const BasisViewDev<Ti> &basis, const NetworkDev<Ti, Tv> &net, const double *thetas, const Tv *lp, const Tv *rp, Tv *grads)
{
    CUDA_CHECK(cudaMemset(grads, 0, net.host_sorted_idxs.size() * sizeof(Tv)));
    const CudaBatchGradMultiGroupOp<Tv> op{thetas, lp, rp, grads};

    dispatch_cuda_multi_group_chunks_by_rank<0>(basis, net.diag_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<1>(basis, net.pure_a_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<2>(basis, net.pure_b_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<3>(basis, net.mixed_groups, op);
}
