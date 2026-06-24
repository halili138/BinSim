#pragma once
#include "cuda_launch.cuh"
#include <cmath>

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
        CudaExpmSingleGroupOp<Tv> op{theta, std::cos(theta) - 1.0, std::sin(theta), dev_vec, nullptr};
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
