#pragma once
#include "cuda_framework.cuh"

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
static inline void launch_cuda_single_group_rank(const BasisSliceDev<Ti> &basis_slice, const GroupsSliceDev<Ti, Tv> &groups, int64 pos, Op op, int max_tasks)
{
    constexpr int block_size = 256;
    if (max_tasks <= 0)
        return;

    dim3 grid(basis_slice.num_blocks, max_tasks);
    cuda_single_group_sharedtile_kernel<Rank, TypeCode, Ti, Tv, Op><<<grid, block_size>>>(basis_slice, groups, pos, op);
}

template <int TypeCode, typename Ti, typename Tv, typename Op>
static inline void launch_cuda_single_group_by_rank(const BasisSliceDev<Ti> &basis_slice, const GroupsSliceDev<Ti, Tv> &groups, int64 pos, Op op, int host_rank, int max_tasks)
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
static inline void dispatch_cuda_network_group(const BasisSliceDev<Ti> &basis_slice, const NetworkDev<Ti, Tv> &net, int64 idx, int max_tasks, Launcher launcher)
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
    void operator()(const BasisSliceDev<Ti> &basis_slice, const GroupsViewDev<Ti, Tv> &groups, int64 pos, int max_tasks) const
    {
        CudaExpmSingleGroupOp<Tv> op{theta, std::cos(theta) - 1.0, std::sin(theta), dev_vec, nullptr};
        launch_cuda_single_group_by_rank<TypeCode, Ti, Tv>(basis_slice, make_groups_slice(groups), pos, op, groups.host_ranks[pos], max_tasks);
    }
};

template <typename Ti, typename Tv>
void expm_svd_network_otf_gpu(const BasisSliceDev<Ti> &basis_slice, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, Tv *dev_vec, int max_tasks)
{
    const CudaExpmLauncher<Ti, Tv> launcher{theta, dev_vec};
    dispatch_cuda_network_group<Ti, Tv>(basis_slice, net, idx, max_tasks, launcher);
}

template <typename Ti, typename Tv>
void expm_svd_network_otf_gpu(const BasisViewDev<Ti> &basis, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, Tv *dev_vec)
{
    const BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    expm_svd_network_otf_gpu<Ti, Tv>(basis_slice, net, idx, theta, dev_vec, cuda_single_group_max_tasks(basis));
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
    void operator()(const BasisSliceDev<Ti> &basis_slice, const GroupsViewDev<Ti, Tv> &groups, int64 pos, int max_tasks) const
    {
        CudaGradSingleGroupOp<Tv> op{theta, -std::sin(theta), std::cos(theta), lp, rp, d_res};
        launch_cuda_single_group_by_rank<TypeCode, Ti, Tv>(basis_slice, make_groups_slice(groups), pos, op, groups.host_ranks[pos], max_tasks);
    }
};

template <typename Ti, typename Tv>
Tv grad_svd_network_otf_gpu(const BasisSliceDev<Ti> &basis_slice, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, const Tv *lp, const Tv *rp, int max_tasks)
{
    Tv h_res = {};
    Tv *d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(Tv)));
    CUDA_CHECK(cudaMemset(d_res, 0, sizeof(Tv)));

    const CudaGradLauncher<Ti, Tv> launcher{theta, lp, rp, d_res};
    dispatch_cuda_network_group<Ti, Tv>(basis_slice, net, idx, max_tasks, launcher);

    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(Tv), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    return h_res;
}

template <typename Ti, typename Tv>
Tv grad_svd_network_otf_gpu(const BasisViewDev<Ti> &basis, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, const Tv *lp, const Tv *rp)
{
    const BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    return grad_svd_network_otf_gpu<Ti, Tv>(basis_slice, net, idx, theta, lp, rp, cuda_single_group_max_tasks(basis));
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
    void operator()(const BasisSliceDev<Ti> &basis_slice, const GroupsViewDev<Ti, Tv> &groups, int64 pos, int max_tasks) const
    {
        CudaBackgradSingleGroupOp<Tv> op{theta, std::cos(theta) - 1.0, -std::sin(theta), -std::sin(theta), std::cos(theta), lp, rp, d_res};
        launch_cuda_single_group_by_rank<TypeCode, Ti, Tv>(basis_slice, make_groups_slice(groups), pos, op, groups.host_ranks[pos], max_tasks);
    }
};

template <typename Ti, typename Tv>
Tv backgrad_svd_network_otf_gpu(const BasisSliceDev<Ti> &basis_slice, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, Tv *lp, Tv *rp, int max_tasks)
{
    Tv h_res = {};
    Tv *d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(Tv)));
    CUDA_CHECK(cudaMemset(d_res, 0, sizeof(Tv)));

    const CudaBackgradLauncher<Ti, Tv> launcher{theta, lp, rp, d_res};
    dispatch_cuda_network_group<Ti, Tv>(basis_slice, net, idx, max_tasks, launcher);

    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(Tv), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    return h_res;
}

template <typename Ti, typename Tv>
Tv backgrad_svd_network_otf_gpu(const BasisViewDev<Ti> &basis, const NetworkDev<Ti, Tv> &net, int64 idx, double theta, Tv *lp, Tv *rp)
{
    const BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    return backgrad_svd_network_otf_gpu<Ti, Tv>(basis_slice, net, idx, theta, lp, rp, cuda_single_group_max_tasks(basis));
}

template <int Rank, int TypeCode, typename Ti, typename Tv, typename Op>
static inline void launch_cuda_multi_group_rank(const BasisSliceDev<Ti> &basis_slice, const GroupsSliceDev<Ti, Tv> &groups, dim3 grid_size, int block_size, Op op)
{
    if constexpr (TypeCode == 0 && Op::SkipRealDiagonal && std::is_arithmetic_v<Tv>)
        return;
    cuda_multi_group_tile_kernel<Rank, TypeCode, Ti, Tv, Op><<<grid_size, block_size>>>(basis_slice, groups, op);
}

template <int TypeCode, typename Ti, typename Tv, typename Op>
static inline void dispatch_cuda_multi_group_chunks_by_rank(const BasisSliceDev<Ti> &basis_slice, int num_active_blocks, const GroupsViewDev<Ti, Tv> &groups, Op op)
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

template <int TypeCode, typename Ti, typename Tv>
static inline void dispatch_chunks_by_rank_gpu(const BasisSliceDev<Ti> &basis_slice, int num_active_blocks, const GroupsViewDev<Ti, Tv> &groups, const Tv *src_vec, Tv *dst_vec)
{
    const CudaHVecMultiGroupOp<Tv> op{src_vec, dst_vec};
    dispatch_cuda_multi_group_chunks_by_rank<TypeCode>(basis_slice, num_active_blocks, groups, op);
}

template <typename Ti, typename Tv>
void cuda_hvec(const BasisViewDev<Ti> &basis, const NetworkDev<Ti, Tv> &net, const Tv *src_vec, Tv *dst_vec)
{
    CUDA_CHECK(cudaMemset(dst_vec, 0, basis.dim * sizeof(Tv)));
    const BasisSliceDev<Ti> slice = make_basis_slice(basis);
    const CudaHVecMultiGroupOp<Tv> op{src_vec, dst_vec};

    dispatch_cuda_multi_group_chunks_by_rank<0>(slice, basis.num_blocks, net.diag_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<1>(slice, basis.num_blocks, net.pure_a_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<2>(slice, basis.num_blocks, net.pure_b_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<3>(slice, basis.num_blocks, net.mixed_groups, op);
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
    const BasisSliceDev<Ti> slice = make_basis_slice(basis);
    const CudaBatchGradMultiGroupOp<Tv> op{thetas, lp, rp, grads};

    dispatch_cuda_multi_group_chunks_by_rank<0>(slice, basis.num_blocks, net.diag_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<1>(slice, basis.num_blocks, net.pure_a_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<2>(slice, basis.num_blocks, net.pure_b_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<3>(slice, basis.num_blocks, net.mixed_groups, op);
}
