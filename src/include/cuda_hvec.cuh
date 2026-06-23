#pragma once
#include "cuda_contract.cuh"

template <int TypeCode, typename Ti, typename Tv>
static inline void dispatch_chunks_by_rank_gpu(
    const BasisSliceDev<Ti> &basis_slice,
    int num_active_blocks,
    const GroupsViewDev<Ti, Tv> &groups,
    const Tv *src_vec,
    Tv *dst_vec)
{
    const CudaHVecMultiGroupOp<Tv> op{src_vec, dst_vec};
    dispatch_cuda_multi_group_chunks_by_rank<TypeCode>(basis_slice, num_active_blocks, groups, op);
}

template <typename Ti, typename Tv>
void cuda_hvec(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    const Tv *src_vec,
    Tv *dst_vec)
{
    CUDA_CHECK(cudaMemset(dst_vec, 0, basis.dim * sizeof(Tv)));
    const BasisSliceDev<Ti> slice = make_basis_slice(basis);
    const CudaHVecMultiGroupOp<Tv> op{src_vec, dst_vec};

    dispatch_cuda_multi_group_chunks_by_rank<0>(slice, basis.num_blocks, net.diag_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<1>(slice, basis.num_blocks, net.pure_a_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<2>(slice, basis.num_blocks, net.pure_b_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<3>(slice, basis.num_blocks, net.mixed_groups, op);
}
