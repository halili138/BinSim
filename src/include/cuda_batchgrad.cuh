#pragma once
#include "cuda_contract.cuh"

template <typename Ti, typename Tv>
void cuda_batchgrad(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    const double *thetas,
    const Tv *lp,
    const Tv *rp,
    Tv *grads)
{
    CUDA_CHECK(cudaMemset(grads, 0, net.host_sorted_idxs.size() * sizeof(Tv)));
    const BasisSliceDev<Ti> slice = make_basis_slice(basis);
    const CudaBatchGradMultiGroupOp<Tv> op{thetas, lp, rp, grads};

    dispatch_cuda_multi_group_chunks_by_rank<0>(slice, basis.num_blocks, net.diag_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<1>(slice, basis.num_blocks, net.pure_a_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<2>(slice, basis.num_blocks, net.pure_b_groups, op);
    dispatch_cuda_multi_group_chunks_by_rank<3>(slice, basis.num_blocks, net.mixed_groups, op);
}
