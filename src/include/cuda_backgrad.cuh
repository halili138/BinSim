#pragma once
#include "cuda_contract.cuh"

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
Tv backgrad_svd_network_otf_gpu(
    const BasisSliceDev<Ti> &basis_slice,
    const NetworkDev<Ti, Tv> &net,
    int64 idx, double theta,
    Tv *lp,
    Tv *rp,
    int max_tasks)
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
Tv backgrad_svd_network_otf_gpu(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    int64 idx, double theta,
    Tv *lp,
    Tv *rp)
{
    const BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    return backgrad_svd_network_otf_gpu<Ti, Tv>(basis_slice, net, idx, theta, lp, rp, cuda_single_group_max_tasks(basis));
}
