#pragma once
#include "cuda_contract.cuh"

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
void expm_svd_network_otf_gpu(
    const BasisSliceDev<Ti> &basis_slice,
    const NetworkDev<Ti, Tv> &net,
    int64 idx,
    double theta,
    Tv *dev_vec,
    int max_tasks)
{
    const CudaExpmLauncher<Ti, Tv> launcher{theta, dev_vec};
    dispatch_cuda_network_group<Ti, Tv>(basis_slice, net, idx, max_tasks, launcher);
}

template <typename Ti, typename Tv>
void expm_svd_network_otf_gpu(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    int64 idx, double theta,
    Tv *dev_vec)
{
    const BasisSliceDev<Ti> basis_slice = make_basis_slice(basis);
    expm_svd_network_otf_gpu<Ti, Tv>(basis_slice, net, idx, theta, dev_vec, cuda_single_group_max_tasks(basis));
}
