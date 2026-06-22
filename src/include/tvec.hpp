#pragma once
#include "otf_contract.hpp"

template <typename Tv>
struct TVecContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = false;

    const Tv *src_vec;
    Tv *dst_vec;

    FORCE_INLINE void diag(Result &, Tv vt, int64 di) const
    {
        dst_vec[di] = src_vec[di] * vt;
    }

    FORCE_INLINE void offdiag(Result &, Tv vt, int64 si, int64 di) const
    {
        tvec_update<Tv>(src_vec + si, src_vec + di, dst_vec + si, dst_vec + di, vt);
    }
};

template <int Rank, int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void launch_tvec_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    const Tv *src_vec, Tv *dst_vec)
{
    const TVecContractOp<Tv> op{src_vec, dst_vec};
    (void)otf_contract_single_group_impl<Rank, TypeCode>(basis, group, op);
}

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_tvec_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    const Tv *src_vec, Tv *dst_vec)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    switch (rank)
    {
    case 1:
        launch_tvec_contract_group<1, TypeCode>(basis, group, src_vec, dst_vec);
        break;
    case 2:
        launch_tvec_contract_group<2, TypeCode>(basis, group, src_vec, dst_vec);
        break;
    default:
        launch_tvec_contract_group<0, TypeCode>(basis, group, src_vec, dst_vec);
        break;
    }
}

template <typename Ti, typename Tv>
void tvec_svd_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, const Tv *src_vec, Tv *dst_vec)
{
    const uint8 type = net->excit_types[idx];
    const int64 pos = net->sorted_idxs[idx];
    const SVDGroup_OTF<Ti, Tv> *group = group_by_type(net, type, pos);
    if (group == nullptr)
    {
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in tvec_svd" << std::endl;
        return;
    }

#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
        dst_vec[i] = {};

    switch (type)
    {
    case 0:
        dispatch_tvec_contract_group<0>(basis, *group, src_vec, dst_vec);
        break;
    case 1:
        dispatch_tvec_contract_group<1>(basis, *group, src_vec, dst_vec);
        break;
    case 2:
        dispatch_tvec_contract_group<2>(basis, *group, src_vec, dst_vec);
        break;
    case 3:
        dispatch_tvec_contract_group<3>(basis, *group, src_vec, dst_vec);
        break;
    }
}
