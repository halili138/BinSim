#pragma once
#include "otf_contract.hpp"
#include <type_traits>


template <int TypeCode, typename Ti, typename Tv, typename Op>
static FORCE_INLINE typename Op::Result dispatch_contract_group_by_rank(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> &group,
    Op op)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;

    switch (rank)
    {
    case 1:
        return otf_contract_single_group_impl<1, TypeCode>(basis, group, op);
    case 2:
        return otf_contract_single_group_impl<2, TypeCode>(basis, group, op);
    default:
        return otf_contract_single_group_impl<0, TypeCode>(basis, group, op);
    }
}

template <int TypeCode, typename Ti, typename Tv, typename OpFactory>
static FORCE_INLINE auto dispatch_contract_group(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> &group,
    OpFactory make_op)
{
    return dispatch_contract_group_by_rank<TypeCode>(basis, group, make_op());
}

template <int TypeCode, typename Ti, typename Tv, typename OpFactory>
static FORCE_INLINE auto dispatch_contract_network(
    const BasisManager<Ti> *basis,
    const Network_OTF<Ti, Tv> *net,
    int64 idx,
    OpFactory make_op)
{
    const int64 pos = net->sorted_idxs[idx];
    const SVDGroup_OTF<Ti, Tv> *group = group_by_type(net, TypeCode, pos);
    using Result = decltype(dispatch_contract_group<TypeCode>(basis, *group, make_op));
    if (group == nullptr)
    {
        std::cerr << "Error: Unexpected type = " << TypeCode << " in dispatch_contract_network" << std::endl;
        if constexpr (std::is_void_v<Result>)
            return;
        else
            return Result{};
    }

    return dispatch_contract_group<TypeCode>(basis, *group, make_op);
}

template <typename Ti, typename Tv, typename OpFactory>
static FORCE_INLINE auto dispatch_contract_network(
    const BasisManager<Ti> *basis,
    const Network_OTF<Ti, Tv> *net,
    int64 idx,
    OpFactory make_op)
{
    const uint8 type = net->excit_types[idx];

    switch (type)
    {
    case 0:
        return dispatch_contract_network<0>(basis, net, idx, make_op);
    case 1:
        return dispatch_contract_network<1>(basis, net, idx, make_op);
    case 2:
        return dispatch_contract_network<2>(basis, net, idx, make_op);
    case 3:
        return dispatch_contract_network<3>(basis, net, idx, make_op);
    default:
    {
        using Result = decltype(dispatch_contract_network<0>(basis, net, idx, make_op));
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in dispatch_contract_network" << std::endl;
        if constexpr (std::is_void_v<Result>)
            return;
        else
            return Result{};
    }
    }
}

template <typename Tv>
struct ExpmContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = false;

    double theta;
    double cd;
    double co;
    Tv *vec;

    FORCE_INLINE void diag(Result &, Tv vt, int64 di) const
    {
        vec[di] *= fast_diag_exp<Tv>(vt, theta);
    }

    FORCE_INLINE void offdiag(Result &, Tv vt, int64 si, int64 di) const
    {
        expm_update<Tv>(vec + si, vec + di, vt, cd, co);
    }
};

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_expm_contract_group(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> &group,
    double theta,
    Tv *vec)
{
    const auto make_op = [=]() { return ExpmContractOp<Tv>{theta, std::cos(theta) - 1.0, std::sin(theta), vec}; };
    (void)dispatch_contract_group<TypeCode>(basis, group, make_op);
}

template <int TypeCode, typename Ti, typename Tv>
void expm_svd_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *vec)
{
    const auto make_op = [=]() { return ExpmContractOp<Tv>{theta, std::cos(theta) - 1.0, std::sin(theta), vec}; };
    (void)dispatch_contract_network<TypeCode>(basis, net, idx, make_op);
}

template <typename Ti, typename Tv>
void expm_svd_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *vec)
{
    const auto make_op = [=]() { return ExpmContractOp<Tv>{theta, std::cos(theta) - 1.0, std::sin(theta), vec}; };
    (void)dispatch_contract_network(basis, net, idx, make_op);
}

template <typename Tv>
struct GradContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = true;

    double theta;
    double cd;
    double co;
    const Tv *lp;
    const Tv *rp;

    FORCE_INLINE void diag(Result &res, Tv vt, int64 di) const
    {
        const Tv du = fast_diag_grad<Tv>(vt, theta);
        res += math_conj(lp[di] * du) * rp[di];
    }

    FORCE_INLINE void offdiag(Result &res, Tv vt, int64 si, int64 di) const
    {
        grad_update<Tv>(res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
    }
};

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE Tv dispatch_grad_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, const Tv *lp, const Tv *rp)
{
    const auto make_op = [=]() { return GradContractOp<Tv>{theta, -std::sin(theta), std::cos(theta), lp, rp}; };
    return dispatch_contract_group<TypeCode>(basis, group, make_op);
}

template <int TypeCode, typename Ti, typename Tv>
Tv grad_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, const Tv *lp, const Tv *rp)
{
    const auto make_op = [=]() { return GradContractOp<Tv>{theta, -std::sin(theta), std::cos(theta), lp, rp}; };
    return dispatch_contract_network<TypeCode>(basis, net, idx, make_op);
}

template <typename Ti, typename Tv>
Tv grad_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, const Tv *lp, const Tv *rp)
{
    const auto make_op = [=]() { return GradContractOp<Tv>{theta, -std::sin(theta), std::cos(theta), lp, rp}; };
    return dispatch_contract_network(basis, net, idx, make_op);
}

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

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_tvec_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    const Tv *src_vec, Tv *dst_vec)
{
    const auto make_op = [=]() { return TVecContractOp<Tv>{src_vec, dst_vec}; };
    (void)dispatch_contract_group<TypeCode>(basis, group, make_op);
}

template <int TypeCode, typename Ti, typename Tv>
void tvec_svd_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, const Tv *src_vec, Tv *dst_vec)
{
#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
        dst_vec[i] = {};

    const auto make_op = [=]() { return TVecContractOp<Tv>{src_vec, dst_vec}; };
    (void)dispatch_contract_network<TypeCode>(basis, net, idx, make_op);
}

template <typename Ti, typename Tv>
void tvec_svd_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, const Tv *src_vec, Tv *dst_vec)
{
#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
        dst_vec[i] = {};

    const auto make_op = [=]() { return TVecContractOp<Tv>{src_vec, dst_vec}; };
    (void)dispatch_contract_network(basis, net, idx, make_op);
}

template <typename Tv>
struct BackgradContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = true;

    double theta;
    double ecd;
    double eco;
    double gcd;
    double gco;
    Tv *lp;
    Tv *rp;

    FORCE_INLINE void diag(Result &res, Tv vt, int64 di) const
    {
        const Tv u = fast_diag_exp<Tv>(vt, -theta);
        const Tv du = fast_diag_grad<Tv>(vt, theta);
        lp[di] *= u;
        res += math_conj(lp[di] * du) * rp[di];
        rp[di] *= u;
    }

    FORCE_INLINE void offdiag(Result &res, Tv vt, int64 si, int64 di) const
    {
        backgrad_update<Tv>(res, lp + si, lp + di, rp + si, rp + di, vt, ecd, eco, gcd, gco);
    }
};

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE Tv dispatch_backgrad_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, Tv *lp, Tv *rp)
{
    const auto make_op = [=]() { return BackgradContractOp<Tv>{theta, std::cos(theta) - 1.0, -std::sin(theta), -std::sin(theta), std::cos(theta), lp, rp}; };
    return dispatch_contract_group<TypeCode>(basis, group, make_op);
}

template <int TypeCode, typename Ti, typename Tv>
Tv backgrad_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *lp, Tv *rp)
{
    const auto make_op = [=]() { return BackgradContractOp<Tv>{theta, std::cos(theta) - 1.0, -std::sin(theta), -std::sin(theta), std::cos(theta), lp, rp}; };
    return dispatch_contract_network<TypeCode>(basis, net, idx, make_op);
}

template <typename Ti, typename Tv>
Tv backgrad_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *lp, Tv *rp)
{
    const auto make_op = [=]() { return BackgradContractOp<Tv>{theta, std::cos(theta) - 1.0, -std::sin(theta), -std::sin(theta), std::cos(theta), lp, rp}; };
    return dispatch_contract_network(basis, net, idx, make_op);
}

template <typename Tv>
struct BacktranContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = false;

    double theta;
    double ecd;
    double eco;
    Tv *lp;
    Tv *rp;
    Tv *bp;

    FORCE_INLINE void diag(Result &, Tv vt, int64 di) const
    {
        const Tv u = fast_diag_exp<Tv>(vt, theta);
        lp[di] *= u;
        rp[di] *= u;
        bp[di] = lp[di] * vt;
    }

    FORCE_INLINE void offdiag(Result &, Tv vt, int64 si, int64 di) const
    {
        backtran_update<Tv>(lp + si, lp + di, rp + si, rp + di, bp + si, bp + di, vt, ecd, eco);
    }
};

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_backtran_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, Tv *lp, Tv *rp, Tv *bp)
{
    const auto make_op = [=]() { return BacktranContractOp<Tv>{theta, std::cos(theta) - 1.0, -std::sin(theta), lp, rp, bp}; };
    (void)dispatch_contract_group<TypeCode>(basis, group, make_op);
}

template <int TypeCode, typename Ti, typename Tv>
void backtran_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *lp, Tv *rp, Tv *bp)
{
#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
        bp[i] = {};

    const auto make_op = [=]() { return BacktranContractOp<Tv>{theta, std::cos(theta) - 1.0, -std::sin(theta), lp, rp, bp}; };
    (void)dispatch_contract_network<TypeCode>(basis, net, idx, make_op);
}

template <typename Ti, typename Tv>
void backtran_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *lp, Tv *rp, Tv *bp)
{
#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
        bp[i] = {};

    const auto make_op = [=]() { return BacktranContractOp<Tv>{theta, std::cos(theta) - 1.0, -std::sin(theta), lp, rp, bp}; };
    (void)dispatch_contract_network(basis, net, idx, make_op);
}

template <typename Tv>
struct BatchedExpmContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = false;

    double theta;
    double cd;
    double co;
    Tv *matrix;
    int ld;
    int num_vecs;

    FORCE_INLINE void diag(Result &, Tv vt, int64 di) const
    {
        expm_batch_update_diag<Tv>(matrix, ld, num_vecs, di, fast_diag_exp<Tv>(vt, theta));
    }

    FORCE_INLINE void offdiag(Result &, Tv vt, int64 si, int64 di) const
    {
        expm_batch_update_matrix<Tv>(matrix, ld, num_vecs, si, di, vt, cd, co);
    }
};

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_expm_batched_contract_group(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> &group,
    double theta,
    Tv *matrix, int ld, int num_vecs)
{
    const auto make_op = [=]() { return BatchedExpmContractOp<Tv>{theta, std::cos(theta) - 1.0, std::sin(theta), matrix, ld, num_vecs}; };
    (void)dispatch_contract_group<TypeCode>(basis, group, make_op);
}

template <int TypeCode, typename Ti, typename Tv>
void expm_svd_batched_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta,
    Tv *matrix, int ld, int num_vecs)
{
    const auto make_op = [=]() { return BatchedExpmContractOp<Tv>{theta, std::cos(theta) - 1.0, std::sin(theta), matrix, ld, num_vecs}; };
    (void)dispatch_contract_network<TypeCode>(basis, net, idx, make_op);
}

template <typename Ti, typename Tv>
void expm_svd_batched_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta,
    Tv *matrix, int ld, int num_vecs)
{
    const auto make_op = [=]() { return BatchedExpmContractOp<Tv>{theta, std::cos(theta) - 1.0, std::sin(theta), matrix, ld, num_vecs}; };
    (void)dispatch_contract_network(basis, net, idx, make_op);
}
