#include "net_build.hpp"
#include "net_hvec.hpp"
#include "net_tvec.hpp"
#include "net_grad.hpp"

extern "C"
{
    void *create_svd_network_f64(
        void *basis_ptr,
        int64 ngs,
        const uint32 *axs,
        const uint32 *bxs,
        const int64 *ranks,
        const int64 *num_as,
        const int64 *num_bs,
        const uint32 *flat_azs,
        const uint32 *flat_bzs,
        const double *flat_wa,
        const double *flat_wb)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);

        return create_svd_network<uint32, double>(
            basis,
            ngs, axs, bxs,
            ranks, num_as, num_bs,
            flat_azs, flat_bzs, flat_wa, flat_wb);
    }

    void destroy_svd_network_f64(void *net_ptr)
    {
        if (!net_ptr)
            return;

        SVDNetwork<uint32, double> *net = static_cast<SVDNetwork<uint32, double> *>(net_ptr);

        destroy_svd_network<uint32, double>(net);
    }

    void hvec_svd_network_f64(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        const double *__restrict__ src,
        double *__restrict__ dst)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const SVDNetwork<uint32, double> *net = static_cast<const SVDNetwork<uint32, double> *>(net_ptr);

        hvec_svd_network<uint32, double>(basis, net, src, dst);
    }

    void tvec_svd_network_f64(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        const int64 idx,
        const double theta,
        double *__restrict__ vec)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const SVDNetwork<uint32, double> *net = static_cast<const SVDNetwork<uint32, double> *>(net_ptr);

        tvec_svd_network<uint32, double>(basis, net, idx, theta, vec);
    }

    double grad_svd_network_f64(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        const int64 idx,
        const double theta,
        const double *__restrict__ lp,
        const double *__restrict__ rp)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const SVDNetwork<uint32, double> *net = static_cast<const SVDNetwork<uint32, double> *>(net_ptr);

        return grad_svd_network<uint32, double>(basis, net, idx, theta, lp, rp);
    }
}

extern "C"
{
    void *create_svd_network_c64(
        void *basis_ptr,
        int64 ngs,
        const uint32 *axs,
        const uint32 *bxs,
        const int64 *ranks,
        const int64 *num_as,
        const int64 *num_bs,
        const uint32 *flat_azs,
        const uint32 *flat_bzs,
        const complexf64 *flat_wa,
        const complexf64 *flat_wb)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);

        return create_svd_network<uint32, complexf64>(
            basis,
            ngs, axs, bxs,
            ranks, num_as, num_bs,
            flat_azs, flat_bzs, flat_wa, flat_wb);
    }

    void destroy_svd_network_c64(void *net_ptr)
    {
        if (!net_ptr)
            return;

        SVDNetwork<uint32, complexf64> *net = static_cast<SVDNetwork<uint32, complexf64> *>(net_ptr);

        destroy_svd_network<uint32, complexf64>(net);
    }

    void hvec_svd_network_c64(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        const complexf64 *__restrict__ src,
        complexf64 *__restrict__ dst)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const SVDNetwork<uint32, complexf64> *net = static_cast<const SVDNetwork<uint32, complexf64> *>(net_ptr);

        hvec_svd_network<uint32, complexf64>(basis, net, src, dst);
    }

    void tvec_svd_network_c64(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        const int64 idx,
        const double theta,
        complexf64 *__restrict__ vec)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const SVDNetwork<uint32, complexf64> *net = static_cast<const SVDNetwork<uint32, complexf64> *>(net_ptr);

        tvec_svd_network<uint32, complexf64>(basis, net, idx, theta, vec);
    }

    complexf64 grad_svd_network_c64(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        const int64 idx,
        const double theta,
        const complexf64 *__restrict__ lp,
        const complexf64 *__restrict__ rp)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const SVDNetwork<uint32, complexf64> *net = static_cast<const SVDNetwork<uint32, complexf64> *>(net_ptr);

        return grad_svd_network<uint32, complexf64>(basis, net, idx, theta, lp, rp);
    }
}
