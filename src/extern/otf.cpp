#include "otf.hpp"
#include "hvec.hpp"
#include "tvec.hpp"
#include "expm.hpp"
#include "grad.hpp"
#include "backgrad.hpp"
#include "backtran.hpp"
#include "batchexpm.hpp"
#include "batchgrad.hpp"
#include "batchtran.hpp"

extern "C"
{
    void destroy_network_otf_f64(void *net_ptr)
    {
        if (net_ptr == nullptr)
            return;

        Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        net->clear();
        delete net;
    }

    void *build_network_otf_f64(
        void *basis_ptr,
        int64 norb,
        int64 ngs,
        const uint32 *axs,
        const uint32 *bxs,
        const int64 *ranks,
        const int64 *num_zas,
        const int64 *num_zbs,
        const uint32 *flat_zas,
        const uint32 *flat_zbs,
        const double *flat_wa,
        const double *flat_wb)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);

        return build_network_otf<uint32, double>(
            basis,
            norb, ngs, axs, bxs,
            ranks, num_zas, num_zbs,
            flat_zas, flat_zbs, flat_wa, flat_wb);
    }

    void hvec_gather_contract_otf_f64(
        void *basis_ptr,
        void *net_ptr,
        const double *__restrict__ src,
        double *__restrict__ dst)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);
        contract_network_otf<uint32, double>(basis, net, src, dst);
    }

    void expm_contract_otf_f64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const double theta,
        double *__restrict__ vec)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        expm_svd_network_otf<uint32, double>(basis, net, idx, theta, vec);
    }

    double grad_contract_otf_f64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const double theta,
        const double *__restrict__ lp,
        const double *__restrict__ rp)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        return grad_svd_network_otf<uint32, double>(basis, net, idx, theta, lp, rp);
    }

    double backgrad_contract_otf_f64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const double theta,
        double *__restrict__ lp,
        double *__restrict__ rp)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        return backgrad_svd_network_otf<uint32, double>(basis, net, idx, theta, lp, rp);
    }

    void backtran_contract_otf_f64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const double theta,
        double *__restrict__ lp,
        double *__restrict__ rp,
        double *__restrict__ bp)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        backtran_svd_network_otf<uint32, double>(basis, net, idx, theta, lp, rp, bp);
    }

    void tvec_contract_otf_f64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const double *__restrict__ src,
        double *__restrict__ dst)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        tvec_svd_network_otf<uint32, double>(basis, net, idx, src, dst);
    }

    void batch_expm_contract_otf_f64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const double theta,
        double *__restrict__ matrix, int ld, int num_vecs)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        expm_svd_batched_network_otf<uint32, double>(basis, net, idx, theta, matrix, ld, num_vecs);
    }

    void batch_grad_contract_otf_f64(
        void *basis_ptr,
        void *net_ptr,
        const double *__restrict__ thetas,
        const double *__restrict__ lp,
        const double *__restrict__ rp,
        double *__restrict__ grads)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        grad_pool_network_batched_otf<uint32, double>(basis, net, thetas, lp, rp, grads);
    }

    void batch_tran_contract_otf_f64(
        void *basis_ptr,
        void *net_ptr,
        const double *__restrict__ lp,
        const double *__restrict__ rp,
        double *__restrict__ trans)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        tran_pool_network_batched_otf<uint32, double>(basis, net, lp, rp, trans);
    }

    void get_diags_elements_f64(void *basis_ptr, void *net_ptr, double *diags)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        get_diags_elements<uint32, double>(basis, net, diags);
    }
}

extern "C"
{
    void destroy_network_otf_c64(void *net_ptr)
    {
        if (net_ptr == nullptr)
            return;

        Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        net->clear();
        delete net;
    }

    void *build_network_otf_c64(
        void *basis_ptr,
        int64 norb,
        int64 ngs,
        const uint32 *axs,
        const uint32 *bxs,
        const int64 *ranks,
        const int64 *num_zas,
        const int64 *num_zbs,
        const uint32 *flat_zas,
        const uint32 *flat_zbs,
        const complexf64 *flat_wa,
        const complexf64 *flat_wb)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);

        return build_network_otf<uint32, complexf64>(
            basis,
            norb, ngs, axs, bxs,
            ranks, num_zas, num_zbs,
            flat_zas, flat_zbs, flat_wa, flat_wb);
    }

    void hvec_gather_contract_otf_c64(
        void *basis_ptr,
        void *net_ptr,
        const complexf64 *__restrict__ src,
        complexf64 *__restrict__ dst)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);
        contract_network_otf<uint32, complexf64>(basis, net, src, dst);
    }

    void expm_contract_otf_c64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const double theta,
        complexf64 *__restrict__ vec)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        expm_svd_network_otf<uint32, complexf64>(basis, net, idx, theta, vec);
    }

    complexf64 grad_contract_otf_c64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const double theta,
        const complexf64 *__restrict__ lp,
        const complexf64 *__restrict__ rp)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        return grad_svd_network_otf<uint32, complexf64>(basis, net, idx, theta, lp, rp);
    }

    complexf64 backgrad_contract_otf_c64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const double theta,
        complexf64 *__restrict__ lp,
        complexf64 *__restrict__ rp)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        return backgrad_svd_network_otf<uint32, complexf64>(basis, net, idx, theta, lp, rp);
    }

    void backtran_contract_otf_c64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const double theta,
        complexf64 *__restrict__ lp,
        complexf64 *__restrict__ rp,
        complexf64 *__restrict__ bp)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        backtran_svd_network_otf<uint32, complexf64>(basis, net, idx, theta, lp, rp, bp);
    }

    void tvec_contract_otf_c64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const complexf64 *__restrict__ src,
        complexf64 *__restrict__ dst)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        tvec_svd_network_otf<uint32, complexf64>(basis, net, idx, src, dst);
    }

    void batch_expm_contract_otf_c64(
        void *basis_ptr,
        void *net_ptr,
        const int64 idx,
        const double theta,
        complexf64 *__restrict__ matrix, int ld, int num_vecs)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        expm_svd_batched_network_otf<uint32, complexf64>(basis, net, idx, theta, matrix, ld, num_vecs);
    }

    void batch_grad_contract_otf_c64(
        void *basis_ptr,
        void *net_ptr,
        const double *__restrict__ thetas,
        const complexf64 *__restrict__ lp,
        const complexf64 *__restrict__ rp,
        complexf64 *__restrict__ grads)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        grad_pool_network_batched_otf<uint32, complexf64>(basis, net, thetas, lp, rp, grads);
    }

    void batch_tran_contract_otf_c64(
        void *basis_ptr,
        void *net_ptr,
        const complexf64 *__restrict__ lp,
        const complexf64 *__restrict__ rp,
        complexf64 *__restrict__ trans)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        tran_pool_network_batched_otf<uint32, complexf64>(basis, net, lp, rp, trans);
    }

    void get_diags_elements_c64(void *basis_ptr, void *net_ptr, complexf64 *diags)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        get_diags_elements<uint32, complexf64>(basis, net, diags);
    }
}
