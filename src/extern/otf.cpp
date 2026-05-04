#include "otf.hpp"

extern "C"
{
    void *build_network_otf_f64(
        void *__restrict__ basis_ptr,
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

    void destroy_network_otf_f64(void *net_ptr)
    {
        Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);
        destroy_network_otf<uint32, double>(net);
    }

    void hvec_gather_contract_otf_f64(
        void *__restrict__ basis_ptr,
        void *net_ptr,
        const double *__restrict__ src,
        double *__restrict__ dst)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, double> *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);
        contract_network_otf<uint32, double>(basis, net, src, dst);
    }
}

extern "C"
{
    void *build_network_otf_c64(
        void *__restrict__ basis_ptr,
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

    void destroy_network_otf_c64(void *net_ptr)
    {
        Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);
        destroy_network_otf<uint32, complexf64>(net);
    }

    void hvec_gather_contract_otf_c64(
        void *__restrict__ basis_ptr,
        void *net_ptr,
        const complexf64 *__restrict__ src,
        complexf64 *__restrict__ dst)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        const Network_OTF<uint32, complexf64> *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);
        contract_network_otf<uint32, complexf64>(basis, net, src, dst);
    }
}
