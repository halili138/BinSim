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

    void destroy_network_otf_f64(void *otf_ptr)
    {
        Network_OTF<uint32, double> *otf = static_cast<Network_OTF<uint32, double> *>(otf_ptr);
        destroy_network_otf<uint32, double>(otf);
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

    void destroy_network_otf_c64(void *otf_ptr)
    {
        Network_OTF<uint32, complexf64> *otf = static_cast<Network_OTF<uint32, complexf64> *>(otf_ptr);
        destroy_network_otf<uint32, complexf64>(otf);
    }
}