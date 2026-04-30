#include "agg_build.hpp"
#include "agg_hvec.hpp"

extern "C"
{
    void init_likwid()
    {
        LIKWID_MARKER_INIT;
    }

    void close_likwid()
    {
        LIKWID_MARKER_CLOSE;
    }
}

extern "C"
{
    void *build_direct_agg_network_f64(
        void *basis_ptr,
        int64 ncs,
        int64 ngs,
        const uint32 *axs,
        const uint32 *bxs,
        const uint32 *azs,
        const uint32 *bzs,
        const double *cs,
        const int64 *gs,
        const int64 *ranks,
        const int64 *num_as,
        const int64 *num_bs,
        const uint32 *flat_azs,
        const uint32 *flat_bzs,
        const double *flat_wa,
        const double *flat_wb,
        const int64 *orbsym)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);

        return build_direct_agg_network<uint32, double>(
            basis,
            ncs, ngs,
            axs, bxs, azs, bzs, cs, gs,
            ranks, num_as, num_bs,
            flat_azs, flat_bzs, flat_wa, flat_wb, orbsym);
    }

    void destroy_direct_agg_network_f64(void *agg_ptr)
    {
        if (!agg_ptr)
            return;

        AggSVDNetwork<uint32, double> *agg = static_cast<AggSVDNetwork<uint32, double> *>(agg_ptr);

        destroy_direct_agg_network<uint32, double>(agg);
    }

    void get_diagonal_elements_agg_f64(
        void *basis_ptr,
        void *agg_ptr,
        double *__restrict__ diags)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const AggSVDNetwork<uint32, double> *agg = static_cast<const AggSVDNetwork<uint32, double> *>(agg_ptr);

        get_diagonal_elements_agg<uint32, double>(basis, agg, diags);
    }

    void hvec_direct_agg_network_f64(
        void *__restrict__ basis_ptr,
        void *__restrict__ agg_ptr,
        const double *__restrict__ src,
        double *__restrict__ dst)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const AggSVDNetwork<uint32, double> *agg = static_cast<const AggSVDNetwork<uint32, double> *>(agg_ptr);

        hvec_direct_agg_network<uint32, double>(basis, agg, src, dst);
    }

    void hvec_direct_agg_network_benchmark_f64(
        void *__restrict__ basis_ptr,
        void *__restrict__ agg_ptr,
        const double *__restrict__ src,
        double *__restrict__ dst,
        int enable_likwid)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const AggSVDNetwork<uint32, double> *agg = static_cast<const AggSVDNetwork<uint32, double> *>(agg_ptr);

        hvec_direct_agg_network_benchmark<uint32, double>(basis, agg, src, dst, enable_likwid);
    }

    void print_agg_network_info_f64(void *agg_ptr)
    {
        if (!agg_ptr)
            return;

        const AggSVDNetwork<uint32, double> *agg = static_cast<const AggSVDNetwork<uint32, double> *>(agg_ptr);

        print_agg_network_info<uint32, double>(agg);
    }
}

extern "C"
{
    void *build_direct_agg_network_c64(
        void *basis_ptr,
        int64 ncs,
        int64 ngs,
        const uint32 *axs,
        const uint32 *bxs,
        const uint32 *azs,
        const uint32 *bzs,
        const complexf64 *cs,
        const int64 *gs,
        const int64 *ranks,
        const int64 *num_as,
        const int64 *num_bs,
        const uint32 *flat_azs,
        const uint32 *flat_bzs,
        const complexf64 *flat_wa,
        const complexf64 *flat_wb,
        const int64 *orbsym)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);

        return build_direct_agg_network<uint32, complexf64>(
            basis,
            ncs, ngs,
            axs, bxs, azs, bzs, cs, gs,
            ranks, num_as, num_bs,
            flat_azs, flat_bzs, flat_wa, flat_wb, orbsym);
    }

    void destroy_direct_agg_network_c64(void *agg_ptr)
    {
        if (!agg_ptr)
            return;

        AggSVDNetwork<uint32, complexf64> *agg = static_cast<AggSVDNetwork<uint32, complexf64> *>(agg_ptr);

        destroy_direct_agg_network<uint32, complexf64>(agg);
    }

    void get_diagonal_elements_agg_c64(
        void *basis_ptr,
        void *agg_ptr,
        complexf64 *__restrict__ diags)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const AggSVDNetwork<uint32, complexf64> *agg = static_cast<const AggSVDNetwork<uint32, complexf64> *>(agg_ptr);

        get_diagonal_elements_agg<uint32, complexf64>(basis, agg, diags);
    }

    void hvec_direct_agg_network_c64(
        void *__restrict__ basis_ptr,
        void *__restrict__ agg_ptr,
        const complexf64 *__restrict__ src,
        complexf64 *__restrict__ dst)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const AggSVDNetwork<uint32, complexf64> *agg = static_cast<const AggSVDNetwork<uint32, complexf64> *>(agg_ptr);

        hvec_direct_agg_network<uint32, complexf64>(basis, agg, src, dst);
    }

    void hvec_direct_agg_network_benchmark_c64(
        void *__restrict__ basis_ptr,
        void *__restrict__ agg_ptr,
        const complexf64 *__restrict__ src,
        complexf64 *__restrict__ dst,
        int enable_likwid)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        const AggSVDNetwork<uint32, complexf64> *agg = static_cast<const AggSVDNetwork<uint32, complexf64> *>(agg_ptr);

        hvec_direct_agg_network_benchmark<uint32, complexf64>(basis, agg, src, dst, enable_likwid);
    }

    void print_agg_network_info_c64(void *agg_ptr)
    {
        if (!agg_ptr)
            return;

        const AggSVDNetwork<uint32, complexf64> *agg = static_cast<const AggSVDNetwork<uint32, complexf64> *>(agg_ptr);

        print_agg_network_info<uint32, complexf64>(agg);
    }
}
