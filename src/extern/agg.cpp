#include "net.hpp"
#include "agg.hpp"
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <utility>
#include <likwid-marker.h>

extern "C"
{
    void *build_direct_agg_network(
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
        const double *flat_wb,
        const int64 *orbsym)
    {
        BasisManager *basis = static_cast<BasisManager *>(basis_ptr);
        AggSVDNetwork *agg = new AggSVDNetwork();

        agg->ngs = ngs;
        agg->excit_types = new uint8[ngs]();

        agg->arenas = new GroupArena[ngs]();
        agg->mixed_b_rev_r1 = new TransR1 *[ngs]();
        agg->mixed_b_rev_r2 = new TransR2 *[ngs]();
        agg->mixed_b_rev_rn = new TransRN *[ngs]();

        agg->num_blocks = basis->num_blocks;
        agg->blocks = new AggBlock[agg->num_blocks]();

        for (int64 i = 0; i < agg->num_blocks; ++i)
        {
            agg->blocks[i].num_a = basis->blocks[i].num_a;
            agg->blocks[i].num_b = basis->blocks[i].num_b;
            agg->blocks[i].offset_dst = basis->blocks[i].offset;
        }

        build_all_agg_edges(ngs, axs, bxs, ranks, num_as, num_bs,
                            flat_azs, flat_bzs, flat_wa, flat_wb,
                            orbsym, basis, agg);

        return static_cast<void *>(agg);
    }

    void destroy_direct_agg_network(void *agg_ptr)
    {
        if (!agg_ptr)
            return;
        AggSVDNetwork *agg = static_cast<AggSVDNetwork *>(agg_ptr);

        if (agg->blocks)
        {
            for (uint64 i = 0; i < agg->num_blocks; ++i)
            {
                AggBlock &b = agg->blocks[i];
                // 释放 CSR Offsets 和 Edges
                delete[] b.pure_a_r1_offsets;
                delete[] b.pure_a_r1_edges;
                delete[] b.pure_a_r2_offsets;
                delete[] b.pure_a_r2_edges;
                delete[] b.pure_a_rn_offsets;
                delete[] b.pure_a_rn_edges;

                delete[] b.pure_b_r1_edges;
                delete[] b.pure_b_r2_edges;
                delete[] b.pure_b_rn_edges;

                delete[] b.mixed_r1_ax_offsets;
                delete[] b.mixed_r1_ax_nodes;
                delete[] b.mixed_r1_g_leaves;
                delete[] b.mixed_r2_ax_offsets;
                delete[] b.mixed_r2_ax_nodes;
                delete[] b.mixed_r2_g_leaves;
                delete[] b.mixed_rn_ax_offsets;
                delete[] b.mixed_rn_ax_nodes;
                delete[] b.mixed_rn_g_leaves;
            }
            delete[] agg->blocks;
        }

        if (agg->arenas)
        {
            for (uint64 g = 0; g < agg->ngs; ++g)
            {
                GroupArena &a = agg->arenas[g];
                delete[] a.r1_jumps;
                delete[] a.r1_phases;
                delete[] a.r2_jumps;
                delete[] a.r2_phases;
                delete[] a.rn_jumps;
                delete[] a.rn_weights;
                delete[] a.rn_phases;
            }
            delete[] agg->arenas;
        }

        if (agg->mixed_b_rev_r1)
        {
            for (uint64 g = 0; g < agg->ngs; ++g)
                delete[] agg->mixed_b_rev_r1[g];
            delete[] agg->mixed_b_rev_r1;
        }
        if (agg->mixed_b_rev_r2)
        {
            for (uint64 g = 0; g < agg->ngs; ++g)
                delete[] agg->mixed_b_rev_r2[g];
            delete[] agg->mixed_b_rev_r2;
        }
        if (agg->mixed_b_rev_rn)
        {
            for (uint64 g = 0; g < agg->ngs; ++g)
                delete[] agg->mixed_b_rev_rn[g];
            delete[] agg->mixed_b_rev_rn;
        }

        delete[] agg->excit_types;
        delete agg;
    }

    void get_diagonal_elements_agg(
        void *basis_ptr,
        void *agg_ptr,
        const uint32 *__restrict__ azs,
        const uint32 *__restrict__ bzs,
        const double *__restrict__ cs,
        const int64 *__restrict__ gs,
        double *__restrict__ diags)
    {
        const BasisManager *basis = static_cast<BasisManager *>(basis_ptr);
        const AggSVDNetwork *agg = static_cast<const AggSVDNetwork *>(agg_ptr);

        for (int64 g = 0; g < agg->ngs; ++g)
        {
            if (agg->excit_types[g] == 0)
            {
                const int64 lb = gs[g];
                const int64 rb = gs[g + 1];
#pragma omp parallel
                {
                    for (int64 i = 0; i < basis->num_blocks; ++i)
                    {
                        const BlockDesc &block = basis->blocks[i];
#pragma omp for schedule(guided)
                        for (int64 a = 0; a < block.num_a; ++a)
                        {
                            uint32 astr = block.astrs[a];
                            int64 row_ptr = block.offset + a * block.num_b;
                            for (int64 b = 0; b < block.num_b; ++b)
                            {
                                uint32 bstr = block.bstrs[b];

                                double vt = 0.0;
                                for (int64 k = lb; k < rb; ++k)
                                {
                                    vt += cs[k] * phase(azs[k] & astr) * phase(bzs[k] & bstr);
                                }

                                diags[row_ptr + b] += vt;
                            }
                        }
                    }
                }
            }
        }
    }

    void hvec_direct_agg_network(
        void *__restrict__ basis_ptr,
        void *__restrict__ agg_ptr,
        const uint32 *__restrict__ azs,
        const uint32 *__restrict__ bzs,
        const double *__restrict__ cs,
        const int64 *__restrict__ gs,
        const double *__restrict__ src,
        double *__restrict__ dst)
    {
        const BasisManager *basis = static_cast<const BasisManager *>(basis_ptr);
        const AggSVDNetwork *agg = static_cast<const AggSVDNetwork *>(agg_ptr);

#pragma omp parallel for schedule(static)
        for (int64 i = 0; i < basis->dim; ++i)
            dst[i] = 0.0;

        for (int64 g = 0; g < agg->ngs; ++g)
        {
            const int type = agg->excit_types[g];
            const int64 lb = gs[g];
            const int64 rb = gs[g + 1];
            const int64 n_terms = rb - lb;

            if (n_terms == 0)
                continue;

            if (type == 0)
                apply_diag_terms(basis, azs + lb, bzs + lb, cs + lb, n_terms, src, dst);
        }

        hvec_aggregated_svd(agg, src, dst);
    }

    void hvec_direct_agg_network_benchmark(
        void *__restrict__ basis_ptr,
        void *__restrict__ agg_ptr,
        const uint32 *__restrict__ azs,
        const uint32 *__restrict__ bzs,
        const double *__restrict__ cs,
        const int64 *__restrict__ gs,
        const double *__restrict__ src,
        double *__restrict__ dst,
        int enable_likwid)
    {
        if (enable_likwid)
        {
#pragma omp parallel
            {
                LIKWID_MARKER_START("hvec_direct_agg");
            }
        }

        const BasisManager *basis = static_cast<const BasisManager *>(basis_ptr);
        const AggSVDNetwork *agg = static_cast<const AggSVDNetwork *>(agg_ptr);

#pragma omp parallel for schedule(static)
        for (int64 i = 0; i < basis->dim; ++i)
            dst[i] = 0.0;

        for (int64 g = 0; g < agg->ngs; ++g)
        {
            const int type = agg->excit_types[g];
            const int64 lb = gs[g];
            const int64 rb = gs[g + 1];
            const int64 n_terms = rb - lb;

            if (n_terms == 0)
                continue;

            if (type == 0)
                apply_diag_terms(basis, azs + lb, bzs + lb, cs + lb, n_terms, src, dst);
        }

        hvec_aggregated_svd(agg, src, dst);

        if (enable_likwid)
        {
#pragma omp parallel
            {
                LIKWID_MARKER_STOP("hvec_direct_agg");
            }
        }
    }

    void print_agg_network_info(void *agg_ptr)
    {
        const AggSVDNetwork *agg = static_cast<const AggSVDNetwork *>(agg_ptr);

        if (!agg)
            return;

        uint64_t c_pa_r1 = 0, c_pa_r2 = 0, c_pa_rn = 0;
        uint64_t c_pb_r1 = 0, c_pb_r2 = 0, c_pb_rn = 0;
        uint64_t c_mx_r1 = 0, c_mx_r2 = 0, c_mx_rn = 0;

        size_t mem_blocks = agg->num_blocks * sizeof(AggBlock);

        for (uint64_t i = 0; i < agg->num_blocks; ++i)
        {
            const AggBlock &b = agg->blocks[i];

            c_pb_r1 += b.num_pure_b_r1;
            mem_blocks += b.num_pure_b_r1 * sizeof(PureB_Edge_R1);
            c_pb_r2 += b.num_pure_b_r2;
            mem_blocks += b.num_pure_b_r2 * sizeof(PureB_Edge_R2);
            c_pb_rn += b.num_pure_b_rn;
            mem_blocks += b.num_pure_b_rn * sizeof(PureB_Edge_RN);

            mem_blocks += (b.num_a + 1) * sizeof(uint32) * 9;

            if (b.pure_a_r1_offsets)
            {
                uint32 n = b.pure_a_r1_offsets[b.num_a];
                c_pa_r1 += n;
                mem_blocks += n * sizeof(PureA_Edge_R1);
            }
            if (b.pure_a_r2_offsets)
            {
                uint32 n = b.pure_a_r2_offsets[b.num_a];
                c_pa_r2 += n;
                mem_blocks += n * sizeof(PureA_Edge_R2);
            }
            if (b.pure_a_rn_offsets)
            {
                uint32 n = b.pure_a_rn_offsets[b.num_a];
                c_pa_rn += n;
                mem_blocks += n * sizeof(PureA_Edge_RN);
            }

            if (b.mixed_r1_ax_offsets)
            {
                uint32 n_nodes = b.mixed_r1_ax_offsets[b.num_a];
                uint32 n_leaves = n_nodes ? (b.mixed_r1_ax_nodes[n_nodes - 1].leaf_offset + b.mixed_r1_ax_nodes[n_nodes - 1].num_leaves) : 0;
                c_mx_r1 += n_leaves;
                mem_blocks += n_nodes * sizeof(Mixed_Ax_Node) + n_leaves * sizeof(Mixed_G_Leaf_R1);
            }
            if (b.mixed_r2_ax_offsets)
            {
                uint32 n_nodes = b.mixed_r2_ax_offsets[b.num_a];
                uint32 n_leaves = n_nodes ? (b.mixed_r2_ax_nodes[n_nodes - 1].leaf_offset + b.mixed_r2_ax_nodes[n_nodes - 1].num_leaves) : 0;
                c_mx_r2 += n_leaves;
                mem_blocks += n_nodes * sizeof(Mixed_Ax_Node) + n_leaves * sizeof(Mixed_G_Leaf_R2);
            }
            if (b.mixed_rn_ax_offsets)
            {
                uint32 n_nodes = b.mixed_rn_ax_offsets[b.num_a];
                uint32 n_leaves = n_nodes ? (b.mixed_rn_ax_nodes[n_nodes - 1].leaf_offset + b.mixed_rn_ax_nodes[n_nodes - 1].num_leaves) : 0;
                c_mx_rn += n_leaves;
                mem_blocks += n_nodes * sizeof(Mixed_Ax_Node) + n_leaves * sizeof(Mixed_G_Leaf_RN);
            }
        }

        size_t mem_arenas = agg->ngs * sizeof(GroupArena);
        size_t mem_rev = agg->ngs * sizeof(TransR1 *) * 3;

        for (uint64 g = 0; g < agg->ngs; ++g)
        {
            const GroupArena &a = agg->arenas[g];
            mem_arenas += a.num_r1_jumps * sizeof(TransR1) + a.num_r1_phases * sizeof(double) +
                          a.num_r2_jumps * sizeof(TransR2) + a.num_r2_phases * sizeof(double) +
                          a.num_rn_jumps * sizeof(TransRN) + a.num_rn_weights * sizeof(double) + a.num_rn_phases * sizeof(double);
        }

        size_t mem_base = sizeof(AggSVDNetwork) + agg->ngs * sizeof(uint8);
        size_t mem_total = mem_arenas + mem_blocks + mem_rev + mem_base;

        double gb = 1024.0 * 1024.0 * 1024.0;
        double mb = 1024.0 * 1024.0;

        std::cout << "\n===========================================================\n";
        std::cout << "     [ AGG Network Memory & Topology Report(CSR version)]  \n";
        std::cout << "===========================================================\n";
        std::cout << "[1] General Information\n";
        std::cout << "  - Number of Groups (ngs) : " << agg->ngs << "\n";
        std::cout << "  - Number of Symm Blocks  : " << agg->num_blocks << "\n\n";

        std::cout << "[2] Edge Topology (Valid Edges Count)\n";
        std::cout << "  - Pure A Edges : R1(" << c_pa_r1 << "), R2(" << c_pa_r2 << "), RN(" << c_pa_rn << ")\n";
        std::cout << "  - Pure B Edges : R1(" << c_pb_r1 << "), R2(" << c_pb_r2 << "), RN(" << c_pb_rn << ")\n";
        std::cout << "  - Mixed Edges  : R1(" << c_mx_r1 << "), R2(" << c_mx_r2 << "), RN(" << c_mx_rn << ")\n\n";

        std::cout << "[3] Memory Footprint (Absolute Raw Pointer Cost)\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  - SVD Arenas (Weights/Phases) : " << (mem_arenas / gb) << " GB\n";
        std::cout << "  - Agg Blocks (CSR Edges)      : " << (mem_blocks / gb) << " GB\n";
        std::cout << "  - Mixed Rev Jumps Pointers    : " << (mem_rev / mb) << " MB\n";
        std::cout << "  - Base Struct Overhead        : " << (mem_base / mb) << " MB\n";
        std::cout << "  ---------------------------------------------------------\n";
        std::cout << "  - TOTAL PHYSICAL MEMORY       : " << (mem_total / gb) << " GB\n";
        std::cout << "===========================================================\n\n";
    }

    void init_likwid()
    {
        LIKWID_MARKER_INIT;
    }

    void close_likwid()
    {
        LIKWID_MARKER_CLOSE;
    }
}
