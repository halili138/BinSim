#include "distribute.hpp"

extern "C"
{

    // ─── DistributedBasisManager (double) ───────────────────────────────────────
    void *create_distributed_basis_f64(MPI_Comm comm, void *basis_ptr, int64 norb)
    {
        auto *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        auto *db = new DistributedBasisManager<uint32, double>(
            comm, norb, basis->num_irreps, basis->num_blocks,
            basis->blocks, basis->block_map, basis->orbsym);
        return static_cast<void *>(db);
    }

    void destroy_distributed_basis_f64(void *ptr)
    {
        delete static_cast<DistributedBasisManager<uint32, double> *>(ptr);
    }

    int64 distributed_basis_local_dim_f64(void *ptr)
    {
        return static_cast<DistributedBasisManager<uint32, double> *>(ptr)->local_dim;
    }

    int64 distributed_basis_num_ranks_f64(void *ptr)
    {
        return static_cast<DistributedBasisManager<uint32, double> *>(ptr)->num_ranks;
    }

    int64 distributed_basis_my_rank_f64(void *ptr)
    {
        return static_cast<DistributedBasisManager<uint32, double> *>(ptr)->my_rank;
    }

    void distributed_basis_get_local_src_f64(void *ptr, double *out)
    {
        auto *db = static_cast<DistributedBasisManager<uint32, double> *>(ptr);
        std::copy(db->local_src_vec, db->local_src_vec + db->local_dim, out);
    }

    void distributed_basis_sync_local_src_f64(void *ptr, double *inp)
    {
        auto *db = static_cast<DistributedBasisManager<uint32, double> *>(ptr);
        std::copy(inp, inp + db->local_dim, db->local_src_vec);
    }

    void distributed_extract_local_vec_f64(void *db_ptr, const double *global_vec, double *out)
    {
        auto *db = static_cast<DistributedBasisManager<uint32, double> *>(db_ptr);
        for (int64 bi = 0; bi < db->num_local_blocks; ++bi)
        {
            const auto &lb = db->local_blocks[bi];
            int64 bid = lb.asym * db->num_irreps + lb.bsym;
            int64 gbi = db->global_block_map[bid];
            if (gbi < 0)
                continue;
            const auto &gb = db->global_blocks[gbi];
            for (int32 a = 0; a < lb.num_a; ++a)
                for (int32 b = 0; b < lb.num_b; ++b)
                    out[lb.offset + a * lb.num_b + b] = global_vec[gb.offset + a * gb.num_b + b];
        }
    }

    // ─── DistributedNetwork_OTF (double) ─────────────────────────────────────────
    void *build_distributed_net_f64(void *net_ptr, const int64 *orbsym)
    {
        auto *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);
        return static_cast<void *>(build_distributed_network<uint32, double>(net, orbsym));
    }

    void destroy_distributed_net_f64(void *ptr)
    {
        delete static_cast<DistributedNetwork_OTF<uint32, double> *>(ptr);
    }

    int64 distributed_net_num_groups_f64(void *ptr)
    {
        return static_cast<DistributedNetwork_OTF<uint32, double> *>(ptr)->num_groups;
    }

    // ─── hvec (double, distributed) ──────────────────────────────────────────────
    void hvec_gather_contract_otf_distributed_f64(
        void *dbasis_ptr, void *dnet_ptr, double *local_src, double *local_dst)
    {
        auto *db = static_cast<DistributedBasisManager<uint32, double> *>(dbasis_ptr);
        auto *dn = static_cast<DistributedNetwork_OTF<uint32, double> *>(dnet_ptr);
        contract_network_otf_distributed<uint32, double>(db, dn, local_src, local_dst);
    }

    void distributed_compute_local_diags_f64(
        void *db_ptr,
        const uint32 *azs, const uint32 *bzs,
        const double *cs, int64 nterms,
        double *out)
    {
        auto *db = static_cast<DistributedBasisManager<uint32, double> *>(db_ptr);
        if (nterms == 0)
            return;

#pragma omp parallel
        {
            bool *parity_a = new bool[nterms];

            for (int64 bi = 0; bi < db->num_local_blocks; ++bi)
            {
                const auto &block = db->local_blocks[bi];
#pragma omp for schedule(guided) nowait
                for (int32 a = 0; a < block.num_a; ++a)
                {
                    const uint32 astr = block.astrs[a];
                    const int64 row_ptr = block.offset + a * block.num_b;

                    for (int64 k = 0; k < nterms; ++k)
                    {
                        parity_a[k] = std::popcount(azs[k] & astr) & 1;
                    }

                    for (int32 b = 0; b < block.num_b; ++b)
                    {
                        const uint32 bstr = block.bstrs[b];

                        double vt = {};
                        for (int64 k = 0; k < nterms; ++k)
                        {
                            const bool parity_b = std::popcount(bzs[k] & bstr) & 1;
                            const bool parity = parity_a[k] ^ parity_b;
                            vt += parity ? -cs[k] : cs[k];
                        }

                        out[row_ptr + b] += vt;
                    }
                }
            }

            delete[] parity_a;
        }
    }

    // ─── DistributedBasisManager (complex double) ────────────────────────────────
    void *create_distributed_basis_c64(MPI_Comm comm, void *basis_ptr, int64 norb)
    {
        auto *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        auto *db = new DistributedBasisManager<uint32, complexf64>(
            comm, norb, basis->num_irreps, basis->num_blocks,
            basis->blocks, basis->block_map, basis->orbsym);
        return static_cast<void *>(db);
    }

    void destroy_distributed_basis_c64(void *ptr)
    {
        delete static_cast<DistributedBasisManager<uint32, complexf64> *>(ptr);
    }

    int64 distributed_basis_local_dim_c64(void *ptr)
    {
        return static_cast<DistributedBasisManager<uint32, complexf64> *>(ptr)->local_dim;
    }

    int64 distributed_basis_num_ranks_c64(void *ptr)
    {
        return static_cast<DistributedBasisManager<uint32, complexf64> *>(ptr)->num_ranks;
    }

    int64 distributed_basis_my_rank_c64(void *ptr)
    {
        return static_cast<DistributedBasisManager<uint32, complexf64> *>(ptr)->my_rank;
    }

    void distributed_basis_get_local_src_c64(void *ptr, complexf64 *out)
    {
        auto *db = static_cast<DistributedBasisManager<uint32, complexf64> *>(ptr);
        std::copy(db->local_src_vec, db->local_src_vec + db->local_dim, out);
    }

    void distributed_basis_sync_local_src_c64(void *ptr, complexf64 *inp)
    {
        auto *db = static_cast<DistributedBasisManager<uint32, complexf64> *>(ptr);
        std::copy(inp, inp + db->local_dim, db->local_src_vec);
    }

    void distributed_extract_local_vec_c64(void *db_ptr, const complexf64 *global_vec, complexf64 *out)
    {
        auto *db = static_cast<DistributedBasisManager<uint32, complexf64> *>(db_ptr);
        for (int64 bi = 0; bi < db->num_local_blocks; ++bi)
        {
            const auto &lb = db->local_blocks[bi];
            int64 bid = lb.asym * db->num_irreps + lb.bsym;
            int64 gbi = db->global_block_map[bid];
            if (gbi < 0)
                continue;
            const auto &gb = db->global_blocks[gbi];
            for (int32 a = 0; a < lb.num_a; ++a)
                for (int32 b = 0; b < lb.num_b; ++b)
                    out[lb.offset + a * lb.num_b + b] = global_vec[gb.offset + a * gb.num_b + b];
        }
    }

    // ─── DistributedNetwork_OTF (complex double) ─────────────────────────────────
    void *build_distributed_net_c64(void *net_ptr, const int64 *orbsym)
    {
        auto *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);
        return static_cast<void *>(build_distributed_network<uint32, complexf64>(net, orbsym));
    }

    void destroy_distributed_net_c64(void *ptr)
    {
        delete static_cast<DistributedNetwork_OTF<uint32, complexf64> *>(ptr);
    }

    int64 distributed_net_num_groups_c64(void *ptr)
    {
        return static_cast<DistributedNetwork_OTF<uint32, complexf64> *>(ptr)->num_groups;
    }

    void hvec_gather_contract_otf_distributed_c64(
        void *dbasis_ptr, void *dnet_ptr, complexf64 *local_src, complexf64 *local_dst)
    {
        auto *db = static_cast<DistributedBasisManager<uint32, complexf64> *>(dbasis_ptr);
        auto *dn = static_cast<DistributedNetwork_OTF<uint32, complexf64> *>(dnet_ptr);
        contract_network_otf_distributed<uint32, complexf64>(db, dn, local_src, local_dst);
    }

    void distributed_compute_local_diags_c64(
        void *db_ptr,
        const uint32 *azs, const uint32 *bzs,
        const complexf64 *cs, int64 nterms,
        complexf64 *out)
    {
        auto *db = static_cast<DistributedBasisManager<uint32, complexf64> *>(db_ptr);
        if (nterms == 0)
            return;

#pragma omp parallel
        {
            bool *parity_a = new bool[nterms];

            for (int64 bi = 0; bi < db->num_local_blocks; ++bi)
            {
                const auto &block = db->local_blocks[bi];
#pragma omp for schedule(guided) nowait
                for (int32 a = 0; a < block.num_a; ++a)
                {
                    const uint32 astr = block.astrs[a];
                    const int64 row_ptr = block.offset + a * block.num_b;

                    for (int64 k = 0; k < nterms; ++k)
                    {
                        parity_a[k] = std::popcount(azs[k] & astr) & 1;
                    }

                    for (int32 b = 0; b < block.num_b; ++b)
                    {
                        const uint32 bstr = block.bstrs[b];

                        complexf64 vt = {};
                        for (int64 k = 0; k < nterms; ++k)
                        {
                            const bool parity_b = std::popcount(bzs[k] & bstr) & 1;
                            const bool parity = parity_a[k] ^ parity_b;
                            complexf64 val = parity ? -cs[k] : cs[k];
                            vt += complexf64(val.real(), 0.0);
                        }

                        out[row_ptr + b] += vt;
                    }
                }
            }

            delete[] parity_a;
        }
    }

} // extern "C"
