#include "dist.hpp"

extern "C"
{
    // ==========================================
    // 1. 全局内存图 (GlobalMemMap) 接口
    // ==========================================
    void *build_global_map_otf_f64(void *basis_ptr, int map_rank, int map_size)
    {
        auto basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        return static_cast<void *>(build_global_map<uint32>(basis, map_rank, map_size));
    }

    void destroy_global_map_otf(void *gmap_ptr)
    {
        if (gmap_ptr)
            delete static_cast<GlobalMemMap *>(gmap_ptr);
    }

    int64_t get_local_dim_otf_gmap(void *gmap_ptr)
    {
        return static_cast<const GlobalMemMap *>(gmap_ptr)->local_dim;
    }

    void get_rank_block_counts_otf_gmap(void *gmap_ptr, int *counts)
    {
        auto gmap = static_cast<const GlobalMemMap *>(gmap_ptr);
        auto rank_block_counts = get_rank_block_counts(gmap);
        std::copy(rank_block_counts.begin(), rank_block_counts.end(), counts);
    }

    int get_max_rank_num_blocks_otf_gmap(void *basis_ptr, void *gmap_ptr)
    {
        auto basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        auto gmap = static_cast<const GlobalMemMap *>(gmap_ptr);
        return get_max_rank_num_blocks<uint32>(basis, gmap);
    }

    // 将物理基态精准切分到各节点的局部内存中
    void scatter_global_v_f64(void *gmap_ptr, void *basis_ptr, const double *global_v, double *local_v)
    {
        auto gmap = static_cast<const GlobalMemMap *>(gmap_ptr);
        auto basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        int64 num_irreps = basis->num_irreps;
        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            int64 h = basis->blocks[i].asym * num_irreps + basis->blocks[i].bsym;
            if (gmap->block_to_rank[h] == gmap->mpi_rank)
            {
                const auto &blk = basis->blocks[i];
                std::copy(global_v + blk.offset, global_v + blk.offset + blk.num_a * blk.num_b, local_v + gmap->block_local_offsets[h]);
            }
        }
    }

    // ==========================================
    // 2. 分段通信账本 (SubTopology) 接口
    // ==========================================
    void *build_sub_topology_otf_f64(void *basis_ptr, void *subnet_ptr, void *gmap_ptr, int num_phases, int phase_idx)
    {
        auto basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        auto subnet = static_cast<const Network_OTF<uint32, double> *>(subnet_ptr);
        auto gmap = static_cast<const GlobalMemMap *>(gmap_ptr);
        return static_cast<void *>(build_sub_topology<uint32, double>(basis, subnet, gmap, num_phases, phase_idx));
    }

    void destroy_sub_topology_otf(void *topo_ptr)
    {
        if (topo_ptr)
            delete static_cast<SubTopology *>(topo_ptr);
    }

    void get_sub_topology_info_otf(void *topo_ptr, int64_t *send_dim, int64_t *recv_dim)
    {
        auto topo = static_cast<const SubTopology *>(topo_ptr);
        *send_dim = topo->send_dim;
        *recv_dim = topo->recv_dim;
    }

    void get_sub_topology_mpi_counts_otf(void *topo_ptr, int *send_counts, int *recv_counts)
    {
        auto topo = static_cast<const SubTopology *>(topo_ptr);
        std::copy(topo->send_counts.begin(), topo->send_counts.end(), send_counts);
        std::copy(topo->recv_counts.begin(), topo->recv_counts.end(), recv_counts);
    }

    void set_local_det_coeff_f64(void *gmap_ptr, void *basis_ptr, uint32 astr, uint32 bstr, double coeff, double *local_vec)
    {
        set_local_det_coeff<uint32, double>(
            static_cast<const GlobalMemMap *>(gmap_ptr),
            static_cast<const BasisManager<uint32> *>(basis_ptr),
            astr, bstr, coeff, local_vec);
    }



    void *build_network_otf_with_idxs_f64(
        void *basis_ptr, int64_t norb, int64_t ngs,
        const int64_t *original_idxs, const uint32_t *axs, const uint32_t *bxs,
        const int64_t *ranks, const int64_t *num_zas, const int64_t *num_zbs,
        const uint32_t *flat_zas, const uint32_t *flat_zbs,
        const double *flat_wa, const double *flat_wb)
    {
        auto basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        auto net = static_cast<Network_OTF<uint32, double> *>(build_network_otf<uint32, double>(
            basis, norb, ngs, axs, bxs, ranks, num_zas, num_zbs, flat_zas, flat_zbs, flat_wa, flat_wb));

        for (auto *bucket : {&net->diag_groups, &net->pure_a_groups, &net->pure_b_groups, &net->mixed_groups})
        {
            for (auto &g : *bucket)
                g.original_idx = original_idxs[g.original_idx];
        }
        return static_cast<void *>(net);
    }

    // ==========================================
    // 3. 计算与打包核心接口
    // ==========================================
    void pack_send_buffer_f64_sub(void *topo_ptr, const double *local_v, double *send_buffer)
    {
        auto topo = static_cast<const SubTopology *>(topo_ptr);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < topo->pack_jobs.size(); ++i)
        {
            const auto &job = topo->pack_jobs[i];
            std::copy(local_v + job.src_offset, local_v + job.src_offset + job.size, send_buffer + job.dst_offset);
        }
    }

    void compute_hvec_sub_chunk_f64(void *basis, void *subnet, void *topo, const double *chunk_cache, double *local_w)
    {
        compute_hvec_sub_chunk(static_cast<const BasisManager<uint32> *>(basis),
                               static_cast<const Network_OTF<uint32, double> *>(subnet),
                               static_cast<const SubTopology *>(topo),
                               chunk_cache, local_w);
    }

    double compute_grad_sub_chunk_f64(void *basis, void *subnet, void *topo, int64_t idx, double theta, const double *lp, const double *rp)
    {
        return compute_grad_sub_chunk(static_cast<const BasisManager<uint32> *>(basis),
                                      static_cast<const Network_OTF<uint32, double> *>(subnet),
                                      static_cast<const SubTopology *>(topo),
                                      idx, theta, lp, rp);
    }

    void compute_expm_sub_chunk_f64(void *basis, void *subnet, void *topo, int64_t idx, double theta, const double *chunk_cache, double *local_v)
    {
        compute_expm_sub_chunk(static_cast<const BasisManager<uint32> *>(basis),
                               static_cast<const Network_OTF<uint32, double> *>(subnet),
                               static_cast<const SubTopology *>(topo),
                               idx, theta, chunk_cache, local_v);
    }
}