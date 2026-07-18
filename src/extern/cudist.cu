#include "cuda/dist.cuh"

extern "C"
{
    void *build_sub_topology_gpu_f64(void *basis_ptr, void *subnet_ptr, void *gmap_ptr, int num_phases, int phase_idx)
    {
        auto basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        auto subnet = static_cast<const Network_OTF<uint32, double> *>(subnet_ptr);
        auto gmap = static_cast<const GlobalMemMap *>(gmap_ptr);
        return static_cast<void *>(build_sub_topology_gpu_impl<uint32, double>(basis, subnet, gmap, num_phases, phase_idx));
    }

    void destroy_sub_topology_gpu(void *topo_ptr)
    {
        if (topo_ptr)
            delete static_cast<SubTopologyDev *>(topo_ptr);
    }

    void get_sub_topology_info_gpu(void *topo_ptr, int64 *send_dim, int64 *recv_dim)
    {
        auto topo = static_cast<const SubTopologyDev *>(topo_ptr);
        *send_dim = topo->send_dim;
        *recv_dim = topo->recv_dim;
    }

    void get_sub_topology_mpi_counts_gpu(void *topo_ptr, int *send_counts, int *recv_counts)
    {
        auto topo = static_cast<const SubTopologyDev *>(topo_ptr);
        std::copy(topo->send_counts.begin(), topo->send_counts.end(), send_counts);
        std::copy(topo->recv_counts.begin(), topo->recv_counts.end(), recv_counts);
    }

    void pack_send_buffer_gpu_f64(void *topo_ptr, const double *d_local_v, double *d_send_buffer)
    {
        auto topo = static_cast<const SubTopologyDev *>(topo_ptr);
        if (topo->num_pack_jobs > 0)
        {
            pack_send_buffer_gpu_kernel<double><<<topo->num_pack_jobs, 256>>>(topo->d_pack_jobs, topo->num_pack_jobs, d_local_v, d_send_buffer);
        }
    }

    void compute_hvec_sub_chunk_gpu_f64(void *basis_ptr, void *subnet_ptr, void *topo_ptr, const double *d_chunk_cache, double *d_local_w)
    {
        auto basis = static_cast<const BasisViewDev<uint32> *>(basis_ptr);
        auto subnet = static_cast<const NetworkDev<uint32, double> *>(subnet_ptr);
        auto topo = static_cast<const SubTopologyDev *>(topo_ptr);

        if (topo->num_targets == 0)
            return;

        BasisSliceDev<uint32> slice = make_basis_slice(*basis);
        slice.block_offsets = topo->d_topo_offsets;
        slice.target_bids = topo->d_target_bids;

        dispatch_chunks_by_rank_gpu<0>(slice, topo->num_targets, subnet->diag_groups, d_chunk_cache, d_local_w);
        dispatch_chunks_by_rank_gpu<1>(slice, topo->num_targets, subnet->pure_a_groups, d_chunk_cache, d_local_w);
        dispatch_chunks_by_rank_gpu<2>(slice, topo->num_targets, subnet->pure_b_groups, d_chunk_cache, d_local_w);
        dispatch_chunks_by_rank_gpu<3>(slice, topo->num_targets, subnet->mixed_groups, d_chunk_cache, d_local_w);
    }



    void compute_expm_sub_chunk_gpu_f64(void *basis_ptr, void *subnet_ptr, void *topo_ptr, const int64 idx, const double theta, double *d_chunk_cache)
    {
        auto basis = static_cast<const BasisViewDev<uint32> *>(basis_ptr);
        auto subnet = static_cast<const NetworkDev<uint32, double> *>(subnet_ptr);
        auto topo = static_cast<const SubTopologyDev *>(topo_ptr);
        if (topo->num_targets == 0)
            return;
        expm_svd_sub_chunk_gpu<uint32, double>(*basis, *subnet, *topo, idx, theta, d_chunk_cache);
    }

    double compute_backgrad_sub_chunk_gpu_f64(void *basis_ptr, void *subnet_ptr, void *topo_ptr, const double theta, double *d_left_cache, double *d_right_cache)
    {
        auto basis = static_cast<const BasisViewDev<uint32> *>(basis_ptr);
        auto subnet = static_cast<const NetworkDev<uint32, double> *>(subnet_ptr);
        auto topo = static_cast<const SubTopologyDev *>(topo_ptr);
        if (topo->num_targets == 0)
            return 0.0;
        return backgrad_svd_sub_chunk_gpu<uint32, double>(*basis, *subnet, *topo, theta, d_left_cache, d_right_cache);
    }

    void set_local_det_coeff_gpu_f64(void *gmap_ptr, void *basis_ptr, uint32 astr, uint32 bstr, double coeff, double *d_local_vec)
    {
        auto gmap = static_cast<const GlobalMemMap *>(gmap_ptr);
        auto basis = static_cast<const BasisManager<uint32> *>(basis_ptr);

        int64 asym = get_string_sym(astr, basis->orbsym);
        int64 bsym = get_string_sym(bstr, basis->orbsym);

        if (asym >= basis->num_irreps || bsym >= basis->num_irreps)
            return;

        int64 h = asym * basis->num_irreps + bsym;

        if (gmap->block_to_rank[h] != gmap->mpi_rank)
            return;

        int64 block_idx = basis->block_map[h];
        if (block_idx == -1)
            return;

        const BlockDesc<uint32> &block = basis->blocks[block_idx];

        int64 ia = find_index(block.astrs, block.num_a, astr);
        if (ia == -1)
            return;

        int64 ib = find_index(block.bstrs, block.num_b, bstr);
        if (ib == -1)
            return;

        int64 local_gid = gmap->block_local_offsets[h] + ia * block.num_b + ib;

        cudaMemcpy(d_local_vec + local_gid, &coeff, sizeof(double), cudaMemcpyHostToDevice);
    }

    void compute_expm_sub_chunk_gpu_2d_f64(void *basis_ptr, void *subnet_ptr, void *topo_ptr, const int64 idx, const double theta, double *d_chunk_cache)
    {
        auto basis = static_cast<const BasisViewDev<uint32> *>(basis_ptr);
        auto subnet = static_cast<const NetworkDev<uint32, double> *>(subnet_ptr);
        auto topo = static_cast<const SubTopologyDev *>(topo_ptr);
        if (topo->num_targets == 0)
            return;
        expm_svd_sub_chunk_gpu_2d<uint32, double>(*basis, *subnet, *topo, idx, theta, d_chunk_cache);
    }

    double compute_backgrad_sub_chunk_gpu_2d_f64(void *basis_ptr, void *subnet_ptr, void *topo_ptr, const double theta, double *d_left_cache, double *d_right_cache)
    {
        auto basis = static_cast<const BasisViewDev<uint32> *>(basis_ptr);
        auto subnet = static_cast<const NetworkDev<uint32, double> *>(subnet_ptr);
        auto topo = static_cast<const SubTopologyDev *>(topo_ptr);
        if (topo->num_targets == 0)
            return 0.0;
        return backgrad_svd_sub_chunk_gpu_2d<uint32, double>(*basis, *subnet, *topo, theta, d_left_cache, d_right_cache);
    }
}
