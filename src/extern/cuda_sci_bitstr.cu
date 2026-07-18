#include "cuda/sci_hvec.cuh"

extern "C"
{
    void *upload_sci_src_f64(void *basis_ptr)
    {
        return static_cast<void *>(
            upload_sci_src_basis<uint32>(static_cast<SciBasisManager<uint32> *>(basis_ptr)));
    }

    void destroy_sci_src_f64(void *dev_ptr)
    {
        destroy_sci_src_dev<uint32>(static_cast<SciBasisViewDev<uint32> *>(dev_ptr));
    }

    void *build_basisdev_from_sci_f64(void *sci_ptr)
    {
        auto *sci = static_cast<SciBasisManager<uint32> *>(sci_ptr);
        void *raw = create_custom_basis_manager_tmpl<uint32>(
            sci->norb,
            sci->all_astrs, sci_num_alpha_strings(sci),
            sci->all_bstrs, sci_num_beta_strings(sci),
            sci->orbsym, sci->total_sym, sci->num_irreps);
        auto *reg = static_cast<BasisManager<uint32> *>(raw);
        void *dev = upload_basis<uint32>(reg);
        reg->clear(); delete reg;
        return dev;
    }

    int64 cuda_sci_select_external_block_f64(
        void *tgt_basis, void *src_dev, void *net_dev,
        const bool *is_new_a, const bool *is_new_b,
        int64 block_idx, const double *host_src_vec,
        const double *host_candidate_diags, double variational_energy,
        int /*chunk_size*/, double eps,
        uint32 *out_a, uint32 *out_b, double *out_v,
        int64 max_entries)
    {
        return cuda_sci_select_external_block<uint32, double>(
            static_cast<SciBasisManager<uint32> *>(tgt_basis),
            static_cast<SciBasisViewDev<uint32> *>(src_dev),
            static_cast<NetworkDev<uint32, double> *>(net_dev),
            is_new_a, is_new_b,
            block_idx,
            host_src_vec, host_candidate_diags,
            variational_energy, eps,
            out_a, out_b, out_v,
            max_entries);
    }

    void cuda_hvec_sci_full_f64(
        void *basis_dev, void *net_dev, int64 basis_dim,
        const double *src_vec, double *dst_vec)
    {
        cuda_hvec_sci_full<uint32, double>(
            static_cast<BasisViewDev<uint32> *>(basis_dev),
            static_cast<NetworkDev<uint32, double> *>(net_dev),
            basis_dim,
            src_vec, dst_vec);
    }
}
