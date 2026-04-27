#include "common.hpp"

extern "C"
{
    int64 get_subspace_dim(void *basis_ptr)
    {
        const BasisManager *basis = static_cast<BasisManager *>(basis_ptr);
        return basis->dim;
    }

    void destroy_basis_manager(void *basis_ptr)
    {
        if (basis_ptr)
        {
            BasisManager *basis = static_cast<BasisManager *>(basis_ptr);

            delete[] basis->all_astrs;
            delete[] basis->all_bstrs;
            delete[] basis->astrs_vec;
            delete[] basis->bstrs_vec;
            delete[] basis->num_astrs;
            delete[] basis->num_bstrs;
            delete[] basis->blocks;
            delete[] basis->block_map;
            delete basis;
        }
    }

    void *create_basis_manager(
        const int64 norb,
        const int64 na,
        const int64 nb,
        const int64 total_sym,
        const int64 *__restrict__ orbsym,
        const int64 num_irreps)
    {
        BasisManager *basis = new BasisManager();

        basis->all_astrs = nullptr;
        basis->all_bstrs = nullptr;
        basis->astrs_vec = nullptr;
        basis->bstrs_vec = nullptr;
        basis->num_astrs = nullptr;
        basis->num_bstrs = nullptr;
        basis->blocks = nullptr;
        basis->block_map = nullptr;

        try
        {
            basis->num_irreps = num_irreps;
            basis->dim = 0;
            basis->num_blocks = 0;

            basis->num_astrs = new int64[num_irreps]();
            basis->num_bstrs = new int64[num_irreps]();

            uint32 a_str = (1UL << na) - 1;
            uint32 a_max = 1UL << norb;
            int64 total_a_strings = 0;
            while (a_str < a_max)
            {
                int64 sym = get_string_sym(a_str, orbsym);
                if (sym < num_irreps)
                {
                    basis->num_astrs[sym]++;
                    total_a_strings++;
                }
                if (na == 0)
                    break;
                a_str = next_combination(a_str);
            }

            uint32 b_str = (1UL << nb) - 1;
            uint32 b_max = 1UL << norb;
            int64 total_b_strings = 0;
            while (b_str < b_max)
            {
                int64 sym = get_string_sym(b_str, orbsym);
                if (sym < num_irreps)
                {
                    basis->num_bstrs[sym]++;
                    total_b_strings++;
                }
                if (nb == 0)
                    break;
                b_str = next_combination(b_str);
            }

            for (int64 asym = 0; asym < num_irreps; ++asym)
            {
                int64 bsym = total_sym ^ asym;
                if (bsym < num_irreps && basis->num_astrs[asym] > 0 && basis->num_bstrs[bsym] > 0)
                {
                    basis->num_blocks++;
                }
            }

            basis->all_astrs = new uint32[total_a_strings];
            basis->all_bstrs = new uint32[total_b_strings];

            basis->astrs_vec = new uint32 *[num_irreps];
            basis->bstrs_vec = new uint32 *[num_irreps];

            basis->blocks = new BlockDesc[basis->num_blocks];
            basis->block_map = new int64[num_irreps * num_irreps];
            std::fill_n(basis->block_map, num_irreps * num_irreps, -1);

            int64 a_offset = 0;
            int64 b_offset = 0;
            for (int64 i = 0; i < num_irreps; ++i)
            {
                basis->astrs_vec[i] = basis->all_astrs + a_offset;
                a_offset += basis->num_astrs[i];

                basis->bstrs_vec[i] = basis->all_bstrs + b_offset;
                b_offset += basis->num_bstrs[i];
            }

            int64 *a_idx = new int64[num_irreps]();
            a_str = (1UL << na) - 1;
            while (a_str < a_max)
            {
                int64 sym = get_string_sym(a_str, orbsym);
                if (sym < num_irreps)
                {
                    basis->astrs_vec[sym][a_idx[sym]++] = a_str;
                }
                if (na == 0)
                    break;
                a_str = next_combination(a_str);
            }
            delete[] a_idx;

            int64 *b_idx = new int64[num_irreps]();
            b_str = (1UL << nb) - 1;
            while (b_str < b_max)
            {
                int64 sym = get_string_sym(b_str, orbsym);
                if (sym < num_irreps)
                {
                    basis->bstrs_vec[sym][b_idx[sym]++] = b_str;
                }
                if (nb == 0)
                    break;
                b_str = next_combination(b_str);
            }
            delete[] b_idx;

            int64 block_counter = 0;
            basis->dim = 0;
            for (int64 asym = 0; asym < num_irreps; ++asym)
            {
                int64 bsym = total_sym ^ asym;
                if (bsym >= num_irreps)
                    continue;

                if (basis->num_astrs[asym] > 0 && basis->num_bstrs[bsym] > 0)
                {
                    BlockDesc &block = basis->blocks[block_counter];
                    block.asym = asym;
                    block.bsym = bsym;
                    block.num_a = basis->num_astrs[asym];
                    block.num_b = basis->num_bstrs[bsym];
                    block.astrs = basis->astrs_vec[asym];
                    block.bstrs = basis->bstrs_vec[bsym];
                    block.offset = basis->dim;

                    basis->block_map[asym * num_irreps + bsym] = block_counter;

                    block_counter++;
                    basis->dim += block.num_a * block.num_b;
                }
            }
        }
        catch (...)
        {
            destroy_basis_manager(basis);
            throw;
        }

        return static_cast<void *>(basis);
    }

    void set_det_coeff(
        void *__restrict__ basis_ptr,
        const uint32 target_astr,
        const uint32 target_bstr,
        const double coeff,
        const int64 *__restrict__ orbsym,
        double *vec)
    {
        const BasisManager *basis = static_cast<const BasisManager *>(basis_ptr);

        int64 asym = get_string_sym(target_astr, orbsym);
        int64 bsym = get_string_sym(target_bstr, orbsym);

        if (asym >= basis->num_irreps || bsym >= basis->num_irreps)
            return;

        // 访问一维展开的 block_map
        int64 block_idx = basis->block_map[asym * basis->num_irreps + bsym];
        if (block_idx == -1)
            return;

        const BlockDesc &block = basis->blocks[block_idx];

        int64 ia = find_index(block.astrs, block.num_a, target_astr);
        if (ia == -1)
            return;

        int64 ib = find_index(block.bstrs, block.num_b, target_bstr);
        if (ib == -1)
            return;

        int64 gid = block.offset + ia * block.num_b + ib;

        *(vec + gid) += coeff;
    }
}
