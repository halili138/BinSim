#pragma once
#include "cuda_common.cuh"
#include "cuda_basis.cuh"
#include "cuda_otf.cuh"
#include "cuda_utils.cuh"
#include "cuda_hvec.cuh"
#include "cuda_dict.cuh"
#include "sci_basis.hpp"

// ─── Data structures ───────────────────────────────────────────

template <typename Ti>
struct SciBasisViewDev
{
    int num_blocks = 0, num_irreps = 0;
    int64 dim = 0;
    const int64 *block_offsets = nullptr;
    const int *block_num_a = nullptr, *block_num_b = nullptr;
    std::vector<int> host_block_num_a, host_block_num_b;
    const int *block_asym = nullptr, *block_bsym = nullptr;
    const Ti *astrs_flat = nullptr, *bstrs_flat = nullptr;
    const int64 *astrs_start = nullptr, *bstrs_start = nullptr;
    const int *block_map = nullptr;
    const Ti *dict_a_keys = nullptr;
    int *dict_a_vals = nullptr;
    const Ti *dict_b_keys = nullptr;
    int *dict_b_vals = nullptr;
    size_t dict_capacity = 0;
    const bool *is_new_a = nullptr, *is_new_b = nullptr;
    const void *candidate_diags = nullptr;

    SciBasisViewDev() = default;
    SciBasisViewDev(const SciBasisViewDev &) = delete;
    SciBasisViewDev &operator=(const SciBasisViewDev &) = delete;
    SciBasisViewDev(SciBasisViewDev &&o) noexcept { *this = std::move(o); }
    ~SciBasisViewDev() { clear(); }
    void clear()
    {
        if (block_offsets)
        {
            cudaFree(const_cast<int64 *>(block_offsets));
            block_offsets = nullptr;
        }
        if (block_num_a)
        {
            cudaFree(const_cast<int *>(block_num_a));
            block_num_a = nullptr;
        }
        if (block_num_b)
        {
            cudaFree(const_cast<int *>(block_num_b));
            block_num_b = nullptr;
        }
        if (block_asym)
        {
            cudaFree(const_cast<int *>(block_asym));
            block_asym = nullptr;
        }
        if (block_bsym)
        {
            cudaFree(const_cast<int *>(block_bsym));
            block_bsym = nullptr;
        }
        if (astrs_flat)
        {
            cudaFree(const_cast<Ti *>(astrs_flat));
            astrs_flat = nullptr;
        }
        if (bstrs_flat)
        {
            cudaFree(const_cast<Ti *>(bstrs_flat));
            bstrs_flat = nullptr;
        }
        if (astrs_start)
        {
            cudaFree(const_cast<int64 *>(astrs_start));
            astrs_start = nullptr;
        }
        if (bstrs_start)
        {
            cudaFree(const_cast<int64 *>(bstrs_start));
            bstrs_start = nullptr;
        }
        if (block_map)
        {
            cudaFree(const_cast<int *>(block_map));
            block_map = nullptr;
        }
        if (dict_a_keys)
        {
            cudaFree(const_cast<Ti *>(dict_a_keys));
            dict_a_keys = nullptr;
        }
        if (dict_a_vals)
        {
            cudaFree(dict_a_vals);
            dict_a_vals = nullptr;
        }
        if (dict_b_keys)
        {
            cudaFree(const_cast<Ti *>(dict_b_keys));
            dict_b_keys = nullptr;
        }
        if (dict_b_vals)
        {
            cudaFree(dict_b_vals);
            dict_b_vals = nullptr;
        }
        if (is_new_a)
        {
            cudaFree(const_cast<bool *>(is_new_a));
            is_new_a = nullptr;
        }
        if (is_new_b)
        {
            cudaFree(const_cast<bool *>(is_new_b));
            is_new_b = nullptr;
        }
        if (candidate_diags)
        {
            cudaFree(const_cast<void *>(candidate_diags));
            candidate_diags = nullptr;
        }
        dict_capacity = 0;
        host_block_num_a.clear();
        host_block_num_b.clear();
    }
    SciBasisViewDev &operator=(SciBasisViewDev &&o) noexcept
    {
        if (this != &o)
        {
            clear();
            num_blocks = o.num_blocks;
            num_irreps = o.num_irreps;
            dim = o.dim;
            block_offsets = o.block_offsets;
            block_num_a = o.block_num_a;
            block_num_b = o.block_num_b;
            host_block_num_a = std::move(o.host_block_num_a);
            host_block_num_b = std::move(o.host_block_num_b);
            block_asym = o.block_asym;
            block_bsym = o.block_bsym;
            astrs_flat = o.astrs_flat;
            bstrs_flat = o.bstrs_flat;
            astrs_start = o.astrs_start;
            bstrs_start = o.bstrs_start;
            block_map = o.block_map;
            dict_a_keys = o.dict_a_keys;
            dict_a_vals = o.dict_a_vals;
            dict_b_keys = o.dict_b_keys;
            dict_b_vals = o.dict_b_vals;
            dict_capacity = o.dict_capacity;
            is_new_a = o.is_new_a;
            is_new_b = o.is_new_b;
            candidate_diags = o.candidate_diags;
            o.block_offsets = nullptr;
            o.block_num_a = nullptr;
            o.block_num_b = nullptr;
            o.block_asym = nullptr;
            o.block_bsym = nullptr;
            o.astrs_flat = nullptr;
            o.bstrs_flat = nullptr;
            o.astrs_start = nullptr;
            o.bstrs_start = nullptr;
            o.block_map = nullptr;
            o.dict_a_keys = nullptr;
            o.dict_a_vals = nullptr;
            o.dict_b_keys = nullptr;
            o.dict_b_vals = nullptr;
            o.dict_capacity = 0;
            o.is_new_a = nullptr;
            o.is_new_b = nullptr;
            o.candidate_diags = nullptr;
        }
        return *this;
    }
};

template <typename Ti, typename Tv>
struct SciBasisSliceDev
{
    int num_blocks = 0, num_irreps = 0;
    int64 dim = 0;
    const int64 *block_offsets = nullptr;
    const int *block_num_a = nullptr, *block_num_b = nullptr;
    const int *block_asym = nullptr, *block_bsym = nullptr;
    const Ti *astrs_flat = nullptr, *bstrs_flat = nullptr;
    const int64 *astrs_start = nullptr, *bstrs_start = nullptr;
    const int *block_map = nullptr;
    const Ti *dict_a_keys = nullptr;
    int *dict_a_vals = nullptr;
    const Ti *dict_b_keys = nullptr;
    int *dict_b_vals = nullptr;
    size_t dict_capacity = 0;
};

// ─── Dict helpers ────────────────────────────────────────────────

template <typename Ti>
static void create_sci_dict_tables(Ti *&d_keys, int *&d_vals, size_t &capacity, size_t n)
{
    capacity = 1ull;
    while (capacity < n * 2)
        capacity <<= 1;
    CUDA_CHECK(cudaMalloc(&d_keys, capacity * sizeof(Ti)));
    CUDA_CHECK(cudaMalloc(&d_vals, capacity * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_keys, 0xFF, capacity * sizeof(Ti)));
    CUDA_CHECK(cudaMemset(d_vals, 0xFF, capacity * sizeof(int))); // -1 = not found
}

template <typename Ti>
static void build_sci_gpu_dict(
    const SciBasisManager<Ti> *hb,
    Ti *&d_keys_a, int *&d_vals_a, Ti *&d_keys_b, int *&d_vals_b, size_t &dcap)
{
    int64 ta = 0, tb = 0;
    for (int64 i = 0; i < hb->num_blocks; ++i)
    {
        ta += hb->blocks[i].num_a;
        tb += hb->blocks[i].num_b;
    }
    size_t ca, cb;
    create_sci_dict_tables<Ti>(d_keys_a, d_vals_a, ca, (size_t)ta);
    create_sci_dict_tables<Ti>(d_keys_b, d_vals_b, cb, (size_t)tb);
    dcap = ca > cb ? ca : cb;

    std::vector<Ti> aka, bkb;
    std::vector<int> ava, bvb;
    // Value = local block index (same meaning as astr2idx/bstr2idx in BasisViewDev)
    for (int64 i = 0; i < hb->num_blocks; ++i)
    {
        for (int a = 0; a < hb->blocks[i].num_a; ++a)
        {
            aka.push_back(hb->blocks[i].astrs[a]);
            ava.push_back(a);
        }
        for (int b = 0; b < hb->blocks[i].num_b; ++b)
        {
            bkb.push_back(hb->blocks[i].bstrs[b]);
            bvb.push_back(b);
        }
    }

    Ti *d_ika = up(aka.data(), (int64)aka.size());
    int *d_iva = up(ava.data(), (int64)ava.size());
    Ti *d_ikb = up(bkb.data(), (int64)bkb.size());
    int *d_ivb = up(bvb.data(), (int64)bvb.size());
    uint32_t *d_st;
    CUDA_CHECK(cudaMalloc(&d_st, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_st, 0, sizeof(uint32_t)));
    cuda_dict::launch_batch_insert(d_keys_a, d_vals_a, d_ika, d_iva, (size_t)aka.size(), ca, d_st);
    cuda_dict::launch_batch_insert(d_keys_b, d_vals_b, d_ikb, d_ivb, (size_t)bkb.size(), cb, d_st);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_ika);
    cudaFree(d_iva);
    cudaFree(d_ikb);
    cudaFree(d_ivb);
    cudaFree(d_st);
}

// ─── Upload ──────────────────────────────────────────────────────

template <typename Ti>
static void build_sci_block_layout(
    const SciBasisManager<Ti> *hb, int filter_bid,
    std::vector<int64> &ho, std::vector<int> &hna, std::vector<int> &hnb,
    std::vector<int> &hasy, std::vector<int> &hbsy,
    std::vector<int64> &has, std::vector<int64> &hbs,
    std::vector<Ti> &haf, std::vector<Ti> &hbf, int &ni, int &nb)
{
    ni = (int)hb->num_irreps;
    nb = (filter_bid >= 0) ? 1 : (int)hb->num_blocks;
    ho.resize(nb);
    hna.resize(nb);
    hnb.resize(nb);
    hasy.resize(nb);
    hbsy.resize(nb);
    has.resize(nb);
    hbs.resize(nb);
    int64 ta = 0, tb = 0;
    int bi = 0;
    for (int64 i = 0; i < hb->num_blocks; ++i)
    {
        if (filter_bid >= 0 && i != filter_bid)
            continue;
        const auto &b = hb->blocks[i];
        ho[bi] = (filter_bid >= 0) ? 0 : b.offset;
        hna[bi] = (int)b.num_a;
        hnb[bi] = (int)b.num_b;
        hasy[bi] = (int)b.asym;
        hbsy[bi] = (int)b.bsym;
        has[bi] = ta;
        hbs[bi] = tb;
        ta += hna[bi];
        tb += hnb[bi];
        ++bi;
    }
    haf.resize(ta);
    hbf.resize(tb);
    bi = 0;
    for (int64 i = 0; i < hb->num_blocks; ++i)
    {
        if (filter_bid >= 0 && i != filter_bid)
            continue;
        for (int j = 0; j < hb->blocks[i].num_a; ++j)
            haf[has[bi] + j] = hb->blocks[i].astrs[j];
        for (int j = 0; j < hb->blocks[i].num_b; ++j)
            hbf[hbs[bi] + j] = hb->blocks[i].bstrs[j];
        ++bi;
    }
}

template <typename Ti>
static SciBasisViewDev<Ti> *upload_sci_src_basis(const SciBasisManager<Ti> *hb)
{
    auto *db = new SciBasisViewDev<Ti>();
    int ni, nb;
    std::vector<int64> ho, has, hbs;
    std::vector<int> hna, hnb, hasy, hbsy;
    std::vector<Ti> haf, hbf;
    build_sci_block_layout<Ti>(hb, -1, ho, hna, hnb, hasy, hbsy, has, hbs, haf, hbf, ni, nb);
    db->num_blocks = nb;
    db->num_irreps = ni;
    db->dim = hb->dim;
    db->host_block_num_a = hna;
    db->host_block_num_b = hnb;
    db->block_offsets = up(ho.data(), nb);
    db->block_num_a = up(hna.data(), nb);
    db->block_num_b = up(hnb.data(), nb);
    db->block_asym = up(hasy.data(), nb);
    db->block_bsym = up(hbsy.data(), nb);
    db->astrs_flat = up(haf.data(), (int64)haf.size());
    db->bstrs_flat = up(hbf.data(), (int64)hbf.size());
    db->astrs_start = up(has.data(), nb);
    db->bstrs_start = up(hbs.data(), nb);
    std::vector<int> hbm(ni * ni, -1);
    for (int i = 0; i < nb; ++i)
        hbm[hasy[i] * ni + hbsy[i]] = i;
    db->block_map = up(hbm.data(), ni * ni);
    build_sci_gpu_dict<Ti>(hb, const_cast<Ti *&>(db->dict_a_keys), db->dict_a_vals,
                           const_cast<Ti *&>(db->dict_b_keys), db->dict_b_vals, db->dict_capacity);
    return db;
}

template <typename Ti, typename Tv>
static SciBasisViewDev<Ti> *upload_sci_tgt_block(
    const SciBasisManager<Ti> *hb, int64 block_idx,
    const bool *is_new_a, const bool *is_new_b, const Tv *host_diags)
{
    auto *db = new SciBasisViewDev<Ti>();
    int ni, nb;
    std::vector<int64> ho, has, hbs;
    std::vector<int> hna, hnb, hasy, hbsy;
    std::vector<Ti> haf, hbf;
    build_sci_block_layout<Ti>(hb, (int)block_idx, ho, hna, hnb, hasy, hbsy, has, hbs, haf, hbf, ni, nb);
    const auto &blk = hb->blocks[block_idx];
    db->num_blocks = nb;
    db->num_irreps = ni;
    db->dim = (int64)blk.num_a * blk.num_b;
    db->host_block_num_a = hna;
    db->host_block_num_b = hnb;
    db->block_offsets = up(ho.data(), nb);
    db->block_num_a = up(hna.data(), nb);
    db->block_num_b = up(hnb.data(), nb);
    db->block_asym = up(hasy.data(), nb);
    db->block_bsym = up(hbsy.data(), nb);
    db->astrs_flat = up(haf.data(), (int64)haf.size());
    db->bstrs_flat = up(hbf.data(), (int64)hbf.size());
    db->astrs_start = up(has.data(), nb);
    db->bstrs_start = up(hbs.data(), nb);
    std::vector<int> hbm(ni * ni, -1);
    hbm[hasy[0] * ni + hbsy[0]] = 0;
    db->block_map = up(hbm.data(), ni * ni);
    build_sci_gpu_dict<Ti>(hb, const_cast<Ti *&>(db->dict_a_keys), db->dict_a_vals,
                           const_cast<Ti *&>(db->dict_b_keys), db->dict_b_vals, db->dict_capacity);
    if (is_new_a)
    {
        int64 o = blk.astrs - hb->all_astrs;
        db->is_new_a = up(is_new_a + o, blk.num_a);
    }
    if (is_new_b)
    {
        int64 o = blk.bstrs - hb->all_bstrs;
        db->is_new_b = up(is_new_b + o, blk.num_b);
    }
    if (host_diags)
        db->candidate_diags = (const void *)up(host_diags + blk.offset, blk.num_a * blk.num_b);
    return db;
}

template <typename Ti, typename Tv>
static void make_sci_slice(const SciBasisViewDev<Ti> *b, SciBasisSliceDev<Ti, Tv> &s)
{
    s.num_blocks = b->num_blocks;
    s.num_irreps = b->num_irreps;
    s.dim = b->dim;
    s.block_offsets = b->block_offsets;
    s.block_num_a = b->block_num_a;
    s.block_num_b = b->block_num_b;
    s.block_asym = b->block_asym;
    s.block_bsym = b->block_bsym;
    s.astrs_flat = b->astrs_flat;
    s.bstrs_flat = b->bstrs_flat;
    s.astrs_start = b->astrs_start;
    s.bstrs_start = b->bstrs_start;
    s.block_map = b->block_map;
    s.dict_a_keys = b->dict_a_keys;
    s.dict_a_vals = b->dict_a_vals;
    s.dict_b_keys = b->dict_b_keys;
    s.dict_b_vals = b->dict_b_vals;
    s.dict_capacity = b->dict_capacity;
}

// ─── Kernel (copy of hvec_gather_offdiag_kernel, 2 changes) ─────

template <int Rank, int TypeCode, typename Ti, typename Tv>
__global__ void sci_hvec_gather_offdiag_kernel(
    const BasisSliceDev<Ti> basis,
    const SciBasisSliceDev<Ti, Tv> src_basis,
    const GroupsSliceDev<Ti, Tv> groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int bid = basis.target_bids ? basis.target_bids[blockIdx.x] : blockIdx.x;
    const int total_groups = groups.num_groups;

    constexpr int SHARED_MEM_SIZE = BATCH_GROUP_SHARED_MEM<Rank>;
    constexpr int BATCH_SIZE = BATCH_GROUP_SIZE<Rank>;
    constexpr int IDX_MEM_SIZE = BATCH_GROUP_IDX_MEM<Rank>; // always use full for SCI

    __shared__ Tv sh_pb[SHARED_MEM_SIZE];
    __shared__ int sh_sa_b[IDX_MEM_SIZE];
    __shared__ int sh_src_bid[BATCH_SIZE];
    __shared__ int sh_valid_b[BATCH_SIZE];

    const int num_chunks = (total_groups + BATCH_SIZE - 1) / BATCH_SIZE;
    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int asym = basis.block_asym[bid];
    const int bsym = basis.block_bsym[bid];
    const int nirp = basis.num_irreps;

    const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
    const int num_a_tiles = (n_a + TILE_A - 1) / TILE_A;
    const int total_tiles = num_b_tiles * num_a_tiles;

    const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
    const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];
    Tv *dst_vec_bid = dst_vec + basis.block_offsets[bid];

    for (int task_idx = blockIdx.y; task_idx < total_tiles; task_idx += gridDim.y)
    {
        const int b_tile_idx = task_idx % num_b_tiles;
        const int a_tile_idx = task_idx / num_b_tiles;
        const int b_tile_start = b_tile_idx * TILE_B;
        const int current_tile_b = min(TILE_B, n_b - b_tile_start);
        const Ti *bstrs_tile_start = bstrs + b_tile_start;

        const int a_tile_start = a_tile_idx * TILE_A;
        const int a_tile_end = min(n_a, a_tile_start + TILE_A);

        const int a = a_tile_start + threadIdx.x;
        const bool valid_a = (a < a_tile_end);
        const Ti astr = valid_a ? astrs[a] : 0;
        Tv *dst_base = valid_a ? (dst_vec_bid + (int64)a * n_b) : nullptr;

        Tv accum[TILE_B] = {};

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);

            for (int g_offset = threadIdx.x; g_offset < current_chunk_groups; g_offset += blockDim.x)
            {
                const int g = chunk_start_g + g_offset;
                const int h = compute_sym_hash<TypeCode>(asym, bsym, groups.asyms[g], groups.bsyms[g], nirp);
                const int src_bid = src_basis.block_map[h];
                sh_src_bid[g_offset] = src_bid;
                sh_valid_b[g_offset] = (src_bid == -1) ? 0 : n_b;
            }
            __syncthreads();

            const int total_sh_elements = current_chunk_groups * current_tile_b;
            for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
            {
                const int g_offset = sh_idx / current_tile_b;
                const int b_offset = sh_idx % current_tile_b;

                if (sh_valid_b[g_offset] == 0)
                    continue;

                const int g = chunk_start_g + g_offset;
                const Ti dst_b_str = bstrs_tile_start[b_offset];
                const Ti sas_b = dst_b_str ^ groups.bxs[g];
                const int sh_flat_offset = g_offset * TILE_B + b_offset;

                // CHANGE 2: b_idx_map → dict lookup
                sh_sa_b[sh_flat_offset] = cuda_dict::lookup_device(
                    src_basis.dict_b_keys, src_basis.dict_b_vals,
                    sas_b, src_basis.dict_capacity, -1);

                const Ti *zbs = groups.flat_zbs + groups.zb_start[g];
                const Tv *wb = groups.flat_wb + groups.wb_start[g];
                const int num_zb = groups.num_zbs[g];
                const int rank = groups.ranks[g];
                Tv *sh_pb_ptr = sh_pb + (g_offset * TILE_B + b_offset);
                compute_phase_dev<Rank, Ti, Tv>(sas_b, zbs, num_zb, wb, sh_pb_ptr, BATCH_SIZE * TILE_B, rank);
            }
            __syncthreads();

            if (valid_a)
            {
                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    if (sh_valid_b[g_offset] == 0)
                        continue;

                    const int g = chunk_start_g + g_offset;
                    const Ti sas_a = astr ^ groups.axs[g];

                    // CHANGE 1: a_idx_map → dict lookup (no TypeCode optimizations)
                    const int sa = cuda_dict::lookup_device(
                        src_basis.dict_a_keys, src_basis.dict_a_vals,
                        sas_a, src_basis.dict_capacity, -1);

                    if (sa != -1)
                    {
                        const int nza = groups.num_zas[g];
                        const Ti *zas = groups.flat_zas + groups.za_start[g];
                        const Tv *wa = groups.flat_wa + groups.wa_start[g];
                        const int rank = groups.ranks[g];

                        constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
                        Tv pa[STACK_SIZE] = {};
                        compute_phase_dev<Rank, Ti, Tv>(sas_a, zas, nza, wa, pa, 1, rank);

                        const int sbi = sh_src_bid[g_offset];
                        const int src_n_b = src_basis.block_num_b[sbi];
                        const Tv *src_base = src_vec + src_basis.block_offsets[sbi] + (int64)sa * src_n_b;
                        const Tv *pb = sh_pb + (g_offset * TILE_B);
                        const int *sh_sa_b_task = sh_sa_b + g_offset * TILE_B;

                        if (current_tile_b == TILE_B)
                        {
#pragma unroll
                            for (int b_offset = 0; b_offset < TILE_B; ++b_offset)
                            {
                                const int sa_b = sh_sa_b_task[b_offset];
                                if (sa_b != -1)
                                {
                                    const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                                    accum[b_offset] += __ldg(&src_base[sa_b]) * vt;
                                }
                            }
                        }
                        else
                        {
                            for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                            {
                                const int sa_b = sh_sa_b_task[b_offset];
                                if (sa_b != -1)
                                {
                                    const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                                    accum[b_offset] += __ldg(&src_base[sa_b]) * vt;
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (valid_a)
        {
            for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
            {
                const int actual_b_idx = b_tile_start + b_offset;
                dst_base[actual_b_idx] += accum[b_offset];
            }
        }
    }
}

// ─── Dispatch (copy of dispatch_chunks_by_rank_gpu, SCI kernel) ──

template <int TypeCode, typename Ti, typename Tv>
static inline void sci_dispatch_chunks_by_rank_gpu(
    const BasisSliceDev<Ti> &basis_slice, int num_active_blocks,
    const SciBasisSliceDev<Ti, Tv> &src_slice,
    const GroupsViewDev<Ti, Tv> &groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int64 total_ngs = groups.num_groups;
    if (total_ngs == 0)
        return;

    int64 start = 0;
    while (start < total_ngs)
    {
        const int current_rank = groups.host_ranks[start];
        const int dispatch_rank = (current_rank == 1 || current_rank == 2) ? current_rank : 0;

        int64 end = start + 1;
        while (end < total_ngs)
        {
            const int next_rank = groups.host_ranks[end];
            if (((next_rank == 1 || next_rank == 2) ? next_rank : 0) != dispatch_rank)
                break;
            end++;
        }

        const int64 chunk_size = end - start;
        GroupsSliceDev<Ti, Tv> slice;
        slice.num_groups = chunk_size;
        slice.axs = groups.axs + start;
        slice.bxs = groups.bxs + start;
        slice.asyms = groups.asyms + start;
        slice.bsyms = groups.bsyms + start;
        slice.ranks = groups.ranks + start;
        slice.num_zas = groups.num_zas + start;
        slice.num_zbs = groups.num_zbs + start;
        slice.za_start = groups.za_start + start;
        slice.zb_start = groups.zb_start + start;
        slice.wa_start = groups.wa_start + start;
        slice.wb_start = groups.wb_start + start;
        slice.flat_zas = groups.flat_zas;
        slice.flat_zbs = groups.flat_zbs;
        slice.flat_wa = groups.flat_wa;
        slice.flat_wb = groups.flat_wb;
        slice.original_idx = groups.original_idx + start;

        int block_size = 256;
        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        dim3 grid_size(num_active_blocks, num_sms * 4);

        switch (dispatch_rank)
        {
        case 1:
            sci_hvec_gather_offdiag_kernel<1, TypeCode, Ti, Tv><<<grid_size, block_size>>>(basis_slice, src_slice, slice, src_vec, dst_vec);
            break;
        case 2:
            sci_hvec_gather_offdiag_kernel<2, TypeCode, Ti, Tv><<<grid_size, block_size>>>(basis_slice, src_slice, slice, src_vec, dst_vec);
            break;
        default:
            sci_hvec_gather_offdiag_kernel<0, TypeCode, Ti, Tv><<<grid_size, block_size>>>(basis_slice, src_slice, slice, src_vec, dst_vec);
            break;
        }
        CUDA_CHECK(cudaGetLastError());
        start = end;
    }
}

template <typename Ti, typename Tv>
void cuda_sci_hvec(
    const BasisSliceDev<Ti> &basis_slice,
    const SciBasisSliceDev<Ti, Tv> &src_slice,
    const NetworkDev<Ti, Tv> &net,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    cudaMemset(dst_vec, 0, basis_slice.dim * sizeof(Tv));

    sci_dispatch_chunks_by_rank_gpu<1>(basis_slice, basis_slice.num_blocks, src_slice, net.pure_a_groups, src_vec, dst_vec);
    sci_dispatch_chunks_by_rank_gpu<2>(basis_slice, basis_slice.num_blocks, src_slice, net.pure_b_groups, src_vec, dst_vec);
    sci_dispatch_chunks_by_rank_gpu<3>(basis_slice, basis_slice.num_blocks, src_slice, net.mixed_groups, src_vec, dst_vec);
}

// ─── Selection kernel ────────────────────────────────────────────

template <typename Ti, typename Tv>
__global__ void sci_select_entries_kernel(
    const BasisSliceDev<Ti> tgt_basis,
    const Tv *__restrict__ dst_vec,
    const void *__restrict__ candidate_diags,
    const bool *__restrict__ is_new_a, const bool *__restrict__ is_new_b,
    Tv e_var, double eps,
    Ti *__restrict__ out_a, Ti *__restrict__ out_b, Tv *__restrict__ out_v,
    int64 *__restrict__ out_count, int64 max_entries)
{
    const int bid = blockIdx.x;
    if (tgt_basis.target_bids)
        return;
    const int n_a = tgt_basis.block_num_a[bid];
    const int n_b = tgt_basis.block_num_b[bid];
    const int64 offset = tgt_basis.block_offsets[bid];
    const Ti *astrs = tgt_basis.astrs_flat + tgt_basis.astrs_start[bid];
    const Ti *bstrs = tgt_basis.bstrs_flat + tgt_basis.bstrs_start[bid];
    const int64 astr_start = tgt_basis.astrs_start[bid];
    const int64 bstr_start = tgt_basis.bstrs_start[bid];
    const Tv *block_dst = dst_vec + offset;
    const Tv *block_diags = ((const Tv *)candidate_diags) ? (((const Tv *)candidate_diags) + offset) : nullptr;
    const double eps_sq = eps * eps;

    for (int idx = threadIdx.x + blockIdx.y * blockDim.x; idx < n_a * n_b; idx += blockDim.x * gridDim.y)
    {
        const int a = idx / n_b, b = idx % n_b;
        const Tv acc = block_dst[(int64)a * n_b + b];
        if (acc == Tv{})
            continue;
        if (!is_new_a[astr_start + a] && !is_new_b[bstr_start + b])
            continue;
        const Tv haa = block_diags[(int64)a * n_b + b];
        const Tv denom = e_var - haa;
        double dn_sq;
        if constexpr (std::is_arithmetic_v<Tv>)
            dn_sq = (double)(denom * denom);
        else
            dn_sq = (double)(denom.real() * denom.real() + denom.imag() * denom.imag());
        if (dn_sq == 0.0)
            continue;
        double a_sq;
        if constexpr (std::is_arithmetic_v<Tv>)
            a_sq = (double)(acc * acc);
        else
            a_sq = (double)(acc.real() * acc.real() + acc.imag() * acc.imag());
        if (a_sq / dn_sq <= eps_sq)
            continue;
        const int64 pos = atomicAdd((unsigned long long *)out_count, 1ull);
        if (pos < max_entries)
        {
            out_a[pos] = astrs[a];
            out_b[pos] = bstrs[b];
            out_v[pos] = acc;
        }
    }
}

// ─── Top-level ───────────────────────────────────────────────────

template <typename Ti, typename Tv>
int64 cuda_sci_select_external_block(
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisViewDev<Ti> *src_dev,
    const NetworkDev<Ti, Tv> *net_dev,
    const bool *is_new_a, const bool *is_new_b,
    int64 blk_idx,
    const Tv *host_sv,
    const Tv *host_diags,
    Tv e_var, double eps,
    Ti *ho_a, Ti *ho_b, Tv *ho_v,
    int64 max_entries)
{
    if (blk_idx < 0 || blk_idx >= tgt_basis->num_blocks)
        return 0;
    const auto &blk = tgt_basis->blocks[blk_idx];
    if (blk.num_a <= 0 || blk.num_b <= 0)
        return 0;

    auto *tgt_dev = upload_sci_tgt_block<Ti, Tv>(tgt_basis, blk_idx, is_new_a, is_new_b, host_diags);

    Tv *sv_dev = nullptr;
    {
        std::vector<Tv> svh(src_dev->dim);
        std::copy_n(host_sv, src_dev->dim, svh.data());
        sv_dev = up(svh.data(), src_dev->dim);
    }

    int64 bdim = blk.num_a * blk.num_b;
    Tv *dv_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dv_dev, bdim * sizeof(Tv)));

    BasisSliceDev<Ti> ts;
    ts.num_blocks = 1;
    ts.num_irreps = tgt_dev->num_irreps;
    ts.dim = bdim;
    ts.block_offsets = tgt_dev->block_offsets;
    ts.block_num_a = tgt_dev->block_num_a;
    ts.block_num_b = tgt_dev->block_num_b;
    ts.block_asym = tgt_dev->block_asym;
    ts.block_bsym = tgt_dev->block_bsym;
    ts.astrs_flat = tgt_dev->astrs_flat;
    ts.bstrs_flat = tgt_dev->bstrs_flat;
    ts.astrs_start = tgt_dev->astrs_start;
    ts.bstrs_start = tgt_dev->bstrs_start;
    ts.block_map = tgt_dev->block_map;
    ts.target_bids = nullptr;

    SciBasisSliceDev<Ti, Tv> ss;
    make_sci_slice<Ti, Tv>(src_dev, ss);

    cuda_sci_hvec(ts, ss, *net_dev, sv_dev, dv_dev);
    CUDA_CHECK(cudaDeviceSynchronize());

    int64 *oc_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&oc_dev, sizeof(int64)));
    CUDA_CHECK(cudaMemset(oc_dev, 0, sizeof(int64)));
    Ti *oa = nullptr, *ob = nullptr;
    Tv *ov = nullptr;
    CUDA_CHECK(cudaMalloc(&oa, max_entries * sizeof(Ti)));
    CUDA_CHECK(cudaMalloc(&ob, max_entries * sizeof(Ti)));
    CUDA_CHECK(cudaMalloc(&ov, max_entries * sizeof(Tv)));

    sci_select_entries_kernel<Ti, Tv><<<dim3(1, 256), 256>>>(
        ts, dv_dev, tgt_dev->candidate_diags,
        tgt_dev->is_new_a, tgt_dev->is_new_b,
        e_var, eps, oa, ob, ov, oc_dev, max_entries);
    CUDA_CHECK(cudaDeviceSynchronize());

    int64 oc = 0;
    CUDA_CHECK(cudaMemcpy(&oc, oc_dev, sizeof(int64), cudaMemcpyDeviceToHost));
    oc = std::min(oc, max_entries);
    if (oc > 0)
    {
        CUDA_CHECK(cudaMemcpy(ho_a, oa, oc * sizeof(Ti), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ho_b, ob, oc * sizeof(Ti), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ho_v, ov, oc * sizeof(Tv), cudaMemcpyDeviceToHost));
    }

    cudaFree(dv_dev);
    cudaFree(oc_dev);
    cudaFree(oa);
    cudaFree(ob);
    cudaFree(ov);
    cudaFree(sv_dev);
    delete tgt_dev;
    return oc;
}

template <typename Ti, typename Tv>
void cuda_hvec_sci_full(
    const BasisViewDev<Ti> *basis_dev,
    const NetworkDev<Ti, Tv> *net_dev,
    int64 basis_dim,
    const Tv *host_src_vec,
    Tv *host_dst_vec)
{
    const Tv *src_dev = up(host_src_vec, basis_dim);
    Tv *dst_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dst_dev, basis_dim * sizeof(Tv)));
    CUDA_CHECK(cudaMemset(dst_dev, 0, basis_dim * sizeof(Tv)));
    cuda_hvec<Ti, Tv>(*basis_dev, *net_dev, src_dev, dst_dev);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_dst_vec, dst_dev, basis_dim * sizeof(Tv), cudaMemcpyDeviceToHost));
    cudaFree(const_cast<Tv *>(src_dev));
    cudaFree(dst_dev);
}

template <typename Ti>
static void destroy_sci_src_dev(SciBasisViewDev<Ti> *dev)
{
    if (dev)
    {
        dev->clear();
        delete dev;
    }
}
