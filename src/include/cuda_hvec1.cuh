#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <utility>
#include "otf.hpp"

inline constexpr int TILE_A = 256;
inline constexpr int TILE_B = 32;
inline constexpr int BATCH_SIZE_SH1 = 64;
inline constexpr int BATCH_SIZE_SH2 = 32;
inline constexpr int BATCH_SIZE_SH3 = 1;
inline constexpr int KERNEL_MAX_RANK = 128;

__device__ __forceinline__ int count_ones(uint32 v) { return __popc(v); }
__device__ __forceinline__ int count_ones(uint64 v) { return __popcll(v); }

template <typename T>
__device__ __forceinline__ T dev_conj(const T &x)
{
    if constexpr (std::is_arithmetic_v<T>)
        return x;
    else
        return T(x.real(), -x.imag());
}

void check_cuda(cudaError_t err, const char *f, int l)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", f, l, cudaGetErrorString(err));
        exit(1);
    }
}

#define CUDA_CHECK(err) check_cuda(err, __FILE__, __LINE__)

template <typename T>
static const T *up(const T *h, int64 n)
{
    T *dev_ptr = nullptr;
    if (n > 0 && h != nullptr)
    {
        CUDA_CHECK(cudaMalloc(&dev_ptr, n * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(dev_ptr, h, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    return dev_ptr;
}

// 1. 显存物理生命周期管理器（禁止拷贝，支持移动）
template <typename Ti>
struct BasisViewDev
{
    int num_blocks = 0;
    int num_irreps = 0;
    int max_a_count = 0;
    int max_b_count = 0;
    int64 dim = 0;
    const int64 *block_offsets = nullptr; // [num_blocks]
    const int *block_num_a = nullptr;     // [num_blocks]
    const int *block_num_b = nullptr;     // [num_blocks]
    const int *block_asym = nullptr;      // [num_blocks]
    const int *block_bsym = nullptr;      // [num_blocks]
    const Ti *astrs_flat = nullptr;       // [total_astrs]
    const Ti *bstrs_flat = nullptr;       // [total_bstrs]
    const int64 *astrs_start = nullptr;   // [num_blocks]
    const int64 *bstrs_start = nullptr;   // [num_blocks]
    const int *block_map = nullptr;       // [num_irreps * num_irreps]
    const int *astr2idx = nullptr;
    const int *bstr2idx = nullptr;

    BasisViewDev() = default;
    BasisViewDev(const BasisViewDev &) = delete;
    BasisViewDev &operator=(const BasisViewDev &) = delete;
    BasisViewDev(BasisViewDev &&other) noexcept
    {
        *this = std::move(other);
    }

    void clear()
    {
        if (block_offsets) { cudaFree(const_cast<int64 *>(block_offsets)); block_offsets = nullptr; }
        if (block_num_a) { cudaFree(const_cast<int *>(block_num_a)); block_num_a = nullptr; }
        if (block_num_b) { cudaFree(const_cast<int *>(block_num_b)); block_num_b = nullptr; }
        if (block_asym) { cudaFree(const_cast<int *>(block_asym)); block_asym = nullptr; }
        if (block_bsym) { cudaFree(const_cast<int *>(block_bsym)); block_bsym = nullptr; }
        if (astrs_flat) { cudaFree(const_cast<Ti *>(astrs_flat)); astrs_flat = nullptr; }
        if (bstrs_flat) { cudaFree(const_cast<Ti *>(bstrs_flat)); bstrs_flat = nullptr; }
        if (astrs_start) { cudaFree(const_cast<int64 *>(astrs_start)); astrs_start = nullptr; }
        if (bstrs_start) { cudaFree(const_cast<int64 *>(bstrs_start)); bstrs_start = nullptr; }
        if (block_map) { cudaFree(const_cast<int *>(block_map)); block_map = nullptr; }
        if (astr2idx) { cudaFree(const_cast<int *>(astr2idx)); astr2idx = nullptr; }
        if (bstr2idx) { cudaFree(const_cast<int *>(bstr2idx)); bstr2idx = nullptr; }
    }

    ~BasisViewDev() { clear(); }

    BasisViewDev &operator=(BasisViewDev &&other) noexcept
    {
        if (this != &other)
        {
            clear();
            num_blocks = other.num_blocks;
            num_irreps = other.num_irreps;
            max_a_count = other.max_a_count;
            max_b_count = other.max_b_count;
            dim = other.dim;
            block_offsets = other.block_offsets;
            block_num_a = other.block_num_a;
            block_num_b = other.block_num_b;
            block_asym = other.block_asym;
            block_bsym = other.block_bsym;
            astrs_flat = other.astrs_flat;
            bstrs_flat = other.bstrs_flat;
            astrs_start = other.astrs_start;
            bstrs_start = other.bstrs_start;
            block_map = other.block_map;
            astr2idx = other.astr2idx;
            bstr2idx = other.bstr2idx;

            other.num_blocks = 0; other.num_irreps = 0; other.max_a_count = 0; other.max_b_count = 0; other.dim = 0;
            other.block_offsets = nullptr; other.block_num_a = nullptr; other.block_num_b = nullptr;
            other.block_asym = nullptr; other.block_bsym = nullptr; other.astrs_flat = nullptr;
            other.bstrs_flat = nullptr; other.astrs_start = nullptr; other.bstrs_start = nullptr;
            other.block_map = nullptr; other.astr2idx = nullptr; other.bstr2idx = nullptr;
        }
        return *this;
    }
};

// 2. 新增的轻量级物理切片体（无析构函数，专供内核按值传递避免参数拷贝拦截）
template <typename Ti>
struct BasisSliceDev
{
    int num_blocks = 0;
    int num_irreps = 0;
    int max_a_count = 0;
    int max_b_count = 0;
    int64 dim = 0;
    const int64 *block_offsets = nullptr;
    const int *block_num_a = nullptr;
    const int *block_num_b = nullptr;
    const int *block_asym = nullptr;
    const int *block_bsym = nullptr;
    const Ti *astrs_flat = nullptr;
    const Ti *bstrs_flat = nullptr;
    const int64 *astrs_start = nullptr;
    const int64 *bstrs_start = nullptr;
    const int *block_map = nullptr;
    const int *astr2idx = nullptr;
    const int *bstr2idx = nullptr;
};

template <typename Ti, typename Tv>
struct GroupsViewDev
{
    int num_groups = 0;
    const Ti *axs = nullptr;         
    const Ti *bxs = nullptr;         
    const int *asyms = nullptr;      
    const int *bsyms = nullptr;      
    const int *ranks = nullptr;      
    const int *num_zas = nullptr;    
    const int *num_zbs = nullptr;    
    const Ti *flat_zas = nullptr;    
    const Ti *flat_zbs = nullptr;    
    const Tv *flat_wa = nullptr;     
    const Tv *flat_wb = nullptr;     
    const int64 *za_start = nullptr; 
    const int64 *zb_start = nullptr; 
    const int64 *wa_start = nullptr; 
    const int64 *wb_start = nullptr; 
    const int *original_idx = nullptr;
    const int *excit_types = nullptr;
    std::vector<int> host_ranks;

    GroupsViewDev() = default;
    GroupsViewDev(const GroupsViewDev &) = delete;
    GroupsViewDev &operator=(const GroupsViewDev &) = delete;
    GroupsViewDev(GroupsViewDev &&other) noexcept
    {
        *this = std::move(other);
    }

    ~GroupsViewDev() { clear(); }

    void clear()
    {
        if (axs) { cudaFree(const_cast<Ti *>(axs)); axs = nullptr; }
        if (bxs) { cudaFree(const_cast<Ti *>(bxs)); bxs = nullptr; }
        if (asyms) { cudaFree(const_cast<int *>(asyms)); asyms = nullptr; }
        if (bsyms) { cudaFree(const_cast<int *>(bsyms)); bsyms = nullptr; }
        if (ranks) { cudaFree(const_cast<int *>(ranks)); ranks = nullptr; }
        if (num_zas) { cudaFree(const_cast<int *>(num_zas)); num_zas = nullptr; }
        if (num_zbs) { cudaFree(const_cast<int *>(num_zbs)); num_zbs = nullptr; }
        if (flat_zas) { cudaFree(const_cast<Ti *>(flat_zas)); flat_zas = nullptr; }
        if (flat_zbs) { cudaFree(const_cast<Ti *>(flat_zbs)); flat_zbs = nullptr; }
        if (flat_wa) { cudaFree(const_cast<Tv *>(flat_wa)); flat_wa = nullptr; }
        if (flat_wb) { cudaFree(const_cast<Tv *>(flat_wb)); flat_wb = nullptr; }
        if (za_start) { cudaFree(const_cast<int64 *>(za_start)); za_start = nullptr; }
        if (zb_start) { cudaFree(const_cast<int64 *>(zb_start)); zb_start = nullptr; }
        if (wa_start) { cudaFree(const_cast<int64 *>(wa_start)); wa_start = nullptr; }
        if (wb_start) { cudaFree(const_cast<int64 *>(wb_start)); wb_start = nullptr; }
        if (original_idx) { cudaFree(const_cast<int *>(original_idx)); original_idx = nullptr; }
        if (excit_types) { cudaFree(const_cast<int *>(excit_types)); excit_types = nullptr; }

        host_ranks.clear();
        num_groups = 0;
    }

    GroupsViewDev &operator=(GroupsViewDev &&other) noexcept
    {
        if (this != &other)
        {
            clear();
            axs = other.axs; bxs = other.bxs; asyms = other.asyms; bsyms = other.bsyms; ranks = other.ranks;
            num_zas = other.num_zas; num_zbs = other.num_zbs; flat_zas = other.flat_zas; flat_zbs = other.flat_zbs;
            flat_wa = other.flat_wa; flat_wb = other.flat_wb; za_start = other.za_start; zb_start = other.zb_start;
            wa_start = other.wa_start; wb_start = other.wb_start; original_idx = other.original_idx;
            excit_types = other.excit_types; num_groups = other.num_groups; host_ranks = std::move(other.host_ranks);

            other.axs = nullptr; other.bxs = nullptr; other.asyms = nullptr; other.bsyms = nullptr; other.ranks = nullptr;
            other.num_zas = nullptr; other.num_zbs = nullptr; other.flat_zas = nullptr; other.flat_zbs = nullptr;
            other.flat_wa = nullptr; other.flat_wb = nullptr; other.za_start = nullptr; other.zb_start = nullptr;
            other.wa_start = nullptr; other.wb_start = nullptr; other.original_idx = nullptr; other.excit_types = nullptr;
            other.num_groups = 0;
        }
        return *this;
    }
};

template <typename Ti, typename Tv>
struct GroupsSliceDev
{
    int num_groups;
    const Ti *axs, *bxs;
    const int *asyms, *bsyms, *ranks, *num_zas, *num_zbs;
    const Ti *flat_zas, *flat_zbs;
    const Tv *flat_wa, *flat_wb;
    const int64 *za_start, *zb_start, *wa_start, *wb_start;
};

template <typename Ti, typename Tv>
struct NetworkDev
{
    GroupsViewDev<Ti, Tv> diag_groups = {};
    GroupsViewDev<Ti, Tv> pure_a_groups = {};
    GroupsViewDev<Ti, Tv> pure_b_groups = {};
    GroupsViewDev<Ti, Tv> mixed_groups = {};

    NetworkDev(const NetworkDev &) = delete;
    NetworkDev &operator=(const NetworkDev &) = delete;
    NetworkDev(NetworkDev &&) noexcept = default;
    NetworkDev &operator=(NetworkDev &&) noexcept = default;

    NetworkDev() = default;
    ~NetworkDev() = default;

    void clear() noexcept
    {
        diag_groups.clear();
        pure_a_groups.clear();
        pure_b_groups.clear();
        mixed_groups.clear();
    }
};

template <typename Ti>
void *upload_basis(const BasisManager<Ti> *hb)
{
    BasisViewDev<Ti> *db = new BasisViewDev<Ti>();

    int nb = hb->num_blocks, ni = hb->num_irreps;

    db->num_blocks = nb;
    db->num_irreps = ni;
    db->dim = hb->dim;
    db->max_a_count = (int)hb->max_a_count;
    db->max_b_count = (int)hb->max_b_count;

    std::vector<int64> ho(nb), has(nb), hbs(nb);
    std::vector<int> hna(nb), hnb(nb), hasy(nb), hbsy(nb);
    int64 ta = 0, tb = 0;
    for (int i = 0; i < nb; ++i)
    {
        ho[i] = hb->blocks[i].offset;
        hna[i] = (int)hb->blocks[i].num_a;
        hnb[i] = (int)hb->blocks[i].num_b;
        hasy[i] = (int)hb->blocks[i].asym;
        hbsy[i] = (int)hb->blocks[i].bsym;
        has[i] = ta;
        hbs[i] = tb;
        ta += hna[i];
        tb += hnb[i];
    }

    std::vector<Ti> haf(ta), hbf(tb);
    for (int i = 0; i < nb; ++i)
    {
        for (int j = 0; j < hb->blocks[i].num_a; ++j)
            haf[has[i] + j] = hb->blocks[i].astrs[j];
        for (int j = 0; j < hb->blocks[i].num_b; ++j)
            hbf[hbs[i] + j] = hb->blocks[i].bstrs[j];
    }

    std::vector<int> hbm(ni * ni);
    for (int i = 0; i < ni * ni; i++)
        hbm[i] = (int)hb->block_map[i];

    int ms = 1 << hb->norb;

    db->block_offsets = up(ho.data(), nb);
    db->block_num_a = up(hna.data(), nb);
    db->block_num_b = up(hnb.data(), nb);
    db->block_asym = up(hasy.data(), nb);
    db->block_bsym = up(hbsy.data(), nb);
    db->astrs_flat = up(haf.data(), ta);
    db->bstrs_flat = up(hbf.data(), tb);
    db->astrs_start = up(has.data(), nb);
    db->bstrs_start = up(hbs.data(), nb);
    db->block_map = up(hbm.data(), ni * ni);
    db->astr2idx = up(hb->a_idx_map, ms);
    db->bstr2idx = up(hb->b_idx_map, ms);

    return static_cast<void *>(db);
}

template <typename Ti, typename Tv>
void *upload_network(const Network_OTF<Ti, Tv> *hn)
{
    NetworkDev<Ti, Tv> *dn = new NetworkDev<Ti, Tv>();

    auto flatten_bucket = [&](const auto &src_bucket, GroupsViewDev<Ti, Tv> &dst_view)
    {
        int count = src_bucket.size();
        dst_view.num_groups = count;
        if (count == 0)
            return;

        size_t total_fza = 0; size_t total_fzb = 0; size_t total_fwa = 0; size_t total_fwb = 0;
        for (int i = 0; i < count; ++i)
        {
            const auto &g = src_bucket[i];
            total_fza += g.num_za; total_fzb += g.num_zb;
            total_fwa += (size_t)g.num_za * g.rank; total_fwb += (size_t)g.num_zb * g.rank;
        }

        std::vector<Ti> axs(count), bxs(count);
        std::vector<int> asyms(count), bsyms(count);
        std::vector<int> ranks(count), nza(count), nzb(count);
        std::vector<int64> zas(count), zbs(count), was(count), wbs(count);
        std::vector<int> original_idxs(count);
        std::vector<int> excit_types(count);

        std::vector<Ti> fza(total_fza), fzb(total_fzb);
        std::vector<Tv> fwa(total_fwa), fwb(total_fwb);

        int64 zo = 0, zbo = 0, wo = 0, wbo = 0;
        size_t idx_fza = 0, idx_fzb = 0, idx_fwa = 0, idx_fwb = 0;

        for (int i = 0; i < count; ++i)
        {
            const auto &g = src_bucket[i];
            axs[i] = g.ax; bxs[i] = g.bx;
            asyms[i] = (int)g.asym; bsyms[i] = (int)g.bsym;
            ranks[i] = g.rank; nza[i] = g.num_za; nzb[i] = g.num_zb;
            zas[i] = zo; zbs[i] = zbo; was[i] = wo; wbs[i] = wbo;
            original_idxs[i] = (int)g.original_idx;
            excit_types[i] = (int)hn->excit_types[g.original_idx];

            if (g.num_za > 0) { std::copy_n(g.unique_zas, g.num_za, &fza[idx_fza]); idx_fza += g.num_za; zo += g.num_za; }
            if (g.num_zb > 0) { std::copy_n(g.unique_zbs, g.num_zb, &fzb[idx_fzb]); idx_fzb += g.num_zb; zbo += g.num_zb; }
            size_t size_wa = (size_t)g.num_za * g.rank;
            if (size_wa > 0) { std::copy_n(g.wa, size_wa, &fwa[idx_fwa]); idx_fwa += size_wa; wo += size_wa; }
            size_t size_wb = (size_t)g.num_zb * g.rank;
            if (size_wb > 0) { std::copy_n(g.wb, size_wb, &fwb[idx_fwb]); idx_fwb += size_wb; wbo += size_wb; }
        }

        auto upload_to_device = [](const auto &host_vec)
        {
            using T = typename std::decay_t<decltype(host_vec)>::value_type;
            T *dev_ptr = nullptr;
            if (!host_vec.empty())
            {
                CUDA_CHECK(cudaMalloc(&dev_ptr, host_vec.size() * sizeof(T)));
                CUDA_CHECK(cudaMemcpy(dev_ptr, host_vec.data(), host_vec.size() * sizeof(T), cudaMemcpyHostToDevice));
            }
            return const_cast<const T *>(dev_ptr);
        };

        dst_view.host_ranks = ranks;
        dst_view.axs = upload_to_device(axs); dst_view.bxs = upload_to_device(bxs);
        dst_view.asyms = upload_to_device(asyms); dst_view.bsyms = upload_to_device(bsyms);
        dst_view.ranks = upload_to_device(ranks); dst_view.num_zas = upload_to_device(nza);
        dst_view.num_zbs = upload_to_device(nzb); dst_view.za_start = upload_to_device(zas);
        dst_view.zb_start = upload_to_device(zbs); dst_view.wa_start = upload_to_device(was);
        dst_view.wb_start = upload_to_device(wbs); dst_view.flat_zas = upload_to_device(fza);
        dst_view.flat_zbs = upload_to_device(fzb); dst_view.flat_wa = upload_to_device(fwa);
        dst_view.flat_wb = upload_to_device(fwb);
        dst_view.original_idx = upload_to_device(original_idxs);
        dst_view.excit_types = upload_to_device(excit_types);
    };

    flatten_bucket(hn->diag_groups, dn->diag_groups);
    flatten_bucket(hn->pure_a_groups, dn->pure_a_groups);
    flatten_bucket(hn->pure_b_groups, dn->pure_b_groups);
    flatten_bucket(hn->mixed_groups, dn->mixed_groups);

    return static_cast<void *>(dn);
}

template <int Rank, typename Ti, typename Tv>
__device__ __forceinline__ void compute_phase_dev(
    Ti str,
    const Ti *__restrict__ zs, int num_zs,
    const Tv *__restrict__ w0, Tv *p0, int stride, int rank)
{
    if constexpr (Rank == 1)
    {
        Tv v0 = {};
        for (int k = 0; k < num_zs; ++k)
        {
            bool parity = count_ones(str & zs[k]) & 1;
            v0 += parity ? -w0[k] : w0[k];
        }
        p0[0] = v0;
    }
    else if constexpr (Rank == 2)
    {
        const Tv *w1 = w0 + num_zs;
        Tv v0 = {}; Tv v1 = {};
        for (int k = 0; k < num_zs; ++k)
        {
            bool parity = count_ones(str & zs[k]) & 1;
            v0 += parity ? -w0[k] : w0[k];
            v1 += parity ? -w1[k] : w1[k];
        }
        p0[0] = v0;
        p0[stride] = v1;
    }
    else
    {
        for (int r = 0; r < rank; ++r)
        {
            const Tv *wr = w0 + r * num_zs;
            Tv vr = {};
            for (int k = 0; k < num_zs; ++k)
            {
                bool parity = count_ones(str & zs[k]) & 1;
                vr += parity ? -wr[k] : wr[k];
            }
            p0[r * stride] = vr;
        }
    }
}

template <int Rank, typename Tv>
__device__ __forceinline__ Tv compute_coeff_dev(
    const Tv *__restrict__ pa,
    const Tv *__restrict__ pb, int stride, int rank, int b)
{
    if constexpr (Rank == 1)
        return pa[0] * pb[b];
    else if constexpr (Rank == 2)
        return pa[0] * pb[b] + pa[1] * pb[stride + b];
    else
    {
        Tv vt = {};
        for (int r = 0; r < rank; ++r)
            vt += pa[r] * pb[r * stride + b];
        return vt;
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void hvec_gather_diag_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int bid = blockIdx.x;
    const int total_groups = groups.num_groups;

    constexpr int SHARED_MEM_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1 * TILE_B
        : Rank == 2 ? BATCH_SIZE_SH2 * TILE_B * 2
                    : BATCH_SIZE_SH3 * TILE_B * KERNEL_MAX_RANK;

    constexpr int BATCH_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1
        : Rank == 2 ? BATCH_SIZE_SH2
                    : BATCH_SIZE_SH3;

    __shared__ Tv sh_pb[SHARED_MEM_SIZE];

    const int num_chunks = (total_groups + BATCH_SIZE - 1) / BATCH_SIZE;
    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
    const int num_a_tiles = (n_a + TILE_A - 1) / TILE_A;
    const int total_tiles = num_b_tiles * num_a_tiles;
    const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
    const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];
    const Tv *src_vec_bid = src_vec + basis.block_offsets[bid];
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
        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);
            const int total_sh_elements = current_chunk_groups * current_tile_b;
            for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
            {
                const int g_offset = sh_idx / current_tile_b;
                const int b_offset = sh_idx % current_tile_b;
                const int g = chunk_start_g + g_offset;
                const Ti bstr = bstrs_tile_start[b_offset];
                const Ti *zbs = groups.flat_zbs + groups.zb_start[g];
                const Tv *wb = groups.flat_wb + groups.wb_start[g];
                const int num_zb = groups.num_zbs[g];
                const int rank = groups.ranks[g];
                Tv *sh_pb_ptr = sh_pb + (g_offset * TILE_B + b_offset);
                compute_phase_dev<Rank, Ti, Tv>(bstr, zbs, num_zb, wb, sh_pb_ptr, BATCH_SIZE * TILE_B, rank);
            }

            __syncthreads();

            for (int a = a_tile_start + threadIdx.x; a < a_tile_end; a += blockDim.x)
            {
                const Ti astr = astrs[a];
                const Tv *src_base = src_vec_bid + (int64)a * n_b;
                Tv *dst_base = dst_vec_bid + (int64)a * n_b;
                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    const int g = chunk_start_g + g_offset;
                    const int nza = groups.num_zas[g];
                    const Ti *zas = groups.flat_zas + groups.za_start[g];
                    const Tv *wa = groups.flat_wa + groups.wa_start[g];
                    const int rank = groups.ranks[g];
                    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
                    Tv pa[STACK_SIZE] = {};
                    compute_phase_dev<Rank, Ti, Tv>(astr, zas, nza, wa, pa, 1, rank);
                    const Tv *pb = sh_pb + (g_offset * TILE_B);
                    for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                    {
                        const int actual_b_idx = b_tile_start + b_offset;
                        const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                        dst_base[actual_b_idx] += src_base[actual_b_idx] * vt;
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void hvec_gather_mixed_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int bid = blockIdx.x;
    const int total_groups = groups.num_groups;
    const int *a_idx_map = basis.astr2idx;
    const int *b_idx_map = basis.bstr2idx;

    constexpr int SHARED_MEM_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1 * TILE_B
        : Rank == 2 ? BATCH_SIZE_SH2 * TILE_B * 2
                    : BATCH_SIZE_SH3 * TILE_B * KERNEL_MAX_RANK;

    constexpr int BATCH_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1
        : Rank == 2 ? BATCH_SIZE_SH2
                    : BATCH_SIZE_SH3;

    constexpr int IDX_MEM_SIZE = BATCH_SIZE * TILE_B;

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

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);

            for (int g_offset = threadIdx.x; g_offset < current_chunk_groups; g_offset += blockDim.x)
            {
                const int g = chunk_start_g + g_offset;
                const int h = (asym ^ groups.asyms[g]) * nirp + (bsym ^ groups.bsyms[g]);
                const int src_bid = basis.block_map[h];
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
                const Ti bx = groups.bxs[g];
                const Ti sas_b = dst_b_str ^ bx;

                const int sh_flat_offset = g_offset * TILE_B + b_offset;
                sh_sa_b[sh_flat_offset] = b_idx_map[sas_b];

                const Ti *zbs = groups.flat_zbs + groups.zb_start[g];
                const Tv *wb = groups.flat_wb + groups.wb_start[g];
                const int num_zb = groups.num_zbs[g];
                const int rank = groups.ranks[g];

                Tv *sh_pb_ptr = sh_pb + (g_offset * TILE_B + b_offset);
                compute_phase_dev<Rank, Ti, Tv>(sas_b, zbs, num_zb, wb, sh_pb_ptr, BATCH_SIZE * TILE_B, rank);
            }

            __syncthreads();

            for (int a = a_tile_start + threadIdx.x; a < a_tile_end; a += blockDim.x)
            {
                const Ti astr = astrs[a];
                Tv *dst_base = dst_vec_bid + (int64)a * n_b;

                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    if (sh_valid_b[g_offset] == 0)
                        continue;

                    const int g = chunk_start_g + g_offset;
                    const Ti ax = groups.axs[g];
                    int sa = -1;
                    Ti sas_a = 0;

                    if (ax == 0)
                    {
                        sa = a;
                        sas_a = astr;
                    }
                    else
                    {
                        sas_a = astr ^ ax;
                        sa = a_idx_map[sas_a];
                    }

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
                        const Tv *src_base = src_vec + basis.block_offsets[sbi] + (int64)sa * basis.block_num_b[sbi];
                        const Tv *pb = sh_pb + (g_offset * TILE_B);

                        const int sh_task_base_offset = g_offset * TILE_B;
                        const int *sh_sa_b_task = sh_sa_b + sh_task_base_offset;

                        for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                        {
                            const int sa_b = sh_sa_b_task[b_offset];
                            if (sa_b != -1)
                            {
                                const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                                const int actual_b_idx = b_tile_start + b_offset;
                                dst_base[actual_b_idx] += src_base[sa_b] * vt;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void hvec_gather_pure_a_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int bid = blockIdx.x; 
    const int total_groups = groups.num_groups;
    const int *a_idx_map = basis.astr2idx;

    constexpr int SHARED_MEM_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1 * TILE_B
        : Rank == 2 ? BATCH_SIZE_SH2 * TILE_B * 2
                    : BATCH_SIZE_SH3 * TILE_B * KERNEL_MAX_RANK;

    constexpr int BATCH_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1
        : Rank == 2 ? BATCH_SIZE_SH2
                    : BATCH_SIZE_SH3;

    __shared__ Tv sh_pb[SHARED_MEM_SIZE];
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

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);

            for (int g_offset = threadIdx.x; g_offset < current_chunk_groups; g_offset += blockDim.x)
            {
                const int g = chunk_start_g + g_offset;
                const int h = (asym ^ groups.asyms[g]) * nirp + bsym; 
                const int src_bid = basis.block_map[h];

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
                const Ti bstr = bstrs_tile_start[b_offset];

                const Ti *zbs = groups.flat_zbs + groups.zb_start[g];
                const Tv *wb = groups.flat_wb + groups.wb_start[g];
                const int num_zb = groups.num_zbs[g];
                const int rank = groups.ranks[g];

                Tv *sh_pb_ptr = sh_pb + (g_offset * TILE_B + b_offset);
                compute_phase_dev<Rank, Ti, Tv>(bstr, zbs, num_zb, wb, sh_pb_ptr, BATCH_SIZE * TILE_B, rank);
            }

            __syncthreads();

            for (int a = a_tile_start + threadIdx.x; a < a_tile_end; a += blockDim.x)
            {
                const Ti astr = astrs[a];
                Tv *dst_base = dst_vec_bid + (int64)a * n_b;

                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    if (sh_valid_b[g_offset] == 0)
                        continue;

                    const int g = chunk_start_g + g_offset;
                    const Ti ax = groups.axs[g];
                    int sa = -1;
                    Ti sas_a = 0;

                    if (ax == 0)
                    {
                        sa = a;
                        sas_a = astr;
                    }
                    else
                    {
                        sas_a = astr ^ ax;
                        sa = a_idx_map[sas_a];
                    }

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
                        const int src_n_b = basis.block_num_b[sbi];
                        const Tv *src_base = src_vec + basis.block_offsets[sbi] + (int64)sa * src_n_b;

                        const Tv *pb = sh_pb + (g_offset * TILE_B);

                        for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                        {
                            const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                            const int actual_b_idx = b_tile_start + b_offset;
                            dst_base[actual_b_idx] += src_base[actual_b_idx] * vt;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void hvec_gather_pure_b_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int bid = blockIdx.x; 
    const int total_groups = groups.num_groups;
    const int *b_idx_map = basis.bstr2idx;

    constexpr int SHARED_MEM_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1 * TILE_B
        : Rank == 2 ? BATCH_SIZE_SH2 * TILE_B * 2
                    : BATCH_SIZE_SH3 * TILE_B * KERNEL_MAX_RANK;

    constexpr int BATCH_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1
        : Rank == 2 ? BATCH_SIZE_SH2
                    : BATCH_SIZE_SH3;

    constexpr int IDX_MEM_SIZE = BATCH_SIZE * TILE_B;

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

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);

            for (int g_offset = threadIdx.x; g_offset < current_chunk_groups; g_offset += blockDim.x)
            {
                const int g = chunk_start_g + g_offset;
                const int h = asym * nirp + (bsym ^ groups.bsyms[g]); 
                const int src_bid = basis.block_map[h];
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
                const Ti bx = groups.bxs[g];
                const Ti sas_b = dst_b_str ^ bx;

                const int sh_flat_offset = g_offset * TILE_B + b_offset;
                sh_sa_b[sh_flat_offset] = b_idx_map[sas_b]; 

                const Ti *zbs = groups.flat_zbs + groups.zb_start[g];
                const Tv *wb = groups.flat_wb + groups.wb_start[g];
                const int num_zb = groups.num_zbs[g];
                const int rank = groups.ranks[g];

                Tv *sh_pb_ptr = sh_pb + (g_offset * TILE_B + b_offset);
                compute_phase_dev<Rank, Ti, Tv>(sas_b, zbs, num_zb, wb, sh_pb_ptr, BATCH_SIZE * TILE_B, rank);
            }

            __syncthreads();

            for (int a = a_tile_start + threadIdx.x; a < a_tile_end; a += blockDim.x)
            {
                const Ti astr = astrs[a];
                Tv *dst_base = dst_vec_bid + (int64)a * n_b;

                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    if (sh_valid_b[g_offset] == 0)
                        continue;

                    const int g = chunk_start_g + g_offset;
                    const int sa = a;      
                    const Ti sas_a = astr; 

                    const int nza = groups.num_zas[g];
                    const Ti *zas = groups.flat_zas + groups.za_start[g];
                    const Tv *wa = groups.flat_wa + groups.wa_start[g];
                    const int rank = groups.ranks[g];

                    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
                    Tv pa[STACK_SIZE] = {};
                    compute_phase_dev<Rank, Ti, Tv>(sas_a, zas, nza, wa, pa, 1, rank);

                    const int sbi = sh_src_bid[g_offset];
                    const int src_n_b = basis.block_num_b[sbi]; 
                    const Tv *src_base = src_vec + basis.block_offsets[sbi] + (int64)sa * src_n_b;
                    const Tv *pb = sh_pb + (g_offset * TILE_B);

                    const int sh_task_base_offset = g_offset * TILE_B;
                    const int *sh_sa_b_task = sh_sa_b + sh_task_base_offset;

                    for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                    {
                        const int sa_b = sh_sa_b_task[b_offset]; 
                        if (sa_b != -1)
                        {
                            const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                            const int actual_b_idx = b_tile_start + b_offset;
                            dst_base[actual_b_idx] += src_base[sa_b] * vt;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <int TypeCode, typename Ti, typename Tv>
static inline void dispatch_chunks_by_rank_gpu(
    const BasisViewDev<Ti> &basis,
    const GroupsViewDev<Ti, Tv> &groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int64 total_ngs = groups.num_groups;
    if (total_ngs == 0)
        return;

    // 创建不带 RAII 析构函数的轻量切片体值对象，用于安全浅拷贝传入内核
    BasisSliceDev<Ti> basis_slice;
    basis_slice.num_blocks = basis.num_blocks;
    basis_slice.num_irreps = basis.num_irreps;
    basis_slice.max_a_count = basis.max_a_count;
    basis_slice.max_b_count = basis.max_b_count;
    basis_slice.dim = basis.dim;
    basis_slice.block_offsets = basis.block_offsets;
    basis_slice.block_num_a = basis.block_num_a;
    basis_slice.block_num_b = basis.block_num_b;
    basis_slice.block_asym = basis.block_asym;
    basis_slice.block_bsym = basis.block_bsym;
    basis_slice.astrs_flat = basis.astrs_flat;
    basis_slice.bstrs_flat = basis.bstrs_flat;
    basis_slice.astrs_start = basis.astrs_start;
    basis_slice.bstrs_start = basis.bstrs_start;
    basis_slice.block_map = basis.block_map;
    basis_slice.astr2idx = basis.astr2idx;
    basis_slice.bstr2idx = basis.bstr2idx;

    int64 start = 0;
    while (start < total_ngs)
    {
        const int current_rank = groups.host_ranks[start];
        const int dispatch_rank = (current_rank == 1 || current_rank == 2) ? current_rank : 0;

        int64 end = start + 1;
        while (end < total_ngs)
        {
            const int next_rank = groups.host_ranks[end];
            const int next_dispatch_rank = (next_rank == 1 || next_rank == 2) ? next_rank : 0;
            if (next_dispatch_rank != dispatch_rank)
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

        int block_size = 256;
        int num_blocks = basis.num_blocks;

        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        dim3 grid_size(num_blocks, num_sms * 4);

        if constexpr (TypeCode == 0)
        {
            switch (dispatch_rank)
            {
            case 1:
                hvec_gather_diag_kernel<1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            case 2:
                hvec_gather_diag_kernel<2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            default:
                hvec_gather_diag_kernel<0, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            }
        }
        else if constexpr (TypeCode == 1)
        {
            switch (dispatch_rank)
            {
            case 1:
                hvec_gather_pure_a_kernel<1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            case 2:
                hvec_gather_pure_a_kernel<2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            default:
                hvec_gather_pure_a_kernel<0, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            }
        }
        else if constexpr (TypeCode == 2)
        {
            switch (dispatch_rank)
            {
            case 1:
                hvec_gather_pure_b_kernel<1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            case 2:
                hvec_gather_pure_b_kernel<2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            default:
                hvec_gather_pure_b_kernel<0, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            }
        }
        else if constexpr (TypeCode == 3)
        {
            switch (dispatch_rank)
            {
            case 1:
                hvec_gather_mixed_kernel<1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            case 2:
                hvec_gather_mixed_kernel<2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            default:
                hvec_gather_mixed_kernel<0, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            }
        }

        start = end;
    }
}

template <typename Ti, typename Tv>
void cuda_hvec(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    cudaMemset(dst_vec, 0, basis.dim * sizeof(Tv));

    dispatch_chunks_by_rank_gpu<0>(basis, net.diag_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank_gpu<1>(basis, net.pure_a_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank_gpu<2>(basis, net.pure_b_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank_gpu<3>(basis, net.mixed_groups, src_vec, dst_vec);
}

