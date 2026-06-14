#pragma once
#include "cuda_common.cuh"
#include "basis.hpp"

// 1. 显存物理生命周期管理器（禁止拷贝, 支持移动）
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
        if (astr2idx)
        {
            cudaFree(const_cast<int *>(astr2idx));
            astr2idx = nullptr;
        }
        if (bstr2idx)
        {
            cudaFree(const_cast<int *>(bstr2idx));
            bstr2idx = nullptr;
        }
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

            other.num_blocks = 0;
            other.num_irreps = 0;
            other.max_a_count = 0;
            other.max_b_count = 0;
            other.dim = 0;
            other.block_offsets = nullptr;
            other.block_num_a = nullptr;
            other.block_num_b = nullptr;
            other.block_asym = nullptr;
            other.block_bsym = nullptr;
            other.astrs_flat = nullptr;
            other.bstrs_flat = nullptr;
            other.astrs_start = nullptr;
            other.bstrs_start = nullptr;
            other.block_map = nullptr;
            other.astr2idx = nullptr;
            other.bstr2idx = nullptr;
        }
        return *this;
    }
};

// 2. 新增的轻量级物理切片体（无析构函数, 专供内核按值传递避免参数拷贝拦截）
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
    // 分布式目标块映射. 如果为空, 则为原生单卡模式; 若不为空, 则为分布式模式
    const int *target_bids = nullptr;
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

template <typename Ti>
FORCE_INLINE BasisSliceDev<Ti> make_basis_slice(const BasisViewDev<Ti> &basis)
{
    BasisSliceDev<Ti> s;
    s.num_blocks = basis.num_blocks;
    s.num_irreps = basis.num_irreps;
    s.max_a_count = basis.max_a_count;
    s.max_b_count = basis.max_b_count;
    s.dim = basis.dim;
    s.block_offsets = basis.block_offsets;
    s.block_num_a = basis.block_num_a;
    s.block_num_b = basis.block_num_b;
    s.block_asym = basis.block_asym;
    s.block_bsym = basis.block_bsym;
    s.astrs_flat = basis.astrs_flat;
    s.bstrs_flat = basis.bstrs_flat;
    s.astrs_start = basis.astrs_start;
    s.bstrs_start = basis.bstrs_start;
    s.block_map = basis.block_map;
    s.astr2idx = basis.astr2idx;
    s.bstr2idx = basis.bstr2idx;
    s.target_bids = nullptr; // 单卡默认无映射
    return s;
}
