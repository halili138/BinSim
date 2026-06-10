#pragma once
#include "cuda_common.cuh"
#include "otf.hpp"

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
        if (axs)
        {
            cudaFree(const_cast<Ti *>(axs));
            axs = nullptr;
        }
        if (bxs)
        {
            cudaFree(const_cast<Ti *>(bxs));
            bxs = nullptr;
        }
        if (asyms)
        {
            cudaFree(const_cast<int *>(asyms));
            asyms = nullptr;
        }
        if (bsyms)
        {
            cudaFree(const_cast<int *>(bsyms));
            bsyms = nullptr;
        }
        if (ranks)
        {
            cudaFree(const_cast<int *>(ranks));
            ranks = nullptr;
        }
        if (num_zas)
        {
            cudaFree(const_cast<int *>(num_zas));
            num_zas = nullptr;
        }
        if (num_zbs)
        {
            cudaFree(const_cast<int *>(num_zbs));
            num_zbs = nullptr;
        }
        if (flat_zas)
        {
            cudaFree(const_cast<Ti *>(flat_zas));
            flat_zas = nullptr;
        }
        if (flat_zbs)
        {
            cudaFree(const_cast<Ti *>(flat_zbs));
            flat_zbs = nullptr;
        }
        if (flat_wa)
        {
            cudaFree(const_cast<Tv *>(flat_wa));
            flat_wa = nullptr;
        }
        if (flat_wb)
        {
            cudaFree(const_cast<Tv *>(flat_wb));
            flat_wb = nullptr;
        }
        if (za_start)
        {
            cudaFree(const_cast<int64 *>(za_start));
            za_start = nullptr;
        }
        if (zb_start)
        {
            cudaFree(const_cast<int64 *>(zb_start));
            zb_start = nullptr;
        }
        if (wa_start)
        {
            cudaFree(const_cast<int64 *>(wa_start));
            wa_start = nullptr;
        }
        if (wb_start)
        {
            cudaFree(const_cast<int64 *>(wb_start));
            wb_start = nullptr;
        }
        if (original_idx)
        {
            cudaFree(const_cast<int *>(original_idx));
            original_idx = nullptr;
        }
        if (excit_types)
        {
            cudaFree(const_cast<int *>(excit_types));
            excit_types = nullptr;
        }

        host_ranks.clear();
        num_groups = 0;
    }

    GroupsViewDev &operator=(GroupsViewDev &&other) noexcept
    {
        if (this != &other)
        {
            clear();
            axs = other.axs;
            bxs = other.bxs;
            asyms = other.asyms;
            bsyms = other.bsyms;
            ranks = other.ranks;
            num_zas = other.num_zas;
            num_zbs = other.num_zbs;
            flat_zas = other.flat_zas;
            flat_zbs = other.flat_zbs;
            flat_wa = other.flat_wa;
            flat_wb = other.flat_wb;
            za_start = other.za_start;
            zb_start = other.zb_start;
            wa_start = other.wa_start;
            wb_start = other.wb_start;
            original_idx = other.original_idx;
            excit_types = other.excit_types;
            num_groups = other.num_groups;
            host_ranks = std::move(other.host_ranks);

            other.axs = nullptr;
            other.bxs = nullptr;
            other.asyms = nullptr;
            other.bsyms = nullptr;
            other.ranks = nullptr;
            other.num_zas = nullptr;
            other.num_zbs = nullptr;
            other.flat_zas = nullptr;
            other.flat_zbs = nullptr;
            other.flat_wa = nullptr;
            other.flat_wb = nullptr;
            other.za_start = nullptr;
            other.zb_start = nullptr;
            other.wa_start = nullptr;
            other.wb_start = nullptr;
            other.original_idx = nullptr;
            other.excit_types = nullptr;
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

    std::vector<int64> host_sorted_idxs;
    std::vector<uint8> host_excit_types;

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
        host_sorted_idxs.clear();
        host_excit_types.clear();
    }
};

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

        size_t total_fza = 0;
        size_t total_fzb = 0;
        size_t total_fwa = 0;
        size_t total_fwb = 0;
        for (int i = 0; i < count; ++i)
        {
            const auto &g = src_bucket[i];
            total_fza += g.num_za;
            total_fzb += g.num_zb;
            total_fwa += (size_t)g.num_za * g.rank;
            total_fwb += (size_t)g.num_zb * g.rank;
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
            axs[i] = g.ax;
            bxs[i] = g.bx;
            asyms[i] = (int)g.asym;
            bsyms[i] = (int)g.bsym;
            ranks[i] = g.rank;
            nza[i] = g.num_za;
            nzb[i] = g.num_zb;
            zas[i] = zo;
            zbs[i] = zbo;
            was[i] = wo;
            wbs[i] = wbo;
            original_idxs[i] = (int)g.original_idx;
            excit_types[i] = (int)hn->excit_types[g.original_idx];

            if (g.num_za > 0)
            {
                std::copy_n(g.unique_zas, g.num_za, &fza[idx_fza]);
                idx_fza += g.num_za;
                zo += g.num_za;
            }
            if (g.num_zb > 0)
            {
                std::copy_n(g.unique_zbs, g.num_zb, &fzb[idx_fzb]);
                idx_fzb += g.num_zb;
                zbo += g.num_zb;
            }
            size_t size_wa = (size_t)g.num_za * g.rank;
            if (size_wa > 0)
            {
                std::copy_n(g.wa, size_wa, &fwa[idx_fwa]);
                idx_fwa += size_wa;
                wo += size_wa;
            }
            size_t size_wb = (size_t)g.num_zb * g.rank;
            if (size_wb > 0)
            {
                std::copy_n(g.wb, size_wb, &fwb[idx_fwb]);
                idx_fwb += size_wb;
                wbo += size_wb;
            }
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
        dst_view.axs = upload_to_device(axs);
        dst_view.bxs = upload_to_device(bxs);
        dst_view.asyms = upload_to_device(asyms);
        dst_view.bsyms = upload_to_device(bsyms);
        dst_view.ranks = upload_to_device(ranks);
        dst_view.num_zas = upload_to_device(nza);
        dst_view.num_zbs = upload_to_device(nzb);
        dst_view.za_start = upload_to_device(zas);
        dst_view.zb_start = upload_to_device(zbs);
        dst_view.wa_start = upload_to_device(was);
        dst_view.wb_start = upload_to_device(wbs);
        dst_view.flat_zas = upload_to_device(fza);
        dst_view.flat_zbs = upload_to_device(fzb);
        dst_view.flat_wa = upload_to_device(fwa);
        dst_view.flat_wb = upload_to_device(fwb);
        dst_view.original_idx = upload_to_device(original_idxs);
        dst_view.excit_types = upload_to_device(excit_types);
    };

    flatten_bucket(hn->diag_groups, dn->diag_groups);
    flatten_bucket(hn->pure_a_groups, dn->pure_a_groups);
    flatten_bucket(hn->pure_b_groups, dn->pure_b_groups);
    flatten_bucket(hn->mixed_groups, dn->mixed_groups);

    dn->host_sorted_idxs.assign(hn->sorted_idxs, hn->sorted_idxs + hn->num_groups);
    dn->host_excit_types.assign(hn->excit_types, hn->excit_types + hn->num_groups);

    return static_cast<void *>(dn);
}

template <typename Ti, typename Tv>
FORCE_INLINE GroupsSliceDev<Ti, Tv> make_groups_slice(const GroupsViewDev<Ti, Tv> &view)
{
    GroupsSliceDev<Ti, Tv> s;
    s.num_groups = view.num_groups;
    s.axs = view.axs;
    s.bxs = view.bxs;
    s.asyms = view.asyms;
    s.bsyms = view.bsyms;
    s.ranks = view.ranks;
    s.num_zas = view.num_zas;
    s.num_zbs = view.num_zbs;
    s.flat_zas = view.flat_zas;
    s.flat_zbs = view.flat_zbs;
    s.flat_wa = view.flat_wa;
    s.flat_wb = view.flat_wb;
    s.za_start = view.za_start;
    s.zb_start = view.zb_start;
    s.wa_start = view.wa_start;
    s.wb_start = view.wb_start;
    return s;
}
