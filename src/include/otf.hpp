#pragma once
#include "common.hpp"

inline constexpr int BATCH_SIZE1 = 256;
inline constexpr int BATCH_SIZE2 = 128;
inline constexpr int BATCH_SIZE3 = 1;
inline constexpr int RANK3 = 64;

template <typename Ti,
          typename Tv>
struct SVDGroup_OTF
{
    Ti ax = {};
    Ti bx = {};
    int rank = {};
    int64 original_idx = {};
    int64 asym = {};
    int64 bsym = {};

    int num_za = {};
    Ti *unique_zas = nullptr;
    int num_zb = {};
    Ti *unique_zbs = nullptr;

    Tv *wa = nullptr;
    Tv *wb = nullptr;

    inline Tv *ptr_wa(int r) const { return wa + r * num_za; }
    inline Tv *ptr_wb(int r) const { return wb + r * num_zb; }

    void clear()
    {
        if (unique_zas)
        {
            delete[] unique_zas;
            unique_zas = nullptr;
        }
        if (unique_zbs)
        {
            delete[] unique_zbs;
            unique_zbs = nullptr;
        }
        if (wa)
        {
            delete[] wa;
            wa = nullptr;
        }
        if (wb)
        {
            delete[] wb;
            wb = nullptr;
        }
    }
};

template <typename Ti,
          typename Tv>
struct Network_OTF
{
    int64 num_groups = {};

    std::vector<SVDGroup_OTF<Ti, Tv>> diag_groups;
    std::vector<SVDGroup_OTF<Ti, Tv>> pure_a_groups;
    std::vector<SVDGroup_OTF<Ti, Tv>> pure_b_groups;
    std::vector<SVDGroup_OTF<Ti, Tv>> mixed_groups;

    int64 *sorted_idxs = nullptr;
    uint8 *excit_types = nullptr;

    void clear()
    {
        if (sorted_idxs)
        {
            delete[] sorted_idxs;
            sorted_idxs = nullptr;
        }
        if (excit_types)
        {
            delete[] excit_types;
            excit_types = nullptr;
        }

        auto clear_bucket = [](std::vector<SVDGroup_OTF<Ti, Tv>> &bucket)
        {
            for (auto &g : bucket)
                g.clear();
            bucket.clear();
        };

        clear_bucket(diag_groups);
        clear_bucket(pure_a_groups);
        clear_bucket(pure_b_groups);
        clear_bucket(mixed_groups);

        num_groups = 0;
    }

    ~Network_OTF()
    {
        clear();
    }
};

template <typename Ti,
          typename Tv>
void *build_network_otf(
    const BasisManager<Ti> *basis,
    int64 norb, int64 ngs,
    const Ti *axs,
    const Ti *bxs,
    const int64 *ranks,
    const int64 *num_zas,
    const int64 *num_zbs,
    const Ti *flat_zas,
    const Ti *flat_zbs,
    const Tv *flat_wa,
    const Tv *flat_wb)
{
    Network_OTF<Ti, Tv> *net = new Network_OTF<Ti, Tv>();
    net->num_groups = ngs;

    net->excit_types = new uint8[ngs];
    net->sorted_idxs = new int64[ngs];

    struct Entry
    {
        SVDGroup_OTF<Ti, Tv> group;
        int64 orig_idx;
        uint8 type_code;
    };
    std::vector<Entry> diag_entries, pure_a_entries, pure_b_entries, mixed_entries;

    uint64 z_offset_a = 0, z_offset_b = 0;
    uint64 w_offset_a = 0, w_offset_b = 0;

    for (int64 g = 0; g < ngs; ++g)
    {
        SVDGroup_OTF<Ti, Tv> group;
        group.ax = axs[g];
        group.bx = bxs[g];
        group.rank = (int)ranks[g];
        group.num_za = (int)num_zas[g];
        group.num_zb = (int)num_zbs[g];
        group.asym = get_string_sym(group.ax, basis->orbsym);
        group.bsym = get_string_sym(group.bx, basis->orbsym);

        group.unique_zas = new Ti[group.num_za];
        std::copy(flat_zas + z_offset_a, flat_zas + z_offset_a + group.num_za, group.unique_zas);
        z_offset_a += group.num_za;

        group.unique_zbs = new Ti[group.num_zb];
        std::copy(flat_zbs + z_offset_b, flat_zbs + z_offset_b + group.num_zb, group.unique_zbs);
        z_offset_b += group.num_zb;

        uint64 wa_size = (uint64)group.num_za * group.rank;
        group.wa = new Tv[wa_size];
        std::copy(flat_wa + w_offset_a, flat_wa + w_offset_a + wa_size, group.wa);
        w_offset_a += wa_size;

        uint64 wb_size = (uint64)group.num_zb * group.rank;
        group.wb = new Tv[wb_size];
        std::copy(flat_wb + w_offset_b, flat_wb + w_offset_b + wb_size, group.wb);
        w_offset_b += wb_size;

        uint8 type;
        if (group.ax == 0 && group.bx == 0)
            type = 0;
        else if (group.ax != 0 && group.bx == 0)
            type = 1;
        else if (group.ax == 0 && group.bx != 0)
            type = 2;
        else
            type = 3;

        group.original_idx = g;
        net->excit_types[g] = type;
        Entry entry = {group, g, type};

        if (type == 0)
            diag_entries.push_back(entry);
        else if (type == 1)
            pure_a_entries.push_back(entry);
        else if (type == 2)
            pure_b_entries.push_back(entry);
        else
            mixed_entries.push_back(entry);
    }

    auto rank_cmp = [](const Entry &a, const Entry &b)
    {
        return a.group.rank < b.group.rank;
    };

    std::sort(diag_entries.begin(), diag_entries.end(), rank_cmp);
    std::sort(pure_a_entries.begin(), pure_a_entries.end(), rank_cmp);
    std::sort(pure_b_entries.begin(), pure_b_entries.end(), rank_cmp);
    std::sort(mixed_entries.begin(), mixed_entries.end(), rank_cmp);

    auto fill = [](auto &entries, auto &groups_vec)
    {
        groups_vec.reserve(entries.size());
        for (auto &e : entries)
        {
            groups_vec.push_back(e.group);
        }
    };

    fill(diag_entries, net->diag_groups);
    fill(pure_a_entries, net->pure_a_groups);
    fill(pure_b_entries, net->pure_b_groups);
    fill(mixed_entries, net->mixed_groups);

    for (int64 i = 0; i < ngs; ++i)
        net->sorted_idxs[i] = -1;
    for (int64 i = 0; i < (int64)net->diag_groups.size(); ++i)
        net->sorted_idxs[net->diag_groups[i].original_idx] = i;
    for (int64 i = 0; i < (int64)net->pure_a_groups.size(); ++i)
        net->sorted_idxs[net->pure_a_groups[i].original_idx] = i;
    for (int64 i = 0; i < (int64)net->pure_b_groups.size(); ++i)
        net->sorted_idxs[net->pure_b_groups[i].original_idx] = i;
    for (int64 i = 0; i < (int64)net->mixed_groups.size(); ++i)
        net->sorted_idxs[net->mixed_groups[i].original_idx] = i;

    return static_cast<void *>(net);
}
