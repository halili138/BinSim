#pragma once
#include "basis.hpp"
#include "otf.hpp"

inline std::vector<int> get_rank_block_counts(const GlobalMemMap *gmap)
{
    std::vector<int> counts(gmap->mpi_size, 0);
    for (int rank : gmap->block_to_rank)
    {
        if (rank >= 0 && rank < gmap->mpi_size)
            counts[rank]++;
    }
    return counts;
}

template <typename Ti>
inline int get_max_rank_num_blocks(const BasisManager<Ti> *basis, const GlobalMemMap *gmap)
{
    std::vector<int> counts(gmap->mpi_size, 0);
    const int64 num_irreps = basis->num_irreps;

    // Keep this in lockstep with build_sub_topology(): phases are assigned
    // from each real basis block's local ordinal on its destination rank.
    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        const int64 h = basis->blocks[i].asym * num_irreps + basis->blocks[i].bsym;
        const int dest_r = gmap->block_to_rank[h];
        if (dest_r >= 0 && dest_r < gmap->mpi_size)
            counts[dest_r]++;
    }

    return counts.empty() ? 0 : *std::max_element(counts.begin(), counts.end());
}

// =================================================================
// 2. 分段通信账本：针对单个 (asym, bsym) 子片段的专属配置
// =================================================================
struct SubTopology
{
    int64 send_dim;
    int64 recv_dim;
    std::vector<int> send_counts;
    std::vector<int> recv_counts;

    // 该片段计算时的虚假内存偏移 (指向 cache 数组)
    std::vector<int64> block_offsets_in_cache;
    std::vector<int> target_blocks;

    std::vector<PackJob> pack_jobs;
};

template <typename Ti, typename Tv>
SubTopology *build_sub_topology(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *sub_net, const GlobalMemMap *gmap, int num_phases, int phase_idx)
{
    SubTopology *topo = new SubTopology();
    int rank = gmap->mpi_rank;
    int size = gmap->mpi_size;
    int64 num_irreps = basis->num_irreps;
    int64 max_h = num_irreps * num_irreps;

    // 因为 sub_net 只包含一种对称性片段，op_syms 将极为纯净
    std::vector<std::pair<int64, int64>> op_syms;
    auto add_syms = [&](const auto &groups)
    {
        for (const auto &g : groups)
            op_syms.push_back({g.asym, g.bsym});
    };
    add_syms(sub_net->diag_groups);
    add_syms(sub_net->pure_a_groups);
    add_syms(sub_net->pure_b_groups);
    add_syms(sub_net->mixed_groups);
    std::sort(op_syms.begin(), op_syms.end());
    op_syms.erase(std::unique(op_syms.begin(), op_syms.end()), op_syms.end());

    std::vector<std::vector<int64>> send_reqs(size);
    std::vector<std::vector<int64>> recv_reqs(size);

    // =================================================================
    // 【终极切片修复】：按 Local Index 分发 Phase，确保每张卡完美平分！
    // =================================================================
    std::vector<int> phase_of_block(basis->num_blocks, 0);
    std::vector<int> rank_block_count(size, 0);

    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        int64 h = basis->blocks[i].asym * num_irreps + basis->blocks[i].bsym;
        int dest_r = gmap->block_to_rank[h];
        // 关键：基于该块在自己 Rank 里的排行来分配 Phase
        phase_of_block[i] = rank_block_count[dest_r] % num_phases;
        rank_block_count[dest_r]++;
    }

    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        // 时间切片拦截
        if (phase_of_block[i] != phase_idx)
            continue;

        int64 t_asym = basis->blocks[i].asym;
        int64 t_bsym = basis->blocks[i].bsym;
        int64 h_tgt = t_asym * num_irreps + t_bsym;
        int tgt_rank = gmap->block_to_rank[h_tgt];

        for (const auto &op : op_syms)
        {
            int64 s_asym = t_asym ^ op.first;
            int64 s_bsym = t_bsym ^ op.second;
            int64 h_src = s_asym * num_irreps + s_bsym;
            int src_rank = gmap->block_to_rank[h_src];

            if (src_rank != -1 && src_rank != tgt_rank)
            {
                if (tgt_rank == rank)
                    recv_reqs[src_rank].push_back(h_src);
                if (src_rank == rank)
                    send_reqs[tgt_rank].push_back(h_src);
            }
        }
    }

    // 去重与合并
    for (int r = 0; r < size; ++r)
    {
        std::sort(recv_reqs[r].begin(), recv_reqs[r].end());
        recv_reqs[r].erase(std::unique(recv_reqs[r].begin(), recv_reqs[r].end()), recv_reqs[r].end());
        std::sort(send_reqs[r].begin(), send_reqs[r].end());
        send_reqs[r].erase(std::unique(send_reqs[r].begin(), send_reqs[r].end()), send_reqs[r].end());
    }

    topo->send_counts.assign(size, 0);
    topo->recv_counts.assign(size, 0);
    topo->send_dim = 0;
    topo->recv_dim = 0;

    for (int r = 0; r < size; ++r)
    {
        for (int64 h : send_reqs[r])
        {
            int64 bsize = basis->blocks[basis->block_map[h]].num_a * basis->blocks[basis->block_map[h]].num_b;
            topo->pack_jobs.push_back({gmap->block_local_offsets[h], topo->send_dim, bsize});
            topo->send_dim += bsize;
            topo->send_counts[r] += bsize;
        }
        for (int64 h : recv_reqs[r])
        {
            int64 bsize = basis->blocks[basis->block_map[h]].num_a * basis->blocks[basis->block_map[h]].num_b;
            topo->recv_dim += bsize;
            topo->recv_counts[r] += bsize;
        }
    }

    // 生成该片段专用的虚拟视口偏移 (前半段是 local_v, 后半段是本次独享的 recv_buffer)
    topo->block_offsets_in_cache.assign(max_h, -1);
    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        int64 h = basis->blocks[i].asym * num_irreps + basis->blocks[i].bsym;

        if (gmap->block_to_rank[h] == rank)
        {
            topo->block_offsets_in_cache[h] = gmap->block_local_offsets[h];

            if (phase_of_block[i] == phase_idx)
            {
                topo->target_blocks.push_back((int)h);
            }
        }
    }

    int64 current_recv_offset = gmap->local_dim;
    for (int r = 0; r < size; ++r)
    {
        for (int64 h : recv_reqs[r])
        {
            topo->block_offsets_in_cache[h] = current_recv_offset;
            int64 bsize = basis->blocks[basis->block_map[h]].num_a * basis->blocks[basis->block_map[h]].num_b;
            current_recv_offset += bsize;
        }
    }

    return topo;
}

template <typename Ti, typename Tv>
void set_local_det_coeff(
    const GlobalMemMap *gmap,
    const BasisManager<Ti> *basis,
    const Ti target_astr,
    const Ti target_bstr,
    const Tv coeff,
    Tv *local_vec)
{
    int64 asym = get_string_sym(target_astr, basis->orbsym);
    int64 bsym = get_string_sym(target_bstr, basis->orbsym);

    if (asym >= basis->num_irreps || bsym >= basis->num_irreps)
        return;

    int64 h = asym * basis->num_irreps + bsym;

    // 【核心拦截 1】：判断这个波函数块是否由当前 MPI 进程负责
    if (gmap->block_to_rank[h] != gmap->mpi_rank)
        return;

    int64 block_idx = basis->block_map[h];
    if (block_idx == -1)
        return;

    const BlockDesc<Ti> &block = basis->blocks[block_idx];

    int64 ia = find_index(block.astrs, block.num_a, target_astr);
    if (ia == -1)
        return;

    int64 ib = find_index(block.bstrs, block.num_b, target_bstr);
    if (ib == -1)
        return;

    // 【核心映射 2】：使用 GlobalMemMap 里的局部偏移，取代原有的 block.offset
    int64 local_gid = gmap->block_local_offsets[h] + ia * block.num_b + ib;

    local_vec[local_gid] += coeff;
}

// ==========================================
// 3. 指针版的按 Rank 分发器
// ==========================================
template <int TypeCode, typename Ti, typename Tv>
static inline void dispatch_subchunks_by_rank(const BasisView<Ti> &view, const SVDGroup_OTF<Ti, Tv> *groups_ptr, int64 total_ngs, const Tv *src_vec, Tv *dst_vec)
{
    if (total_ngs == 0)
        return;
    int64 start = 0;
    while (start < total_ngs)
    {
        const int current_rank = groups_ptr[start].rank;
        const int dispatch_rank = (current_rank == 1 || current_rank == 2) ? current_rank : 0;
        int64 end = start + 1;
        while (end < total_ngs)
        {
            const int next_rank = groups_ptr[end].rank;
            const int next_dispatch_rank = (next_rank == 1 || next_rank == 2) ? next_rank : 0;
            if (next_dispatch_rank != dispatch_rank)
                break;
            end++;
        }
        const int64 chunk_size = end - start;
        const SVDGroup_OTF<Ti, Tv> *chunk_ptr = groups_ptr + start;

        switch (dispatch_rank)
        {
        case 1:
            gather_contract_batched_impl<1, TypeCode>(view, chunk_ptr, chunk_size, src_vec, dst_vec);
            break;
        case 2:
            gather_contract_batched_impl<2, TypeCode>(view, chunk_ptr, chunk_size, src_vec, dst_vec);
            break;
        default:
            gather_contract_batched_impl<0, TypeCode>(view, chunk_ptr, chunk_size, src_vec, dst_vec);
            break;
        }
        start = end;
    }
}

// =================================================================
// 4. 最终无状态计算入口：传入完整的 [local_v | 专属 recv_buffer]
// =================================================================
template <typename Ti, typename Tv>
void compute_hvec_sub_chunk(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *sub_net, const SubTopology *topo, const Tv *chunk_cache, Tv *local_w)
{
    int64 num_irreps = basis->num_irreps;
    std::vector<BlockDesc<Ti>> virtual_blocks;
    std::vector<int64> virtual_block_map(num_irreps * num_irreps, -1);

    int64 next_idx = 0;
    for (int h : topo->target_blocks)
    {
        BlockDesc<Ti> blk = basis->blocks[basis->block_map[h]];
        blk.offset = topo->block_offsets_in_cache[h];
        virtual_blocks.push_back(blk);
        virtual_block_map[h] = next_idx++;
    }
    for (int h = 0; h < num_irreps * num_irreps; ++h)
    {
        if (virtual_block_map[h] == -1 && topo->block_offsets_in_cache[h] != -1)
        {
            BlockDesc<Ti> blk = basis->blocks[basis->block_map[h]];
            blk.offset = topo->block_offsets_in_cache[h];
            virtual_blocks.push_back(blk);
            virtual_block_map[h] = next_idx++;
        }
    }

    BasisView<Ti> view = basis->view;
    view.blocks = virtual_blocks.data();
    view.block_map = virtual_block_map.data();
    view.num_blocks = topo->target_blocks.size();

    // 在这个阶段，子 OTF 的算子完全保持着 otf.hpp 原生的高效 rank 排序！
    dispatch_subchunks_by_rank<0>(view, sub_net->diag_groups.data(), sub_net->diag_groups.size(), chunk_cache, local_w);
    dispatch_subchunks_by_rank<1>(view, sub_net->pure_a_groups.data(), sub_net->pure_a_groups.size(), chunk_cache, local_w);
    dispatch_subchunks_by_rank<2>(view, sub_net->pure_b_groups.data(), sub_net->pure_b_groups.size(), chunk_cache, local_w);
    dispatch_subchunks_by_rank<3>(view, sub_net->mixed_groups.data(), sub_net->mixed_groups.size(), chunk_cache, local_w);
}
