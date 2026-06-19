#pragma once
#include <vector>
#include <algorithm>
#include "dist.hpp"
#include "cuda_hvec.cuh"
#include "cuda_expm.cuh"
#include "cuda_backgrad.cuh"

struct SubTopologyDev
{
    int64 send_dim = 0;
    int64 recv_dim = 0;
    std::vector<int> send_counts;
    std::vector<int> recv_counts;

    // --- GPU VRAM 驻留数据 ---
    int num_targets = 0;
    int *d_target_bids = nullptr;
    int64 *d_topo_offsets = nullptr;

    int num_pack_jobs = 0;
    PackJob *d_pack_jobs = nullptr;

    // Virtual chunk basis for distributed expm/backgrad.  Entries are ordered
    // as CPU dist.cpp does: local target blocks first, then ghost cache blocks.
    int virt_num_blocks = 0;
    int *d_virt_block_map = nullptr;
    int64 *d_virt_block_offsets = nullptr;
    int *d_virt_block_num_a = nullptr;
    int *d_virt_block_num_b = nullptr;
    int *d_virt_block_asym = nullptr;
    int *d_virt_block_bsym = nullptr;
    int64 *d_virt_astrs_start = nullptr;
    int64 *d_virt_bstrs_start = nullptr;

    ~SubTopologyDev()
    {
        if (d_target_bids)
        {
            cudaFree(d_target_bids);
            d_target_bids = nullptr;
        }
        if (d_topo_offsets)
        {
            cudaFree(d_topo_offsets);
            d_topo_offsets = nullptr;
        }
        if (d_pack_jobs)
        {
            cudaFree(d_pack_jobs);
            d_pack_jobs = nullptr;
        }
        if (d_virt_block_map) cudaFree(d_virt_block_map);
        if (d_virt_block_offsets) cudaFree(d_virt_block_offsets);
        if (d_virt_block_num_a) cudaFree(d_virt_block_num_a);
        if (d_virt_block_num_b) cudaFree(d_virt_block_num_b);
        if (d_virt_block_asym) cudaFree(d_virt_block_asym);
        if (d_virt_block_bsym) cudaFree(d_virt_block_bsym);
        if (d_virt_astrs_start) cudaFree(d_virt_astrs_start);
        if (d_virt_bstrs_start) cudaFree(d_virt_bstrs_start);
    }
};

template <typename Ti, typename Tv>
SubTopologyDev *build_sub_topology_gpu_impl(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *sub_net, const GlobalMemMap *gmap, int num_phases, int phase_idx)
{
    SubTopology *h_topo = build_sub_topology<Ti, Tv>(basis, sub_net, gmap, num_phases, phase_idx);

    SubTopologyDev *d_topo = new SubTopologyDev();

    d_topo->send_dim = h_topo->send_dim;
    d_topo->recv_dim = h_topo->recv_dim;
    d_topo->send_counts = std::move(h_topo->send_counts);
    d_topo->recv_counts = std::move(h_topo->recv_counts);

    // =========================================================================
    // 【核心修复】：将 CPU 松散的对称性 ID (h) 映射为 GPU 紧凑的 Block Index (bid)
    // =========================================================================

    // 1. 映射 target_bids
    d_topo->num_targets = h_topo->target_blocks.size();
    std::vector<int> gpu_target_bids;
    gpu_target_bids.reserve(d_topo->num_targets);
    for (int h : h_topo->target_blocks)
    {
        // 通过 block_map 将物理 h 转为连续的 bid
        gpu_target_bids.push_back((int)basis->block_map[h]);
    }

    // 2. 映射 topo_offsets
    std::vector<int64> gpu_topo_offsets(basis->num_blocks, -1);
    for (int i = 0; i < basis->num_blocks; ++i)
    {
        int64 h = basis->blocks[i].asym * basis->num_irreps + basis->blocks[i].bsym;
        // 把索引为 h 的偏移量，塞进索引为 bid (即 i) 的数组中
        gpu_topo_offsets[i] = h_topo->block_offsets_in_cache[h];
    }
    // =========================================================================


    // Build the same virtual chunk basis as the CPU distributed expm/backgrad
    // path: targets are the only destination blocks, while all cached ghost
    // blocks remain addressable through the virtual block map.
    std::vector<int> virtual_physical_bids;
    std::vector<int> virtual_block_map(basis->num_irreps * basis->num_irreps, -1);
    auto append_virtual_block = [&](int h) {
        const int physical_bid = (int)basis->block_map[h];
        if (physical_bid == -1)
            return;
        virtual_block_map[h] = (int)virtual_physical_bids.size();
        virtual_physical_bids.push_back(physical_bid);
    };
    for (int h : h_topo->target_blocks)
        append_virtual_block(h);
    const int num_virtual_targets = (int)virtual_physical_bids.size();
    for (int h = 0; h < basis->num_irreps * basis->num_irreps; ++h)
    {
        if (virtual_block_map[h] == -1 && h_topo->block_offsets_in_cache[h] != -1)
            append_virtual_block(h);
    }

    d_topo->virt_num_blocks = (int)virtual_physical_bids.size();
    std::vector<int64> virt_offsets(d_topo->virt_num_blocks);
    std::vector<int> virt_num_a(d_topo->virt_num_blocks);
    std::vector<int> virt_num_b(d_topo->virt_num_blocks);
    std::vector<int> virt_asym(d_topo->virt_num_blocks);
    std::vector<int> virt_bsym(d_topo->virt_num_blocks);
    std::vector<int64> virt_astrs_start(d_topo->virt_num_blocks);
    std::vector<int64> virt_bstrs_start(d_topo->virt_num_blocks);
    for (int vi = 0; vi < d_topo->virt_num_blocks; ++vi)
    {
        const int bid = virtual_physical_bids[vi];
        const auto &blk = basis->blocks[bid];
        const int h = (int)(blk.asym * basis->num_irreps + blk.bsym);
        virt_offsets[vi] = h_topo->block_offsets_in_cache[h];
        virt_num_a[vi] = (int)blk.num_a;
        virt_num_b[vi] = (int)blk.num_b;
        virt_asym[vi] = (int)blk.asym;
        virt_bsym[vi] = (int)blk.bsym;
        virt_astrs_start[vi] = 0;
        virt_bstrs_start[vi] = 0;
        for (int bj = 0; bj < bid; ++bj)
        {
            virt_astrs_start[vi] += basis->blocks[bj].num_a;
            virt_bstrs_start[vi] += basis->blocks[bj].num_b;
        }
    }
    if (!virtual_block_map.empty()) d_topo->d_virt_block_map = up(virtual_block_map.data(), virtual_block_map.size());
    if (d_topo->virt_num_blocks > 0)
    {
        d_topo->d_virt_block_offsets = up(virt_offsets.data(), d_topo->virt_num_blocks);
        d_topo->d_virt_block_num_a = up(virt_num_a.data(), d_topo->virt_num_blocks);
        d_topo->d_virt_block_num_b = up(virt_num_b.data(), d_topo->virt_num_blocks);
        d_topo->d_virt_block_asym = up(virt_asym.data(), d_topo->virt_num_blocks);
        d_topo->d_virt_block_bsym = up(virt_bsym.data(), d_topo->virt_num_blocks);
        d_topo->d_virt_astrs_start = up(virt_astrs_start.data(), d_topo->virt_num_blocks);
        d_topo->d_virt_bstrs_start = up(virt_bstrs_start.data(), d_topo->virt_num_blocks);
    }
    d_topo->num_targets = num_virtual_targets;

    d_topo->num_pack_jobs = h_topo->pack_jobs.size();

    // 将映射后的干净数据上传到 GPU
    if (d_topo->num_targets > 0)
    {
        d_topo->d_target_bids = up(gpu_target_bids.data(), d_topo->num_targets);
    }

    if (basis->num_blocks > 0)
    {
        d_topo->d_topo_offsets = up(gpu_topo_offsets.data(), basis->num_blocks);
    }

    if (d_topo->num_pack_jobs > 0)
    {
        d_topo->d_pack_jobs = up(h_topo->pack_jobs.data(), d_topo->num_pack_jobs);
    }

    delete h_topo; // 清理 CPU 临时拓扑

    return d_topo;
}

template <typename Ti>
BasisSliceDev<Ti> make_dist_virtual_basis_slice(const BasisViewDev<Ti> &basis, const SubTopologyDev &topo)
{
    BasisSliceDev<Ti> slice = make_basis_slice(basis);
    slice.num_blocks = topo.num_targets;
    slice.block_offsets = topo.d_virt_block_offsets;
    slice.block_num_a = topo.d_virt_block_num_a;
    slice.block_num_b = topo.d_virt_block_num_b;
    slice.block_asym = topo.d_virt_block_asym;
    slice.block_bsym = topo.d_virt_block_bsym;
    slice.astrs_start = topo.d_virt_astrs_start;
    slice.bstrs_start = topo.d_virt_bstrs_start;
    slice.block_map = topo.d_virt_block_map;
    slice.target_bids = nullptr;
    return slice;
}

template <typename Ti, typename Tv>
void expm_svd_sub_chunk_gpu(const BasisViewDev<Ti> &basis, const NetworkDev<Ti, Tv> &net, const SubTopologyDev &topo, int64 idx, double theta, Tv *dev_vec)
{
    BasisSliceDev<Ti> slice = make_dist_virtual_basis_slice(basis, topo);
    expm_svd_network_otf_gpu<Ti, Tv>(slice, net, idx, theta, dev_vec);
}

template <typename Ti, typename Tv>
Tv backgrad_svd_sub_chunk_gpu(const BasisViewDev<Ti> &basis, const NetworkDev<Ti, Tv> &net, const SubTopologyDev &topo, double theta, Tv *lp, Tv *rp)
{
    BasisSliceDev<Ti> slice = make_dist_virtual_basis_slice(basis, topo);
    return backgrad_svd_network_otf_gpu<Ti, Tv>(slice, net, 0, theta, lp, rp);
}

// =================================================================
// GPU 高速显存打包内核
// =================================================================
template <typename Tv>
__global__ void pack_send_buffer_gpu_kernel(const PackJob *jobs, int num_jobs, const Tv *__restrict__ local_v, Tv *__restrict__ send_buffer)
{
    const int job_idx = blockIdx.x;
    if (job_idx < num_jobs)
    {
        PackJob job = jobs[job_idx];
        for (int64 i = threadIdx.x; i < job.size; i += blockDim.x)
        {
            send_buffer[job.dst_offset + i] = local_v[job.src_offset + i];
        }
    }
}
