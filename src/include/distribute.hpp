#pragma once
#include <mpi.h>
#include "otf.hpp"
#include "hvec.hpp"

// ─── MPI type mapping ───────────────────────────────────────────────────────
template <typename Tv>
inline MPI_Datatype mpi_dtype()
{
    if constexpr (std::is_same_v<Tv, double>)
        return MPI_DOUBLE;
    else if constexpr (std::is_same_v<Tv, std::complex<double>>)
        return MPI_C_DOUBLE_COMPLEX;
    else if constexpr (std::is_same_v<Tv, float>)
        return MPI_FLOAT;
    else if constexpr (std::is_same_v<Tv, std::complex<float>>)
        return MPI_C_FLOAT_COMPLEX;
    else
        static_assert(sizeof(Tv) == 0, "Unsupported MPI type");
}

// ─── Ax-Group: pre-built at init time, read-only in hot path ─────────────────
template <typename Ti, typename Tv>
struct AxGroup
{
    Ti ax;
    int64 bxsym = 0;
    int dispatch_rank = 0;
    std::vector<SVDGroup_OTF<Ti, Tv>> groups;
};

// ─── Block location in the cluster ──────────────────────────────────────────
struct BlockLocation
{
    int mpi_rank;
    int64 local_offset;
    int32 num_a;
    int32 num_b;
    inline bool is_local(int my_rank) const { return mpi_rank == my_rank; }
};

// ─── Distributed basis manager ──────────────────────────────────────────────
template <typename Ti, typename Tv>
struct DistributedBasisManager
{
    int my_rank, num_ranks;
    int64 local_dim, num_irreps, norb;

    MPI_Comm comm;
    int64 global_num_blocks;
    BlockDesc<Ti> *global_blocks;
    int64 *global_block_map;
    int64 *global_orbsym;

    int64 num_local_blocks;
    BlockDesc<Ti> *local_blocks;
    BlockLocation *routing_table;

    int32 *a_idx_map;
    int32 *b_idx_map;

    Tv *local_src_vec;
    MPI_Win win_src_vec;

    DistributedBasisManager() = default;

    DistributedBasisManager(MPI_Comm _comm,
                            int64 _norb, int64 _num_irreps,
                            int64 _global_num_blocks,
                            const BlockDesc<Ti> *_global_blocks,
                            const int64 *_global_block_map,
                            const int64 *_global_orbsym)
        : norb(_norb), num_irreps(_num_irreps), global_num_blocks(_global_num_blocks)
    {
        MPI_Comm_dup(_comm, &this->comm);
        MPI_Comm_rank(this->comm, &my_rank);
        MPI_Comm_size(this->comm, &num_ranks);

        int64 map_entries = num_irreps * num_irreps;
        global_blocks = new BlockDesc<Ti>[global_num_blocks];
        global_block_map = new int64[map_entries];
        global_orbsym = new int64[norb];
        routing_table = new BlockLocation[map_entries];

        std::copy(_global_blocks, _global_blocks + global_num_blocks, global_blocks);
        std::copy(_global_block_map, _global_block_map + map_entries, global_block_map);
        std::copy(_global_orbsym, _global_orbsym + norb, global_orbsym);

        int32 map_size = 1 << norb;
        a_idx_map = new int32[map_size];
        b_idx_map = new int32[map_size];
        std::fill_n(a_idx_map, map_size, int32(-1));
        std::fill_n(b_idx_map, map_size, int32(-1));
        for (int64 i = 0; i < global_num_blocks; ++i)
        {
            const auto &blk = global_blocks[i];
            for (int32 a = 0; a < blk.num_a; ++a)
                a_idx_map[blk.astrs[a]] = a;
            for (int32 b = 0; b < blk.num_b; ++b)
                b_idx_map[blk.bstrs[b]] = b;
        }

        // ── Asym-safe greedy partition ──
        std::fill_n(routing_table, map_entries, BlockLocation{-1, -1, 0, 0});

        int64 total_global_dim = 0;
        for (int64 i = 0; i < global_num_blocks; ++i)
            total_global_dim += (int64)global_blocks[i].num_a * global_blocks[i].num_b;

        int64 target_dim_per_rank = total_global_dim / num_ranks;
        std::vector<int64> rank_offsets(num_ranks, 0);
        std::vector<int64> rank_dims(num_ranks, 0);
        std::vector<BlockDesc<Ti>> local_tmp;

        int cur_rank = 0;
        int64 cur_rank_acc = 0;

        for (int64 i = 0; i < global_num_blocks; ++i)
        {
            const auto &blk = global_blocks[i];
            int64 block_size = (int64)blk.num_a * blk.num_b;
            int64 bid = blk.asym * num_irreps + blk.bsym;

            bool asym_changed = (i > 0 && blk.asym != global_blocks[i - 1].asym);

            if (cur_rank < num_ranks - 1 &&
                cur_rank_acc + block_size > target_dim_per_rank &&
                cur_rank_acc > 0 && asym_changed)
            {
                cur_rank++;
                cur_rank_acc = 0;
            }

            BlockLocation loc{cur_rank, rank_offsets[cur_rank], (int32)blk.num_a, (int32)blk.num_b};
            routing_table[bid] = loc;

            if (cur_rank == my_rank)
            {
                BlockDesc<Ti> lb = blk;
                lb.offset = loc.local_offset;
                local_tmp.push_back(lb);
            }

            rank_offsets[cur_rank] += block_size;
            cur_rank_acc += block_size;
            rank_dims[cur_rank] += block_size;
        }

        num_local_blocks = local_tmp.size();
        local_dim = rank_dims[my_rank];
        local_blocks = new BlockDesc<Ti>[num_local_blocks];
        std::copy(local_tmp.begin(), local_tmp.end(), local_blocks);

        MPI_Alloc_mem(local_dim * sizeof(Tv), MPI_INFO_NULL, &local_src_vec);
        std::fill_n(local_src_vec, local_dim, Tv{});

        if (num_ranks > 1)
            MPI_Win_create(local_src_vec, local_dim * sizeof(Tv), sizeof(Tv),
                           MPI_INFO_NULL, this->comm, &win_src_vec);
        else
            win_src_vec = MPI_WIN_NULL;
    }

    ~DistributedBasisManager()
    {
        if (win_src_vec != MPI_WIN_NULL)
        {
            MPI_Win_free(&win_src_vec);
            win_src_vec = MPI_WIN_NULL;
        }
        if (local_src_vec)
        {
            MPI_Free_mem(local_src_vec);
            local_src_vec = nullptr;
        }
        if (comm != MPI_COMM_NULL)
        {
            MPI_Comm_free(&comm);
            comm = MPI_COMM_NULL;
        }
        delete[] a_idx_map;
        delete[] b_idx_map;
        delete[] global_blocks;
        delete[] global_block_map;
        delete[] global_orbsym;
        delete[] local_blocks;
        delete[] routing_table;
    }

    DistributedBasisManager(const DistributedBasisManager &) = delete;
    DistributedBasisManager &operator=(const DistributedBasisManager &) = delete;
};

// ─── Ghost buffer for remote fetches ────────────────────────────────────────
template <typename Tv>
struct GhostBuffer
{
    Tv *data = nullptr;
    int64 capacity = 0;

    GhostBuffer() = default;
    GhostBuffer(const GhostBuffer &) = delete;
    GhostBuffer &operator=(const GhostBuffer &) = delete;

    void ensure(int64 count)
    {
        if (count <= capacity)
            return;
        if (data)
            delete[] data;
        data = new Tv[count];
        capacity = count;
    }
    ~GhostBuffer()
    {
        if (data)
            delete[] data;
    }

    void fetch(int target_rank, int64 target_offset, int64 count, MPI_Win win)
    {
        ensure(count);
        MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, win);
        MPI_Get(data, count, mpi_dtype<Tv>(), target_rank, target_offset, count, mpi_dtype<Tv>(), win);
        MPI_Win_unlock(target_rank, win);
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// DistributedNetwork_OTF — pre-built at init, read-only in hot path
// ═══════════════════════════════════════════════════════════════════════════════
template <typename Ti, typename Tv>
struct DistributedNetwork_OTF
{
    IndexMap map;
    std::vector<SVDGroup_OTF<Ti, Tv>> diag_groups;
    std::vector<SVDGroup_OTF<Ti, Tv>> pure_b_groups;
    std::vector<AxGroup<Ti, Tv>> pure_a_ax;
    std::vector<AxGroup<Ti, Tv>> mixed_ax;
    int64 num_groups = 0;
};

template <typename Ti, typename Tv>
DistributedNetwork_OTF<Ti, Tv> *build_distributed_network(
    const Network_OTF<Ti, Tv> *net, const int64 *orbsym)
{
    auto *dnet = new DistributedNetwork_OTF<Ti, Tv>();
    dnet->map = net->map;
    dnet->diag_groups = net->diag_groups;
    dnet->pure_b_groups = net->pure_b_groups;
    dnet->num_groups = dnet->diag_groups.size() + dnet->pure_b_groups.size();

    auto group_by_ax = [orbsym](const std::vector<SVDGroup_OTF<Ti, Tv>> &src,
                                std::vector<AxGroup<Ti, Tv>> &dst,
                                bool is_mixed)
    {
        std::map<uint64_t, AxGroup<Ti, Tv>> ag_map;
        for (const auto &g : src)
        {
            int64 kb = is_mixed ? get_string_sym(g.bx, orbsym) : 0;
            int dr = (g.rank == 1 || g.rank == 2) ? (int)g.rank : 0;
            uint64_t key = ((uint64_t)g.ax << 32) | ((uint64_t)(kb & 0xFFFF) << 16) | (dr & 0xFF);
            auto it = ag_map.find(key);
            if (it == ag_map.end())
            {
                AxGroup<Ti, Tv> ag;
                ag.ax = g.ax;
                ag.bxsym = kb;
                ag.dispatch_rank = dr;
                ag.groups.push_back(g);
                ag_map[key] = ag;
            }
            else
            {
                it->second.groups.push_back(g);
            }
        }
        for (auto &kv : ag_map)
            dst.push_back(std::move(kv.second));
    };

    group_by_ax(net->pure_a_groups, dnet->pure_a_ax, false);
    group_by_ax(net->mixed_groups, dnet->mixed_ax, true);

    for (auto &ag : dnet->pure_a_ax)
        dnet->num_groups += ag.groups.size();
    for (auto &ag : dnet->mixed_ax)
        dnet->num_groups += ag.groups.size();

    return dnet;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Distributed contraction kernels
// ═══════════════════════════════════════════════════════════════════════════════

// ── diag: zero communication ─────────────────────────────────────────────────
template <int Rank, typename Ti, typename Tv>
static inline void gather_contract_diag_distributed_impl(
    const DistributedBasisManager<Ti, Tv> *__restrict__ dbasis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> *__restrict__ groups, int64 num_groups,
    const Tv *__restrict__ local_src,
    Tv *__restrict__ local_dst)
{
    const BlockDesc<Ti> *blocks = dbasis->local_blocks;
    const int64 num_blocks = dbasis->num_local_blocks;

    int max_a_count = 0, max_b_count = 0, max_rank = 0;
    get_upper<Rank, Ti, Tv>(blocks, groups, num_blocks, num_groups,
                            max_a_count, max_b_count, max_rank);

    std::vector<Tv> phase_b(BATCH_SIZE * max_b_count * max_rank);

#pragma omp parallel
    {
        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min(BATCH_SIZE, num_groups - batch_start);
#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    Tv *pb0 = phase_b.data() + batch_idx * max_b_count * max_rank;
                    compute_phases<Rank, 1, Ti, Tv>(
                        block.bstrs, block.num_b,
                        group.unique_zbs, group.num_zb, group.wb,
                        pb0, max_b_count, group.rank);
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < block.num_a; ++a)
                {
                    const Ti str_a = block.astrs[a];
                    const Tv *src = local_src + block.offset + a * block.num_b;
                    Tv *dst = local_dst + block.offset + a * block.num_b;
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const Tv *pb0 = phase_b.data() + batch_idx * max_b_count * max_rank;
                        Tv pa0 = {}, pa1 = {}, pan[64] = {};
                        compute_a_phase<Rank, Ti, Tv>(
                            str_a, group.unique_zas, group.num_za,
                            group.wa, group.rank, pa0, pa1, pan);
                        update_dst<Rank, 1, 0, Tv>(
                            block.num_b, group.rank, pa0, pa1, pan,
                            pb0, max_b_count, nullptr, nullptr, src, dst);
                    }
                }
            }
        }
    }
}

// ── pure_b: zero communication (asym-safe partition) ─────────────────────────
template <int Rank, typename Ti, typename Tv>
static inline void gather_contract_pure_b_distributed_impl(
    const DistributedBasisManager<Ti, Tv> *__restrict__ dbasis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> *__restrict__ groups, int64 num_groups,
    const Tv *__restrict__ local_src, Tv *__restrict__ local_dst)
{
    const BlockDesc<Ti> *blocks = dbasis->local_blocks;
    const int64 num_blocks = dbasis->num_local_blocks;
    const int64 *orbsym = dbasis->global_orbsym;
    const int64 num_irreps = dbasis->num_irreps;

    int max_a_count = 0, max_b_count = 0, max_rank = 0;
    const BlockDesc<Ti> *all_blocks = dbasis->global_blocks;
    get_upper<Rank, Ti, Tv>(all_blocks, groups, dbasis->global_num_blocks, num_groups,
                            max_a_count, max_b_count, max_rank);

    SharedBatchBuffer<Tv> batch_buf(BATCH_SIZE, max_b_count, max_rank);

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min(BATCH_SIZE, num_groups - batch_start);
#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    const int64 bxsym = get_string_sym(group.bx, orbsym);
                    const int64 bid = dst_block.asym * num_irreps + (dst_block.bsym ^ bxsym);
                    const int64 src_block_idx = dbasis->global_block_map[bid];
                    batch_buf.src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1)
                    {
                        batch_buf.valid_b_counts[batch_idx] = 0;
                        continue;
                    }

                    batch_buf.valid_b_counts[batch_idx] =
                        compute_phases_symm<Rank, 1, Ti, Tv>(
                            group.bx, idx_map.b_idx_map,
                            dst_block.bstrs, dst_block.num_b,
                            group.unique_zbs, group.num_zb, group.wb,
                            batch_buf.ptr_phase(batch_idx), max_b_count, group.rank,
                            batch_buf.ptr_src_b(batch_idx), batch_buf.ptr_dst_b(batch_idx),
                            false, false);
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti str_a = dst_block.astrs[a];
                    Tv *dst = local_dst + dst_block.offset + a * dst_block.num_b;
                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_b_count = batch_buf.valid_b_counts[batch_idx];
                        if (valid_b_count == 0)
                            continue;

                        const int src_block_idx = batch_buf.src_block_idxs[batch_idx];
                        const BlockDesc<Ti> &src_block = all_blocks[src_block_idx];
                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const Tv *pb0 = batch_buf.ptr_phase(batch_idx);
                        const int *src_b_idx = batch_buf.ptr_src_b(batch_idx);
                        const int *dst_b_idx = batch_buf.ptr_dst_b(batch_idx);

                        const int64 _bxsym = get_string_sym(group.bx, orbsym);
                        const BlockLocation &src_loc = dbasis->routing_table[dst_block.asym * num_irreps + (dst_block.bsym ^ _bxsym)];
                        const Tv *src = local_src + src_loc.local_offset + a * src_block.num_b;

                        Tv pa0 = {}, pa1 = {}, pan[64] = {};
                        compute_a_phase<Rank, Ti, Tv>(
                            str_a, group.unique_zas, group.num_za,
                            group.wa, group.rank, pa0, pa1, pan);
                        update_dst<Rank, 1, 1, Tv>(
                            valid_b_count, group.rank, pa0, pa1, pan,
                            pb0, max_b_count, src_b_idx, dst_b_idx, src, dst);
                    }
                }
            }
        }
    }
}

// ── ax-driven kernel (pure_a + mixed): fetch once per ax-group, then OpenMP ──
template <int Rank, typename Ti, typename Tv>
static inline void gather_contract_ax_distributed_impl(
    const DistributedBasisManager<Ti, Tv> *__restrict__ dbasis,
    const IndexMap &idx_map,
    const AxGroup<Ti, Tv> &ax_group,
    const Tv *__restrict__ local_src, Tv *__restrict__ local_dst,
    GhostBuffer<Tv> &ghost_buf,
    bool is_mixed)
{
    const BlockDesc<Ti> *blocks = dbasis->local_blocks;
    const int64 num_blocks = dbasis->num_local_blocks;
    const int64 *orbsym = dbasis->global_orbsym;
    const int64 num_irreps = dbasis->num_irreps;
    const Ti ax = ax_group.ax;
    const int64 num_bx = ax_group.groups.size();

    if (num_bx == 0)
        return;

    const int64 axsym = get_string_sym(ax, orbsym);
    const int64 bxsym = is_mixed ? ax_group.bxsym : 0;

    int max_a_count = 0, max_b_count = 0, max_rank = 0;
    get_upper<Rank, Ti, Tv>(dbasis->global_blocks, ax_group.groups.data(),
                            dbasis->global_num_blocks, num_bx,
                            max_a_count, max_b_count, max_rank);

    for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
    {
        const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];

        int64 bid = (dst_block.asym ^ axsym) * num_irreps + (dst_block.bsym ^ bxsym);
        int64 src_block_idx = dbasis->global_block_map[bid];
        if (src_block_idx == -1)
            continue;

        // ── Single-threaded resolve & fetch ──
        const BlockLocation &src_loc = dbasis->routing_table[bid];
        const Tv *active_src;
        if (src_loc.is_local(dbasis->my_rank))
            active_src = local_src + src_loc.local_offset;
        else
        {
            int64 fetch_sz = (int64)src_loc.num_a * src_loc.num_b;
            ghost_buf.fetch(src_loc.mpi_rank, src_loc.local_offset, fetch_sz, dbasis->win_src_vec);
            active_src = ghost_buf.data;
        }
        const BlockDesc<Ti> &src_block = dbasis->global_blocks[src_block_idx];

        SharedBatchBuffer<Tv> batch_buf(BATCH_SIZE, max_b_count, max_rank);

        // ── OpenMP parallel over bx groups ──
#pragma omp parallel
        {
            for (int64 batch_start = 0; batch_start < num_bx; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min((int64)BATCH_SIZE, num_bx - batch_start);

#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = ax_group.groups[batch_start + batch_idx];
                    batch_buf.valid_b_counts[batch_idx] =
                        compute_phases_symm<Rank, 1, Ti, Tv>(
                            group.bx, idx_map.b_idx_map,
                            dst_block.bstrs, dst_block.num_b,
                            group.unique_zbs, group.num_zb, group.wb,
                            batch_buf.ptr_phase(batch_idx), max_b_count, group.rank,
                            batch_buf.ptr_src_b(batch_idx), batch_buf.ptr_dst_b(batch_idx),
                            false, false);
                }

#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_block.num_a; ++a)
                {
                    const Ti dst_str_a = dst_block.astrs[a];
                    const Ti src_str_a = dst_str_a ^ ax;
                    const int src_a_idx = idx_map.a_idx_map[src_str_a];
                    if (src_a_idx == -1)
                        continue;

                    Tv *dst = local_dst + dst_block.offset + a * dst_block.num_b;
                    const Tv *src_row = active_src + src_a_idx * src_block.num_b;

                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_b_count = batch_buf.valid_b_counts[batch_idx];
                        if (valid_b_count == 0)
                            continue;

                        const SVDGroup_OTF<Ti, Tv> &group = ax_group.groups[batch_start + batch_idx];
                        const Tv *pb0 = batch_buf.ptr_phase(batch_idx);

                        Tv pa0 = {}, pa1 = {}, pan[64] = {};
                        compute_a_phase<Rank, Ti, Tv>(
                            src_str_a, group.unique_zas, group.num_za,
                            group.wa, group.rank, pa0, pa1, pan);

                        update_dst<Rank, 1, 1, Tv>(
                            valid_b_count, group.rank, pa0, pa1, pan,
                            pb0, max_b_count,
                            batch_buf.ptr_src_b(batch_idx), batch_buf.ptr_dst_b(batch_idx),
                            src_row, dst);
                    }
                }
            }
        } // end omp parallel
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dispatch functions ── zero allocation hot path
// ═══════════════════════════════════════════════════════════════════════════════

// ── Flat dispatch (diag / pure_b): chunk by rank ────────────────────────────
template <int TypeCode, typename Ti, typename Tv>
static inline void dispatch_dist_flat(
    const DistributedBasisManager<Ti, Tv> *dbasis,
    const IndexMap &map,
    const std::vector<SVDGroup_OTF<Ti, Tv>> &groups,
    const Tv *local_src, Tv *local_dst)
{
    int64 total = groups.size();
    if (total == 0)
        return;

    const SVDGroup_OTF<Ti, Tv> *gptr = groups.data();
    int64 start = 0;
    while (start < total)
    {
        int cr = gptr[start].rank;
        int dr = (cr == 1 || cr == 2) ? cr : 0;

        int64 end = start + 1;
        while (end < total)
        {
            int nr = gptr[end].rank;
            int ndr = (nr == 1 || nr == 2) ? nr : 0;
            if (ndr != dr)
                break;
            end++;
        }

        const SVDGroup_OTF<Ti, Tv> *cp = gptr + start;
        int64 sz = end - start;

        if constexpr (TypeCode == 0)
        {
            switch (dr)
            {
            case 1:
                gather_contract_diag_distributed_impl<1>(dbasis, map, cp, sz, local_src, local_dst);
                break;
            case 2:
                gather_contract_diag_distributed_impl<2>(dbasis, map, cp, sz, local_src, local_dst);
                break;
            default:
                gather_contract_diag_distributed_impl<0>(dbasis, map, cp, sz, local_src, local_dst);
                break;
            }
        }
        else
        {
            switch (dr)
            {
            case 1:
                gather_contract_pure_b_distributed_impl<1>(dbasis, map, cp, sz, local_src, local_dst);
                break;
            case 2:
                gather_contract_pure_b_distributed_impl<2>(dbasis, map, cp, sz, local_src, local_dst);
                break;
            default:
                gather_contract_pure_b_distributed_impl<0>(dbasis, map, cp, sz, local_src, local_dst);
                break;
            }
        }
        start = end;
    }
}

// ── Ax-driven dispatch: iterate pre-built AxGroup vector ─────────────────────
template <typename Ti, typename Tv>
static inline void dispatch_dist_ax(
    const DistributedBasisManager<Ti, Tv> *dbasis,
    const IndexMap &map,
    const std::vector<AxGroup<Ti, Tv>> &ax_groups,
    const Tv *local_src, Tv *local_dst,
    GhostBuffer<Tv> &ghost_buf,
    bool is_mixed)
{
    for (const auto &ag : ax_groups)
    {
        int dr = ag.dispatch_rank;
        switch (dr)
        {
        case 1:
            gather_contract_ax_distributed_impl<1>(dbasis, map, ag, local_src, local_dst, ghost_buf, is_mixed);
            break;
        case 2:
            gather_contract_ax_distributed_impl<2>(dbasis, map, ag, local_src, local_dst, ghost_buf, is_mixed);
            break;
        default:
            gather_contract_ax_distributed_impl<0>(dbasis, map, ag, local_src, local_dst, ghost_buf, is_mixed);
            break;
        }
    }
}

// ─── Top-level distributed hvec contract ────────────────────────────────────
template <typename Ti, typename Tv>
void contract_network_otf_distributed(
    const DistributedBasisManager<Ti, Tv> *__restrict__ dbasis,
    const DistributedNetwork_OTF<Ti, Tv> *__restrict__ dnet,
    const Tv *__restrict__ local_src,
    Tv *__restrict__ local_dst)
{
    int64 local_dim = dbasis->local_dim;

    // Sync Julia vector → RMA window, barrier ensures all ranks have synced
    std::copy(local_src, local_src + local_dim, dbasis->local_src_vec);
    if (dbasis->num_ranks > 1)
        MPI_Barrier(dbasis->comm);

#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < local_dim; ++i)
        local_dst[i] = {};

    GhostBuffer<Tv> ghost_buf;

    dispatch_dist_flat<0>(dbasis, dnet->map, dnet->diag_groups, local_src, local_dst);
    dispatch_dist_flat<2>(dbasis, dnet->map, dnet->pure_b_groups, local_src, local_dst);
    dispatch_dist_ax(dbasis, dnet->map, dnet->pure_a_ax, local_src, local_dst, ghost_buf, false);
    dispatch_dist_ax(dbasis, dnet->map, dnet->mixed_ax, local_src, local_dst, ghost_buf, true);
}
