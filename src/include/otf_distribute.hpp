#pragma once
#include <mpi.h>
#include <vector>
#include "common.hpp"
#include "otf.hpp"

// 记录一个 Block 在整个集群中的物理位置 (对应第 1, 2 级分组)
struct BlockLocation
{
    int mpi_rank;       // 这个 Block 存放在哪个节点
    int64 local_offset; // 在该节点本地 src_vec 中的偏移量
    int32 num_a;        // 该 Block 的 alpha 弦数量
    int32 num_b;        // 该 Block 的 beta 弦数量

    inline bool is_local(int my_rank) const { return mpi_rank == my_rank; }
};

#pragma once
#include <mpi.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include "common.hpp"

struct BlockLocation
{
    int mpi_rank;       // 这个 Block 存放在哪个节点
    int64 local_offset; // 在该节点本地 src_vec 中的偏移量
    int32 num_a;        // 该 Block 的 alpha 弦数量
    int32 num_b;        // 该 Block 的 beta 弦数量

    inline bool is_local(int my_rank) const { return mpi_rank == my_rank; }
};

template <typename Ti,
          typename Tv>
struct DistributedBasisManager
{
    int my_rank;
    int num_ranks;
    int64 local_dim; // 当前节点负责的波函数维度大小

    int64 num_irreps;
    int64 num_local_blocks;
    std::vector<BlockDesc<Ti>> local_blocks;

    // 全局路由表：大小为 num_irreps * num_irreps
    // 索引为 bid = asym * num_irreps + bsym
    std::vector<BlockLocation> routing_table;

    // 本地波函数数组 (由于 MPI_Win 生命周期管理，由 Manager 负责分配更安全)
    Tv *local_src_vec = nullptr;

    // 单向 MPI 窗口 (用于 MPI_Get RDMA 拉取)
    MPI_Win win_src_vec = MPI_WIN_NULL;

    // 构造函数：接受全局的 Blocks 信息，并在内部完成分布式切分
    DistributedBasisManager(MPI_Comm comm,
                            int64 global_num_irreps,
                            int64 global_num_blocks,
                            const BlockDesc<Ti> *global_blocks)
        : num_irreps(global_num_irreps)
    {
        MPI_Comm_rank(comm, &my_rank);
        MPI_Comm_size(comm, &num_ranks);

        routing_table.resize(num_irreps * num_irreps, {-1, -1, 0, 0});

        // 1. 简单的负载均衡切分 (贪心/轮询/连续切分)
        // 这里采用连续切分策略，确保连续的 asym 尽量在同一个节点，提升局部性
        std::vector<int64> rank_dims(num_ranks, 0);
        std::vector<int64> rank_offsets(num_ranks, 0); // 记录每个 Rank 分配到的当前偏移量

        int64 total_global_dim = 0;
        for (int64 i = 0; i < global_num_blocks; ++i)
        {
            total_global_dim += (int64)global_blocks[i].num_a * global_blocks[i].num_b;
        }

        int64 target_dim_per_rank = total_global_dim / num_ranks;
        int current_target_rank = 0;
        int64 current_rank_acc = 0;

        for (int64 i = 0; i < global_num_blocks; ++i)
        {
            const auto &block = global_blocks[i];
            int64 block_size = (int64)block.num_a * block.num_b;
            int64 bid = block.asym * num_irreps + block.bsym;

            // 决定这个 block 分配给哪个 rank
            // 如果当前 rank 加上这个 block 严重超出目标负载，且不是最后一个 rank，则切到下一个
            if (current_target_rank < num_ranks - 1 &&
                current_rank_acc + block_size > target_dim_per_rank &&
                current_rank_acc > 0)
            {
                current_target_rank++;
                current_rank_acc = 0;
            }

            // 2. 填写全局路由表 (所有节点都能算出同样的路由表，无需通信)
            BlockLocation loc;
            loc.mpi_rank = current_target_rank;
            loc.local_offset = rank_offsets[current_target_rank]; // 该块在这个 rank 上的局部偏移
            loc.num_a = block.num_a;
            loc.num_b = block.num_b;

            routing_table[bid] = loc;

            // 如果这个 block 属于我，加入本地列表
            if (current_target_rank == my_rank)
            {
                BlockDesc<Ti> local_blk = block;
                // 重写 offset 为局部 offset！
                local_blk.offset = loc.local_offset;
                local_blocks.push_back(local_blk);
            }

            // 更新累加器和偏移量
            rank_offsets[current_target_rank] += block_size;
            current_rank_acc += block_size;
            rank_dims[current_target_rank] += block_size;
        }

        num_local_blocks = local_blocks.size();
        local_dim = rank_dims[my_rank];

        // 3. 使用 MPI_Alloc_mem 分配锁页内存 (Pinned Memory)
        // 这对 RDMA (MPI_Get) 性能至关重要，能避免网卡驱动在传输时做额外的页表复制
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "alloc_shared_noncontig", "true"); // 允许底层做更优的非连续共享内存优化

        MPI_Alloc_mem(local_dim * sizeof(Tv), info, &local_src_vec);

        // 初始化为 0
        std::fill(local_src_vec, local_src_vec + local_dim, Tv{0});

        // 4. 创建单向通信窗口
        // 将 local_src_vec 暴露给同一个通讯域里的其他节点
        MPI_Win_create(local_src_vec,          // 暴露的内存基地址
                       local_dim * sizeof(Tv), // 暴露的总字节数
                       sizeof(Tv),             // 偏移量计算单位（1 个 displacement 对应的字节数）
                       info,                   // 优化提示
                       comm,
                       &win_src_vec); // 输出的窗口句柄

        MPI_Info_free(&info);
    }

    ~DistributedBasisManager()
    {
        if (win_src_vec != MPI_WIN_NULL)
        {
            MPI_Win_free(&win_src_vec);
        }
        if (local_src_vec != nullptr)
        {
            MPI_Free_mem(local_src_vec);
        }
    }
};

template <typename Ti, typename Tv>
struct AxGroup
{
    Ti ax;
    int64 axsym; // ax 的对称性

    // 挂载在这个 ax 下的所有 beta 激发（内部按 rank 排序，退化为 Pure B）
    std::vector<SVDGroup_OTF<Ti, Tv>> bx_groups;
};

template <typename Ti, typename Tv>
struct DistributedNetwork_OTF
{
    // 现有的无通信组，保持不变
    std::vector<SVDGroup_OTF<Ti, Tv>> diag_groups;
    std::vector<SVDGroup_OTF<Ti, Tv>> pure_b_groups;

    // 需要通信的组，按照 ax 重新打包
    std::vector<AxGroup<Ti, Tv>> pure_a_driven_groups;
    std::vector<AxGroup<Ti, Tv>> mixed_driven_groups;

    // ... 构建函数中，先对输入按 ax 排序，相同的 ax 塞进同一个 AxGroup ...
};

template <typename Tv>
struct GhostBuffer
{
    Tv *data = nullptr;
    int64 capacity = 0;
    MPI_Request req = MPI_REQUEST_NULL;

    GhostBuffer(int64 max_block_size)
    {
        capacity = max_block_size;
        // 分配 64 字节对齐内存，极其利好 AVX-512
        data = static_cast<Tv *>(_mm_malloc(capacity * sizeof(Tv), 64));
    }

    ~GhostBuffer()
    {
        if (data)
            _mm_free(data);
    }

    // 发起异步拉取 (单边通信 RDMA 风格)
    void fetch_async(int target_rank, int64 target_offset, int64 count, MPI_Win win)
    {
        MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, win);
        MPI_Get(data, count, mpi_type<Tv>(), target_rank, target_offset, count, mpi_type<Tv>(), win);
        MPI_Win_unlock(target_rank, win);
        // 注：如果是传统双边，这里就是 MPI_Irecv
    }
};

template <int Rank, typename Ti, typename Tv>
static inline void gather_contract_mixed_distributed(
    const DistributedBasisManager<Ti> *__restrict__ basis,
    const AxGroup<Ti, Tv> &ax_group, // 当前处理的一个 ax
    const Tv *__restrict__ local_src,
    Tv *__restrict__ local_dst,
    GhostBuffer<Tv> &ghost_buf)
{
    const BlockDesc<Ti> *local_blocks = basis->local_blocks;
    const int64 num_irreps = basis->num_irreps;

    // 1. 遍历本地所有的 alpha 块
    for (int dst_block_idx = 0; dst_block_idx < basis->num_local_blocks; ++dst_block_idx)
    {
        const BlockDesc<Ti> &dst_block = local_blocks[dst_block_idx];

        // 2. 预测源数据块在哪里 (核心神技：对称性路由)
        const int64 src_asym = dst_block.asym ^ ax_group.axsym;
        const int64 bid_src = src_asym * num_irreps + dst_block.bsym; // 假设 B 对称性暂时不变，用于拉取整块

        const BlockLocation &src_loc = basis->routing_table[bid_src];

        const Tv *active_src_ptr = nullptr;

        // 3. 通信或本地指针绑定
        if (src_loc.is_local(basis->my_rank))
        {
            // 太棒了，数据就在本地！(Zero Communication)
            active_src_ptr = local_src + src_loc.local_offset;
        }
        else
        {
            // 数据在远端，发起 RDMA 拉取
            int64 fetch_size = (int64)src_loc.num_a * src_loc.num_b;
            ghost_buf.fetch_async(src_loc.mpi_rank, src_loc.local_offset, fetch_size, basis->win_src_vec);

            // active_src_ptr 指向 Ghost Buffer (它已经被拉到本地并连续排布了)
            active_src_ptr = ghost_buf.data;
        }

        // 4. 计算阶段：对于这个 block 和这个 ax，遍历所有挂载的 beta 激发！
        // =================================================================
        // 注意看！这里彻底退化成了你之前的 gather_contract_pure_b_batched_impl
        // 我们不再关心 MPI，不再关心网络，只对着 active_src_ptr 狂算！
        // =================================================================

        for (int64 batch_start = 0; batch_start < ax_group.bx_groups.size(); batch_start += BATCH_SIZE)
        {
            // 复用你已经写好的 SOA 阶段计算、位运算奇偶校验
            // ... (复用你的 SharedBatchBuffer 逻辑) ...

            for (int a = 0; a < dst_block.num_a; ++a)
            {
                const Ti dst_str_a = dst_block.astrs[a];
                const Ti src_str_a = dst_str_a ^ ax_group.ax; // 第 3 级分组发挥作用

                // 这里需要一个局部的 a_idx_map，把 src_str_a 映射到 0 ~ src_loc.num_a-1
                const int src_a_idx = local_idx_map.find(src_str_a);

                Tv *dst = local_dst + dst_block.offset + a * dst_block.num_b;

                // 指向正确的数据行 (不管它是来自本地还是 Ghost Buffer)
                const Tv *src_row = active_src_ptr + src_a_idx * src_loc.num_b;

                // 调用你修复后的终极极速 SOA 内核
                // update_vec_indirect_aos<Rank, Tv>(... src_row, dst ...);
            }
        }
    }
}
