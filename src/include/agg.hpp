#pragma once
#include "common.hpp"
#include "net.hpp"

struct PureA_Edge_R1
{
    uint64 src_ptr;
    double pa0;
    const double *b_phases;
};

struct PureA_Edge_R2
{
    uint64 src_ptr;
    double pa0, pa1;
    const double *b_phases;
};

struct PureA_Edge_RN
{
    uint64 src_ptr;
    const double *pa_weights;
    const double *b_phases;
    uint32 rank;
    uint32 _padding;
};

struct PureB_Edge_R1
{
    uint64 src_offset;
    uint64 src_num_b;
    const double *a_phases;
    double pb0;
    uint32 src_b;
    uint32 dst_b;
};

struct PureB_Edge_R2
{
    uint64 src_offset;
    uint64 src_num_b;
    const double *a_phases;
    double pb0, pb1;
    uint32 src_b;
    uint32 dst_b;
};

struct PureB_Edge_RN
{
    uint64 src_offset;
    uint64 src_num_b;
    const double *pb_weights;
    const double *a_phases;
    uint32 src_b;
    uint32 dst_b;
    uint32 rank;
    uint32 _padding;
};

struct Mixed_Ax_Node
{
    int64 src_ptr;      // 8 bytes: 共享的源指针
    uint32 leaf_offset; // 4 bytes: 对应 g_leaves 数组的起点索引
    uint32 num_leaves;  // 4 bytes: 下辖的 g_leaves 数量
}; // 16 bytes

struct Mixed_G_Leaf_R1
{
    double pa0;             // 8 bytes
    const TransR1 *b_jumps; // 8 bytes
    uint32 num_b_jumps;     // 4 bytes
    uint32 _padding;        // 4 bytes
}; // 24 bytes

struct Mixed_G_Leaf_R2
{
    double pa0, pa1;        // 16 bytes
    const TransR2 *b_jumps; // 8 bytes
    uint32 num_b_jumps;     // 4 bytes
    uint32 _padding;        // 4 bytes
}; // 32 bytes

struct Mixed_G_Leaf_RN
{
    const double *pa_weights;     // 8 bytes
    const TransRN *b_jumps;       // 8 bytes
    const double *b_weights_base; // 8 bytes
    uint32 num_b_jumps;           // 4 bytes
    uint32 rank;                  // 4 bytes
}; // 32 bytes

struct AggBlock
{
    uint32 num_a, num_b;
    uint64 offset_dst;

    uint32 *pure_a_r1_offsets;
    PureA_Edge_R1 *pure_a_r1_edges;
    uint32 *pure_a_r2_offsets;
    PureA_Edge_R2 *pure_a_r2_edges;
    uint32 *pure_a_rn_offsets;
    PureA_Edge_RN *pure_a_rn_edges;

    uint32 num_pure_b_r1;
    PureB_Edge_R1 *pure_b_r1_edges;
    uint32 num_pure_b_r2;
    PureB_Edge_R2 *pure_b_r2_edges;
    uint32 num_pure_b_rn;
    PureB_Edge_RN *pure_b_rn_edges;

    uint32 *mixed_r1_ax_offsets;
    Mixed_Ax_Node *mixed_r1_ax_nodes;
    Mixed_G_Leaf_R1 *mixed_r1_g_leaves;
    uint32 *mixed_r2_ax_offsets;
    Mixed_Ax_Node *mixed_r2_ax_nodes;
    Mixed_G_Leaf_R2 *mixed_r2_g_leaves;
    uint32 *mixed_rn_ax_offsets;
    Mixed_Ax_Node *mixed_rn_ax_nodes;
    Mixed_G_Leaf_RN *mixed_rn_g_leaves;
};

struct AggSVDNetwork
{
    uint32 *azs;
    uint32 *bzs;
    double *cs;
    uint64 *gs;
    uint64 ngs;
    uint8 *excit_types;

    AggBlock *blocks;
    uint64 num_blocks;

    GroupArena *arenas;

    TransR1 **mixed_b_rev_r1;
    TransR2 **mixed_b_rev_r2;
    TransRN **mixed_b_rev_rn;
};

void build_all_agg_edges(
    int64 ngs,
    const uint32 *axs,
    const uint32 *bxs,
    const int64 *ranks,
    const int64 *num_as,
    const int64 *num_bs,
    const uint32 *flat_azs,
    const uint32 *flat_bzs,
    const double *flat_wa,
    const double *flat_wb,
    const int64 *orbsym,
    const BasisManager *basis,
    AggSVDNetwork *agg);

void hvec_aggregated_svd(
    const AggSVDNetwork *__restrict__ agg_net,
    const double *__restrict__ src,
    double *__restrict__ dst);
