#pragma once
#include "common.hpp"
#include <vector>

struct TransR1
{
    uint32 src_idx;
    uint32 dst_idx;
    double w0;
};

struct TransR2
{
    uint32 src_idx;
    uint32 dst_idx;
    double w0, w1;
};

struct TransRN
{
    uint32 src_idx;
    uint32 dst_idx;
    uint64 w_offset;
};

struct PureRoute
{
    uint64 jump_offset;
    uint64 phase_offset;
    uint32 n;
    uint16 block_src_idx;
    uint16 block_dst_idx;
};

struct MixedRoute
{
    uint64 a_jump_offset;
    uint64 b_jump_offset;
    uint32 na;
    uint32 nb;
    uint16 block_src_idx;
    uint16 block_dst_idx;
};

struct GroupArena
{
    uint16 rank;

    TransR1 *r1_jumps;
    uint64 num_r1_jumps;
    double *r1_phases;
    uint64 num_r1_phases;

    TransR2 *r2_jumps;
    uint64 num_r2_jumps;
    double *r2_phases;
    uint64 num_r2_phases;

    TransRN *rn_jumps;
    uint64 num_rn_jumps;
    double *rn_weights;
    uint64 num_rn_weights;
    double *rn_phases;
    uint64 num_rn_phases;
};

struct SVDNetwork
{
    uint64 ngs;
    uint8 *excit_types;
    uint16 *group_ranks;

    GroupArena *arenas;

    PureRoute **pure_a_routes;
    uint64 *num_pure_a_routes;
    PureRoute **pure_b_routes;
    uint64 *num_pure_b_routes;
    MixedRoute **mixed_routes;
    uint64 *num_mixed_routes;
};

struct TempArena
{
    std::vector<TransR1> r1_jumps;
    std::vector<double> r1_phases;

    std::vector<TransR2> r2_jumps;
    std::vector<double> r2_phases;

    std::vector<TransRN> rn_jumps;
    std::vector<double> rn_weights;
    std::vector<double> rn_phases;

    std::vector<PureRoute> pure_routes;
    std::vector<MixedRoute> mixed_routes;

    void rollback_jumps(int rank, uint64 j_size, uint64 w_size)
    {
        if (rank == 1)
            r1_jumps.resize(j_size);
        else if (rank == 2)
            r2_jumps.resize(j_size);
        else
        {
            rn_jumps.resize(j_size);
            rn_weights.resize(w_size);
        }
    }
};

#define PURE_A_IDX(blk, idx, b) ((blk).offset + (int64)(idx) * (blk).num_b + (b))
#define PURE_B_IDX(blk, a, idx) ((blk).offset + (int64)(a) * (blk).num_b + (idx))
#define MIXED_IDX(blk, a_idx, b_idx) ((blk).offset + (int64)(a_idx) * (blk).num_b + (b_idx))

void build_pure_a(
    int64 g, uint32 ax, int rank, int64 na, int64 nb,
    int64 offset_az, int64 offset_bz,
    int64 offset_wa, int64 offset_wb,
    const uint32 *flat_azs, const uint32 *flat_bzs,
    const double *flat_wa, const double *flat_wb,
    const int64 *orbsym, const BasisManager *basis, TempArena &temp);

void build_pure_b(
    int64 g, uint32 bx, int rank, int64 na, int64 nb,
    int64 offset_az, int64 offset_bz,
    int64 offset_wa, int64 offset_wb,
    const uint32 *flat_azs, const uint32 *flat_bzs,
    const double *flat_wa, const double *flat_wb,
    const int64 *orbsym, const BasisManager *basis, TempArena &temp);

void build_mixed(
    int64 g, uint32 ax, uint32 bx, int rank, int64 na, int64 nb,
    int64 offset_az, int64 offset_bz,
    int64 offset_wa, int64 offset_wb,
    const uint32 *flat_azs, const uint32 *flat_bzs,
    const double *flat_wa, const double *flat_wb,
    const int64 *orbsym, const BasisManager *basis, TempArena &temp);

void apply_diag_terms(
    const BasisManager *__restrict__ basis,
    const uint32 *__restrict__ azs,
    const uint32 *__restrict__ bzs,
    const double *__restrict__ cs,
    const int64 n_terms,
    const double *__restrict__ src,
    double *__restrict__ dst);

void hvec_pure_a(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst);

void hvec_pure_b(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst);

void hvec_mixed(
    const BasisManager *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst);

void tvec_pure_a(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec);

void tvec_pure_b(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec);

void tvec_mixed(
    const BasisManager *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec);

double grad_pure_a(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    const double *__restrict__ lp,
    const double *__restrict__ rp);

double grad_pure_b(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    const double *__restrict__ lp,
    const double *__restrict__ rp);

double grad_mixed(
    const BasisManager *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    const double *__restrict__ lp,
    const double *__restrict__ rp);
