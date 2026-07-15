#= ============================================================
 Sci v4 — block + 三层 batch select, 基于 sci3.jl 的类型和命名
============================================================ =#

using LinearAlgebra, Combinatorics, Printf

struct SVDGroup{Ti,Tv}
    ax::Ti
    bx::Ti
    rank::Int
    num_za::Int
    num_zb::Int
    unique_zas::Vector{Ti}
    unique_zbs::Vector{Ti}
    wa::Vector{Tv}
    wb::Vector{Tv}
end

struct BlockDesc{Ti}
    asym::Int
    bsym::Int
    num_a::Int
    num_b::Int
    astrs::Vector{Ti}
    bstrs::Vector{Ti}
    offset::Int
end

const Group{Ti,Tv} = Tuple{Int,SVDGroup{Ti,Tv}}
const Phase{Tv} = Tuple{Vararg{Tv}}

struct shared_data{Tv}
    groupd_dst_idxs::Vector{Vector{Int}}
    groupd_src_idxs::Vector{Vector{Int}}
    groupd_src_blk_idxs::Vector{Vector{Int}}
    groupd_phases::Vector{Vector{Phase{Tv}}}
end

function get_symm(str::Ti, orbsym::Vector{Int}) where Ti
    i = 0
    sym = 0
    while str != 0
        if str & 1 != 0
            sym ⊻= orbsym[i+1]
        end
        str >>= 1
        i += 1
    end
    return sym
end

function expand_bitstrings_bitstr(
    src_astrs::Vector{Ti}, src_bstrs::Vector{Ti},
    axs::Vector{Ti}, bxs::Vector{Ti},
    nelec::Tuple{Int,Int}, orbsym::Vector{Int64}, num_irreps::Int=16,
) where Ti
    na, nb = nelec
    a_map = Dict{Ti,Int}(a => 0 for a in src_astrs)
    b_map = Dict{Ti,Int}(b => 0 for b in src_bstrs)
    for (ax, bx) in zip(axs, bxs)
        if ax != 0
            for a in src_astrs
                new_a = a ⊻ ax
                count_ones(new_a) == na && get!(a_map, new_a, 1)
            end
        end
        if bx != 0
            for b in src_bstrs
                new_b = b ⊻ bx
                count_ones(new_b) == nb && get!(b_map, new_b, 1)
            end
        end
    end
    by_sym = [Vector{Ti}() for _ in 1:num_irreps]
    for a in keys(a_map)
        sym = get_symm(a, orbsym)
        0 <= sym < num_irreps && push!(by_sym[sym+1], a)
    end
    dst_astrs = Ti[]
    for s in 1:num_irreps
        sort!(by_sym[s])
        append!(dst_astrs, by_sym[s])
    end
    by_sym = [Vector{Ti}() for _ in 1:num_irreps]
    for b in keys(b_map)
        sym = get_symm(b, orbsym)
        0 <= sym < num_irreps && push!(by_sym[sym+1], b)
    end
    dst_bstrs = Ti[]
    for s in 1:num_irreps
        sort!(by_sym[s])
        append!(dst_bstrs, by_sym[s])
    end
    is_new_a = Bool[a_map[a] == 1 for a in dst_astrs]
    is_new_b = Bool[b_map[b] == 1 for b in dst_bstrs]
    return dst_astrs, dst_bstrs, is_new_a, is_new_b
end

function build_idx_map(blocks::Vector{BlockDesc{Ti}}, side::Symbol) where Ti
    idx_map = Dict{Ti,Tuple{Int,Int}}()
    for (blk_i, blk) in enumerate(blocks)
        ss = (side == :alpha ? blk.astrs : blk.bstrs)
        for (loc, s) in enumerate(ss)
            idx_map[s] = (loc - 1, blk_i - 1)
        end
    end
    return idx_map
end

# ============================================================
# 构建 link (只在 batch 内临时使用)
# ============================================================

function get_old2new_link(new_strs::Vector{Ti}, groups::Vector{SVDGroup{Ti,Tv}}, side::Symbol) where {Ti,Tv}
    link = Vector{Vector{Group{Ti,Tv}}}(undef, length(new_strs))
    for (i, str) in enumerate(new_strs)
        tmp = Group{Ti,Tv}[]
        for (ig, group) in enumerate(groups)
            src_str = (side == :alpha ? group.ax : group.bx) ⊻ str
            !(src_str in new_strs) && push!(tmp, (ig, group))
        end
        link[i] = tmp
    end
    return link
end

function get_old2old_link(old_strs::Vector{Ti}, groups::Vector{SVDGroup{Ti,Tv}}, side::Symbol) where {Ti,Tv}
    link = Vector{Vector{Group{Ti,Tv}}}(undef, length(old_strs))
    for (i, str) in enumerate(old_strs)
        tmp = Group{Ti,Tv}[]
        for (ig, group) in enumerate(groups)
            src_str = (side == :alpha ? group.ax : group.bx) ⊻ str
            src_str in old_strs && push!(tmp, (ig, group))
        end
        link[i] = tmp
    end
    return link
end

function precompute_phase(str::Ti, zs::Vector{Ti}, w::Vector{Tv}, rank::Int) where {Ti,Tv}
    if rank == 1
        v = 0.0
        for i in eachindex(zs)
            v += w[i] * (1 - 2 * (count_ones(str & zs[i]) & 1))
        end
        return (v)
    else
        nz = length(zs)
        v0, v1 = 0.0, 0.0
        for i in eachindex(zs)
            phase = 1 - 2 * (count_ones(str & zs[i]) & 1)
            v0 += w[i] * phase
            v1 += w[nz+i] * phase
        end
        return (v0, v1)
    end
end

function compute_coeff(pa::Vector{Phase{Tv}}, pb::Vector{Phase{Tv}}, rank::Int) where Tv
    v = pa[1] * pb[1]
    rank >= 2 && (v += pa[2] * pb[2])
    return v
end

# ============================================================
# 临时 precompute (只对当前 g_chunk × b_chunk)
# ============================================================

function precompute_shared_data_chunk(
    new_chunk::Vector{Ti},                  # 当前 b batch 的 target strings
    global_offset::Int,                     # b batch 的全局起始索引
    g_chunk::Vector{SVDGroup{Ti,Tv}},       # 当前 groups batch
    g_start::Int,                           # g batch 的全局起始索引
    old_idx_map::Dict{Ti,Tuple{Int,Int}},
    side::Symbol,
) where {Ti,Tv}
    ngs = length(g_chunk)

    dst_idxs = Vector{Vector{Int}}(undef, ngs)
    src_idxs = Vector{Vector{Int}}(undef, ngs)
    src_blks = Vector{Vector{Int}}(undef, ngs)
    phases = Vector{Vector{Phase{Tv}}}(undef, ngs)

    for (i, dst) in enumerate(new_chunk)
        i_global = global_offset + i
        for (g, group) in enumerate(g_chunk)
            src = (side == :alpha ? group.ax : group.bx) ⊻ dst
            val = get(old_idx_map, src, nothing)
            val === nothing && continue
            src_i, src_blk = val

            push!(dst_idxs[g], i_global)
            push!(src_idxs[g], src_i)
            push!(src_blks[g], src_blk)

            if side == :alpha
                p = precompute_phase(src, group.unique_zas, group.wa, group.rank)
            else
                p = precompute_phase(src, group.unique_zbs, group.wb, group.rank)
            end
            push!(phases[g], p)
        end
    end

    return shared_data(dst_idxs, src_idxs, src_blks, phases)
end

# ============================================================
# Pass A: new_α batch → Part 1 + Part 3 (block + batch 版)
# ============================================================

function select_pass_a_batch!(
    # —— 全量数据 ——
    new_α_full::Vector{Ti},
    old_β_full::Vector{Ti},
    new_β_full::Vector{Ti},
    old_α_set::Set{Ti},
    old_a_idx_map::Dict{Ti,Tuple{Int,Int}},
    old_b_idx_map::Dict{Ti,Tuple{Int,Int}},
    all_groups::Vector{SVDGroup{Ti,Tv}},
    src_psi::Vector{Float64},
    src_blocks::Vector{BlockDesc{Ti}},
    E_var::Float64, eps::Float64,

    # —— 三层 batch 大小 ——
    A_CHUNK::Int=32,
    G_CHUNK::Int=2000,
    B_CHUNK::Int=2000,
) where {Ti,Tv}
    n_new_α = length(new_α_full)
    n_old = length(old_β_full)
    n_new = length(new_β_full)
    ngroups = length(all_groups)

    output_p1 = Vector{Tuple{Ti,Ti}}()
    output_p3 = Vector{Tuple{Ti,Ti}}()

    # ==================================================================
    #  Layer 1: new_α batch
    # ==================================================================
    for a_start in 1:A_CHUNK:n_new_α
        a_end = min(a_start + A_CHUNK - 1, n_new_α)
        a_chunk = new_α_full[a_start:a_end]
        n_a = length(a_chunk)

        # ★ accum 在最外层持久化 — 跨 g_chunk 和 b_chunk 自然累加
        accum_old = zeros(Float64, n_a, n_old)
        accum_new = zeros(Float64, n_a, n_new)

        # ==============================================================
        #  Layer 2: groups batch
        # ==============================================================
        for g_start in 1:G_CHUNK:ngroups
            g_end = min(g_start + G_CHUNK - 1, ngroups)
            g_chunk = all_groups[g_start:g_end]
            n_g = g_end - g_start + 1

            # ── 临时 alink [n_a][n_g] — 即时构建, 丢弃 ──
            alink_local = [Int[] for _ in 1:n_a]
            for (ig_local, group) in enumerate(g_chunk)
                ig_global = g_start + ig_local - 1
                for (ia, dst_a) in enumerate(a_chunk)
                    src_a = group.ax ⊻ dst_a
                    src_a in old_α_set || continue
                    push!(alink_local[ia], ig_global)
                end
            end

            # ==========================================================
            #  Layer 3a: old_β batch
            # ==========================================================
            for b_old_start in 1:B_CHUNK:n_old
                b_old_end = min(b_old_start + B_CHUNK - 1, n_old)
                b_old_chunk = old_β_full[b_old_start:b_old_end]
                b_offset = b_old_start - 1

                # 临时 shared_b_old — 即时构建, 丢弃
                shared_old = precompute_shared_data_chunk(
                    b_old_chunk, b_offset, g_chunk, g_start,
                    old_b_idx_map, :beta)

                # Phase 2: 累加进 accum_old (持久)
                for ia in 1:n_a
                    dst_a = a_chunk[ia]
                    for ig_global in alink_local[ia]
                        ig_local = ig_global - g_start + 1
                        group = g_chunk[ig_local]

                        src_a = dst_a ⊻ group.ax
                        t = get(old_a_idx_map, src_a, nothing)
                        t === nothing && continue
                        src_ia, src_a_blk_idx = t

                        pa = precompute_phase(src_a, group.unique_zas, group.wa, group.rank)

                        ds = shared_old.groupd_dst_idxs[ig_local]
                        sc = shared_old.groupd_src_idxs[ig_local]
                        sb = shared_old.groupd_src_blk_idxs[ig_local]
                        ph = shared_old.groupd_phases[ig_local]
                        for j in eachindex(ds)
                            src_b_blk_idx = sb[j]
                            src_a_blk_idx == src_b_blk_idx || continue

                            src_blk = src_blocks[src_a_blk_idx+1]

                            old_ib = ds[j]
                            src_ib = sc[j]
                            pb = ph[j]
                            src_gid = src_blk.offset + src_ia * src_blk.num_b + src_ib + 1
                            accum_old[ia, old_ib] += src_psi[src_gid] * compute_coeff(pa, pb, group.rank)
                        end
                    end
                end
            end  # ================== end old_β batch ==================

            # ==========================================================
            #  Layer 3b: new_β batch (对称于 3a)
            # ==========================================================
            for b_new_start in 1:B_CHUNK:n_new
                b_new_end = min(b_new_start + B_CHUNK - 1, n_new)
                b_new_chunk = new_β_full[b_new_start:b_new_end]
                b_offset = b_new_start - 1

                shared_new = precompute_shared_data_chunk(
                    b_new_chunk, b_offset, g_chunk, g_start,
                    old_b_idx_map, :beta)

                for ia in 1:n_a
                    dst_a = a_chunk[ia]
                    for ig_global in alink_local[ia]
                        ig_local = ig_global - g_start + 1
                        group = g_chunk[ig_local]

                        src_a = dst_a ⊻ group.ax
                        t = get(old_a_idx_map, src_a, nothing)
                        t === nothing && continue
                        src_ia, src_a_blk_idx = t

                        pa = precompute_phase(src_a, group.unique_zas, group.wa, group.rank)

                        ds = shared_new.groupd_dst_idxs[ig_local]
                        sc = shared_new.groupd_src_idxs[ig_local]
                        sb = shared_new.groupd_src_blk_idxs[ig_local]
                        ph = shared_new.groupd_phases[ig_local]
                        for j in eachindex(ds)
                            src_b_blk_idx = sb[j]
                            src_a_blk_idx == src_b_blk_idx || continue

                            src_blk = src_blocks[src_a_blk_idx+1]

                            new_ib = ds[j]
                            src_ib = sc[j]
                            pb = ph[j]
                            src_gid = src_blk.offset + src_ia * src_blk.num_b + src_ib + 1
                            accum_new[ia, new_ib] += src_psi[src_gid] * compute_coeff(pa, pb, group.rank)
                        end
                    end
                end
            end  # ================== end new_β batch ==================

        end  # ================== end groups batch ==================

        # ==============================================================
        #  eps_check: 所有 g 和 b 累加完毕后一次性判定
        # ==============================================================
        for ia in 1:n_a
            global_a_idx = a_start + ia - 1
            dst_a = new_α_full[global_a_idx]

            # Part 1 (old_β)
            for old_ib in 1:n_old
                v = accum_old[ia, old_ib]
                v != 0.0 || continue
                if v^2 / (E_var)^2 > eps^2
                    push!(output_p1, (dst_a, old_β_full[old_ib]))
                end
            end

            # Part 3 (new_β)
            for new_ib in 1:n_new
                v = accum_new[ia, new_ib]
                v != 0.0 || continue
                if v^2 / (E_var)^2 > eps^2
                    push!(output_p3, (dst_a, new_β_full[new_ib]))
                end
            end
        end

    end  # ================== end new_α batch ==================

    return output_p1, output_p3
end

# ============================================================
# Pass B: new_β batch → Part 2 (block + batch 版)
# ============================================================

function select_pass_b_batch!(
    new_β_full::Vector{Ti},
    old_α_full::Vector{Ti},
    old_β_set::Set{Ti},
    old_a_idx_map::Dict{Ti,Tuple{Int,Int}},
    old_b_idx_map::Dict{Ti,Tuple{Int,Int}},
    all_groups::Vector{SVDGroup{Ti,Tv}},
    src_psi::Vector{Float64},
    src_blocks::Vector{BlockDesc{Ti}},
    E_var::Float64, eps::Float64, B_CHUNK::Int=32,
    G_CHUNK::Int=2000,
    A_CHUNK::Int=2000,
) where {Ti,Tv}
    n_new_β = length(new_β_full)
    n_old = length(old_α_full)
    ngroups = length(all_groups)
    output_p2 = Vector{Tuple{Ti,Ti}}()

    # ===== Layer 1: new_β batch =====
    for b_start in 1:B_CHUNK:n_new_β
        b_end = min(b_start + B_CHUNK - 1, n_new_β)
        b_chunk = new_β_full[b_start:b_end]
        n_b = length(b_chunk)

        # ★ accum_old: 跨 g 和 a 持久化
        accum_old = zeros(Float64, n_b, n_old)

        # ===== Layer 2: groups batch =====
        for g_start in 1:G_CHUNK:ngroups
            g_end = min(g_start + G_CHUNK - 1, ngroups)
            g_chunk = all_groups[g_start:g_end]
            n_g = g_end - g_start + 1

            # ── 临时 blink_local [n_b][n_g] ──
            blink_local = [Int[] for _ in 1:n_b]
            for (ig_local, group) in enumerate(g_chunk)
                ig_global = g_start + ig_local - 1
                for (ib, dst_b) in enumerate(b_chunk)
                    src_b = group.bx ⊻ dst_b
                    src_b in old_β_set || continue
                    push!(blink_local[ib], ig_global)
                end
            end

            # ===== Layer 3: old_α batch =====
            for a_old_start in 1:A_CHUNK:n_old
                a_old_end = min(a_old_start + A_CHUNK - 1, n_old)
                a_old_chunk = old_α_full[a_old_start:a_old_end]
                a_offset = a_old_start - 1

                # 临时 shared_a_old — 即时构建, 丢弃
                shared_old = precompute_shared_data_chunk(
                    a_old_chunk, a_offset, g_chunk, g_start,
                    old_a_idx_map, :alpha)

                # Phase 2: 累加进 accum_old
                for ib in 1:n_b
                    dst_b = b_chunk[ib]
                    for ig_global in blink_local[ib]
                        ig_local = ig_global - g_start + 1
                        group = g_chunk[ig_local]

                        src_b = dst_b ⊻ group.bx
                        t = get(old_b_idx_map, src_b, nothing)
                        t === nothing && continue
                        src_ib, src_b_blk_idx = t

                        pb = precompute_phase(src_b, group.unique_zbs, group.wb, group.rank)

                        ds = shared_old.groupd_dst_idxs[ig_local]
                        sc = shared_old.groupd_src_idxs[ig_local]
                        sb = shared_old.groupd_src_blk_idxs[ig_local]
                        ph = shared_old.groupd_phases[ig_local]
                        for j in eachindex(ds)
                            src_a_blk_idx = sb[j]
                            src_b_blk_idx == src_a_blk_idx || continue

                            src_blk = src_blocks[src_b_blk_idx+1]

                            old_ia = ds[j]
                            src_ia = sc[j]
                            pa = ph[j]
                            src_gid = src_blk.offset + src_ia * src_blk.num_b + src_ib + 1
                            accum_old[ib, old_ia] += src_psi[src_gid] * compute_coeff(pa, pb, group.rank)
                        end
                    end
                end
            end  # end old_α batch
        end  # end groups batch

        # ── eps_check ──
        for ib in 1:n_b
            global_b_idx = b_start + ib - 1
            dst_b = new_β_full[global_b_idx]
            for old_ia in 1:n_old
                v = accum_old[ib, old_ia]
                v != 0.0 || continue
                if v^2 / (E_var)^2 > eps^2
                    push!(output_p2, (old_α_full[old_ia], dst_b))
                end
            end
        end
    end  # end new_β batch

    return output_p2
end

# ============================================================
# End-to-end 测试
# ============================================================

function run_sci4_test(
    norb::Int, na::Int, nb::Int,
    orbsym::Vector{Int64}, total_sym::Int,
    all_axs::Vector{UInt32}, all_bxs::Vector{UInt32},
    all_groups::Vector{SVDGroup{UInt32,Float64}},
    eps::Float64;
    max_iter::Int=3, verbose::Bool=true,
    A_CHUNK::Int=32, G_CHUNK::Int=2000, B_CHUNK::Int=2000,
)
    hf_astr = UInt32((1 << na) - 1)
    hf_bstr = UInt32((1 << nb) - 1)
    src_astrs = UInt32[hf_astr]
    src_bstrs = UInt32[hf_bstr]

    blocks, _ = get_sym_blocks(norb, (na, nb), orbsym, total_sym, UInt32)
    src_blocks = blocks
    dim = sum(b.num_a * b.num_b for b in blocks)
    psi = ones(Float64, dim)

    E_var = 0.0

    for iter in 1:max_iter
        new_astrs, new_bstrs, is_new_a, is_new_b = expand_bitstrings_bitstr(
            src_astrs, src_bstrs, all_axs, all_bxs, (na, nb), orbsym)

        old_astrs = new_astrs[.!is_new_a]
        old_bstrs = new_bstrs[.!is_new_b]
        new_α = new_astrs[is_new_a]
        new_β = new_bstrs[is_new_b]

        old_α_set = Set(old_astrs)
        old_β_set = Set(old_bstrs)

        verbose && @printf("\nIter %d: |new_α|=%d  |new_β|=%d  |old_α|=%d  |old_β|=%d\n",
            iter, length(new_α), length(new_β), length(old_astrs), length(old_bstrs))

        old_a_idx_map = build_idx_map(blocks, :alpha)
        old_b_idx_map = build_idx_map(blocks, :beta)

        t_pa = @elapsed pairs_p1, pairs_p3 = select_pass_a_batch!(
            new_α, old_bstrs, new_β, old_α_set,
            old_a_idx_map, old_b_idx_map, all_groups,
            psi, src_blocks, E_var, eps,
            A_CHUNK, G_CHUNK, B_CHUNK)

        t_pb = @elapsed pairs_p2 = select_pass_b_batch!(
            new_β, old_astrs, old_β_set,
            old_a_idx_map, old_b_idx_map, all_groups,
            psi, src_blocks, E_var, eps,
            A_CHUNK, G_CHUNK, B_CHUNK)

        all_pairs = vcat(pairs_p1, pairs_p2, pairs_p3)
        verbose && @printf("  P1=%d  P2=%d  P3=%d  total=%d\n",
            length(pairs_p1), length(pairs_p2), length(pairs_p3), length(all_pairs))
        verbose && @printf("  passA=%.3fs  passB=%.3fs\n", t_pa, t_pb)

        length(all_pairs) == 0 && (verbose && println("Done."); break)

        sel_a = unique!(UInt32[p[1] for p in all_pairs])
        sel_b = unique!(UInt32[p[2] for p in all_pairs])
        src_astrs = sort(union(src_astrs, sel_a))
        src_bstrs = sort(union(src_bstrs, sel_b))

        blocks, _ = get_sym_blocks(norb, (na, nb), orbsym, total_sym, UInt32)
        src_blocks = blocks
    end

    return src_astrs, src_bstrs, psi
end

function get_sym_blocks(norb::Int, nelec::Tuple{Int,Int}, orbsym::Vector{Int}, total_sym::Int, Ti::Type)
    na, nb = nelec
    f_str = (str, pos) -> str | (Ti(1) << pos)
    f_sym = (sym, pos) -> sym ⊻ orbsym[pos+1]

    adict = Dict{Int,Vector{Ti}}()
    bdict = Dict{Int,Vector{Ti}}()

    strs = zeros(Ti, binomial(norb, na))
    syms = zeros(Int, binomial(norb, na))
    c = 0
    for comb in combinations(0:norb-1, na)
        c += 1
        strs[c] = foldl(f_str, comb; init=Ti(0))
        syms[c] = foldl(f_sym, comb; init=0)
    end
    for (s, str) in zip(syms, strs)
        v = get!(Ti[], adict, s)
        push!(v, str)
    end

    strs = zeros(Ti, binomial(norb, nb))
    syms = zeros(Int, binomial(norb, nb))
    c = 0
    for comb in combinations(0:norb-1, nb)
        c += 1
        strs[c] = foldl(f_str, comb; init=Ti(0))
        syms[c] = foldl(f_sym, comb; init=0)
    end
    for (s, str) in zip(syms, strs)
        v = get!(Ti[], bdict, s)
        push!(v, str)
    end

    blocks = BlockDesc{Ti}[]
    max_asym = 0
    max_bsym = 0
    off = 0
    for (asym, astrs) in adict
        bsym = total_sym ⊻ asym
        haskey(bdict, bsym) || continue
        bstrs = bdict[bsym]
        sort!(astrs)
        sort!(bstrs)
        push!(blocks, BlockDesc(asym, bsym, length(astrs), length(bstrs), astrs, bstrs, off))
        off += length(astrs) * length(bstrs)
        max_asym = max(max_asym, asym)
        max_bsym = max(max_bsym, bsym)
    end
    block_map = fill(-1, max_asym + 1, max_bsym + 1)
    for (i, b) in enumerate(blocks)
        block_map[b.asym+1, b.bsym+1] = i - 1
    end
    return blocks, block_map
end
