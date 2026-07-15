#= ============================================================
 Sci v3 — block-aware, non-batch, spin-separated select
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
    groupd_src_blk_idxs::Vector{Vector{Int}}        # ★ NEW: source b block index
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

# ============================================================
# idx_map: str → (local_idx_0idx, blk_idx_0idx)
# ============================================================

function build_idx_map(blocks::Vector{BlockDesc{Ti}}, side::Symbol) where Ti # SCI 一般都是大体系, 所以用字典
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
# 构建 link: target_str → valid group entries
# ============================================================

function get_old2new_link(new_strs::Vector{Ti}, groups::Vector{SVDGroup{Ti,Tv}}, side::Symbol) where {Ti,Tv}
    link = Vector{Vector{Group{Ti,Tv}}}(undef, length(new_strs))
    for (i, str) in enumerate(new_strs)
        tmp = Group{Ti,Tv}[]
        for (ig, group) in enumerate(groups)
            src_str = (side == :alpha ? group.ax : group.bx) ⊻ str
            # new_str XOR ax → 结果不在 new_strs 中, 即是 old
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
# Phase 1: precompute shared_data per-group (block-aware)
# ============================================================

function precompute_shared_data(
    new_strs::Vector{Ti}, old_idx_map::Dict{Ti,Tuple{Int,Int}},
    link::Vector{Vector{Group{Ti,Tv}}}, ngs::Int, side::Symbol,
) where {Ti,Tv}
    dst_idxs = Vector{Vector{Int}}(undef, ngs)
    src_idxs = Vector{Vector{Int}}(undef, ngs)
    src_blks = Vector{Vector{Int}}(undef, ngs)
    phases = Vector{Vector{Phase{Tv}}}(undef, ngs)

    # 因为 new 不需要反向索引获取值, 仅用于计算全局坐标
    # 因此 new_idx_map 直接按顺序即可
    for (i, dst) in enumerate(new_strs)
        for (g, group) in link[i]
            src = (side == :alpha ? group.ax : group.bx) ⊻ dst
            val = get(old_idx_map, src, nothing)
            val === nothing && continue
            src_i, src_blk = val            # ★ unpack block index

            push!(dst_idxs[g], i)
            push!(src_idxs[g], src_i)      # ★ store local index
            push!(src_blks[g], src_blk)    # ★ store block index

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
# Pass A: new_α 驱动 → Part 1 (new_α × old_β) + Part 3 (new_α × new_β)
# ============================================================

function select_pass_a!(
    new_astrs::Vector{Ti},
    new_bstrs::Vector{Ti},
    old_bstrs::Vector{Ti},
    alink::Vector{Vector{Group{Ti,Tv}}},
    old_a_idx_map::Dict{Ti,Tuple{Int,Int}},
    shared_b_old::shared_data{Tv},
    shared_b_new::shared_data{Tv},
    src_psi::Vector{Float64},
    src_blocks::Vector{BlockDesc{Ti}},
    E_var::Float64, eps::Float64,
) where {Ti,Tv}
    output_p1 = Vector{Tuple{Ti,Ti}}()
    output_p3 = Vector{Tuple{Ti,Ti}}()

    Threads.@threads for (ia, dst_a) in enumerate(new_astrs)
        accum_old = zeros(length(old_bstrs))
        accum_new = zeros(length(new_bstrs))

        for (ig, group) in alink[ia]
            src_a = dst_a ⊻ group.ax
            t = get(old_a_idx_map, src_a, nothing)
            t === nothing && continue
            src_ia, src_a_blk_idx = t # ★ unpack a-side block

            pa = precompute_phase(src_a, group.unique_zas, group.wa, group.rank)

            # ── Part 1: old_β (mixed + pure) ──
            ds = shared_b_old.groupd_dst_idxs[ig]
            sc = shared_b_old.groupd_src_idxs[ig]
            sb = shared_b_old.groupd_src_blk_idxs[ig]
            ph = shared_b_old.groupd_phases[ig]
            for j in eachindex(ds)
                src_b_blk_idx = sb[j]                       # ★ b-side block
                src_a_blk_idx == src_b_blk_idx || continue  # ★ same-block guard

                src_blk = src_blocks[src_a_blk_idx+1]

                old_ib = ds[j]
                src_ib = sc[j]
                pb = ph[j]
                src_gid = src_blk.offset + src_ia * src_blk.num_b + src_ib + 1
                accum_old[old_ib] += src_psi[src_gid] * compute_coeff(pa, pb, group.rank)
            end

            # ── Part 3: new_β (mixed only) ──
            ds = shared_b_new.groupd_dst_idxs[ig]
            sc = shared_b_new.groupd_src_idxs[ig]
            sb = shared_b_new.groupd_src_blk_idxs[ig]
            ph = shared_b_new.groupd_phases[ig]
            for j in eachindex(ds)
                src_b_blk_idx = sb[j]
                src_a_blk_idx == src_b_blk_idx || continue

                src_blk = src_blocks[src_a_blk_idx+1]
                new_ib = ds[j]
                src_ib = sc[j]
                pb = ph[j]
                src_gid = src_blk.offset + src_ia * src_blk.num_b + src_ib + 1
                accum_new[new_ib] += src_psi[src_gid] * compute_coeff(pa, pb, group.rank)
            end
        end

        # ── eps_check Part 1 (old_β) ──
        for (old_ib, old_b) in enumerate(old_bstrs)
            v = accum_old[old_ib]
            v != 0.0 || continue
            if v^2 / (E_var)^2 > eps^2
                push!(output_p1, (dst_a, old_b))
            end
        end

        # ── eps_check Part 3 (new_β) ──
        for (new_ib, new_b) in enumerate(new_bstrs)
            v = accum_new[new_ib]
            v != 0.0 || continue
            if v^2 / (E_var)^2 > eps^2
                push!(output_p3, (dst_a, new_b))
            end
        end
    end

    return output_p1, output_p3
end

# ============================================================
# Pass B: new_β 驱动 → Part 2 (old_α × new_β)
# ============================================================

function select_pass_b!(
    new_bstrs::Vector{Ti},
    old_astrs::Vector{Ti},
    blink::Vector{Vector{Group{Ti,Tv}}},
    old_b_idx_map::Dict{Ti,Tuple{Int,Int}},
    shared_a_old::shared_data{Tv},
    src_psi::Vector{Float64},
    src_blocks::Vector{BlockDesc{Ti}},
    E_var::Float64, eps::Float64,
) where {Ti,Tv}
    output_p2 = Vector{Tuple{Ti,Ti}}()

    Threads.@threads for (ib, dst_b) in enumerate(new_bstrs)
        accum_old = zeros(Float64, length(old_astrs))

        for (ig, group) in blink[ib]
            src_b = dst_b ⊻ group.bx
            t = get(old_b_idx_map, src_b, nothing)
            t === nothing && continue
            src_ib, src_b_blk_idx = t # ★ unpack b-side block

            pb = precompute_phase!(src_b, group.unique_zbs, group.wb, group.rank)

            # ── Part 2: old_α (mixed + pure) ──
            ds = shared_a_old.groupd_dst_idxs[ig]
            sc = shared_a_old.groupd_src_idxs[ig]
            sb = shared_a_old.groupd_src_blk_idxs[ig]
            ph = shared_a_old.groupd_phases[ig]
            for j in eachindex(ds)
                src_a_blk_idx = sb[j]                       # ★ a-side block
                src_b_blk_idx == src_a_blk_idx || continue  # ★ same-block guard

                src_blk = src_blocks[src_b_blk_idx+1]

                old_ia = ds[j]
                src_ia = sc[j]
                pa = ph[j]
                src_gid = src_blk.offset + src_ia * src_blk.num_b + src_ib + 1
                accum_old[old_ia] += src_psi[src_gid] * compute_coeff(pa, pb, group.rank)
            end
        end

        # ── eps_check Part 2 (old_α) ──
        for (old_ia, old_a) in enumerate(old_astrs)
            v = accum_old[old_ia]
            v != 0.0 || continue
            if v^2 / (E_var)^2 > eps^2
                push!(output_p2, (old_a, dst_b))
            end
        end
    end

    return output_p2
end

# ============================================================
# End-to-end 测试
# ============================================================

function run_sci3_test(
    norb::Int, na::Int, nb::Int,
    orbsym::Vector{Int64}, total_sym::Int,
    all_axs::Vector{UInt32}, all_bxs::Vector{UInt32},
    all_groups::Vector{SVDGroup{UInt32,Float64}},
    eps::Float64;
    max_iter::Int=3, verbose::Bool=true,
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
    ngs = length(all_groups)

    for iter in 1:max_iter
        new_astrs, new_bstrs, is_new_a, is_new_b = expand_bitstrings_bitstr(
            src_astrs, src_bstrs, all_axs, all_bxs, (na, nb), orbsym)

        old_astrs = new_astrs[.!is_new_a]
        old_bstrs = new_bstrs[.!is_new_b]
        new_α = new_astrs[is_new_a]
        new_β = new_bstrs[is_new_b]

        verbose && @printf("\nIter %d: |new_α|=%d  |new_β|=%d  |old_α|=%d  |old_β|=%d\n",
            iter, length(new_α), length(new_β), length(old_astrs), length(old_bstrs))

        old_a_idx_map = build_idx_map(blocks, :alpha)
        old_b_idx_map = build_idx_map(blocks, :beta)

        a_n2o = get_old2new_link(new_α, all_groups, :alpha)
        a_o2o = get_old2old_link(old_astrs, all_groups, :alpha)
        b_n2o = get_old2new_link(new_β, all_groups, :beta)
        b_o2o = get_old2old_link(old_bstrs, all_groups, :beta)

        t_p1 = @elapsed begin
            shared_b_new = precompute_shared_data(new_β, old_b_idx_map, b_n2o, ngs, :beta)
            shared_b_old = precompute_shared_data(old_bstrs, old_b_idx_map, b_o2o, ngs, :beta)
            shared_a_old = precompute_shared_data(old_astrs, old_a_idx_map, a_o2o, ngs, :alpha)
        end

        t_pa = @elapsed pairs_p1, pairs_p3 = select_pass_a!(
            new_α, new_β, old_bstrs, a_n2o, old_a_idx_map,
            shared_b_old, shared_b_new, psi, src_blocks, E_var, eps)

        t_pb = @elapsed pairs_p2 = select_pass_b!(
            new_β, old_astrs, b_n2o, old_b_idx_map,
            shared_a_old, psi, src_blocks, E_var, eps)

        all_pairs = vcat(pairs_p1, pairs_p2, pairs_p3)
        verbose && @printf("  P1=%d  P2=%d  P3=%d  total=%d\n",
            length(pairs_p1), length(pairs_p2), length(pairs_p3), length(all_pairs))
        verbose && @printf("  phase1=%.3fs  passA=%.3fs  passB=%.3fs\n", t_p1, t_pa, t_pb)

        length(all_pairs) == 0 && (verbose && println("Done."); break)

        sel_a = unique!(UInt32[p[1] for p in all_pairs])
        sel_b = unique!(UInt32[p[2] for p in all_pairs])
        old_len_a, old_len_b = length(old_astrs), length(old_bstrs)
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
