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

function get_symm(str::Ti, orbsym::Vector{Int}) where {Ti}
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

const Group{Ti,Tv} = Tuple{Int, SVDGroup{Ti,Tv}}
const Phase{Tv} = Tuple{Vararg{Tv}}

function get_old2new_link(
    new_strs::Vector{Ti}, groups::Vector{SVDGroup{Ti,Tv}}, side::Symbol,
)
    link = Vector{Vector{Group{Ti,Tv}}}(undef, length(new_strs))
    for (i, str) in enumerate(new_strs)
        tmp = Tuple{Int, SVDGroup}[]
        for (ig, group) in enumerate(groups)
            src_str = (side == :alpha ? group.ax : group.bx) ⊻ str
            if !(src_str in new_strs) # 不在 new 中即是 old, 可以链接到 old, 所以有效.
                push!(tmp, (ig, group))
            end
        end
        link[i] = tmp
    end

    return link
end

function get_old2old_link(
    old_strs::Vector{Ti}, groups::Vector{SVDGroup{Ti,Tv}}, side::Symbol, 
)
    link = Vector{Vector{Group{Ti,Tv}}}(undef, length(old_strs))
    for (i, str) in enumerate(old_strs)
        tmp = Tuple{Int, SVDGroup}[]
        for (ig, group) in enumerate(groups)
            src_str = (side == :alpha ? group.ax : group.bx) ⊻ str
            if src_str in old_strs # 可以链接到 old, 所以有效.
                push!(tmp, (ig, group))
            end
        end
        link[i] = tmp
    end

    return link
end

struct shared_data{Tv}
    groupd_dst_idxs::Vector{Vector{Int}}
    groupd_src_idxs::Vector{Vector{Int}}
    groupd_phases::Vector{Vector{Phase{Tv}}}
end

function precompute_phase(str::Ti, zs::Vector{Ti}, w::Vector{Tv}, rank::Int)
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

function precompute_shared_data(
    new_strs::Vector{Ti}, old_idx_map, 
    link::Vector{Vector{Group{Ti,Tv}}}, ngs::Int, side::Symbol, 
)
    dst_idxs = Vector{Vector{Int}}(undef, ngs)
    src_idxs = Vector{Vector{Int}}(undef, ngs)
    phases   = Vector{Vector{Phase{Tv}}}(undef, ngs)

    for (i, dst) in enumerate(new_strs) # 因为 new 不需要反向索引获取值, 仅用于计算全局坐标, 因此 new_idx_map 直接按顺序即可
        for (g, group) in link[i]
            src   = (side == :alpha ? group.ax : group.bx) ⊻ dst # src 只有是 old 才有效
            src_i = old_idx_map[src]
            src_i == -1 && continue
            push!(dst_idxs[g], i)
            push!(src_idxs[g], src_i)
            if side == :alpha
                p = precompute_phase(src, group.unique_zas, group.wa, group.rank)
                push!(phases[g], p)
            else
                p = precompute_phase(src, group.unique_zbs, group.wb, group.rank)
                push!(phases[g], p)
            end
        end
    end

    return shared_data(dst_idxs, src_idxs, phases)
end

function compute_coeff(pa::Vector{Phase{Tv}}, pb::Vector{Phase{Tv}}, rank::Int)
    v = pa[1] * pb[1]
    rank >= 2 && (v += pa[2] * pb[2])
    return v
end

a_n2o = get_old2new_link(new_astrs, groups, :alpha) # new_α → valid groups
a_o2o = get_old2old_link(old_astrs, groups, :alpha) # old_α → valid groups (Phase 1 a-side)
b_n2o = get_old2new_link(new_bstrs, groups, :beta)  # new_β → valid groups
b_o2o = get_old2old_link(old_bstrs, groups, :beta)  # old_β → valid groups (Phase 1 b-side)

shared_b_new = precompute_shared_data(new_bstrs, old_b_idx_map, blink_new, ngs, :beta)  # Part 3 mixed
shared_b_old = precompute_shared_data(old_bstrs, old_b_idx_map, blink_old, ngs, :beta)  # Part 1 mixed
shared_a_old = precompute_shared_data(old_astrs, old_a_idx_map, alink_old, ngs, :alpha) # Part 2 mixed

function select_pass_a!(
    new_astrs::Vector{Ti}, 
    new_bstrs::Vector{Ti},
    old_bstrs::Vector{Ti},
    alink::Vector{Vector{Group{Ti,Tv}}},    # new_α → valid groups
    old_a_idx_map,                          # source α → (local_idx, blk_idx)
    shared_b_old::shared_data{Tv},          # Phase 1: old_β entries (mixed only)
    shared_b_new::shared_data{Tv},          # Phase 1: new_β entries (mixed only)
    psi,                                    # source wavefunction (no block, simplified)
    E_var::Float64, 
    eps::Float64,
) where {Ti,Tv}

    output_p1 = Vector{Tuple{Ti,Ti}}() # (new_α, old_β) pairs
    output_p3 = Vector{Tuple{Ti,Ti}}() # (new_α, new_β) pairs

    @threads for (ia, dst_a) in enumerate(new_astrs) 
        accum_old = zeros(length(old_bstrs))
        accum_new = zeros(length(new_bstrs))

        for (ig, group) in alink[ia]
            src_a  = dst_a ⊻ group.ax
            src_ia = old_a_idx_map[src_a]  
            src_ia == -1 && continue         
            pa = precompute_phase(src_a, group.unique_zas, group.wa, group.rank)
            # ── mixed: Part 1 (old_β, pb 预计算) ──
            ds = shared_b_old.groupd_dst_bidxs[ig]
            si = shared_b_old.groupd_src_bidxs[ig]
            ph = shared_b_old.groupd_phases[ig]
            for j in eachindex(ds)
                old_ib = ds[j]
                src_ib = si[j]
                pb = ph[j]
                accum_old[old_ib] += psi[src_ia, src_ib] * compute_coeff(pa, pb, group.rank)
            end
            # ── mixed: Part 3 (new_β, pb 预计算) ──
            ds = shared_b_new.groupd_dst_bidxs[ig]
            si = shared_b_new.groupd_src_bidxs[ig]
            ph = shared_b_new.groupd_phases[ig]
            for j in eachindex(ds)
                new_ib = ds[j]
                src_ib = si[j]
                pb = ph[j]
                accum_new[new_ib] += psi[src_ia, src_ib] * compute_coeff(pa, pb, group.rank)
            end
        end
        
        # ── eps_check Part 1 (old_β) ──
        for (old_ib, old_b) in enumerate(old_bstrs)
            v = accum_old[old_ib]
            v != 0 || continue
            if v^2 / (E_var)^2 > eps^2
                push!(output_p1, (dst_a, old_b))
            end
        end

        # ── eps_check Part 3 (new_β) ──
        for (new_ib, new_b) in enumerate(new_bstrs)
            v = accum_new[new_ib]
            v != 0 || continue
            if v^2 / (E_var)^2 > eps^2
                push!(output_p3, (dst_a, new_b))
            end
        end
    end

    return output_p1, output_p3
end

function select_pass_b!(
    new_bstrs::Vector{Ti},
    old_astrs::Vector{Ti},
    blink::Vector{Vector{Group{Ti,Tv}}},                # new_β → valid groups
    old_b_idx_map,
    shared_a_old::shared_data{Tv},                      # Phase 1: old_α entries
    psi, 
    E_var::Float64, 
    eps::Float64,
) where {Ti,Tv}

    output_p2 = Vector{Tuple{Ti,Ti}}() # (old_α, new_β)

    @threads for (ib, dst_b) in enumerate(new_bstrs)
        accum_old = zeros(Float64, length(old_astrs))

        for (ig, group) in blink[ib]
            src_b  = dst_b ⊻ group.bx
            src_ib = old_b_idx_map[src_b]
            src_ib == -1 && continue
            pb = precompute_phase(src_b, group.unique_zbs, group.wb, group.rank)
            # ── mixed: Part 2 (old_α, pa 预计算) ──
            ds = shared_a_old.groupd_dst_bidxs[ig]
            si = shared_a_old.groupd_src_bidxs[ig]
            ph = shared_a_old.groupd_phases[ig]
            for j in eachindex(ds)
                old_ia = ds[j]
                src_ia = si[j]
                pa = ph[j]
                accum_old[old_ia] += psi[src_ia, src_ib] * compute_coeff(pa, pb, group.rank)
            end
        end

        for (old_ia, old_a) in enumerate(old_astrs)
            v = accum_old[old_ia]
            v != 0 || continue
            if v^2 / (E_var)^2 > eps^2
                push!(output_p2, (old_a, dst_b))
            end
        end
    end

    return output_p2
end














function select_pass_a!(
    # —— 全量数据 ——
    new_α_full::Vector{Ti},               # 全部 new α strings
    old_β_full::Vector{Ti},               # 全部 old β strings
    new_β_full::Vector{Ti},               # 全部 new β strings
    old_α_set::Set{Ti},                   # old α hash set (快速成员判断)
    old_a_idx_map::Dict{Ti,Tuple{Int,Int}}, # old α → (local_idx_0idx, blk_idx_0idx)
    old_b_idx_map::Dict{Ti,Tuple{Int,Int}}, # old β → (local_idx_0idx, blk_idx_0idx)
    all_groups::Vector{SVDGroup{Ti,Tv}},
    psi,
    E_var::Float64, eps::Float64,

    # —— 三层 batch 大小 ——
    A_CHUNK::Int = 32,                    # new_α batch
    G_CHUNK::Int = 2000,                  # groups batch
    B_CHUNK::Int = 2000,                  # old_β / new_β batch
) where {Ti,Tv}

    n_new_α = length(new_α_full)
    n_old   = length(old_β_full)
    n_new   = length(new_β_full)
    ngroups = length(all_groups)

    output_p1 = Vector{Tuple{Ti,Ti}}()    # (new_α, old_β)
    output_p3 = Vector{Tuple{Ti,Ti}}()    # (new_α, new_β)

    # ====================================================================
    #  Layer 1: new_α batch
    # ====================================================================
    for a_start in 1:A_CHUNK:n_new_α
        a_end   = min(a_start + A_CHUNK - 1, n_new_α)
        a_chunk = new_α_full[a_start : a_end]
        n_a     = length(a_chunk)

        # ★ accum_old/accum_new 是最外层持久窗口
        #   所有 g_chunk 和 b_chunk 都累加进同一个 (ia, ib) 槽位
        accum_old = zeros(Float64, n_a, n_old)   # (a行, old_β列)
        accum_new = zeros(Float64, n_a, n_new)   # (a行, new_β列)

        # ================================================================
        #  Layer 2: groups batch
        # ================================================================
        for g_start in 1:G_CHUNK:ngroups
            g_end   = min(g_start + G_CHUNK - 1, ngroups)
            g_chunk = all_groups[g_start : g_end]
            n_g     = g_end - g_start + 1

            # ── 临时: alink_local [n_a][n_g] — 对 a_chunk + g_chunk 即时构建 ──
            alink_local = [Int[] for _ in 1:n_a]
            for (ig_local, group) in enumerate(g_chunk)
                ig_global = g_start + ig_local - 1
                for (ia, dst_a) in enumerate(a_chunk)
                    src_a = group.ax ⊻ dst_a
                    src_a in old_α_set || continue
                    push!(alink_local[ia], ig_global)
                end
            end

            # ============================================================
            #  Layer 3a: old_β batch
            # ============================================================
            for b_old_start in 1:B_CHUNK:n_old
                b_old_end   = min(b_old_start + B_CHUNK - 1, n_old)
                b_old_chunk = old_β_full[b_old_start : b_old_end]
                n_b_old     = b_old_end - b_old_start + 1
                b_offset    = b_old_start - 1       # ★ 全局索引偏移

                # ── 临时: shared_b_old [n_g][n_b_old] 三套平展向量 ──
                dst_idxs_chunk = [Int[]   for _ in 1:n_g]
                src_idxs_chunk = [Int[]   for _ in 1:n_g]
                phases_chunk   = [Vector{Float64}[] for _ in 1:n_g]

                for (ib_local, old_b) in enumerate(b_old_chunk)
                    ib_global = b_offset + ib_local           # ★ 全局 old_β 索引
                    for (ig_local, group) in enumerate(g_chunk)
                        src_b   = old_b ⊻ group.bx
                        t = get(old_b_idx_map, src_b, nothing)
                        t === nothing && continue
                        src_ib_local, src_b_blk = t

                        pb = zeros(Float64, group.rank)
                        precompute_phase!(pb, src_b, group.unique_zbs, group.num_zb, group.wb, group.rank)

                        push!(dst_idxs_chunk[ig_local], ib_global)   # ★ 全局索引
                        push!(src_idxs_chunk[ig_local], src_ib_local)
                        push!(phases_chunk[ig_local],   pb)
                    end
                end

                # ── Phase 2: 累加进 accum_old (持久) ──
                for ia in 1:n_a
                    dst_a = a_chunk[ia]
                    for ig_global in alink_local[ia]
                        ig_local = ig_global - g_start + 1
                        group   = g_chunk[ig_local]

                        # pa: 即时算 (依赖 dst_a)
                        src_a = dst_a ⊻ group.ax
                        t = get(old_a_idx_map, src_a, nothing)
                        t === nothing && continue
                        src_ia_local, src_a_blk = t
                        pa = zeros(Float64, group.rank)
                        precompute_phase!(pa, src_a, group.unique_zas, group.num_za, group.wa, group.rank)

                        ds = dst_idxs_chunk[ig_local]
                        sc = src_idxs_chunk[ig_local]
                        ph = phases_chunk[ig_local]
                        for j in eachindex(ds)
                            accum_old[ia, ds[j]] += psi[src_ia_local, sc[j]] * compute_coeff(pa, ph[j], group.rank)
                        end
                    end
                end
            end  # ================== end old_β batch ==================

            # ============================================================
            #  Layer 3b: new_β batch (对称于 3a)
            # ============================================================
            for b_new_start in 1:B_CHUNK:n_new
                b_new_end   = min(b_new_start + B_CHUNK - 1, n_new)
                b_new_chunk = new_β_full[b_new_start : b_new_end]
                n_b_new     = b_new_end - b_new_start + 1
                b_offset    = b_new_start - 1

                dst_idxs_chunk = [Int[]   for _ in 1:n_g]
                src_idxs_chunk = [Int[]   for _ in 1:n_g]
                phases_chunk   = [Vector{Float64}[] for _ in 1:n_g]

                for (ib_local, new_b) in enumerate(b_new_chunk)
                    ib_global = b_offset + ib_local
                    for (ig_local, group) in enumerate(g_chunk)
                        src_b = new_b ⊻ group.bx
                        t = get(old_b_idx_map, src_b, nothing)
                        t === nothing && continue
                        src_ib_local, _ = t
                        pb = zeros(Float64, group.rank)
                        precompute_phase!(pb, src_b, group.unique_zbs, group.num_zb, group.wb, group.rank)
                        push!(dst_idxs_chunk[ig_local], ib_global)
                        push!(src_idxs_chunk[ig_local], src_ib_local)
                        push!(phases_chunk[ig_local],   pb)
                    end
                end

                for ia in 1:n_a
                    dst_a = a_chunk[ia]
                    for ig_global in alink_local[ia]
                        ig_local = ig_global - g_start + 1
                        group   = g_chunk[ig_local]
                        src_a = dst_a ⊻ group.ax
                        t = get(old_a_idx_map, src_a, nothing)
                        t === nothing && continue
                        src_ia_local, _ = t
                        pa = zeros(Float64, group.rank)
                        precompute_phase!(pa, src_a, group.unique_zas, group.num_za, group.wa, group.rank)

                        ds = dst_idxs_chunk[ig_local]
                        sc = src_idxs_chunk[ig_local]
                        ph = phases_chunk[ig_local]
                        for j in eachindex(ds)
                            accum_new[ia, ds[j]] += psi[src_ia_local, sc[j]] * compute_coeff(pa, ph[j], group.rank)
                        end
                    end
                end
            end  # ================== end new_β batch ==================

        end  # ================== end groups batch ==================

        # ================================================================
        #  eps_check: 所有 g 和 b 累加完毕后一次性判定
        # ================================================================
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

function select_pass_b!(
    new_β_full::Vector{Ti},
    old_α_full::Vector{Ti},
    old_β_set::Set{Ti},
    old_a_idx_map::Dict{Ti,Tuple{Int,Int}},
    old_b_idx_map::Dict{Ti,Tuple{Int,Int}},
    all_groups::Vector{SVDGroup{Ti,Tv}},
    psi,
    E_var::Float64, eps::Float64,
    B_CHUNK::Int = 32,
    G_CHUNK::Int = 2000,
    A_CHUNK::Int = 2000,
) where {Ti,Tv}

    n_new_β = length(new_β_full)
    n_old   = length(old_α_full)
    ngroups = length(all_groups)
    output_p2 = Vector{Tuple{Ti,Ti}}()

    # ===== Layer 1: new_β batch =====
    for b_start in 1:B_CHUNK:n_new_β
        b_end   = min(b_start + B_CHUNK - 1, n_new_β)
        b_chunk = new_β_full[b_start : b_end]
        n_b     = length(b_chunk)

        # ★ accum_old: 跨 g 和 a 持久化
        accum_old = zeros(Float64, n_b, n_old)  # (b行, old_α列)

        # ===== Layer 2: groups batch =====
        for g_start in 1:G_CHUNK:ngroups
            g_end   = min(g_start + G_CHUNK - 1, ngroups)
            g_chunk = all_groups[g_start : g_end]
            n_g     = g_end - g_start + 1

            # ── 临时: blink_local [n_b][n_g] ──
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
                a_old_end   = min(a_old_start + A_CHUNK - 1, n_old)
                a_old_chunk = old_α_full[a_old_start : a_old_end]
                n_a_old     = a_old_end - a_old_start + 1
                a_offset    = a_old_start - 1

                # ── 临时: shared_a_old [n_g][n_a_old] ──
                dst_idxs_chunk = [Int[]   for _ in 1:n_g]
                src_idxs_chunk = [Int[]   for _ in 1:n_g]
                phases_chunk   = [Vector{Float64}[] for _ in 1:n_g]

                for (ia_local, old_a) in enumerate(a_old_chunk)
                    ia_global = a_offset + ia_local
                    for (ig_local, group) in enumerate(g_chunk)
                        src_a = old_a ⊻ group.ax
                        t = get(old_a_idx_map, src_a, nothing)
                        t === nothing && continue
                        src_ia_local, _ = t
                        pa = zeros(Float64, group.rank)
                        precompute_phase!(pa, src_a, group.unique_zas, group.num_za, group.wa, group.rank)
                        push!(dst_idxs_chunk[ig_local], ia_global)
                        push!(src_idxs_chunk[ig_local], src_ia_local)
                        push!(phases_chunk[ig_local],   pa)
                    end
                end

                # ── Phase 2: 累加进 accum_old ──
                for ib in 1:n_b
                    dst_b = b_chunk[ib]
                    for ig_global in blink_local[ib]
                        ig_local = ig_global - g_start + 1
                        group   = g_chunk[ig_local]

                        # pb: 即时算 (依赖 dst_b)
                        src_b = dst_b ⊻ group.bx
                        t = get(old_b_idx_map, src_b, nothing)
                        t === nothing && continue
                        src_ib_local, _ = t
                        pb = zeros(Float64, group.rank)
                        precompute_phase!(pb, src_b, group.unique_zbs, group.num_zb, group.wb, group.rank)

                        ds = dst_idxs_chunk[ig_local]
                        sc = src_idxs_chunk[ig_local]
                        ph = phases_chunk[ig_local]
                        for j in eachindex(ds)
                            accum_old[ib, ds[j]] += psi[sc[j], src_ib_local] * compute_coeff(ph[j], pb, group.rank)
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

