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

function cal_strs_syms(norb::Int, ne::Int, orbsym::Vector{Int}, Ti::Type)
    f_str = (str, pos) -> str | (Ti(1) << pos)
    f_sym = (sym, pos) -> sym ⊻ orbsym[pos+1]

    strs = Array{Ti,1}(undef, binomial(norb, ne))
    syms = Array{Int,1}(undef, binomial(norb, ne))

    count = 0
    for comb in combinations([i for i in 0:norb-1], ne)
        count += 1
        strs[count] = foldl(f_str, comb; init=Ti(0))
        syms[count] = foldl(f_sym, comb; init=0)
    end

    dict = Dict{Int,Array{Ti,1}}()
    for (sym, str) in zip(syms, strs)
        index = Base.ht_keyindex2!(dict, sym)
        if index > 0
            @inbounds push!(dict.vals[index], str)
        else
            @inbounds Base._setindex!(dict, Ti[str], sym, -index)
        end
    end

    return dict
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

function get_sym_blocks(norb::Int, nelec::Tuple{Int,Int}, orbsym::Vector{Int}, total_sym::Int, Ti::Type)
    na, nb = nelec
    adict = cal_strs_syms(norb, na, orbsym, Ti)
    bdict = cal_strs_syms(norb, nb, orbsym, Ti)

    blocks = BlockDesc{Ti}[]
    max_asym = 0
    max_bsym = 0
    current_offset = 0

    for (asym, astrs) in adict
        bsym = total_sym ⊻ asym
        if haskey(bdict, bsym)
            bstrs = bdict[bsym]

            sort!(astrs)
            sort!(bstrs)

            num_a = length(astrs)
            num_b = length(bstrs)

            push!(blocks, BlockDesc(asym, bsym, num_a, num_b, astrs, bstrs, current_offset))

            current_offset += num_a * num_b
            (asym > max_asym) && (max_asym = asym)
            (bsym > max_bsym) && (max_bsym = bsym)
        end
    end

    block_map = fill(-1, max_asym + 1, max_bsym + 1)

    for i in eachindex(blocks)
        B = blocks[i]
        asym = B.asym
        bsym = B.bsym
        block_map[asym+1, bsym+1] = i - 1
    end

    return blocks, block_map
end

# ============================================================
# CSR 数据结构
# ============================================================

struct SortedSrcInfo{Ti}
    strings::Vector{Ti}          # 全部位串, 按 block/对称性 排列
    blk_of_str::Vector{Int}      # strings[i] 所属的 block 索引 (0-indexed)
    sym_offsets::Vector{Int}     # 对称性扇区边界, 长度 num_irreps+1
    num_irreps::Int
end

struct Link{Ti}
    main_strs::Vector{Ti}               # 目标位串 (target)
    bounds::Vector{Int}                 # CSR row pointers, 长度 nrows+1
    flatten_strs::Vector{Ti}            # 源位串
    flatten_xs::Vector{Ti}              # ax 或 bx
    flatten_src_idxs::Vector{Int}       # source block 内局部索引 (0-indexed)
    flatten_src_blk_idxs::Vector{Int}   # source block 在 src_basis.blocks 中的索引
end

# ============================================================
# 构建函数
# ============================================================

function build_sorted_src(
    src_strs::Vector{Ti},
    norb::Int, 
    ne::Int,
    orbsym::Vector{Int64},
    total_sym::Int,
    ::Type{Ti},
    spin::Symbol,               # :alpha or :beta
) where {Ti}
    blocks, block_map = get_sym_blocks(norb, (ne, ne), orbsym, total_sym, Ti)

    nstrs = length(src_strs)
    blk_of_str = Vector{Int}(undef, nstrs)
    fill!(blk_of_str, -1)

    for (blk_idx, blk) in enumerate(blocks)
        strs = (spin === :alpha ? blk.astrs : blk.bstrs)
        for (local_idx, str) in enumerate(strs)
            global_idx = searchsortedfirst(src_strs, str)
            if global_idx <= nstrs && src_strs[global_idx] == str
                blk_of_str[global_idx] = blk_idx - 1
            end
        end
    end

    # sym_offsets: 按 get_symm 分组边界
    num_irreps = maximum(orbsym) + 1
    sym_offsets = zeros(Int, num_irreps + 1)
    for (i, str) in enumerate(src_strs)
        s = get_symm(str, orbsym)
        if sym_offsets[s+2] == 0
            sym_offsets[s+2] = i
        end
    end
    sym_offsets[1] = 0
    for s in 2:num_irreps
        if sym_offsets[s] == 0
            sym_offsets[s] = sym_offsets[s-1]
        end
    end
    sym_offsets[num_irreps+1] = nstrs

    return SortedSrcInfo{Ti}(src_strs, blk_of_str, sym_offsets, num_irreps)
end

function build_single_side_link(
    sorted_src::SortedSrcInfo{Ti},
    unique_xs::Vector{Ti},
    new_set::Set{Ti},
    orbsym::Vector{Int64},
) where {Ti}
    n_src = length(sorted_src.strings)

    # ===== Step A: src → {(x, dst, src_local, src_blk)} =====
    src_to_dst = Dict{Ti,Vector{Tuple{Ti,Ti,Int,Int}}}()

    for src_local in 1:n_src
        src = sorted_src.strings[src_local]
        src_blk = sorted_src.blk_of_str[src_local]
        src_blk == -1 && continue
        entries = Vector{Tuple{Ti,Ti,Int,Int}}()
        for x in unique_xs
            x == Ti(0) && continue
            dst = src ⊻ x
            if dst in new_set
                push!(entries, (x, dst, src_local, src_blk))
            end
        end
        !isempty(entries) && (src_to_dst[src] = entries)
    end

    # ===== Step B: 反转为 dst 侧 =====
    dst_to_src = Dict{Ti,Vector{Tuple{Ti,Ti,Int,Int}}}()
    total = 0
    for (src, entries) in src_to_dst
        for (x, dst, src_local, src_blk) in entries
            vec = get!(Vector{Tuple{Ti,Ti,Int,Int}}(), dst_to_src, dst)
            push!(vec, (x, src, src_local, src_blk))
            total += 1
        end
    end

    # ===== Step C: 扁化为 Link =====
    sorted_pairs = sort!(collect(pairs(dst_to_src)))
    nrows = length(sorted_pairs)

    main_strs = Vector{Ti}(undef, nrows)
    bounds = Vector{Int}(undef, nrows + 1)
    f_strs = Vector{Ti}(undef, total)
    f_xs = Vector{Ti}(undef, total)
    f_src_idxs = Vector{Int}(undef, total)
    f_src_blks = Vector{Int}(undef, total)

    count = 0
    for (i, (dst, entries)) in enumerate(sorted_pairs)
        main_strs[i] = dst
        bounds[i] = count
        for (x, src, src_local, src_blk) in entries
            f_strs[count+1] = src
            f_xs[count+1] = x
            f_src_idxs[count+1] = src_local
            f_src_blks[count+1] = src_blk
            count += 1
        end
    end
    bounds[nrows+1] = total

    return Link{Ti}(main_strs, bounds, f_strs, f_xs, f_src_idxs, f_src_blks)
end

function build_mixed_csr(
    old_strs::Vector{Ti},
    unique_xs::Vector{Ti},
    sorted_src::SortedSrcInfo{Ti},
    orbsym::Vector{Int64},
) where {Ti}
    # Returns: Dict{Ti, Vector{Tuple{Int,Int,Int}}}
    #          x → [(old_local_1based, src_local_1based, src_blk)]
    src_strings = sorted_src.strings
    src_sym_off = sorted_src.sym_offsets
    num_irreps = sorted_src.num_irreps

    mixed = Dict{Ti,Vector{Tuple{Int,Int,Int}}}()

    for x in unique_xs
        x == Ti(0) && continue
        sym_x = get_symm(x, orbsym)
        entries = Vector{Tuple{Int,Int,Int}}()

        for (old_local, old_str) in enumerate(old_strs)
            sym_old = get_symm(old_str, orbsym)
            sym_src = sym_old ⊻ sym_x
            if sym_src < 0 || sym_src >= num_irreps
                continue
            end
            src_str = old_str ⊻ x

            lo = src_sym_off[sym_src+1] + 1
            hi = src_sym_off[sym_src+2]
            if lo <= hi
                idx = searchsortedfirst(src_strings, src_str, lo, hi, Base.Order.Forward)
                if idx <= hi && src_strings[idx] == src_str
                    src_blk = sorted_src.blk_of_str[idx]
                    push!(entries, (old_local, idx, src_blk))
                end
            end
        end
        !isempty(entries) && (mixed[x] = entries)
    end
    return mixed
end

# ============================================================
# 探索用数据结构 (独立于 binisim 环境)
# ============================================================

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

function build_groups_2d(groups::Vector{SVDGroup{UInt32,Float64}})
    g2d = Dict{UInt32,Dict{UInt32,SVDGroup{UInt32,Float64}}}()
    for g in groups
        d = get!(Dict{UInt32,SVDGroup{UInt32,Float64}}(), g2d, g.ax)
        d[g.bx] = g
    end
    return g2d
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

function build_csr_select_data(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32},
    expanded_astrs::Vector{UInt32},
    expanded_bstrs::Vector{UInt32},
    is_new_a::Vector{Bool},
    is_new_b::Vector{Bool},
    all_axs::Vector{UInt32},
    all_bxs::Vector{UInt32},
    groups::Vector{SVDGroup{UInt32,Float64}},
    orbsym::Vector{Int64},
    norb::Int, na::Int, nb::Int,
    total_sym::Int,
)
    old_astrs = expanded_astrs[.!is_new_a]
    old_bstrs = expanded_bstrs[.!is_new_b]
    new_astrs = expanded_astrs[is_new_a]
    new_bstrs = expanded_bstrs[is_new_b]

    new_α_set = Set{UInt32}(new_astrs)
    new_β_set = Set{UInt32}(new_bstrs)

    sorted_src_a = build_sorted_src(src_astrs, norb, na, orbsym, total_sym, UInt32, :alpha)
    sorted_src_b = build_sorted_src(src_bstrs, norb, nb, orbsym, total_sym, UInt32, :beta)

    alink = build_single_side_link(sorted_src_a, all_axs, new_α_set, orbsym)
    blink = build_single_side_link(sorted_src_b, all_bxs, new_β_set, orbsym)
    mixed_b_csr = build_mixed_csr(old_bstrs, all_bxs, sorted_src_b, orbsym)
    mixed_a_csr = build_mixed_csr(old_astrs, all_axs, sorted_src_a, orbsym)
    groups_2d = build_groups_2d(groups)

    return (;
        alink, blink, mixed_a_csr, mixed_b_csr, groups_2d,
        new_α_set, new_β_set, old_astrs, old_bstrs,
        new_astrs, new_bstrs, sorted_src_a, sorted_src_b,
    )
end

# ============================================================
# 相位 / contract 辅助
# ============================================================

function precompute_phase_str!(
    dst::Vector{Float64},
    str::UInt32,
    zas::Vector{UInt32},
    nza::Int,
    wa::Vector{Float64},
    rank::Int,
)
    if rank == 1
        v = 0.0
        for i in 1:nza
            phase = 1 - 2 * (count_ones(str & zas[i]) & 1)
            v += wa[i] * phase
        end
        dst[1] = v
    else
        v0, v1 = 0.0, 0.0
        for i in 1:nza
            phase = 1 - 2 * (count_ones(str & zas[i]) & 1)
            v0 += wa[i]      * phase
            v1 += wa[nza+i]   * phase
        end
        dst[1] = v0
        dst[2] = v1
    end
end

function contract_group(
    group::SVDGroup{UInt32,Float64},
    a_str::UInt32,
    b_str::UInt32,
)
    pa = zeros(group.rank)
    pb = zeros(group.rank)
    precompute_phase_str!(pa, a_str, group.unique_zas, group.num_za, group.wa, group.rank)
    precompute_phase_str!(pb, b_str, group.unique_zbs, group.num_zb, group.wb, group.rank)
    return dot(pa, pb)
end

function precompute_diag_phases(
    astrs::Vector{UInt32},
    bstrs::Vector{UInt32},
    diag_groups::Vector{SVDGroup{UInt32,Float64}},
)
    total_rank = sum(g.rank for g in diag_groups; init=0)
    total_rank == 0 && return (Vector{Float64}(), Vector{Float64}(), 0)

    na = length(astrs)
    nb = length(bstrs)
    a_phase = zeros(Float64, na * total_rank)
    b_phase = zeros(Float64, nb * total_rank)

    offset = 0
    for group in diag_groups
        for (i, a) in enumerate(astrs)
            pa = zeros(group.rank)
            precompute_phase_str!(pa, a, group.unique_zas, group.num_za, group.wa, group.rank)
            for r in 1:group.rank
                a_phase[(i-1)*total_rank + offset + r] = pa[r]
            end
        end
        for (i, b) in enumerate(bstrs)
            pb = zeros(group.rank)
            precompute_phase_str!(pb, b, group.unique_zbs, group.num_zb, group.wb, group.rank)
            for r in 1:group.rank
                b_phase[(i-1)*total_rank + offset + r] = pb[r]
            end
        end
        offset += group.rank
    end
    return a_phase, b_phase, total_rank
end

# ============================================================
# 三个 Select Part
# ============================================================

function hvec_select_part1!(
    alink::Link{UInt32},
    new_α_rows::Vector{Int},
    old_β_strs::Vector{UInt32},
    old_β_count::Int,
    mixed_b_csr::Dict{UInt32,Vector{Tuple{Int,Int,Int}}},
    groups_2d::Dict{UInt32,Dict{UInt32,SVDGroup{UInt32,Float64}}},
    src_psi::Vector{Float64},
    src_blocks::Vector{BlockDesc{UInt32}},
    a_diag_phase::Vector{Float64},
    b_diag_phase::Vector{Float64},
    total_rank::Int,
    E_var::Float64, eps::Float64,
)
    n_new_α = length(new_α_rows)
    nthreads = Threads.nthreads()
    output_all = [Vector{Tuple{UInt32,UInt32}}() for _ in 1:nthreads]

    Threads.@threads for a_local in 1:n_new_α
        new_α_row = new_α_rows[a_local]
        new_α_str = alink.main_strs[new_α_row]
        accum_b = zeros(Float64, old_β_count)

        for j in alink.bounds[new_α_row]:alink.bounds[new_α_row+1]-1
            ax = alink.flatten_xs[j]
            src_a_idx = alink.flatten_src_idxs[j]
            src_a_blk = alink.flatten_src_blk_idxs[j] + 1
            blk = src_blocks[src_a_blk]

            # --- pure_a (bx=0): 直扫全部 old_β ---
            group = get(groups_2d[ax], UInt32(0), nothing)
            if group !== nothing
                gid_base = blk.offset + src_a_idx * blk.num_b
                for old_b_local in 1:old_β_count
                    src_val = src_psi[gid_base + old_b_local]
                    H_contrib = contract_group(group, new_α_str, old_β_strs[old_b_local])
                    accum_b[old_b_local] += src_val * H_contrib
                end
            end

            # --- mixed (bx≠0): CSR 过滤 ---
            for (bx, b_entries) in mixed_b_csr
                group = get(groups_2d[ax], bx, nothing)
                group === nothing && continue
                gid_base = blk.offset + src_a_idx * blk.num_b
                for (old_b_local, src_b_idx, src_b_blk) in b_entries
                    src_b_blk + 1 == src_a_blk || continue  # ★ 同 block 检查
                    src_val = src_psi[gid_base + src_b_idx]
                    H_contrib = contract_group(group, new_α_str, old_β_strs[old_b_local])
                    accum_b[old_b_local] += src_val * H_contrib
                end
            end
        end

        tid = Threads.threadid()
        a_phase = @view a_diag_phase[(new_α_row-1)*total_rank .+ (1:total_rank)]
        for old_b_local in 1:old_β_count
            val = accum_b[old_b_local]
            val != 0.0 || continue
            b_phase = @view b_diag_phase[(old_b_local-1)*total_rank .+ (1:total_rank)]
            haa = dot(a_phase, b_phase)
            if val^2 / (E_var - haa)^2 > eps^2
                push!(output_all[tid], (new_α_str, old_β_strs[old_b_local]))
            end
        end
    end
    return vcat(output_all...)
end

function hvec_select_part2!(
    blink::Link{UInt32},
    new_β_rows::Vector{Int},
    old_α_strs::Vector{UInt32},
    old_α_count::Int,
    mixed_a_csr::Dict{UInt32,Vector{Tuple{Int,Int,Int}}},
    groups_2d::Dict{UInt32,Dict{UInt32,SVDGroup{UInt32,Float64}}},
    src_psi::Vector{Float64},
    src_blocks::Vector{BlockDesc{UInt32}},
    a_diag_phase::Vector{Float64},
    b_diag_phase::Vector{Float64},
    total_rank::Int,
    E_var::Float64, eps::Float64,
)
    n_new_β = length(new_β_rows)
    nthreads = Threads.nthreads()
    output_all = [Vector{Tuple{UInt32,UInt32}}() for _ in 1:nthreads]

    Threads.@threads for b_local in 1:n_new_β
        new_β_row = new_β_rows[b_local]
        new_β_str = blink.main_strs[new_β_row]
        accum_a = zeros(Float64, old_α_count)

        for k in blink.bounds[new_β_row]:blink.bounds[new_β_row+1]-1
            bx = blink.flatten_xs[k]
            src_b_idx = blink.flatten_src_idxs[k]
            src_b_blk = blink.flatten_src_blk_idxs[k] + 1
            blk = src_blocks[src_b_blk]

            # --- pure_b (ax=0): 直扫全部 old_α ---
            group = get(groups_2d[UInt32(0)], bx, nothing)
            if group !== nothing
                gid_base = blk.offset + src_b_idx
                for old_a_local in 1:old_α_count
                    src_val = src_psi[gid_base + (old_a_local - 1) * blk.num_b]
                    H_contrib = contract_group(group, old_α_strs[old_a_local], new_β_str)
                    accum_a[old_a_local] += src_val * H_contrib
                end
            end

            # --- mixed (ax≠0): mixed_a_csr 过滤 ---
            for (ax, a_entries) in mixed_a_csr
                group = get(groups_2d[ax], bx, nothing)
                group === nothing && continue
                gid_base = blk.offset + src_b_idx
                for (old_a_local, src_a_idx, src_a_blk) in a_entries
                    src_a_blk + 1 == src_b_blk || continue  # ★ 同 block
                    gid = gid_base + src_a_idx * blk.num_b  # 1-indexed
                    src_val = src_psi[gid]
                    H_contrib = contract_group(group, old_α_strs[old_a_local], new_β_str)
                    accum_a[old_a_local] += src_val * H_contrib
                end
            end
        end

        tid = Threads.threadid()
        b_phase = @view b_diag_phase[(new_β_row-1)*total_rank .+ (1:total_rank)]
        for old_a_local in 1:old_α_count
            val = accum_a[old_a_local]
            val != 0.0 || continue
            a_phase = @view a_diag_phase[(old_a_local-1)*total_rank .+ (1:total_rank)]
            haa = dot(a_phase, b_phase)
            if val^2 / (E_var - haa)^2 > eps^2
                push!(output_all[tid], (old_α_strs[old_a_local], new_β_str))
            end
        end
    end
    return vcat(output_all...)
end

function hvec_select_part3!(
    alink::Link{UInt32},
    blink::Link{UInt32},
    new_α_rows::Vector{Int},
    new_β_rows::Vector{Int},
    new_α_strs::Vector{UInt32},
    new_β_strs::Vector{UInt32},
    groups_2d::Dict{UInt32,Dict{UInt32,SVDGroup{UInt32,Float64}}},
    src_psi::Vector{Float64},
    src_blocks::Vector{BlockDesc{UInt32}},
    a_diag_phase::Vector{Float64},
    b_diag_phase::Vector{Float64},
    total_rank::Int,
    E_var::Float64,
    eps::Float64,
)
    n_new_α = length(new_α_rows)
    n_new_β = length(new_β_rows)
    nthreads = Threads.nthreads()

    # ============ Step A: 预建 b_list[bx] ============
    b_list = Dict{UInt32,Vector{Tuple{Int,Int,Int,Int}}}()
    for (b_local, b_row) in enumerate(new_β_rows)
        si = blink.bounds[b_row]
        ei = blink.bounds[b_row+1]
        for k in si:ei-1
            bx = blink.flatten_xs[k]
            src_b_idx = blink.flatten_src_idxs[k]
            src_b_blk = blink.flatten_src_blk_idxs[k]
            vec = get!(Vector{Tuple{Int,Int,Int,Int}}(), b_list, bx)
            push!(vec, (b_local, b_row, src_b_blk, src_b_idx))
        end
    end

    # ============ Step B: per-thread output ============
    output_all = Vector{Vector{Tuple{UInt32,UInt32}}}(undef, nthreads)
    for t in 1:nthreads
        output_all[t] = Vector{Tuple{UInt32,UInt32}}()
    end

    # ============ Step C: 外层 new_α 并行 ============
    Threads.@threads for a_local in 1:n_new_α
        new_α_row = new_α_rows[a_local]
        new_α_str = new_α_strs[a_local]
        accum_b = zeros(Float64, n_new_β)
        a_phase_ptr = (new_α_row - 1) * total_rank

        si = alink.bounds[new_α_row]
        ei = alink.bounds[new_α_row+1]

        for j in si:ei-1
            ax = alink.flatten_xs[j]
            src_a_idx = alink.flatten_src_idxs[j]
            src_a_blk = alink.flatten_src_blk_idxs[j]

            g2d_ax = groups_2d[ax]
            for (bx, group) in g2d_ax
                bx == 0 && continue

                b_entries = get(b_list, bx, nothing)
                b_entries === nothing && continue

                pa_vec = Vector{Float64}(undef, group.rank)
                precompute_phase_str!(
                    pa_vec, new_α_str,
                    group.unique_zas, group.num_za, group.wa,
                    group.rank)

                blk = src_blocks[src_a_blk + 1]
                gid_base = blk.offset + src_a_idx * blk.num_b

                for (new_b_local, new_b_row, src_b_blk, src_b_idx) in b_entries
                    src_b_blk == src_a_blk || continue  # ★ 同 block 检查
                    src_val = src_psi[gid_base + src_b_idx]

                    pb_vec = Vector{Float64}(undef, group.rank)
                    precompute_phase_str!(
                        pb_vec, new_β_strs[new_b_local],
                        group.unique_zbs, group.num_zb, group.wb,
                        group.rank)
                    H_contrib = dot(pa_vec, pb_vec)

                    accum_b[new_b_local] += src_val * H_contrib
                end
            end
        end

        # ============ eps_check ============
        tid = Threads.threadid()
        a_phase = @view a_diag_phase[a_phase_ptr .+ (1:total_rank)]

        for b_local in 1:n_new_β
            val = accum_b[b_local]
            val != 0.0 || continue

            new_b_row = new_β_rows[b_local]
            b_phase = @view b_diag_phase[(new_b_row-1)*total_rank .+ (1:total_rank)]
            haa = dot(a_phase, b_phase)

            if val^2 / (E_var - haa)^2 > eps^2
                push!(output_all[tid], (new_α_str, new_β_strs[b_local]))
            end
        end
    end

    return vcat(output_all...)
end

# ============================================================
# End-to-end 测试
# ============================================================

function run_sci_csr_test(
    norb::Int, na::Int, nb::Int,
    orbsym::Vector{Int64},
    total_sym::Int,
    all_axs::Vector{UInt32},
    all_bxs::Vector{UInt32},
    groups::Vector{SVDGroup{UInt32,Float64}},
    eps::Float64;
    max_iter::Int=5,
    verbose::Bool=true,
)
    # 初始化 — Hartree-Fock 行列式
    hf_astr = UInt32((1 << na) - 1)
    hf_bstr = UInt32((1 << nb) - 1)

    src_astrs = UInt32[hf_astr]
    src_bstrs = UInt32[hf_bstr]

    # 源波函数 (1 维, 系数 = 1)
    dim = 1
    psi = Float64[1.0]

    # source blocks (临时: 单 block, 1×1)
    src_blocks = [BlockDesc{UInt32}(0, 0, 1, 1, UInt32[hf_astr], UInt32[hf_bstr], 0)]

    E_var = contract_group(groups[1], hf_astr, hf_bstr)  # Haa for HF

    verbose && @printf("Initial basis: dim=%d  E0=%.10f\n", dim, E_var)

    for iter in 1:max_iter
        # --- expand ---
        new_astrs, new_bstrs, is_new_a, is_new_b = expand_bitstrings_bitstr(
            src_astrs, src_bstrs, all_axs, all_bxs, (na, nb), orbsym
        )

        verbose && @printf("\nIteration %d: |new_α|=%d  |new_β|=%d\n",
                           iter, sum(is_new_a), sum(is_new_b))

        # --- 构建 CSR 数据 ---
        data = build_csr_select_data(
            src_astrs, src_bstrs,
            new_astrs, new_bstrs, is_new_a, is_new_b,
            all_axs, all_bxs, groups, orbsym,
            norb, na, nb, total_sym,
        )

        # --- 预计算 diag phases ---
        diag_groups = filter(g -> g.ax == UInt32(0) && g.bx == UInt32(0), groups)
        a_phase, b_phase, total_rank = precompute_diag_phases(
            data.new_astrs, data.new_bstrs, diag_groups,
        )

        # --- 映射 new α/β 位串到 alink/blink 行号 ---
        new_a_to_row = Dict{UInt32,Int}()
        for (i, s) in enumerate(data.alink.main_strs)
            new_a_to_row[s] = i
        end
        new_α_rows = Int[new_a_to_row[s] for s in data.new_astrs if s in new_a_to_row]

        new_b_to_row = Dict{UInt32,Int}()
        for (i, s) in enumerate(data.blink.main_strs)
            new_b_to_row[s] = i
        end
        new_β_rows = Int[new_b_to_row[s] for s in data.new_bstrs if s in new_b_to_row]

        old_β_strs = data.old_bstrs
        old_α_strs = data.old_astrs
        old_β_count = length(old_β_strs)
        old_α_count = length(old_α_strs)

        # --- Part 1: new_α × old_β ---
        @time pairs_p1 = hvec_select_part1!(
            data.alink, new_α_rows, old_β_strs, old_β_count,
            data.mixed_b_csr, data.groups_2d,
            psi, src_blocks, a_phase, b_phase, total_rank,
            E_var, eps,
        )

        # --- Part 2: old_α × new_β ---
        @time pairs_p2 = hvec_select_part2!(
            data.blink, new_β_rows, old_α_strs, old_α_count,
            data.mixed_a_csr, data.groups_2d,
            psi, src_blocks, a_phase, b_phase, total_rank,
            E_var, eps,
        )

        # --- Part 3: new_α × new_β ---
        @time pairs_p3 = hvec_select_part3!(
            data.alink, data.blink,
            new_α_rows, new_β_rows,
            data.new_astrs, data.new_bstrs,
            data.groups_2d,
            psi, src_blocks,
            a_phase, b_phase, total_rank,
            E_var, eps,
        )

        all_pairs = vcat(pairs_p1, pairs_p2, pairs_p3)
        n_sel = length(all_pairs)
        verbose && @printf("  Part1=%d  Part2=%d  Part3=%d  Total=%d\n",
                           length(pairs_p1), length(pairs_p2), length(pairs_p3), n_sel)

        n_sel == 0 && (verbose && println("No new states, done."); break)

        # --- 抽出新选的 alpha/beta 串 ---
        sel_astrs = unique!(UInt32[p[1] for p in all_pairs])
        sel_bstrs = unique!(UInt32[p[2] for p in all_pairs])

        # --- merge: old + selected new ---
        new_src_astrs = sort(union(src_astrs, sel_astrs))
        new_src_bstrs = sort(union(src_bstrs, sel_bstrs))

        # --- 重建 source blocks (简化: 单 block, 不处理对称性变化) ---
        new_dim = length(new_src_astrs) * length(new_src_bstrs)
        psi_new = zeros(Float64, new_dim)
        # remap: 旧系数复制, 新 entry = 0
        for old_a_idx in eachindex(src_astrs)
            a_local = searchsortedfirst(new_src_astrs, src_astrs[old_a_idx])
            for old_b_idx in eachindex(src_bstrs)
                b_local = searchsortedfirst(new_src_bstrs, src_bstrs[old_b_idx])
                gid = (a_local - 1) * length(new_src_bstrs) + b_local
                old_gid = (old_a_idx - 1) * length(src_bstrs) + old_b_idx
                psi_new[gid] = psi[old_gid]
            end
        end

        src_astrs = new_src_astrs
        src_bstrs = new_src_bstrs
        src_blocks = [BlockDesc{UInt32}(0, 0, length(src_astrs), length(src_bstrs), src_astrs, src_bstrs, 0)]
        psi = psi_new

        # --- 对角化 (简化: 精确纯对角化, 小空间下可行) ---
        H = diags_only(src_astrs, src_bstrs, groups)
        psi = H \ psi  # placeholder — 实际 SCI 用 Davidson

        # --- 更新 E_var ---
        E_var = psi' * H * psi
        verbose && @printf("  Energy: %.14f\n", E_var)
    end

    return src_astrs, src_bstrs, psi, E_var
end


precompute_bstrs  = Vector{Vector{Ti}}(undef, ngroups)
precompute_bidxs  = Vector{Vector{Int}}(undef, ngroups)
precompute_phases = Vector{Vector{Tv}}(undef, ngroups)
for (ib, dst_b) in enumerate(new_bstrs) 
    valid_groups = blink[ib]
    for (ig, group) in valid_groups
        src_b = group.bx ⊻ dst_b
        src_ib = b_idx_map[src_b]
        pb = precompute(group, src_b)
        push!(precompute_bstrs[ig],  src_b)
        push!(precompute_bidxs[ig],  src_ib)
        push!(precompute_phases[ig], pb)
    end
end

for (ia, dst_a) in enumerate(new_astrs)
    valid_groups = alink[ia]
    for (ig, group) in valid_groups
        src_a = group.ax ⊻ dst_a
        src_ia = a_idx_map[src_a]
        pa = precompute(group, src_a)
        valid_bstrs = precompute_bstrs[ig] 
        valid_bidxs = precompute_bidxs[ig] 
        valid_pbs   = precompute_phases[ig] 
        for ib in eachindex(valid_bstrs)
            src_b  = valid_bstrs[ib]
            src_ib = valid_bidxs[ib]
            pb     = valid_pbs[ib]
            output[ia, ib] += input[src_ia, src_ib] * compute_coeff(group, pa, pb)
        end
    end
end
function ()
    
end
