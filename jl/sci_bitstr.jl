const LIB_SCI_BITSTR = joinpath(libpath, "libsci_otf_bitstr.so")

mutable struct SciBasisManagerBitstr
    ptr::Ptr{Cvoid}
    dim::Int64
    norb::Int64
    num_blocks::Int64
    astrs::Vector{UInt32}
    bstrs::Vector{UInt32}
    a_idx_map::Dict{UInt32,Int}
    b_idx_map::Dict{UInt32,Int}
end

function SciBasisManagerBitstr()
    return SciBasisManagerBitstr(C_NULL, 0, 0, 0, UInt32[], UInt32[],
                                  Dict{UInt32,Int}(), Dict{UInt32,Int}())
end

function get_string_sym(str::UInt32, orbsym::Vector{Int64})::Int64
    sym = Int64(0)
    pos = 0
    while str > 0
        if (str & 1) != 0
            sym ⊻= orbsym[pos + 1]
        end
        str >>= 1
        pos += 1
    end
    return sym
end

function sort_by_sym(arr::Vector{UInt32}, orbsym::Vector{Int64})::Vector{UInt32}
    num_irreps = length(unique(orbsym))
    by_sym = [Vector{UInt32}() for _ in 1:num_irreps]
    for a in arr
        sym = get_string_sym(a, orbsym) + 1
        if sym <= num_irreps
            push!(by_sym[sym], a)
        end
    end
    result = UInt32[]
    for s in 1:num_irreps
        sort!(by_sym[s])
        append!(result, by_sym[s])
    end
    return result
end

function extract_ax_bx(svd_groups::Vector{<:Any})
    axs = UInt32[]
    bxs = UInt32[]
    for g in svd_groups
        push!(axs, g.ax)
        push!(bxs, g.bx)
    end
    return axs, bxs
end

function SciBasisManagerBitstr(
    astrs::Vector{UInt32}, bstrs::Vector{UInt32},
    norb::Int64, total_sym::Int64, orbsym::Vector{Int64};
    sorted::Bool=false)

    if !sorted
        astrs = sort_by_sym(astrs, orbsym)
        bstrs = sort_by_sym(bstrs, orbsym)
    end

    na = length(astrs)
    nb = length(bstrs)
    a_idx_map = Dict{UInt32,Int}(astrs[i] => i-1 for i in 1:na)
    b_idx_map = Dict{UInt32,Int}(bstrs[i] => i-1 for i in 1:nb)

    na64 = Int64(na)
    nb64 = Int64(nb)
    num_irreps = length(unique(orbsym))
    nirp64 = Int64(num_irreps)
    ptr = @ccall LIB_SCI_BITSTR.create_sci_basis_manager_bitstr_f64(
        astrs::Ptr{UInt32}, na64::Int64,
        bstrs::Ptr{UInt32}, nb64::Int64,
        norb::Int64, orbsym::Ptr{Int64},
        total_sym::Int64, nirp64::Int64
    )::Ptr{Cvoid}
    ptr == C_NULL && error("Failed to create bitstr basis manager.")
    dim = @ccall LIB_SCI_BITSTR.sci_basis_dim_bitstr(ptr::Ptr{Cvoid})::Int64
    nbk = @ccall LIB_SCI_BITSTR.sci_basis_num_blocks_bitstr(ptr::Ptr{Cvoid})::Int64

    obj = SciBasisManagerBitstr(ptr, dim, norb, nbk, astrs, bstrs, a_idx_map, b_idx_map)
    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_SCI_BITSTR.destroy_sci_basis_manager_bitstr_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end
    return obj
end

function OTF_bitstr(orbsym::Vector{Int64}, norb::Int64, A::BinaryQubitAABB{Ti,Tv,K,V};
                    tol::Float64=1e-12) where {Ti,Tv,K,V}
    groups = compress_by_svd(A, tol)
    ngs = length(groups)
    axs    = Vector{Ti}(undef, ngs);    bxs    = Vector{Ti}(undef, ngs)
    ranks  = Vector{Int64}(undef, ngs); num_as = Vector{Int64}(undef, ngs)
    num_bs = Vector{Int64}(undef, ngs)
    flat_azs = Ti[]; flat_bzs = Ti[]; flat_wa = Tv[]; flat_wb = Tv[]
    for (g, group) in enumerate(groups)
        axs[g]=group.ax; bxs[g]=group.bx; ranks[g]=group.rank
        num_as[g]=length(group.azs); num_bs[g]=length(group.bzs)
        append!(flat_azs, group.azs); append!(flat_bzs, group.bzs)
        append!(flat_wa, vec(group.wa)); append!(flat_wb, vec(group.wb))
    end
    ptr = @ccall LIB_SCI_BITSTR.build_network_otf_sci_bitstr_f64(
        orbsym::Ptr{Int64}, norb::Int64, ngs::Int64,
        axs::Ptr{Ti}, bxs::Ptr{Ti}, ranks::Ptr{Int64},
        num_as::Ptr{Int64}, num_bs::Ptr{Int64},
        flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti},
        flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv}
    )::Ptr{Cvoid}
    ptr == C_NULL && error("Failed to build bitstr OTF.")
    obj = OTF(ptr, 0, ngs)
    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_OTF.destroy_network_otf_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end
    return obj
end

function expand_bitstrings_bitstr(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32},
    axs::Vector{UInt32}, bxs::Vector{UInt32},
    orbsym::Vector{Int64})

    a_map = Dict{UInt32,Int}(a => 0 for a in src_astrs)
    b_map = Dict{UInt32,Int}(b => 0 for b in src_bstrs)

    for (ax, bx) in zip(axs, bxs)
        if ax != 0
            for a in src_astrs
                get!(a_map, a ⊻ ax, 1)
            end
        end
        if bx != 0
            for b in src_bstrs
                get!(b_map, b ⊻ bx, 1)
            end
        end
    end

    dst_astrs = sort_by_sym(collect(keys(a_map)), orbsym)
    dst_bstrs = sort_by_sym(collect(keys(b_map)), orbsym)

    is_new_a = Bool[a_map[a] == 1 for a in dst_astrs]
    is_new_b = Bool[b_map[b] == 1 for b in dst_bstrs]

    return dst_astrs, dst_bstrs, is_new_a, is_new_b
end

function sci_hvec_select_for_block_bitstr!(
    tgt::SciBasisManagerBitstr, src::SciBasisManagerBitstr, otf::OTF,
    blk::Integer, psi::Vector{Float64}, chunk_size::Int, eps::Float64,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    max_per_block = tgt.dim
    buf_a = Vector{UInt32}(undef, max_per_block)
    buf_b = Vector{UInt32}(undef, max_per_block)
    buf_v = Vector{Float64}(undef, max_per_block)
    mb64 = Int64(max_per_block)
    blk64 = Int64(blk)
    n = @ccall LIB_SCI_BITSTR.sci_hvec_select_for_block_bitstr_f64(
        tgt.ptr::Ptr{Cvoid}, src.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        blk64::Int64, psi::Ptr{Float64}, chunk_size::Cint, eps::Cdouble,
        buf_a::Ptr{UInt32}, buf_b::Ptr{UInt32}, buf_v::Ptr{Float64},
        mb64::Int64
    )::Int64
    append!(sel_a, view(buf_a, 1:n))
    append!(sel_b, view(buf_b, 1:n))
    append!(sel_v, view(buf_v, 1:n))
    return n
end

function filter_new_selected(
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64},
    dst_a_idx_map::Dict{UInt32,Int}, dst_b_idx_map::Dict{UInt32,Int},
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool})

    new_a = UInt32[]; new_b = UInt32[]; new_v = Float64[]
    for i in 1:length(sel_a)
        a_local = get(dst_a_idx_map, sel_a[i], -1)
        b_local = get(dst_b_idx_map, sel_b[i], -1)
        if a_local == -1 || b_local == -1
            continue
        end
        if is_new_a[a_local + 1] || is_new_b[b_local + 1]
            push!(new_a, sel_a[i])
            push!(new_b, sel_b[i])
            push!(new_v, sel_v[i])
        end
    end
    return new_a, new_b, new_v
end

function merge_bitstrings(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32},
    orbsym::Vector{Int64})

    a_union = sort_by_sym(collect(union(Set(src_astrs), Set(sel_a))), orbsym)
    b_union = sort_by_sym(collect(union(Set(src_bstrs), Set(sel_b))), orbsym)
    return a_union, b_union
end

function remap_wavefunction_bitstr!(
    old::SciBasisManagerBitstr, old_psi::Vector{Float64},
    new::SciBasisManagerBitstr, new_psi::Vector{Float64},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    nn64 = Int64(length(sel_a))
    @ccall LIB_SCI_BITSTR.remap_wavefunction_sci_bitstr_f64(
        old.ptr::Ptr{Cvoid}, old_psi::Ptr{Float64},
        new.ptr::Ptr{Cvoid}, new_psi::Ptr{Float64},
        sel_a::Ptr{UInt32}, sel_b::Ptr{UInt32}, sel_v::Ptr{Float64},
        nn64::Int64
    )::Cvoid
end

function get_diags_bitstr!(basis::SciBasisManagerBitstr, otf::OTF, diags::Vector{Float64})
    @ccall LIB_SCI_BITSTR.get_diags_elements_sci_bitstr_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64})::Cvoid
end

function hvec_sci_full_bitstr!(basis::SciBasisManagerBitstr, otf::OTF,
                                src::Vector{Float64}, dst::Vector{Float64})
    @ccall LIB_SCI_BITSTR.hvec_sci_full_bitstr_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        src::Ptr{Float64}, dst::Ptr{Float64})::Cvoid
end

function destroy_sci_basis_manager_bitstr(sb::SciBasisManagerBitstr)
    if sb.ptr != C_NULL
        @ccall LIB_SCI_BITSTR.destroy_sci_basis_manager_bitstr_f64(sb.ptr::Ptr{Cvoid})::Cvoid
        sb.ptr = C_NULL
    end
end

function create_sci_basis_from_standard_bitstr(basis::BasisManager)
    ptr = @ccall LIB_SCI_BITSTR.create_sci_basis_from_standard_bitstr_f64(
        basis.ptr::Ptr{Cvoid})::Ptr{Cvoid}
    dim = @ccall LIB_SCI_BITSTR.sci_basis_dim_bitstr(ptr::Ptr{Cvoid})::Int64
    nb  = @ccall LIB_SCI_BITSTR.sci_basis_num_blocks_bitstr(ptr::Ptr{Cvoid})::Int64
    obj = SciBasisManagerBitstr(ptr, dim, basis.norb, nb, UInt32[], UInt32[],
                                 Dict{UInt32,Int}(), Dict{UInt32,Int}())
    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_SCI_BITSTR.destroy_sci_basis_manager_bitstr_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end
    return obj
end

function run_sci_bitstr(mole::Mole;
    max_iter::Int=20, max_size::Int=10000, eps::Float64=1e-6,
    chunk_size::Int=256, davidson_tol::Float64=1e-5, verbose::Bool=true)

    na, nb = mole.nelec

    verbose && print("Building OTF ... ")
    t0 = @elapsed begin
        ham = JW_hamiltonian(mole; verbose=false)
        svd_groups = compress_by_svd(ham)
        all_axs, all_bxs = extract_ax_bx(svd_groups)
        ham_otf = OTF_bitstr(mole.orbsym, mole.norb, ham)
    end
    verbose && @printf("Done in %.4f s  (ngroups=%d)\n", t0, length(svd_groups))

    hf_astr = UInt32((1 << na) - 1)
    hf_bstr = UInt32((1 << nb) - 1)
    basis = SciBasisManagerBitstr([hf_astr], [hf_bstr], mole.norb, 0, mole.orbsym)
    psi = Float64[1.0]
    diags = zeros(Float64, 1)
    get_diags_bitstr!(basis, ham_otf, diags)
    verbose && @printf("Initial basis: dim=%d  E0=%.10f\n", basis.dim, diags[1])

    for iter in 1:max_iter
        t_iter = @elapsed begin
            dst_a, dst_b, is_new_a, is_new_b = expand_bitstrings_bitstr(
                basis.astrs, basis.bstrs, all_axs, all_bxs, mole.orbsym)
            tgt = SciBasisManagerBitstr(dst_a, dst_b, mole.norb, 0, mole.orbsym; sorted=true)

            sel_a = UInt32[]; sel_b = UInt32[]; sel_v = Float64[]
            for blk in 0:tgt.num_blocks-1
                sci_hvec_select_for_block_bitstr!(tgt, basis, ham_otf, blk, psi,
                                                   chunk_size, eps, sel_a, sel_b, sel_v)
            end
            verbose && @printf("  [%d] expand %d→%d  raw_sel=%d  ",
                               iter, basis.dim, tgt.dim, length(sel_v))

            sel_a, sel_b, sel_v = filter_new_selected(
                sel_a, sel_b, sel_v, tgt.a_idx_map, tgt.b_idx_map, is_new_a, is_new_b)
            nsel = length(sel_v)
            verbose && @printf("new_sel=%d  ", nsel)

            if nsel == 0
                verbose && println("No new states, done.")
                destroy_sci_basis_manager_bitstr(tgt)
                break
            end

            new_a, new_b = merge_bitstrings(basis.astrs, basis.bstrs,
                                             sel_a, sel_b, mole.orbsym)
            new_basis = SciBasisManagerBitstr(new_a, new_b, mole.norb, 0, mole.orbsym; sorted=true)
            verbose && @printf("merged=%d  ", new_basis.dim)

            new_psi = zeros(Float64, new_basis.dim)
            remap_wavefunction_bitstr!(basis, psi, new_basis, new_psi, sel_a, sel_b, sel_v)
            new_diags = zeros(Float64, new_basis.dim)
            get_diags_bitstr!(new_basis, ham_otf, new_diags)

            n = new_basis.dim
            hvec_fn = (v, Hv) -> hvec_sci_full_bitstr!(new_basis, ham_otf, v, Hv)
            E, psi_new = davidson(hvec_fn, new_psi, new_diags;
                                  tol=davidson_tol, ncv=1, maxspace=min(max_size, n+20),
                                  verbose=false)
            verbose && @printf("E=%.10f  err=%.1e\n", E, norm(psi_new - new_psi))

            destroy_sci_basis_manager_bitstr(tgt)
            destroy_sci_basis_manager_bitstr(basis)
            basis, psi, diags = new_basis, psi_new, new_diags
        end
        verbose && @printf("  iter %d time=%.2f s\n", iter, t_iter)
    end
    return basis, psi, diags
end
