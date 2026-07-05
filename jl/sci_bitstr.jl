const SCI_BITSTR_NUM_IRREPS = 16

mutable struct SciBasisManagerBitstr
    ptr::Ptr{Cvoid}
    dim::Int64
    norb::Int64
    num_blocks::Int64
    na::Int64
    nb::Int64
    astrs::Vector{UInt32}
    bstrs::Vector{UInt32}
    a_idx_map::Dict{UInt32,Int}
    b_idx_map::Dict{UInt32,Int}
end

function SciBasisManagerBitstr()
    SciBasisManagerBitstr(C_NULL, 0, 0, 0, -1, -1, UInt32[], UInt32[], Dict{UInt32,Int}(), Dict{UInt32,Int}())
end

function sort_by_sym(arr::Vector{UInt32}, orbsym::Vector{Int64}, num_irreps::Int)
    by_sym = [Vector{UInt32}() for _ in 1:num_irreps]
    for a in arr
        sym = get_symm(a, orbsym)
        if 0 <= sym < num_irreps
            push!(by_sym[sym+1], a)
        end
    end
    result = UInt32[]
    for s in 1:num_irreps
        sort!(by_sym[s])
        append!(result, by_sym[s])
    end
    return result
end

function extract_ax_bx(svd_groups)
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
    norb::Int64, total_sym::Int64, orbsym::Vector{Int64}, na::Int, nb::Int;
    sorted::Bool=false,
    num_irreps::Int=SCI_BITSTR_NUM_IRREPS
)
    astrs = UInt32[a for a in astrs if count_ones(a) == na]
    bstrs = UInt32[b for b in bstrs if count_ones(b) == nb]

    @assert all(count_ones.(astrs) .== na)
    @assert all(count_ones.(bstrs) .== nb)

    if !sorted
        astrs = sort_by_sym(astrs, orbsym, num_irreps)
        bstrs = sort_by_sym(bstrs, orbsym, num_irreps)
    end

    num_a = length(astrs)
    num_b = length(bstrs)

    a_idx_map = Dict{UInt32,Int}(astrs[i] => i - 1 for i in 1:num_a)
    b_idx_map = Dict{UInt32,Int}(bstrs[i] => i - 1 for i in 1:num_b)

    ptr = @ccall LIB_SCI_BITSTR.create_sci_basis_manager_bitstr_f64(
        astrs::Ptr{UInt32}, num_a::Int64,
        bstrs::Ptr{UInt32}, num_b::Int64,
        norb::Int64, orbsym::Ptr{Int64},
        total_sym::Int64, num_irreps::Int64
    )::Ptr{Cvoid}

    ptr == C_NULL && error("Failed to create bitstr basis manager.")

    dim = @ccall LIB_SCI_BITSTR.sci_basis_dim_bitstr(ptr::Ptr{Cvoid})::Int64
    nbk = @ccall LIB_SCI_BITSTR.sci_basis_num_blocks_bitstr(ptr::Ptr{Cvoid})::Int64

    obj = SciBasisManagerBitstr(ptr, dim, norb, nbk, na, nb, astrs, bstrs, a_idx_map, b_idx_map)
    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_SCI_BITSTR.destroy_sci_basis_manager_bitstr_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end

    return obj
end

function OTF_bitstr(orbsym::Vector{Int64}, norb::Int64, A::BinaryQubitAABB{Ti,Tv,K,V}; tol::Float64=1e-12) where {Ti,Tv,K,V}
    groups = compress_by_svd(A, tol)
    ngs = length(groups)
    axs = Vector{Ti}(undef, ngs)
    bxs = Vector{Ti}(undef, ngs)
    ranks = Vector{Int64}(undef, ngs)
    num_as = Vector{Int64}(undef, ngs)
    num_bs = Vector{Int64}(undef, ngs)
    flat_azs = Ti[]
    flat_bzs = Ti[]
    flat_wa = Tv[]
    flat_wb = Tv[]
    for (g, group) in enumerate(groups)
        axs[g] = group.ax
        bxs[g] = group.bx
        ranks[g] = group.rank
        num_as[g] = length(group.azs)
        num_bs[g] = length(group.bzs)
        append!(flat_azs, group.azs)
        append!(flat_bzs, group.bzs)
        append!(flat_wa, vec(group.wa))
        append!(flat_wb, vec(group.wb))
    end
    ptr = @ccall LIB_SCI_BITSTR.build_network_otf_sci_bitstr_f64(
        orbsym::Ptr{Int64}, norb::Int64, ngs::Int64,
        axs::Ptr{Ti}, bxs::Ptr{Ti}, ranks::Ptr{Int64},
        num_as::Ptr{Int64}, num_bs::Ptr{Int64},
        flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti},
        flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv}
    )::Ptr{Cvoid}
    ptr == C_NULL && error("Failed to build bitstr OTF.")
    sci_ptr = @ccall LIB_SCI_BITSTR.build_network_sci_bitstr_f64(ptr::Ptr{Cvoid})::Ptr{Cvoid}
    sci_ptr == C_NULL && error("Failed to build SCI network.")
    obj = OTF(ptr, 0, ngs, sci_ptr)
    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_OTF.destroy_network_otf_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
        if o.sci_ptr != C_NULL
            @ccall LIB_SCI_BITSTR.destroy_network_sci_bitstr_f64(o.sci_ptr::Ptr{Cvoid})::Cvoid
            o.sci_ptr = C_NULL
        end
    end
    return obj
end

function expand_bitstrings_bitstr(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32},
    axs::Vector{UInt32}, bxs::Vector{UInt32},
    na::Int, nb::Int, orbsym::Vector{Int64}, num_irreps::Int=SCI_BITSTR_NUM_IRREPS)

    a_map = Dict{UInt32,Int}(a => 0 for a in src_astrs)
    b_map = Dict{UInt32,Int}(b => 0 for b in src_bstrs)

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

    dst_astrs = sort_by_sym(collect(keys(a_map)), orbsym, num_irreps)
    dst_bstrs = sort_by_sym(collect(keys(b_map)), orbsym, num_irreps)

    is_new_a = Bool[a_map[a] == 1 for a in dst_astrs]
    is_new_b = Bool[b_map[b] == 1 for b in dst_bstrs]

    return dst_astrs, dst_bstrs, is_new_a, is_new_b
end

function sci_hvec_select_external_bitstr!(
    tgt::SciBasisManagerBitstr, src::SciBasisManagerBitstr, otf::OTF,
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool},
    blk::Int, psi::Vector{Float64}, candidate_diags::Vector{Float64},
    variational_energy::Float64, chunk_size::Int, eps::Float64,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    max_per_block = tgt.dim
    buf_a = Vector{UInt32}(undef, max_per_block)
    buf_b = Vector{UInt32}(undef, max_per_block)
    buf_v = Vector{Float64}(undef, max_per_block)

    n = @ccall LIB_SCI_BITSTR.sci_hvec_select_external_bitstr_f64(
        tgt.ptr::Ptr{Cvoid}, src.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        is_new_a::Ptr{Bool}, is_new_b::Ptr{Bool},
        blk::Int64, psi::Ptr{Float64}, candidate_diags::Ptr{Float64},
        variational_energy::Cdouble, chunk_size::Cint, eps::Cdouble,
        buf_a::Ptr{UInt32}, buf_b::Ptr{UInt32}, buf_v::Ptr{Float64},
        max_per_block::Int64
    )::Int64

    append!(sel_a, view(buf_a, 1:n))
    append!(sel_b, view(buf_b, 1:n))
    append!(sel_v, view(buf_v, 1:n))

    return n
end

function sci_hvec_select_external_link_all_blocks_bitstr!(
    tgt::SciBasisManagerBitstr, src::SciBasisManagerBitstr, otf::OTF,
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool},
    unique_axs::Vector{UInt32}, unique_bxs::Vector{UInt32},
    psi::Vector{Float64}, candidate_diags::Vector{Float64},
    variational_energy::Float64, chunk_size::Int, eps::Float64,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    max_entries = tgt.dim
    buf_a = Vector{UInt32}(undef, max_entries)
    buf_b = Vector{UInt32}(undef, max_entries)
    buf_v = Vector{Float64}(undef, max_entries)

    n = @ccall LIB_SCI_BITSTR.sci_hvec_select_external_link_all_blocks_with_masks_bitstr_f64(
        tgt.ptr::Ptr{Cvoid}, src.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, otf.sci_ptr::Ptr{Cvoid},
        is_new_a::Ptr{Bool}, is_new_b::Ptr{Bool},
        unique_axs::Ptr{UInt32}, length(unique_axs)::Int64,
        unique_bxs::Ptr{UInt32}, length(unique_bxs)::Int64,
        psi::Ptr{Float64}, candidate_diags::Ptr{Float64},
        variational_energy::Cdouble, chunk_size::Cint, eps::Cdouble,
        buf_a::Ptr{UInt32}, buf_b::Ptr{UInt32}, buf_v::Ptr{Float64},
        max_entries::Int64
    )::Int64

    append!(sel_a, view(buf_a, 1:n))
    append!(sel_b, view(buf_b, 1:n))
    append!(sel_v, view(buf_v, 1:n))

    return n
end

function merge_bitstrings(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32},
    orbsym::Vector{Int64}, num_irreps::Int=SCI_BITSTR_NUM_IRREPS)

    a_union = sort_by_sym(collect(union(Set(src_astrs), Set(sel_a))), orbsym, num_irreps)
    b_union = sort_by_sym(collect(union(Set(src_bstrs), Set(sel_b))), orbsym, num_irreps)
    return a_union, b_union
end

function remap_wavefunction_bitstr!(
    old::SciBasisManagerBitstr, old_psi::Vector{Float64},
    new::SciBasisManagerBitstr, new_psi::Vector{Float64},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    @ccall LIB_SCI_BITSTR.remap_wavefunction_sci_bitstr_f64(
        old.ptr::Ptr{Cvoid}, old_psi::Ptr{Float64},
        new.ptr::Ptr{Cvoid}, new_psi::Ptr{Float64},
        sel_a::Ptr{UInt32}, sel_b::Ptr{UInt32}, sel_v::Ptr{Float64},
        length(sel_a)::Int64
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

function select_external_block_bitstr!(tgt::SciBasisManagerBitstr, src::SciBasisManagerBitstr,
    otf::OTF, psi::Vector{Float64}, candidate_diags::Vector{Float64},
    variational_energy::Float64, chunk_size::Int, eps::Float64,
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    for blk in 0:tgt.num_blocks-1
        sci_hvec_select_external_bitstr!(
            tgt, src, otf, is_new_a, is_new_b,
            blk, psi, candidate_diags, variational_energy,
            chunk_size, eps, sel_a, sel_b, sel_v
        )
    end
end

function select_external_link_bitstr!(tgt::SciBasisManagerBitstr, src::SciBasisManagerBitstr,
    otf::OTF, psi::Vector{Float64}, candidate_diags::Vector{Float64},
    variational_energy::Float64, chunk_size::Int, eps::Float64,
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool},
    unique_axs::Vector{UInt32}, unique_bxs::Vector{UInt32},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    sci_hvec_select_external_link_all_blocks_bitstr!(
        tgt, src, otf, is_new_a, is_new_b, unique_axs, unique_bxs,
        psi, candidate_diags, variational_energy,
        chunk_size, eps, sel_a, sel_b, sel_v
    )
end

function run_sci_bitstr(mole::Mole;
    max_iter::Int=20, max_size::Int=10000, eps::Float64=1e-6,
    chunk_size::Int=256, davidson_tol::Float64=1e-5, verbose::Bool=true, total_sym::Int64=0,
    select_mode::Symbol=:external_link)

    # SCI selection uses a first-order/CIPSI-style amplitude estimate for each
    # candidate determinant: abs(Hψ(candidate) / (E - Haa)) > eps, where E is
    # the current variational energy and Haa is the candidate diagonal matrix
    # element.  Thus eps thresholds an energy-aware estimated CI coefficient,
    # not the raw residual |Hψ(candidate)|.

    na, nb = mole.nelec
    num_irreps = SCI_BITSTR_NUM_IRREPS

    ham = JW_hamiltonian(mole)
    svd_groups = compress_by_svd(ham)
    all_axs, all_bxs = extract_ax_bx(svd_groups)
    unique_axs = unique(all_axs)
    unique_bxs = unique(all_bxs)
    ham_otf = OTF_bitstr(mole.orbsym, mole.norb, ham)

    hf_astr = UInt32((1 << na) - 1)
    hf_bstr = UInt32((1 << nb) - 1)
    basis = SciBasisManagerBitstr([hf_astr], [hf_bstr], mole.norb, total_sym, mole.orbsym, na, nb; num_irreps=num_irreps)
    psi = Float64[1.0]
    diags = zeros(Float64, 1)
    get_diags_bitstr!(basis, ham_otf, diags)
    current_energy = diags[1]
    verbose && @printf("Initial basis: dim=%d  E0=%.10f\n\n", basis.dim, current_energy)

    for iter in 1:max_iter
        t1 = @elapsed dst_a, dst_b, is_new_a, is_new_b = expand_bitstrings_bitstr(basis.astrs, basis.bstrs, all_axs, all_bxs, na, nb, mole.orbsym, num_irreps)
        tgt = SciBasisManagerBitstr(dst_a, dst_b, mole.norb, total_sym, mole.orbsym, na, nb; sorted=true, num_irreps=num_irreps)
        tgt_diags = zeros(Float64, tgt.dim)
        get_diags_bitstr!(tgt, ham_otf, tgt_diags)

        sel_a = UInt32[]
        sel_b = UInt32[]
        sel_v = Float64[]

        t2 = @elapsed if select_mode == :external_block
            select_external_block_bitstr!(
                tgt, basis, ham_otf, psi, tgt_diags,
                current_energy, chunk_size, eps, is_new_a, is_new_b,
                sel_a, sel_b, sel_v
            )
        else
            select_external_link_bitstr!(
                tgt, basis, ham_otf, psi, tgt_diags,
                current_energy, chunk_size, eps, is_new_a, is_new_b,
                unique_axs, unique_bxs, sel_a, sel_b, sel_v
            )
        end

        nsel = length(sel_v)
        if verbose
            @printf("Iteration: %d\n", iter)
            @printf("  Expand           %d → %d\n", basis.dim, tgt.dim)
            @printf("  New pairs        %d\n", nsel)
        end

        if nsel == 0
            verbose && println("No new states, done.")
            destroy_sci_basis_manager_bitstr(tgt)
            break
        end

        t4 = @elapsed new_a, new_b = merge_bitstrings(basis.astrs, basis.bstrs, sel_a, sel_b, mole.orbsym, num_irreps)
        new_basis = SciBasisManagerBitstr(new_a, new_b, mole.norb, total_sym, mole.orbsym, na, nb; sorted=true, num_irreps=num_irreps)

        if verbose
            @printf("  Merged           %d\n", new_basis.dim)
        end

        new_psi = zeros(Float64, new_basis.dim)
        t5 = @elapsed remap_wavefunction_bitstr!(basis, psi, new_basis, new_psi, sel_a, sel_b, sel_v)
        new_diags = zeros(Float64, new_basis.dim)
        t6 = @elapsed get_diags_bitstr!(new_basis, ham_otf, new_diags)
        t7 = @elapsed E, psi_new = davidson(
            (v, Hv) -> hvec_sci_full_bitstr!(new_basis, ham_otf, v, Hv),
            new_psi,
            new_diags;
            tol=davidson_tol,
            ncv=1,
            maxspace=min(max_size, new_basis.dim + 20),
            verbose=false
        )

        if verbose
            @printf("  Energy           %.14f\n", E)
            @printf("  Error            %.3e\n\n", abs(E - mole.e_scale))
        end

        destroy_sci_basis_manager_bitstr(tgt)
        destroy_sci_basis_manager_bitstr(basis)
        basis, psi, diags = new_basis, psi_new, new_diags
        current_energy = E

        if verbose
            @printf("  Expand           %-8.4f seconds\n", t1)
            @printf("  Select           %-8.4f seconds\n", t2)
            @printf("  Merge            %-8.4f seconds\n", t4)
            @printf("  Remap            %-8.4f seconds\n", t5)
            @printf("  Get_diags        %-8.4f seconds\n", t6)
            @printf("  Diag             %-8.4f seconds\n\n", t7)
        end
    end

    return basis, psi, diags
end
