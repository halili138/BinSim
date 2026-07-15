mutable struct SciBasisManager
    ptr::Ptr{Cvoid}
    dim::Int64
    norb::Int64
    num_blocks::Int64
    na::Int64
    nb::Int64
    astrs::Vector{UInt32}
    bstrs::Vector{UInt32}
end

function SciBasisManager()
    SciBasisManager(C_NULL, 0, 0, 0, -1, -1, UInt32[], UInt32[])
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

function SciBasisManager(mole::Mole, astrs::Vector{UInt32}, bstrs::Vector{UInt32};
    sorted::Bool=false, num_irreps::Int=16, total_sym::Int=0
)
    na, nb = mole.nelec
    astrs  = UInt32[a for a in astrs if count_ones(a) == na]
    bstrs  = UInt32[b for b in bstrs if count_ones(b) == nb]

    @assert all(count_ones.(astrs) .== na)
    @assert all(count_ones.(bstrs) .== nb)

    if !sorted
        astrs = sort_by_sym(astrs, mole.orbsym, num_irreps)
        bstrs = sort_by_sym(bstrs, mole.orbsym, num_irreps)
    end

    ptr = @ccall LIB_SCI_BITSTR.create_sci_basis_manager_bitstr_f64(
        astrs::Ptr{UInt32}, length(astrs)::Int64,
        bstrs::Ptr{UInt32}, length(bstrs)::Int64,
        mole.norb::Int64, mole.orbsym::Ptr{Int64},
        total_sym::Int64, num_irreps::Int64
    )::Ptr{Cvoid}

    ptr == C_NULL && error("Failed to create bitstr basis manager.")

    dim = @ccall LIB_SCI_BITSTR.sci_basis_dim_bitstr(ptr::Ptr{Cvoid})::Int64
    nbk = @ccall LIB_SCI_BITSTR.sci_basis_num_blocks_bitstr(ptr::Ptr{Cvoid})::Int64
    obj = SciBasisManager(ptr, dim, mole.norb, nbk, na, nb, astrs, bstrs)

    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_SCI_BITSTR.destroy_sci_basis_manager_bitstr_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end

    return obj
end

function expand_bitstrings_bitstr(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32}, 
    axs::Vector{UInt32}, bxs::Vector{UInt32}, 
    nelec::Tuple{Int,Int}, orbsym::Vector{Int64}, num_irreps::Int=16
)

    na, nb = nelec
    a_map  = Dict{UInt32,Int}(a => 0 for a in src_astrs)
    b_map  = Dict{UInt32,Int}(b => 0 for b in src_bstrs)

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
    dst::SciBasisManager, src::SciBasisManager, otf::OTF,
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool},
    blk::Int, psi::Vector{Float64}, candidate_diags::Vector{Float64},
    variational_energy::Float64, chunk_size::Int, eps::Float64,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64}
)
    buf_a = Vector{UInt32}(undef, dst.dim)
    buf_b = Vector{UInt32}(undef, dst.dim)
    buf_v = Vector{Float64}(undef, dst.dim)

    n = @ccall LIB_SCI_BITSTR.sci_hvec_select_external_bitstr_f64(
        dst.ptr::Ptr{Cvoid}, src.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        is_new_a::Ptr{Bool}, is_new_b::Ptr{Bool},
        blk::Int64, psi::Ptr{Float64}, candidate_diags::Ptr{Float64},
        variational_energy::Cdouble, chunk_size::Cint, eps::Cdouble,
        buf_a::Ptr{UInt32}, buf_b::Ptr{UInt32}, buf_v::Ptr{Float64},
        dst.dim::Int64
    )::Int64

    append!(sel_a, view(buf_a, 1:n))
    append!(sel_b, view(buf_b, 1:n))
    append!(sel_v, view(buf_v, 1:n))

    return n
end

function select_external_block_bitstr!(
    tgt::SciBasisManager, 
    src::SciBasisManager,
    otf::OTF, 
    psi::Vector{Float64}, 
    candidate_diags::Vector{Float64},
    variational_energy::Float64, 
    chunk_size::Int, 
    eps::Float64,
    is_new_a::Vector{Bool}, 
    is_new_b::Vector{Bool},
    sel_a::Vector{UInt32}, 
    sel_b::Vector{UInt32}, 
    sel_v::Vector{Float64}
)
    for blk in 0:tgt.num_blocks-1
        sci_hvec_select_external_bitstr!(
            tgt, src, otf, is_new_a, is_new_b,
            blk, psi, candidate_diags, variational_energy,
            chunk_size, eps, sel_a, sel_b, sel_v
        )
    end
end

function sci_select_block_instant!(
    dst::SciBasisManager, src::SciBasisManager, otf::OTF,
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool},
    blk::Int, psi::Vector{Float64},
    variational_energy::Float64, a_chunk_size::Int, b_chunk_size::Int, eps::Float64,
    selected_a::Vector{Bool}, selected_b::Vector{Bool}
)
    @ccall LIB_SCI_BITSTR.sci_select_instant_bitstr_f64(
        dst.ptr::Ptr{Cvoid}, src.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        is_new_a::Ptr{Bool}, is_new_b::Ptr{Bool},
        blk::Int64, psi::Ptr{Float64},
        variational_energy::Cdouble, a_chunk_size::Cint, b_chunk_size::Cint, eps::Cdouble,
        selected_a::Ptr{Bool}, selected_b::Ptr{Bool}
    )::Cvoid
end

function select_instant_all_blocks!(
    tgt::SciBasisManager, 
    src::SciBasisManager,
    otf::OTF, 
    psi::Vector{Float64}, 
    variational_energy::Float64, 
    a_chunk_size::Int, b_chunk_size::Int,
    eps::Float64,
    is_new_a::Vector{Bool}, 
    is_new_b::Vector{Bool},
    selected_a::Vector{Bool}, 
    selected_b::Vector{Bool}
)
    for blk in 0:tgt.num_blocks-1
        sci_select_block_instant!(
            tgt, src, otf, is_new_a, is_new_b,
            blk, psi, variational_energy,
            a_chunk_size, b_chunk_size, eps,
            selected_a, selected_b
        )
    end
end

function merge_bitstrings(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32},
    orbsym::Vector{Int64}, num_irreps::Int=16
)
    a_union = sort_by_sym(collect(union(Set(src_astrs), Set(sel_a))), orbsym, num_irreps)
    b_union = sort_by_sym(collect(union(Set(src_bstrs), Set(sel_b))), orbsym, num_irreps)

    return a_union, b_union
end

function remap_wavefunction_bitstr!(
    old::SciBasisManager, old_psi::Vector{Float64},
    new::SciBasisManager, new_psi::Vector{Float64},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64}
)
    @ccall LIB_SCI_BITSTR.remap_wavefunction_sci_bitstr_f64(
        old.ptr::Ptr{Cvoid}, old_psi::Ptr{Float64},
        new.ptr::Ptr{Cvoid}, new_psi::Ptr{Float64},
        sel_a::Ptr{UInt32}, sel_b::Ptr{UInt32}, sel_v::Ptr{Float64},
        length(sel_a)::Int64
    )::Cvoid
end

function get_diags_bitstr!(basis::SciBasisManager, otf::OTF, diags::Vector{Float64})
    @ccall LIB_SCI_BITSTR.get_diags_elements_sci_bitstr_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64}
    )::Cvoid
end

function sci3_select_bitstr!(
    new_α::Vector{UInt32}, new_β::Vector{UInt32},
    old_α::Vector{UInt32}, old_β::Vector{UInt32},
    src_basis::SciBasisManager, otf::OTF,
    psi::Vector{Float64}, E_var::Float64, eps::Float64,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}
)
    out_a_ref = Ref{Ptr{UInt32}}(C_NULL)
    out_b_ref = Ref{Ptr{UInt32}}(C_NULL)
    n_ref = Ref{Int64}(0)

    @ccall LIB_SCI_BITSTR.sci3_select_bitstr_f64(
        new_α::Ptr{UInt32}, length(new_α)::Int64,
        new_β::Ptr{UInt32}, length(new_β)::Int64,
        old_α::Ptr{UInt32}, length(old_α)::Int64,
        old_β::Ptr{UInt32}, length(old_β)::Int64,
        src_basis.ptr::Ptr{Cvoid},
        otf.ptr::Ptr{Cvoid},
        psi::Ptr{Float64},
        E_var::Cdouble, eps::Cdouble,
        out_a_ref::Ptr{Ptr{UInt32}}, out_b_ref::Ptr{Ptr{UInt32}}, n_ref::Ptr{Int64}
    )::Cvoid

    n = n_ref[]
    if n > 0
        out_a = unsafe_wrap(Array, out_a_ref[], Int(n); own=false)
        out_b = unsafe_wrap(Array, out_b_ref[], Int(n); own=false)
        append!(sel_a, out_a)
        append!(sel_b, out_b)
        @ccall free(out_a_ref[]::Ptr{Cvoid})::Cvoid
        @ccall free(out_b_ref[]::Ptr{Cvoid})::Cvoid
    end
end

function hvec_svd!(basis::SciBasisManager, otf::OTF, src::Vector{Float64}, dst::Vector{Float64})
    @ccall LIB_SCI_BITSTR.hvec_sci_full_bitstr_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        src::Ptr{Float64}, dst::Ptr{Float64}
    )::Cvoid
end

function destroy_sci_basis_manager_bitstr(sb::SciBasisManager)
    if sb.ptr != C_NULL
        @ccall LIB_SCI_BITSTR.destroy_sci_basis_manager_bitstr_f64(sb.ptr::Ptr{Cvoid})::Cvoid
        sb.ptr = C_NULL
    end
end

function run_sci_bitstr(mole::Mole;
    max_iter::Int=20, eps::Float64=1e-6, a_chunk_size::Int=256, b_chunk_size::Int=256,
    verbose::Bool=true)

    na, nb  = mole.nelec
    ham     = JW_hamiltonian(mole)
    basis   = SciBasisManager(mole, [UInt32(1 << na - 1)], [UInt32(1 << nb - 1)])
    otf     = OTF(ham, mole.orbsym)
    all_axs = sort!(unique(ham.axs))
    all_bxs = sort!(unique(ham.bxs))

    psi     = Float64[1.0]

    diags   = Float64[0.0]
    @ccall LIB_SCI_BITSTR.get_diags_elements_sci_bitstr_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64}
    )::Cvoid

    current_energy = diags[1]

    verbose && @printf("Initial basis: dim=%d  E0=%.10f\n\n", basis.dim, current_energy)

    for iter in 1:max_iter
        t1 = @elapsed new_astrs, new_bstrs, is_new_a, is_new_b = expand_bitstrings_bitstr(
            basis.astrs, basis.bstrs, all_axs, all_bxs, mole.nelec, mole.orbsym
        )
        new_basis = SciBasisManager(mole, new_astrs, new_bstrs, sorted=true)

        selected_a = fill(false, length(new_astrs))
        selected_b = fill(false, length(new_bstrs))

        t2 = @elapsed select_instant_all_blocks!(
            new_basis, basis, otf, psi,
            current_energy, a_chunk_size, b_chunk_size, eps,
            is_new_a, is_new_b,
            selected_a, selected_b
        )

        # extract newly selected strings
        new_sel_astrs = new_astrs[selected_a.&is_new_a]
        new_sel_bstrs = new_bstrs[selected_b.&is_new_b]
        num_sel = length(new_sel_astrs) + length(new_sel_bstrs)

        if verbose
            @printf("Iteration: %d\n", iter)
            @printf("  Expand           %d → %d\n", basis.dim, new_basis.dim)
            @printf("  New pairs        %d\n", num_sel)
        end

        if num_sel == 0
            verbose && println("No new states, done.")
            destroy_sci_basis_manager_bitstr(new_basis)
            break
        end

        t4 = @elapsed all_astrs, all_bstrs = merge_bitstrings(
            basis.astrs, basis.bstrs, new_sel_astrs, new_sel_bstrs, mole.orbsym
        )
        all_basis = SciBasisManager(mole, all_astrs, all_bstrs, sorted=true)

        if verbose
            @printf("  Merged           %d\n", all_basis.dim)
        end

        all_psi = zeros(Float64, all_basis.dim)
        t5 = @elapsed remap_wavefunction_bitstr!(basis, psi, all_basis, all_psi, UInt32[], UInt32[], Float64[])

        destroy_sci_basis_manager_bitstr(new_basis)
        destroy_sci_basis_manager_bitstr(basis)
        basis, psi = all_basis, all_psi

        diags = zeros(Float64, basis.dim)
        @ccall LIB_SCI_BITSTR.get_diags_elements_sci_bitstr_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64}
        )::Cvoid

        hvec = (v, Hv) -> @ccall LIB_SCI_BITSTR.hvec_sci_full_bitstr_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, v::Ptr{Float64}, Hv::Ptr{Float64}
        )::Cvoid

        t6 = @elapsed current_energy, psi = davidson(hvec, psi, diags, verbose=false)

        if verbose
            @printf("  Energy           %.14f\n", current_energy)
            @printf("  Error            %.3e\n\n", abs(current_energy - mole.e_scale))
        end

        if verbose
            @printf("  Expand           %-8.4f seconds\n", t1)
            @printf("  Select           %-8.4f seconds\n", t2)
            @printf("  Merge            %-8.4f seconds\n", t4)
            @printf("  Remap            %-8.4f seconds\n", t5)
            @printf("  Diag             %-8.4f seconds\n\n", t6)
        end
    end

    return basis, psi, diags
end

function run_sci_bitstr_old(mole::Mole;
    max_iter::Int=20, eps::Float64=1e-6, chunk_size::Int=256, verbose::Bool=true)

    # SCI selection uses a first-order/CIPSI-style amplitude estimate for each
    # candidate determinant: abs(Hψ(candidate) / (E - Haa)) > eps, where E is
    # the current variational energy and Haa is the candidate diagonal matrix
    # element.  Thus eps thresholds an energy-aware estimated CI coefficient,
    # not the raw residual |Hψ(candidate)|.

    na, nb  = mole.nelec
    ham     = JW_hamiltonian(mole)
    basis   = SciBasisManager(mole, [UInt32(1 << na - 1)], [UInt32(1 << nb - 1)])
    otf     = OTF(ham, mole.orbsym)
    all_axs = sort!(unique(ham.axs))
    all_bxs = sort!(unique(ham.bxs))

    psi     = Float64[1.0]

    diags   = Float64[0.0]
    @ccall LIB_SCI_BITSTR.get_diags_elements_sci_bitstr_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64}
    )::Cvoid

    current_energy = diags[1]

    verbose && @printf("Initial basis: dim=%d  E0=%.10f\n\n", basis.dim, current_energy)

    for iter in 1:max_iter
        # Expand old basis, get new basis
        t1 = @elapsed new_astrs, new_bstrs, is_new_a, is_new_b = expand_bitstrings_bitstr(
            basis.astrs, basis.bstrs, all_axs, all_bxs, mole.nelec, mole.orbsym
        )
        new_basis = SciBasisManager(mole, new_astrs, new_bstrs, sorted=true)
        new_diags = zeros(Float64, new_basis.dim)
        @ccall LIB_SCI_BITSTR.get_diags_elements_sci_bitstr_f64(
            new_basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, new_diags::Ptr{Float64}
        )::Cvoid


        # Select from new basis, get selected basis
        sel_astrs = UInt32[]
        sel_bstrs = UInt32[]
        sel_vals  = Float64[]
        t2 = @elapsed select_external_block_bitstr!(
            new_basis, basis, otf, psi, new_diags,
            current_energy, chunk_size, eps, is_new_a, is_new_b,
            sel_astrs, sel_bstrs, sel_vals
        )

        if verbose
            @printf("Iteration: %d\n", iter)
            @printf("  Expand           %d → %d\n", basis.dim, new_basis.dim)
            @printf("  New pairs        %d\n", length(sel_vals))
        end

        if length(sel_vals) == 0
            verbose && println("No new states, done.")
            break
        end


        # Merge old basis and selected basis, get subspace basis
        t4 = @elapsed all_astrs, all_bstrs = merge_bitstrings(basis.astrs, basis.bstrs, sel_astrs, sel_bstrs, mole.orbsym)
        all_basis = SciBasisManager(mole, all_astrs, all_bstrs, sorted=true)

        if verbose
            @printf("  Merged           %d\n", all_basis.dim)
        end

        all_psi = zeros(Float64, all_basis.dim)
        t5 = @elapsed remap_wavefunction_bitstr!(basis, psi, all_basis, all_psi, sel_astrs, sel_bstrs, sel_vals)


        # Perform subspace diag on subspace basis
        basis, psi = all_basis, all_psi
        diags = zeros(Float64, basis.dim)
        @ccall LIB_SCI_BITSTR.get_diags_elements_sci_bitstr_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64}
        )::Cvoid

        hvec = (v, Hv) -> @ccall LIB_SCI_BITSTR.hvec_sci_full_bitstr_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, v::Ptr{Float64}, Hv::Ptr{Float64}
        )::Cvoid

        t6 = @elapsed current_energy, psi = davidson(hvec, psi, diags, verbose=false)

        if verbose
            @printf("  Energy           %.14f\n", current_energy)
            @printf("  Error            %.3e\n\n", abs(current_energy - mole.e_scale))
        end

        if verbose
            @printf("  Expand           %-8.4f seconds\n", t1)
            @printf("  Select           %-8.4f seconds\n", t2)
            @printf("  Merge            %-8.4f seconds\n", t4)
            @printf("  Remap            %-8.4f seconds\n", t5)
            @printf("  Diag             %-8.4f seconds\n\n", t6)
        end
    end

    return basis, psi, diags
end

function run_sci_bitstr2(mole::Mole;
    max_iter::Int=20, eps::Float64=1e-6, verbose::Bool=true)

    na, nb  = mole.nelec
    ham     = JW_hamiltonian(mole)
    basis   = SciBasisManager(mole, [UInt32(1 << na - 1)], [UInt32(1 << nb - 1)])
    otf     = OTF(ham, mole.orbsym)
    all_axs = sort!(unique(ham.axs))
    all_bxs = sort!(unique(ham.bxs))

    psi     = Float64[1.0]

    diags   = Float64[0.0]
    @ccall LIB_SCI_BITSTR.get_diags_elements_sci_bitstr_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64}
    )::Cvoid

    current_energy = diags[1]

    verbose && @printf("Initial: dim=%d  E0=%.10f\n\n", basis.dim, current_energy)

    for iter in 1:max_iter
        t1 = @elapsed new_astrs, new_bstrs, is_new_a, is_new_b = expand_bitstrings_bitstr(
            basis.astrs, basis.bstrs, all_axs, all_bxs, mole.nelec, mole.orbsym
        )

        old_α = new_astrs[.!is_new_a]
        old_β = new_bstrs[.!is_new_b]
        new_α = new_astrs[is_new_a]
        new_β = new_bstrs[is_new_b]

        if verbose
            @printf("Iter %d: |newα|=%d |newβ|=%d |oldα|=%d |oldβ|=%d\n",
                iter, length(new_α), length(new_β), length(old_α), length(old_β))
        end

        sel_a = UInt32[]
        sel_b = UInt32[]
        t2 = @elapsed sci3_select_bitstr!(
            new_α, new_β, old_α, old_β,
            basis, otf, psi, current_energy, eps,
            sel_a, sel_b
        )

        unique!(sel_a); unique!(sel_b)
        num_sel = length(sel_a) + length(sel_b)
        verbose && @printf("  select=%d (%da %db)\n", num_sel, length(sel_a), length(sel_b))

        if num_sel == 0
            verbose && println("Done.")
            break
        end

        t4 = @elapsed all_astrs, all_bstrs = merge_bitstrings(
            basis.astrs, basis.bstrs, sel_a, sel_b, mole.orbsym
        )
        all_basis = SciBasisManager(mole, all_astrs, all_bstrs, sorted=true)
        verbose && @printf("  merge=%d\n", all_basis.dim)

        all_psi = zeros(Float64, all_basis.dim)
        t5 = @elapsed remap_wavefunction_bitstr!(basis, psi, all_basis, all_psi, UInt32[], UInt32[], Float64[])

        destroy_sci_basis_manager_bitstr(basis)
        basis, psi = all_basis, all_psi

        diags = zeros(Float64, basis.dim)
        @ccall LIB_SCI_BITSTR.get_diags_elements_sci_bitstr_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64}
        )::Cvoid

        hvec = (v, Hv) -> @ccall LIB_SCI_BITSTR.hvec_sci_full_bitstr_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, v::Ptr{Float64}, Hv::Ptr{Float64}
        )::Cvoid

        t6 = @elapsed current_energy, psi = davidson(hvec, psi, diags, verbose=false)

        if verbose
            @printf("  E=%.14f  err=%.3e  t:exp=%.1fs sel=%.1fs mrg=%.1fs rmp=%.1fs dg=%.1fs\n\n",
                current_energy, abs(current_energy - mole.e_scale),
                t1, t2, t4, t5, t6)
        end
    end

    return basis, psi, diags
end
