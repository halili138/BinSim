const SCI_BITSTR_NUM_IRREPS = 16

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

function SciBasisManager(astrs::Vector{UInt32}, bstrs::Vector{UInt32}, norb::Int64, total_sym::Int64, orbsym::Vector{Int64}, na::Int, nb::Int;
    sorted::Bool=false, num_irreps::Int=SCI_BITSTR_NUM_IRREPS
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

    ptr = @ccall LIB_SCI_BITSTR.create_sci_basis_manager_bitstr_f64(
        astrs::Ptr{UInt32}, num_a::Int64,
        bstrs::Ptr{UInt32}, num_b::Int64,
        norb::Int64, orbsym::Ptr{Int64},
        total_sym::Int64, num_irreps::Int64
    )::Ptr{Cvoid}

    ptr == C_NULL && error("Failed to create bitstr basis manager.")

    dim = @ccall LIB_SCI_BITSTR.sci_basis_dim_bitstr(ptr::Ptr{Cvoid})::Int64
    nbk = @ccall LIB_SCI_BITSTR.sci_basis_num_blocks_bitstr(ptr::Ptr{Cvoid})::Int64

    obj = SciBasisManager(ptr, dim, norb, nbk, na, nb, astrs, bstrs)
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
    tgt::SciBasisManager, src::SciBasisManager, otf::OTF,
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

function merge_bitstrings(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32},
    orbsym::Vector{Int64}, num_irreps::Int=SCI_BITSTR_NUM_IRREPS)

    a_union = sort_by_sym(collect(union(Set(src_astrs), Set(sel_a))), orbsym, num_irreps)
    b_union = sort_by_sym(collect(union(Set(src_bstrs), Set(sel_b))), orbsym, num_irreps)
    return a_union, b_union
end

function remap_wavefunction_bitstr!(
    old::SciBasisManager, old_psi::Vector{Float64},
    new::SciBasisManager, new_psi::Vector{Float64},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    @ccall LIB_SCI_BITSTR.remap_wavefunction_sci_bitstr_f64(
        old.ptr::Ptr{Cvoid}, old_psi::Ptr{Float64},
        new.ptr::Ptr{Cvoid}, new_psi::Ptr{Float64},
        sel_a::Ptr{UInt32}, sel_b::Ptr{UInt32}, sel_v::Ptr{Float64},
        length(sel_a)::Int64
    )::Cvoid
end

function get_diags_bitstr!(basis::SciBasisManager, otf::OTF, diags::Vector{Float64})
    @ccall LIB_SCI_BITSTR.get_diags_elements_sci_bitstr_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64})::Cvoid
end

function hvec_svd!(basis::SciBasisManager, otf::OTF, src::Vector{Float64}, dst::Vector{Float64})
    @ccall LIB_SCI_BITSTR.hvec_sci_full_bitstr_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        src::Ptr{Float64}, dst::Ptr{Float64})::Cvoid
end

function destroy_sci_basis_manager_bitstr(sb::SciBasisManager)
    if sb.ptr != C_NULL
        @ccall LIB_SCI_BITSTR.destroy_sci_basis_manager_bitstr_f64(sb.ptr::Ptr{Cvoid})::Cvoid
        sb.ptr = C_NULL
    end
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

function run_sci_bitstr(mole::Mole;
    max_iter::Int=20, eps::Float64=1e-6, chunk_size::Int=256, verbose::Bool=true)

    # SCI selection uses a first-order/CIPSI-style amplitude estimate for each
    # candidate determinant: abs(Hψ(candidate) / (E - Haa)) > eps, where E is
    # the current variational energy and Haa is the candidate diagonal matrix
    # element.  Thus eps thresholds an energy-aware estimated CI coefficient,
    # not the raw residual |Hψ(candidate)|.

    na, nb = mole.nelec

    ham     = JW_hamiltonian(mole)
    basis   = SciBasisManager([UInt32(1 << na - 1)], [UInt32(1 << nb - 1)], mole.norb, 0, mole.orbsym, na, nb, num_irreps=SCI_BITSTR_NUM_IRREPS)
    otf     = OTF(basis, ham)
    all_axs = sort!(unique(ham.axs))
    all_bxs = sort!(unique(ham.bxs))

    psi     = Float64[1.0]
    diags   = zeros(Float64, 1)
    get_diags_bitstr!(basis, otf, diags)
    current_energy = diags[1]

    verbose && @printf("Initial basis: dim=%d  E0=%.10f\n\n", basis.dim, current_energy)

    for iter in 1:max_iter
        t1 = @elapsed dst_a, dst_b, is_new_a, is_new_b = expand_bitstrings_bitstr(basis.astrs, basis.bstrs, all_axs, all_bxs, na, nb, mole.orbsym, SCI_BITSTR_NUM_IRREPS)
        tgt = SciBasisManager(dst_a, dst_b, mole.norb, total_sym, mole.orbsym, na, nb; sorted=true, num_irreps=SCI_BITSTR_NUM_IRREPS)
        tgt_diags = zeros(Float64, tgt.dim)
        get_diags_bitstr!(tgt, otf, tgt_diags)

        sel_a = UInt32[]
        sel_b = UInt32[]
        sel_v = Float64[]

        t2 = @elapsed select_external_block_bitstr!(
            tgt, basis, otf, psi, tgt_diags,
            current_energy, chunk_size, eps, is_new_a, is_new_b,
            sel_a, sel_b, sel_v
        )

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

        t4 = @elapsed new_a, new_b = merge_bitstrings(basis.astrs, basis.bstrs, sel_a, sel_b, mole.orbsym, SCI_BITSTR_NUM_IRREPS)
        new_basis = SciBasisManager(new_a, new_b, mole.norb, total_sym, mole.orbsym, na, nb; sorted=true, num_irreps=SCI_BITSTR_NUM_IRREPS)

        if verbose
            @printf("  Merged           %d\n", new_basis.dim)
        end

        new_psi = zeros(Float64, new_basis.dim)
        t5 = @elapsed remap_wavefunction_bitstr!(basis, psi, new_basis, new_psi, sel_a, sel_b, sel_v)
        new_diags = zeros(Float64, new_basis.dim)
        t6 = @elapsed get_diags_bitstr!(new_basis, otf, new_diags)
        t7 = @elapsed E, psi_new = davidson(
            (v, Hv) -> hvec_svd!(new_basis, otf, v, Hv),
            new_psi,
            new_diags;
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
