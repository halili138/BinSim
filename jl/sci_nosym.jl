mutable struct SciBasisManagerNosym
    ptr::Ptr{Cvoid}
    dim::Int64
    norb::Int64
    na::Int64
    nb::Int64
    astrs::Vector{UInt32}
    bstrs::Vector{UInt32}
end

function SciBasisManagerNosym()
    SciBasisManagerNosym(C_NULL, 0, 0, -1, -1, UInt32[], UInt32[])
end

function SciBasisManagerNosym(mole::Mole, astrs::Vector{UInt32}, bstrs::Vector{UInt32})
    na, nb = mole.nelec
    astrs  = UInt32[a for a in astrs if count_ones(a) == na]
    bstrs  = UInt32[b for b in bstrs if count_ones(b) == nb]

    @assert all(count_ones.(astrs) .== na)
    @assert all(count_ones.(bstrs) .== nb)

    ptr = @ccall LIB_SCI_NOSYM.create_sci_basis_manager_nosym_f64(
        astrs::Ptr{UInt32}, length(astrs)::Int64,
        bstrs::Ptr{UInt32}, length(bstrs)::Int64,
        mole.norb::Int64
    )::Ptr{Cvoid}

    ptr == C_NULL && error("Failed to create nosym basis manager.")

    dim = @ccall LIB_SCI_NOSYM.sci_basis_nosym_dim_f64(ptr::Ptr{Cvoid})::Int64
    obj = SciBasisManagerNosym(ptr, dim, mole.norb, length(astrs), length(bstrs), astrs, bstrs)

    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_SCI_NOSYM.destroy_sci_basis_manager_nosym_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end
    return obj
end

function expand_bitstrings_bitstr(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32}, 
    axs::Vector{UInt32}, bxs::Vector{UInt32}, 
    nelec::Tuple{Int,Int}
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

    dst_astrs = sort(collect(keys(a_map)))
    dst_bstrs = sort(collect(keys(b_map)))

    is_new_a = Bool[a_map[a] == 1 for a in dst_astrs]
    is_new_b = Bool[b_map[b] == 1 for b in dst_bstrs]

    return dst_astrs, dst_bstrs, is_new_a, is_new_b
end

function merge_bitstrings(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}
)
    a_union = sort(collect(union(Set(src_astrs), Set(sel_a))))
    b_union = sort(collect(union(Set(src_bstrs), Set(sel_b))))
    return a_union, b_union
end

function sci_select_nosym!(
    new_α::Vector{UInt32}, new_β::Vector{UInt32},
    old_α::Vector{UInt32}, old_β::Vector{UInt32},
    src_basis::SciBasisManagerNosym, otf::OTF,
    psi::Vector{Float64}, E_var::Float64, eps::Float64,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}
)
    out_a_ref = Ref{Ptr{UInt32}}(C_NULL)
    out_b_ref = Ref{Ptr{UInt32}}(C_NULL)
    n_ref = Ref{Int64}(0)

    @ccall LIB_SCI_NOSYM.sci_select_nosym_f64(
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

function hvec_svd_nosym!(basis::SciBasisManagerNosym, otf::OTF, src::Vector{Float64}, dst::Vector{Float64})
    @ccall LIB_SCI_NOSYM.hvec_sci_nosym_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        src::Ptr{Float64}, dst::Ptr{Float64}
    )::Cvoid
end

function get_diags_nosym!(basis::SciBasisManagerNosym, otf::OTF, diags::Vector{Float64})
    @ccall LIB_SCI_NOSYM.get_diags_elements_sci_nosym_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64}
    )::Cvoid
end

function remap_wavefunction_nosym!(
    old::SciBasisManagerNosym, old_psi::Vector{Float64},
    new::SciBasisManagerNosym, new_psi::Vector{Float64}
)
    @ccall LIB_SCI_NOSYM.remap_wavefunction_sci_nosym_f64(
        old.ptr::Ptr{Cvoid}, old_psi::Ptr{Float64},
        new.ptr::Ptr{Cvoid}, new_psi::Ptr{Float64}
    )::Cvoid
end

function destroy_sci_basis_manager_nosym(sb::SciBasisManagerNosym)
    if sb.ptr != C_NULL
        @ccall LIB_SCI_NOSYM.destroy_sci_basis_manager_nosym_f64(sb.ptr::Ptr{Cvoid})::Cvoid
        sb.ptr = C_NULL
    end
end

function run_sci_nosym(mole::Mole;
    max_iter::Int=20, eps::Float64=1e-6, verbose::Bool=true)

    na, nb  = mole.nelec
    ham     = JW_hamiltonian(mole)
    basis   = SciBasisManagerNosym(mole, [UInt32(1 << na - 1)], [UInt32(1 << nb - 1)])
    otf     = OTF(ham, mole.orbsym)
    all_axs = sort!(unique(ham.axs))
    all_bxs = sort!(unique(ham.bxs))

    psi     = Float64[1.0]

    diags   = Float64[0.0]
    get_diags_nosym!(basis, otf, diags)
    current_energy = diags[1]

    verbose && @printf("Initial: dim=%d  E0=%.10f\n\n", basis.dim, current_energy)

    for iter in 1:max_iter
        t1 = @elapsed new_astrs, new_bstrs, is_new_a, is_new_b = expand_bitstrings_bitstr(
            basis.astrs, basis.bstrs, all_axs, all_bxs, mole.nelec
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
        t2 = @elapsed sci_select_nosym!(
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
            basis.astrs, basis.bstrs, sel_a, sel_b
        )
        all_basis = SciBasisManagerNosym(mole, all_astrs, all_bstrs)
        verbose && @printf("  merge=%d\n", all_basis.dim)

        all_psi = zeros(Float64, all_basis.dim)
        t5 = @elapsed remap_wavefunction_nosym!(basis, psi, all_basis, all_psi)

        destroy_sci_basis_manager_nosym(basis)
        basis, psi = all_basis, all_psi

        diags = zeros(Float64, basis.dim)
        get_diags_nosym!(basis, otf, diags)

        hvec = (v, Hv) -> hvec_svd_nosym!(basis, otf, v, Hv)
        t6 = @elapsed current_energy, psi = davidson(hvec, psi, diags, verbose=false)

        if verbose
            @printf("  E=%.14f  err=%.3e  t:exp=%.1fs sel=%.1fs mrg=%.1fs rmp=%.1fs dg=%.1fs\n\n",
                current_energy, abs(current_energy - mole.e_scale),
                t1, t2, t4, t5, t6)
        end
    end

    return basis, psi, diags
end
