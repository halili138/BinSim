mutable struct SciBasisManager{Ti}
    ptr::Ptr{Cvoid}
    dim::Int64
    norb::Int64
    num_blocks::Int64
    na::Int64
    nb::Int64
    astrs::Vector{Ti}
    bstrs::Vector{Ti}
end

function SciBasisManager()
    SciBasisManager{UInt32}(C_NULL, 0, 0, 0, -1, -1, UInt32[], UInt32[])
end

_sci_suffix(::Type{UInt32}) = "_ui32"
_sci_suffix(::Type{UInt64}) = "_ui64"
_sci_suffix(::Type{UInt128}) = "_ui128"

for Ti in (UInt32, UInt64, UInt128)
    sfx = _sci_suffix(Ti)
    @eval begin
        function _create_sci_basis_manager(astrs::Vector{$Ti}, bstrs::Vector{$Ti}, norb, orbsym, total_sym, num_irreps)
            @ccall LIB_SCI_SELECT.$(Symbol("create_sci_basis_manager_bitstr$(sfx)_f64"))(
                astrs::Ptr{$Ti}, length(astrs)::Int64,
                bstrs::Ptr{$Ti}, length(bstrs)::Int64,
                norb::Int64, orbsym::Ptr{Int64},
                total_sym::Int64, num_irreps::Int64
            )::Ptr{Cvoid}
        end

        function sci_select_bitstr!(
            new_α::Vector{$Ti}, new_β::Vector{$Ti},
            old_α::Vector{$Ti}, old_β::Vector{$Ti},
            src_basis::SciBasisManager{$Ti}, otf::OTF,
            psi::Vector{Float64}, E_var::Float64, eps::Float64,
            sel_a::Vector{$Ti}, sel_b::Vector{$Ti}
        )
            out_a_ref = Ref{Ptr{$Ti}}(C_NULL)
            out_b_ref = Ref{Ptr{$Ti}}(C_NULL)
            n_ref = Ref{Int64}(0)

            @ccall LIB_SCI_SELECT.$(Symbol("sci_select_bitstr$(sfx)_f64"))(
                new_α::Ptr{$Ti}, length(new_α)::Int64,
                new_β::Ptr{$Ti}, length(new_β)::Int64,
                old_α::Ptr{$Ti}, length(old_α)::Int64,
                old_β::Ptr{$Ti}, length(old_β)::Int64,
                src_basis.ptr::Ptr{Cvoid},
                otf.ptr::Ptr{Cvoid},
                psi::Ptr{Float64},
                E_var::Cdouble, eps::Cdouble,
                out_a_ref::Ptr{Ptr{$Ti}}, out_b_ref::Ptr{Ptr{$Ti}}, n_ref::Ptr{Int64}
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

        function hvec_svd!(basis::SciBasisManager{$Ti}, otf::OTF, src::Vector{Float64}, dst::Vector{Float64})
            @ccall LIB_SCI_SELECT.$(Symbol("hvec_sci_full_bitstr$(sfx)_f64"))(
                basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
                src::Ptr{Float64}, dst::Ptr{Float64}
            )::Cvoid
        end

        function get_diags_bitstr!(basis::SciBasisManager{$Ti}, otf::OTF, diags::Vector{Float64})
            @ccall LIB_SCI_SELECT.$(Symbol("get_diags_elements_sci_bitstr$(sfx)_f64"))(
                basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64}
            )::Cvoid
        end

        function remap_wavefunction_bitstr!(
            old::SciBasisManager{$Ti}, old_psi::Vector{Float64},
            new::SciBasisManager{$Ti}, new_psi::Vector{Float64},
            sel_a::Vector{$Ti}, sel_b::Vector{$Ti}, sel_v::Vector{Float64}
        )
            @ccall LIB_SCI_SELECT.$(Symbol("remap_wavefunction_sci_bitstr$(sfx)_f64"))(
                old.ptr::Ptr{Cvoid}, old_psi::Ptr{Float64},
                new.ptr::Ptr{Cvoid}, new_psi::Ptr{Float64},
                sel_a::Ptr{$Ti}, sel_b::Ptr{$Ti}, sel_v::Ptr{Float64},
                length(sel_a)::Int64
            )::Cvoid
        end

        function destroy_sci_basis_manager_bitstr(sb::SciBasisManager{$Ti})
            if sb.ptr != C_NULL
                @ccall LIB_SCI_SELECT.$(Symbol("destroy_sci_basis_manager_bitstr$(sfx)_f64"))(sb.ptr::Ptr{Cvoid})::Cvoid
                sb.ptr = C_NULL
            end
        end

        function _sci_basis_dim(ptr::Ptr{Cvoid})::Int64
            @ccall LIB_SCI_SELECT.$(Symbol("sci_basis_dim_bitstr$(sfx)_f64"))(ptr::Ptr{Cvoid})::Int64
        end

        function _sci_basis_num_blocks(ptr::Ptr{Cvoid})::Int64
            @ccall LIB_SCI_SELECT.$(Symbol("sci_basis_num_blocks_bitstr$(sfx)_f64"))(ptr::Ptr{Cvoid})::Int64
        end
    end
end

function sort_by_sym(arr::Vector{Ti}, orbsym::Vector{Int64}, num_irreps::Int) where Ti
    by_sym = [Vector{Ti}() for _ in 1:num_irreps]
    for a in arr
        sym = get_symm(a, orbsym)
        if 0 <= sym < num_irreps
            push!(by_sym[sym+1], a)
        end
    end
    result = Ti[]
    for s in 1:num_irreps
        sort!(by_sym[s])
        append!(result, by_sym[s])
    end
    return result
end

function SciBasisManager(mole::Mole, astrs::Vector{Ti}, bstrs::Vector{Ti};
    sorted::Bool=false, num_irreps::Int=16, total_sym::Int=0
) where Ti
    na, nb = mole.nelec
    astrs  = Ti[a for a in astrs if count_ones(a) == na]
    bstrs  = Ti[b for b in bstrs if count_ones(b) == nb]

    @assert all(count_ones.(astrs) .== na)
    @assert all(count_ones.(bstrs) .== nb)

    if !sorted
        astrs = sort_by_sym(astrs, mole.orbsym, num_irreps)
        bstrs = sort_by_sym(bstrs, mole.orbsym, num_irreps)
    end

    ptr = _create_sci_basis_manager(astrs, bstrs, mole.norb, mole.orbsym, total_sym, num_irreps)

    ptr == C_NULL && error("Failed to create bitstr basis manager.")

    dim = _sci_basis_dim(ptr)
    nbk = _sci_basis_num_blocks(ptr)
    obj = SciBasisManager{Ti}(ptr, dim, mole.norb, nbk, na, nb, astrs, bstrs)

    finalizer(obj) do o
        if o.ptr != C_NULL
            destroy_sci_basis_manager_bitstr(o)
            o.ptr = C_NULL
        end
    end

    return obj
end

function expand_bitstrings_bitstr(
    src_astrs::Vector{Ti}, src_bstrs::Vector{Ti}, 
    axs::Vector{Ti}, bxs::Vector{Ti}, 
    nelec::Tuple{Int,Int}, orbsym::Vector{Int64}, num_irreps::Int=16
) where Ti
    na, nb = nelec
    a_map  = Dict{Ti,Int}(a => 0 for a in src_astrs)
    b_map  = Dict{Ti,Int}(b => 0 for b in src_bstrs)

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

function merge_bitstrings(
    src_astrs::Vector{Ti}, src_bstrs::Vector{Ti},
    sel_a::Vector{Ti}, sel_b::Vector{Ti},
    orbsym::Vector{Int64}, num_irreps::Int=16
) where Ti
    a_union = sort_by_sym(collect(union(Set(src_astrs), Set(sel_a))), orbsym, num_irreps)
    b_union = sort_by_sym(collect(union(Set(src_bstrs), Set(sel_b))), orbsym, num_irreps)

    return a_union, b_union
end

function run_sci_bitstr(mole::Mole;
    max_iter::Int=20, eps::Float64=1e-6, verbose::Bool=true)

    na, nb  = mole.nelec
    ham     = JW_hamiltonian(mole)
    basis   = SciBasisManager(mole, [UInt32(1 << na - 1)], [UInt32(1 << nb - 1)])
    otf     = OTF(ham, mole.orbsym)
    all_axs = sort!(unique(ham.axs))
    all_bxs = sort!(unique(ham.bxs))

    psi     = Float64[1.0]

    diags   = Float64[0.0]
    get_diags_bitstr!(basis, otf, diags)

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
        t2 = @elapsed sci_select_bitstr!(
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
        get_diags_bitstr!(basis, otf, diags)

        hvec = (v, Hv) -> hvec_svd!(basis, otf, v, Hv)

        t6 = @elapsed current_energy, psi = davidson(hvec, psi, diags, verbose=false)

        if verbose
            @printf("  E=%.14f  err=%.3e  t:exp=%.1fs sel=%.1fs mrg=%.1fs rmp=%.1fs dg=%.1fs\n\n",
                current_energy, abs(current_energy - mole.e_scale),
                t1, t2, t4, t5, t6)
        end
    end

    return basis, psi, diags
end
