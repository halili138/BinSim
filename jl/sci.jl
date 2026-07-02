const LIB_SCI = joinpath(libpath, "libsci_otf.so")

const SCI_MODE_BITSTRING = 0
const SCI_MODE_PAIR      = 1
const SCI_STRAT_GROW     = 0
const SCI_STRAT_RECOMPETE = 1

mutable struct SciBasisManager
    ptr::Ptr{Cvoid}
    dim::Int64
    norb::Int64
    num_blocks::Int64
end

function SciBasisManager()
    return SciBasisManager(C_NULL, 0, 0, 0)
end

function sci_basis_num_alpha_strings(basis::SciBasisManager)
    return @ccall LIB_SCI.sci_basis_num_alpha_strings(basis.ptr::Ptr{Cvoid})::Int64
end

function sci_basis_num_beta_strings(basis::SciBasisManager)
    return @ccall LIB_SCI.sci_basis_num_beta_strings(basis.ptr::Ptr{Cvoid})::Int64
end

function sci_basis_full_product_dim(basis::SciBasisManager)
    return sci_basis_num_alpha_strings(basis) * sci_basis_num_beta_strings(basis)
end

function SciBasisManager(
    astrs::Vector{UInt32}, bstrs::Vector{UInt32},
    norb::Int64, total_sym::Int64, orbsym::Vector{Int64};
    mode::Int=SCI_MODE_BITSTRING, strat::Int=SCI_STRAT_GROW)

    num_irreps = length(unique(orbsym))

    ptr = @ccall LIB_SCI.create_sci_basis_from_strings_f64(
        astrs::Ptr{UInt32}, length(astrs)::Int64,
        bstrs::Ptr{UInt32}, length(bstrs)::Int64,
        norb::Int64, orbsym::Ptr{Int64},
        total_sym::Int64, num_irreps::Int64,
        mode::Cint, strat::Cint
    )::Ptr{Cvoid}

    ptr == C_NULL && error("Failed to create SCI basis manager.")

    dim = @ccall LIB_SCI.sci_basis_dim(ptr::Ptr{Cvoid})::Int64
    nb  = @ccall LIB_SCI.sci_basis_num_blocks(ptr::Ptr{Cvoid})::Int64

    obj = SciBasisManager(ptr, dim, norb, nb)

    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_SCI.destroy_sci_basis_manager_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end

    return obj
end

function OTF_sci(orbsym::Vector{Int64}, norb::Int64, A::BinaryQubitAABB{Ti,Tv,K,V};
                 tol::Float64=1e-12) where {Ti,Tv,K,V}
    groups = compress_by_svd(A, tol)
    ngs = length(groups)

    axs    = Vector{Ti}(undef, ngs)
    bxs    = Vector{Ti}(undef, ngs)
    ranks  = Vector{Int64}(undef, ngs)
    num_as = Vector{Int64}(undef, ngs)
    num_bs = Vector{Int64}(undef, ngs)

    flat_azs = Ti[]
    flat_bzs = Ti[]
    flat_wa  = Tv[]
    flat_wb  = Tv[]

    for (g, group) in enumerate(groups)
        axs[g]    = group.ax
        bxs[g]    = group.bx
        ranks[g]  = group.rank
        num_as[g] = length(group.azs)
        num_bs[g] = length(group.bzs)

        append!(flat_azs, group.azs)
        append!(flat_bzs, group.bzs)
        append!(flat_wa,  vec(group.wa))
        append!(flat_wb,  vec(group.wb))
    end

    ptr = if Tv <: Complex
        @ccall LIB_SCI.build_network_otf_sci_c64(
            orbsym::Ptr{Int64}, norb::Int64, ngs::Int64,
            axs::Ptr{Ti}, bxs::Ptr{Ti},
            ranks::Ptr{Int64}, num_as::Ptr{Int64}, num_bs::Ptr{Int64},
            flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti},
            flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv}
        )::Ptr{Cvoid}
    else
        @ccall LIB_SCI.build_network_otf_sci_f64(
            orbsym::Ptr{Int64}, norb::Int64, ngs::Int64,
            axs::Ptr{Ti}, bxs::Ptr{Ti},
            ranks::Ptr{Int64}, num_as::Ptr{Int64}, num_bs::Ptr{Int64},
            flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti},
            flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv}
        )::Ptr{Cvoid}
    end

    ptr == C_NULL && error("Failed to build SCI OTF network.")

    obj = OTF(ptr, 0, ngs)

    if Tv <: Complex
        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_OTF.destroy_network_otf_c64(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end
    else
        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_OTF.destroy_network_otf_f64(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end
    end

    return obj
end

function expand_and_build_sci_basis(
    src::SciBasisManager, psi::Vector{Tv},
    otf::OTF, norb::Int64, orbsym::Vector{Int64},
    num_irreps::Int64;
    mode::Int=SCI_MODE_BITSTRING, strat::Int=SCI_STRAT_GROW) where Tv

    ptr = if Tv <: Complex
        @ccall LIB_SCI.expand_and_build_sci_basis_c64(
            src.ptr::Ptr{Cvoid}, psi::Ptr{Tv}, otf.ptr::Ptr{Cvoid},
            norb::Int64, orbsym::Ptr{Int64}, num_irreps::Int64,
            mode::Cint, strat::Cint
        )::Ptr{Cvoid}
    else
        @ccall LIB_SCI.expand_and_build_sci_basis_f64(
            src.ptr::Ptr{Cvoid}, psi::Ptr{Tv}, otf.ptr::Ptr{Cvoid},
            norb::Int64, orbsym::Ptr{Int64}, num_irreps::Int64,
            mode::Cint, strat::Cint
        )::Ptr{Cvoid}
    end

    ptr == C_NULL && error("Failed to expand and build SCI basis.")

    dim = @ccall LIB_SCI.sci_basis_dim(ptr::Ptr{Cvoid})::Int64
    nb  = @ccall LIB_SCI.sci_basis_num_blocks(ptr::Ptr{Cvoid})::Int64

    obj = SciBasisManager(ptr, dim, norb, nb)

    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_SCI.destroy_sci_basis_manager_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end

    return obj
end

function sci_hvec_select_for_block!(
    tgt::SciBasisManager, src::SciBasisManager, otf::OTF,
    blk::Integer, psi::Vector{Tv}, chunk_size::Int, eps::Float64,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Tv}) where Tv

    max_per_block = tgt.dim
    buf_a = Vector{UInt32}(undef, max_per_block)
    buf_b = Vector{UInt32}(undef, max_per_block)
    buf_v = Vector{Tv}(undef, max_per_block)
    max_abs_ref = Ref{Cdouble}(0.0)
    count_gt_eps_ref = Ref{Int64}(0)
    count_gt_1e12_ref = Ref{Int64}(0)

    n = if Tv <: Complex
        @ccall LIB_SCI.sci_hvec_select_for_block_c64(
            tgt.ptr::Ptr{Cvoid}, src.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
            blk::Int64, psi::Ptr{Tv}, chunk_size::Cint, eps::Cdouble,
            buf_a::Ptr{UInt32}, buf_b::Ptr{UInt32}, buf_v::Ptr{Tv},
            max_per_block::Int64,
            max_abs_ref::Ref{Cdouble}, count_gt_eps_ref::Ref{Int64},
            count_gt_1e12_ref::Ref{Int64}
        )::Int64
    else
        @ccall LIB_SCI.sci_hvec_select_for_block_f64(
            tgt.ptr::Ptr{Cvoid}, src.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
            blk::Int64, psi::Ptr{Tv}, chunk_size::Cint, eps::Cdouble,
            buf_a::Ptr{UInt32}, buf_b::Ptr{UInt32}, buf_v::Ptr{Tv},
            max_per_block::Int64,
            max_abs_ref::Ref{Cdouble}, count_gt_eps_ref::Ref{Int64},
            count_gt_1e12_ref::Ref{Int64}
        )::Int64
    end

    append!(sel_a, view(buf_a, 1:n))
    append!(sel_b, view(buf_b, 1:n))
    append!(sel_v, view(buf_v, 1:n))
    return n, max_abs_ref[], count_gt_eps_ref[], count_gt_1e12_ref[]
end

function create_source_basis_from_merge(
    src::SciBasisManager,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Tv},
    norb::Int64, orbsym::Vector{Int64}, total_sym::Int64;
    mode::Int=SCI_MODE_BITSTRING, strat::Int=SCI_STRAT_GROW) where Tv

    num_irreps = length(unique(orbsym))

    ptr = if Tv <: Complex
        @ccall LIB_SCI.create_source_basis_from_merge_c64(
            src.ptr::Ptr{Cvoid},
            sel_a::Ptr{UInt32}, sel_b::Ptr{UInt32}, sel_v::Ptr{Tv},
            length(sel_a)::Int64,
            norb::Int64, orbsym::Ptr{Int64}, total_sym::Int64,
            num_irreps::Int64, mode::Cint, strat::Cint
        )::Ptr{Cvoid}
    else
        @ccall LIB_SCI.create_source_basis_from_merge_f64(
            src.ptr::Ptr{Cvoid},
            sel_a::Ptr{UInt32}, sel_b::Ptr{UInt32}, sel_v::Ptr{Tv},
            length(sel_a)::Int64,
            norb::Int64, orbsym::Ptr{Int64}, total_sym::Int64,
            num_irreps::Int64, mode::Cint, strat::Cint
        )::Ptr{Cvoid}
    end

    ptr == C_NULL && error("Failed to create merged SCI basis.")

    dim = @ccall LIB_SCI.sci_basis_dim(ptr::Ptr{Cvoid})::Int64
    nb  = @ccall LIB_SCI.sci_basis_num_blocks(ptr::Ptr{Cvoid})::Int64

    obj = SciBasisManager(ptr, dim, norb, nb)

    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_SCI.destroy_sci_basis_manager_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end

    return obj
end

function remap_wavefunction!(
    old::SciBasisManager, old_psi::Vector{Tv},
    new::SciBasisManager, new_psi::Vector{Tv},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Tv}) where Tv

    if Tv <: Complex
        @ccall LIB_SCI.remap_wavefunction_sci_c64(
            old.ptr::Ptr{Cvoid}, old_psi::Ptr{Tv},
            new.ptr::Ptr{Cvoid}, new_psi::Ptr{Tv},
            sel_a::Ptr{UInt32}, sel_b::Ptr{UInt32}, sel_v::Ptr{Tv},
            length(sel_a)::Int64,
        )::Cvoid
    else
        @ccall LIB_SCI.remap_wavefunction_sci_f64(
            old.ptr::Ptr{Cvoid}, old_psi::Ptr{Tv},
            new.ptr::Ptr{Cvoid}, new_psi::Ptr{Tv},
            sel_a::Ptr{UInt32}, sel_b::Ptr{UInt32}, sel_v::Ptr{Tv},
            length(sel_a)::Int64,
        )::Cvoid
    end
end

function get_diags_sci!(basis::SciBasisManager, otf::OTF, diags::Vector{Tv}) where Tv
    if Tv <: Complex
        @ccall LIB_SCI.get_diags_elements_sci_c64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Tv}
        )::Cvoid
    else
        @ccall LIB_SCI.get_diags_elements_sci_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Tv}
        )::Cvoid
    end
end

function hvec_sci_full!(basis::SciBasisManager, otf::OTF,
                         src::Vector{Tv}, dst::Vector{Tv}) where Tv
    if Tv <: Complex
        @ccall LIB_SCI.hvec_sci_full_c64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
            src::Ptr{Tv}, dst::Ptr{Tv}
        )::Cvoid
    else
        @ccall LIB_SCI.hvec_sci_full_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
            src::Ptr{Tv}, dst::Ptr{Tv}
        )::Cvoid
    end
end

function destroy_sci_basis_manager(sb::SciBasisManager)
    if sb.ptr != C_NULL
        @ccall LIB_SCI.destroy_sci_basis_manager_f64(sb.ptr::Ptr{Cvoid})::Cvoid
        sb.ptr = C_NULL
    end
end

function create_sci_basis_from_standard(basis::BasisManager; mode=SCI_MODE_BITSTRING, strat=SCI_STRAT_GROW)
    ptr = @ccall LIB_SCI.create_sci_basis_from_standard_f64(
        basis.ptr::Ptr{Cvoid}, mode::Cint, strat::Cint
    )::Ptr{Cvoid}
    ptr == C_NULL && error("Failed to create SCI basis from standard.")
    dim = @ccall LIB_SCI.sci_basis_dim(ptr::Ptr{Cvoid})::Int64
    nb  = @ccall LIB_SCI.sci_basis_num_blocks(ptr::Ptr{Cvoid})::Int64
    obj = SciBasisManager(ptr, dim, basis.norb, nb)
    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_SCI.destroy_sci_basis_manager_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end
    return obj
end

function run_sci(mole::Mole;
    max_iter::Int=20,
    max_size::Int=10000,
    eps::Float64=1e-6,
    chunk_size::Int=256,
    mode::Int=SCI_MODE_BITSTRING,
    strat::Int=SCI_STRAT_GROW,
    davidson_tol::Float64=1e-5,
    verbose::Bool=true)

    na, nb = mole.nelec
    num_irreps = length(unique(mole.orbsym))

    verbose && print("Building OTF ... ")
    t0 = @elapsed begin
        ham = JW_hamiltonian(mole; verbose=false)
        ham_otf = OTF_sci(mole.orbsym, mole.norb, ham)
    end
    verbose && @printf("Done in %.4f s\n", t0)

    hf_astr = UInt32((1 << na) - 1)
    hf_bstr = UInt32((1 << nb) - 1)
    basis = SciBasisManager([hf_astr], [hf_bstr], mole.norb, 0, mole.orbsym;
                            mode=mode, strat=strat)
    psi = Float64[1.0]
    diags = zeros(Float64, 1)
    get_diags_sci!(basis, ham_otf, diags)

    verbose && @printf("Initial basis: dim=%d  E0=%.10f\n", basis.dim, diags[1])

    for iter in 1:max_iter
        t_iter = @elapsed begin
            tgt = expand_and_build_sci_basis(basis, psi, ham_otf,
                  mole.norb, mole.orbsym, num_irreps; mode=mode, strat=strat)
            verbose && @printf("  [%d] expand %d→%d  ", iter, basis.dim, tgt.dim)

            sel_a = UInt32[];  sel_b = UInt32[];  sel_v = Float64[]
            external_max_abs = 0.0
            external_count_gt_eps = 0
            external_count_gt_1e12 = 0
            for blk in 0:tgt.num_blocks-1
                blk_nsel, blk_max_abs, blk_count_gt_eps, blk_count_gt_1e12 =
                    sci_hvec_select_for_block!(tgt, basis, ham_otf, blk, psi,
                                               chunk_size, eps, sel_a, sel_b, sel_v)
                external_max_abs = max(external_max_abs, blk_max_abs)
                external_count_gt_eps += blk_count_gt_eps
                external_count_gt_1e12 += blk_count_gt_1e12
                verbose && @printf("block=%d max|Hψ|=%.3e count>|eps|=%d count>|1e-12|=%d sel=%d  ",
                                   blk, blk_max_abs, blk_count_gt_eps,
                                   blk_count_gt_1e12, blk_nsel)
            end
            nsel = length(sel_v)
            verbose && @printf("sel=%d external_max|Hψ|=%.3e external_count>|eps|=%d external_count>|1e-12|=%d  ",
                               nsel, external_max_abs, external_count_gt_eps, external_count_gt_1e12)
            if verbose && iter == 4
                @printf("plateau_probe_iter4_external_max|Hψ|=%.6e candidates>|1e-12|=%d  ",
                        external_max_abs, external_count_gt_1e12)
            end

            if nsel == 0
                verbose && @printf("\nNo new states selected; external max residual=%.6e candidate determinants=%d (>|1e-12|), count>|eps|=%d.\n",
                                   external_max_abs, external_count_gt_1e12, external_count_gt_eps)
                break
            end

            new_basis = create_source_basis_from_merge(
                basis, sel_a, sel_b, sel_v, mole.norb, mole.orbsym, 0;
                mode=mode, strat=strat)
            verbose && @printf("merged=%d unique_alpha=%d unique_beta=%d active_det=%d full_product_det=%d  ",
                               new_basis.dim, sci_basis_num_alpha_strings(new_basis),
                               sci_basis_num_beta_strings(new_basis), new_basis.dim,
                               sci_basis_full_product_dim(new_basis))

            new_psi = zeros(Float64, new_basis.dim)
            remap_wavefunction!(basis, psi, new_basis, new_psi,
                                sel_a, sel_b, sel_v)

            new_diags = zeros(Float64, new_basis.dim)
            get_diags_sci!(new_basis, ham_otf, new_diags)

            n = new_basis.dim
            hvec_fn = (v, Hv) -> hvec_sci_full!(new_basis, ham_otf, v, Hv)
            E, psi_new = davidson(hvec_fn, new_psi, new_diags;
                                  tol=davidson_tol, ncv=1, maxspace=min(max_size, n+20),
                                  verbose=false)

            verbose && @printf("E=%.10f  err=%.3e\n", E, abs(E - mole.e_scale))

            destroy_sci_basis_manager(tgt)
            destroy_sci_basis_manager(basis)
            basis, psi, diags = new_basis, psi_new, new_diags
        end
        verbose && @printf("  iter %d time=%.2f s\n", iter, t_iter)
    end

    return basis, psi, diags
end
