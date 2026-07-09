function cuda_sci_hvec_select_external_bitstr!(
    tgt::SciBasisManagerBitstr,
    src_dev::Ptr{Cvoid}, net_dev::Ptr{Cvoid},
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool},
    blk::Int, psi::Vector{Float64}, candidate_diags::Vector{Float64},
    variational_energy::Float64, chunk_size::Int, eps::Float64,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    max_per_block = tgt.dim
    buf_a = Vector{UInt32}(undef, max_per_block)
    buf_b = Vector{UInt32}(undef, max_per_block)
    buf_v = Vector{Float64}(undef, max_per_block)

    n = @ccall LIB_CUDA_SCI.cuda_sci_select_external_block_f64(
        tgt.ptr::Ptr{Cvoid}, src_dev::Ptr{Cvoid}, net_dev::Ptr{Cvoid},
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

function cuda_select_external_block_bitstr!(
    tgt::SciBasisManagerBitstr,
    src_dev::Ptr{Cvoid}, net_dev::Ptr{Cvoid},
    psi::Vector{Float64}, candidate_diags::Vector{Float64},
    variational_energy::Float64, chunk_size::Int, eps::Float64,
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    for blk in 0:tgt.num_blocks-1
        cuda_sci_hvec_select_external_bitstr!(
            tgt, src_dev, net_dev, is_new_a, is_new_b,
            blk, psi, candidate_diags, variational_energy,
            chunk_size, eps, sel_a, sel_b, sel_v)
    end
end

function cuda_hvec_sci_full!(
    basis_dev::Ptr{Cvoid}, net_dev::Ptr{Cvoid}, basis_dim::Int64,
    src::Vector{Float64}, dst::Vector{Float64})
    @ccall LIB_CUDA_SCI.cuda_hvec_sci_full_f64(
        basis_dev::Ptr{Cvoid}, net_dev::Ptr{Cvoid}, basis_dim::Int64,
        src::Ptr{Float64}, dst::Ptr{Float64})::Cvoid
end

function run_cuda_sci(mole::Mole;
    max_iter::Int=20, max_size::Int=5000, eps::Float64=1e-6,
    chunk_size::Int=256, davidson_tol::Float64=1e-5, verbose::Bool=true, total_sym::Int64=0,
    diag_mode::Symbol=:cpu)

    na, nb = mole.nelec
    num_irreps = SCI_BITSTR_NUM_IRREPS

    ham = JW_hamiltonian(mole)
    svd_groups = compress_by_svd(ham)
    all_axs, all_bxs = extract_ax_bx(svd_groups)
    ham_otf = OTF_bitstr(mole.orbsym, mole.norb, ham)

    # === Persistent: upload network once ===
    net_dev = @ccall LIB_CUOTF.build_networkdev_f64(ham_otf.ptr::Ptr{Cvoid})::Ptr{Cvoid}
    src_dev = C_NULL

    hf_astr = UInt32((1 << na) - 1)
    hf_bstr = UInt32((1 << nb) - 1)
    basis = SciBasisManagerBitstr([hf_astr], [hf_bstr], mole.norb, total_sym, mole.orbsym, na, nb; num_irreps=num_irreps)
    psi = Float64[1.0]
    diags = zeros(Float64, 1)
    get_diags_bitstr!(basis, ham_otf, diags)
    current_energy = diags[1]
    verbose && @printf("Initial basis: dim=%d  E0=%.10f\n\n", basis.dim, current_energy)

    for iter in 1:max_iter
        t1 = @elapsed dst_a, dst_b, is_new_a, is_new_b = expand_bitstrings_bitstr(
            basis.astrs, basis.bstrs, all_axs, all_bxs, na, nb, mole.orbsym, num_irreps)
        tgt = SciBasisManagerBitstr(dst_a, dst_b, mole.norb, total_sym, mole.orbsym, na, nb; sorted=true, num_irreps=num_irreps)
        tgt_diags = zeros(Float64, tgt.dim)
        get_diags_bitstr!(tgt, ham_otf, tgt_diags)

        # === Persistent: upload source basis once per iteration ===
        if src_dev != C_NULL
            @ccall LIB_CUDA_SCI.destroy_sci_src_f64(src_dev::Ptr{Cvoid})::Cvoid
        end
        src_dev = @ccall LIB_CUDA_SCI.upload_sci_src_f64(basis.ptr::Ptr{Cvoid})::Ptr{Cvoid}

        sel_a = UInt32[]
        sel_b = UInt32[]
        sel_v = Float64[]

        t2 = @elapsed cuda_select_external_block_bitstr!(
            tgt, src_dev, net_dev, psi, tgt_diags,
            current_energy, chunk_size, eps, is_new_a, is_new_b,
            sel_a, sel_b, sel_v)

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

        t7 = @elapsed begin
            if diag_mode == :gpu
                basis_dev = @ccall LIB_CUDA_SCI.build_basisdev_from_sci_f64(new_basis.ptr::Ptr{Cvoid})::Ptr{Cvoid}
                E, psi_new = davidson(
                    (v, Hv) -> cuda_hvec_sci_full!(basis_dev, net_dev, new_basis.dim, v, Hv),
                    new_psi,
                    new_diags;
                    tol=davidson_tol,
                    ncv=1,
                    maxspace=min(max_size, new_basis.dim + 20),
                    verbose=false)
                @ccall LIB_CUOTF.destroy_basisdev_f64(basis_dev::Ptr{Cvoid})::Cvoid
            else
                E, psi_new = davidson(
                    (v, Hv) -> hvec_sci_full_bitstr!(new_basis, ham_otf, v, Hv),
                    new_psi,
                    new_diags;
                    tol=davidson_tol,
                    ncv=1,
                    maxspace=min(max_size, new_basis.dim + 20),
                    verbose=false)
            end
        end

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

    if src_dev != C_NULL
        @ccall LIB_CUDA_SCI.destroy_sci_src_f64(src_dev::Ptr{Cvoid})::Cvoid
    end
    @ccall LIB_CUOTF.destroy_networkdev_f64(net_dev::Ptr{Cvoid})::Cvoid

    return basis, psi, diags
end
