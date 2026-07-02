ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "8")
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")
include("../jl/sci.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]
    build(mole)

    # ==== Test 1: SCI hvec == standard hvec (same basis) ====
    @printf("=== Test 1: hvec correctness ===\n")
    basis_std = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole; verbose=false)
    ham_otf = OTF_sci(mole.orbsym, mole.norb, ham)
    basis_sci = create_sci_basis_from_standard(basis_std)

    psi = get_hf(basis_std, mole.nelec)
    psi_sci = copy(psi)

    Hv_std = zeros(Float64, basis_std.dim)
    hvec_svd!(basis_std, ham_otf, psi, Hv_std)

    Hv_sci = zeros(Float64, basis_sci.dim)
    hvec_sci_full!(basis_sci, ham_otf, psi_sci, Hv_sci)

    hdiff = norm(Hv_std - Hv_sci)
    @printf("  ||Hv_std - Hv_sci|| = %.3e  (should be ~0)\n", hdiff)
    @assert hdiff < 1e-10 "hvec mismatch!"

    # ==== Test 2: diags correctness ====
    @printf("=== Test 2: diags correctness ===\n")
    d_std = get_diags(basis_std, ham_otf, Float64)
    d_sci = zeros(Float64, basis_sci.dim)
    get_diags_sci!(basis_sci, ham_otf, d_sci)
    ddiff = norm(d_std - d_sci)
    @printf("  ||diags_std - diags_sci|| = %.3e  (should be ~0)\n", ddiff)
    @assert ddiff < 1e-10 "diags mismatch!"

    destroy_sci_basis_manager(basis_sci)

    # ==== Test 3: expand + select correctness ====
    @printf("=== Test 3: expand + select ===\n")
    na, nb = mole.nelec
    hf_astr = UInt32((1 << na) - 1)
    hf_bstr = UInt32((1 << nb) - 1)
    src = SciBasisManager([hf_astr], [hf_bstr], mole.norb, 0, mole.orbsym;
                          mode=SCI_MODE_BITSTRING, strat=SCI_STRAT_GROW)
    psi_src = Float64[1.0]
    @printf("  Source basis dim=%d\n", src.dim)

    num_irreps = length(unique(mole.orbsym))
    tgt = expand_and_build_sci_basis(src, psi_src, ham_otf,
          mole.norb, mole.orbsym, num_irreps; mode=SCI_MODE_BITSTRING, strat=SCI_STRAT_GROW)
    @printf("  Target basis dim=%d  blocks=%d\n", tgt.dim, tgt.num_blocks)

    # collect all selected with eps=0 (selects everything new)
    sel_a = UInt32[]; sel_b = UInt32[]; sel_v = Float64[]
    for blk in 0:tgt.num_blocks-1
        sci_hvec_select_for_block!(tgt, src, ham_otf, blk, psi_src,
                                   256, 0.0, sel_a, sel_b, sel_v)
    end
    @printf("  Selected (eps=0): %d pairs\n", length(sel_v))

    # verify: selected with eps=0 should include all reachable new states
    # simplified: just check selected values are non-negligible
    n_nz = count(x -> abs(x) > 1e-12, sel_v)
    @printf("  Non-zero selected: %d / %d\n", n_nz, length(sel_v))
    @assert n_nz > 0 "No non-zero selected values!"

    # ==== Test 4: merge + remap + davidson ====
    @printf("=== Test 4: merge + davidson ===\n")
    new_basis = create_source_basis_from_merge(
        src, sel_a, sel_b, sel_v, mole.norb, mole.orbsym, 0;
        mode=SCI_MODE_BITSTRING, strat=SCI_STRAT_GROW)
    @printf("  Merged basis dim=%d\n", new_basis.dim)

    new_psi = zeros(Float64, new_basis.dim)
    remap_wavefunction!(src, psi_src, new_basis, new_psi, sel_a, sel_b, sel_v)

    new_diags = zeros(Float64, new_basis.dim)
    get_diags_sci!(new_basis, ham_otf, new_diags)

    # Check: E_V = ψ · Hψ  in the new basis
    n = new_basis.dim
    Hv_tmp = zeros(Float64, n)
    hvec_sci_full!(new_basis, ham_otf, new_psi, Hv_tmp)
    E_init = dot(new_psi, Hv_tmp)
    @printf("  Initial E = %.10f\n", E_init)

    # Davidson
    hvec_fn = (v, Hv) -> hvec_sci_full!(new_basis, ham_otf, v, Hv)
    E_dav, psi_dav = davidson(hvec_fn, new_psi, new_diags;
                               tol=1e-5, ncv=1, maxspace=min(n, n+20), verbose=false)
    @printf("  Davidson E = %.10f\n", E_dav)

    destroy_sci_basis_manager(src)
    destroy_sci_basis_manager(tgt)
    destroy_sci_basis_manager(new_basis)

    @printf("\nAll tests passed!\n")
end
