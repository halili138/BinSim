using Test

ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "1")
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")
include("../jl/sci_bitstr.jl")

@testset "SCI bitstring fixed-electron expansion" begin
    mole = Mole()
    mole.name = "n2"
    mole.ratio = 1.0
    mole.basis = "sto-3g"
    build(mole)
    mole.orbsym = Int64.(mole.orbsym .% 10)

    na, nb = mole.nelec
    fci_basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole; verbose=false)
    svd_groups = compress_by_svd(ham)
    all_axs, all_bxs = extract_ax_bx(svd_groups)

    hf_astr = UInt32((1 << na) - 1)
    hf_bstr = UInt32((1 << nb) - 1)
    dst_a, dst_b, _, _ = expand_bitstrings_bitstr(
        [hf_astr], [hf_bstr], all_axs, all_bxs, na, nb, mole.orbsym)
    expanded_basis = SciBasisManagerBitstr(dst_a, dst_b, mole.norb, 0, mole.orbsym, na, nb; sorted=true)

    @test all(count_ones(a) == na for a in dst_a)
    @test all(count_ones(b) == nb for b in dst_b)
    @test expanded_basis.dim <= fci_basis.dim

    sci_basis, _, _ = run_sci_bitstr(mole; max_iter=1, max_size=fci_basis.dim,
                                     eps=1e-6, davidson_tol=1e-4, verbose=false)
    @test sci_basis.dim <= fci_basis.dim

    destroy_sci_basis_manager_bitstr(expanded_basis)
    destroy_sci_basis_manager_bitstr(sci_basis)
end
