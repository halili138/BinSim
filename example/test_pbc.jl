include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    nkx = parse(Int, ARGS[1])

    pbc = Pbc()
    pbc.name = "1d-h"
    pbc.ratio = 1.0
    pbc.basis = "gth-szv"
    pbc.pseudo = "gth-pade"
    pbc.mesh = [nkx, 1, 1]
    pbc.scaled_center = [0, 0, 0]

    build(pbc)
    pbc.orbsym = ones(Int64, pbc.norb)

    basis = BasisManager(pbc.norb, pbc.nelec, pbc.orbsym)
    ham = JW_hamiltonian(pbc)
    ham = apply_constraint(ham, pbc.norb, pbc.nelec, (0.5, 0.5, 0.5))
    v0 = get_hf(basis, pbc.nelec, Tv=ComplexF64)

    pbc.e_scale, _ = run_fci(basis, ham, v0, net="otf")

    # v0 = get_hf(basis, pbc.nelec, Tv=ComplexF64)
    # orbs = Orbitals()
    # kernel(pbc, orbs, generalize=true)
    # pool = FEB(orbs, Tv=ComplexF64, complete=true)

    # run_vqe(basis, ham, pool, v0, pbc.e_scale, net="otf",
    #     options=VQE_OPTIONS(
    #         ftol=1e-10,
    #         gtol=1e-6,
    #         maxiter=100000,
    #         verbose=1),
    # )

    # run_adapt_vqe(basis, ham, pool, v0, pbc.e_scale, net="otf",
    #     adapt_options=ADAPT_OPTIONS(
    #         Gtol=1e-3,
    #         gtol=1e-4,
    #         htol=1e-3,
    #         Δtol=1e-8,
    #     ),
    #     vqe_options=VQE_OPTIONS(
    #         ftol=1e-10,
    #         gtol=1e-6,
    #         maxiter=1000,
    #         verbose=1,
    #     ),
    # )

end

