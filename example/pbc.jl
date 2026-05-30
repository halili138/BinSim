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

    pbc.e_scale, v_fci = run_fci(basis, ham, get_hf(basis, pbc.nelec, Tv=ComplexF64))
    println(eltype(v_fci))

    orbs = Orbitals()
    kernel(pbc, orbs, generalize=false)
    pool = FEB(orbs, Tv=ComplexF64, complete=true)

    run_vqe(basis, ham, pool, get_hf(basis, pbc.nelec, Tv=ComplexF64), pbc.e_scale,
        options=VQE_OPTIONS(
            ftol=1e-10,
            gtol=1e-6,
            maxiter=100000,
            verbose=2),
    )
end

