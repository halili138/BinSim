include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    v0 = get_hf(basis, mole.nelec)

    mole.e_scale, _ = run_fci(basis, ham, v0, net="otf")

    @time run_rk4(basis, ham, v0, mole.e_scale, 
    dτ=0.5, max_step=10000, tol=1e-10, net="agg")
end
