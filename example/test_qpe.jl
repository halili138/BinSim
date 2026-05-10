include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    v0  = get_hf(basis, mole.nelec)
    mole.e_scale, _ = run_fci(basis, ham, v0, net="otf")

    ham = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, ComplexF64.(ham.cs))
    v0 = get_hf(basis, mole.nelec, Tv=ComplexF64)
    run_qpe(basis, ham, v0, dt=0.05, net="otf")
end
