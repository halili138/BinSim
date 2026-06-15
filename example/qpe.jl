include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))
    ham = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, ComplexF64.(ham.cs))
    run_qpe(basis, ham, get_hf(basis, mole.nelec, Tv=ComplexF64), dt=0.05)
end
