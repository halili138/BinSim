include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    # basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    # ham = JW_hamiltonian(mole)

    # run_fci(basis, ham, get_hf(basis, mole.nelec))
    println("Single elect int terms: ", length(findall(x->abs(x)>1e-12, mole.one_body_mo)))
    println("Double elect int terms: ", length(findall(x->abs(x)>1e-12, mole.two_body_mo)))
end
