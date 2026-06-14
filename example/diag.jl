include("../binsim.jl")

function run_fci2(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, v0::Vector{Tv}) where {Ti,Tv,K,V}
    funcs = OTF_Functions(basis, ham, BinaryQubitAABB{Ti,Tv,K,V}[], time_print=true)
    print("Generating Diag elements vector ... ")
    time_ops = @elapsed diags = get_diags(basis, funcs.ham, Tv)
    @printf("Done in %.4f seconds\n", time_ops)

    println("Solving FCI with davidson ... ")

    @time davidson2(funcs.hvec, v0, diags, tol=1e-5, ncv=1, maxspace=3)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name   = ARGS[1]
    mole.ratio  = parse(Float64, ARGS[2])
    mole.basis  = ARGS[3]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)
    
    run_fci2(basis, ham, get_hf(basis, mole.nelec))
end