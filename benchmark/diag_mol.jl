include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    @ccall LIB_AGG.init_likwid()::Cvoid

    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    basis      = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    psi_space  = basis.dim * 8 / (1 << 30)
    @printf("Num symmetry allowed elements: %d    %.4f GB\n\n", basis.dim, psi_space)

    ham   = JW_hamiltonian(mole)
    hf    = get_hf(basis, mole.nelec, mole.orbsym, Tv=ComplexF64)
    diags = get_diags(basis, ham)

    @time agg = AGG(basis, ham, mole.orbsym)
    
    println("Warming up JIT and Hardware...")
    dummy_src = rand(Float64, basis.dim)
    dummy_dst = zeros(Float64, basis.dim)
    hvec_direct_agg_benchmark!(basis, agg, dummy_src, dummy_dst, false) 
    println("Warm-up completed.")

    aop! = (src, dst) -> hvec_direct_agg_benchmark!(basis, agg, src, dst, true) 

    @time davidson(aop!, hf, diags)

    @ccall LIB_AGG.close_likwid()::Cvoid
end

