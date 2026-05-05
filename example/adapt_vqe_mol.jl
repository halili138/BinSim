include("../binsim.jl")


function run_adapt_vqe(mole::Mole; 
    amplitudes::Vector{Float64}=Float64[], selec_idxs::Vector{Int64}=Int64[],
    adapt_options::ADAPT_OPTIONS=ADAPT_OPTIONS(),
    vqe_options::VQE_OPTIONS=VQE_OPTIONS(ftol=1.0e-10, maxiter=1000, verbose=1),
)
    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    println("Num symmetry allowed elements: $(basis.dim)\n")

    ham = JW_hamiltonian(mole)
    ret = @timed ham_net = AGG(basis, ham)
    println("Successifully Generate Ham AGG in $(ret.time) seconds")
    print_info(ham_net)

    orbs = Orbitals(); kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)
    println("Operator pool size: $(length(pool))")
    ret = @timed pool_net = NET(basis, pool)
    println("Successifully Generate Pool NET in $(ret.time) seconds")

    v0 = get_hf(basis, mole.nelec)
    lv = zeros(Float64, basis.dim)
    rv = zeros(Float64, basis.dim)
    idxs = [i for i in eachindex(pool)]
    
    if !isempty(amplitudes) && !isempty(selec_idxs)
        @assert length(amplitudes) == length(selec_idxs)
    else
        amplitudes = Float64[]
        selec_idxs = Int64[]
    end

    _adapt_vqe(
        basis,
        ham_net, 
        pool_net,  
        idxs, 
        v0, 
        lv, 
        rv, 
        mole.e_scale, 
        amplitudes,
        selec_idxs,
        adapt_options,
        vqe_options,
    )
end


if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    # amplitudes, selec_idxs = load_idxs("save_temp.jld2")

    run_adapt_vqe(mole;
                #   amplitudes = amplitudes,  
                #   selec_idxs = selec_idxs,
                  adapt_options=ADAPT_OPTIONS(
                    Gtol=1e-3,
                    gtol=1e-4,
                    htol=1e-3,
                    Δtol=1e-8,
                    # save_path="save_temp.jld2",
                    ),
                  vqe_options=VQE_OPTIONS(
                    ftol=1e-8,
                    gtol=1e-6,
                    maxiter=1000,
                    verbose=1,
                    ),
                  )


    run_adapt_vqe(mole;
                #   amplitudes = amplitudes,  
                #   selec_idxs = selec_idxs,
                  adapt_options=ADAPT_OPTIONS(
                    Gtol=1e-3,
                    gtol=1e-4,
                    htol=1e-3,
                    Δtol=1e-8,
                    # save_path="save_temp.jld2",
                    ),
                  vqe_options=VQE_OPTIONS(
                    ftol=1e-8,
                    gtol=1e-6,
                    maxiter=1000,
                    verbose=1,
                    ),
                  )
end

