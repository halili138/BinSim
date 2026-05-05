include("../binsim.jl")


function run_adapt_vqe(pbc::Pbc; 
    amplitudes::Vector{Float64}=Float64[], selec_idxs::Vector{Int64}=Int64[],
    adapt_options::ADAPT_OPTIONS=ADAPT_OPTIONS(),
    vqe_options::VQE_OPTIONS=VQE_OPTIONS(ftol=1.0e-10, maxiter=1000, verbose=1),
)
    basis = BasisManager(pbc.norb, pbc.nelec, pbc.orbsym)
    println("Num symmetry allowed elements: $(basis.dim)\n")

    ham = JW_hamiltonian(pbc)
    ham = apply_constraint(ham, pbc.norb, pbc.nelec, (0.5, 0.5, 0.5))

    ret = @timed ham_net = AGG(basis, ham)
    println("Successifully Generate Ham AGG in $(ret.time) seconds")
    print_info(ham_net)

    orbs = Orbitals(); kernel(pbc, orbs, generalize=true)
    pool = FEB(orbs, Tv=ComplexF64, complete=true)
    println("Operator pool size: $(length(pool))")

    ret = @timed pool_net = NET(basis, pool)
    println("Successifully Generate Pool NET in $(ret.time) seconds")

    v0 = get_hf(basis, pbc.nelec, Tv=ComplexF64)
    lv = zeros(ComplexF64, basis.dim)
    rv = zeros(ComplexF64, basis.dim)
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
        pbc.e_scale, 
        amplitudes,
        selec_idxs,
        adapt_options,
        vqe_options,
    )
end


if abspath(PROGRAM_FILE) == @__FILE__
    pbc = Pbc()
    pbc.name   = "1d-h"
    pbc.ratio  = 1.0
    pbc.basis  = "gth-szv"
    pbc.pseudo = "gth-pade"
    pbc.mesh   = [4,1,1]
    pbc.scaled_center = [0,0,0]

    build(pbc)

    pbc.e_scale = -0.90257763392980
    pbc.orbsym  = ones(Int64, pbc.norb)
    
    # amplitudes, selec_idxs = load_idxs("save_temp.jld2")

    run_adapt_vqe(pbc;
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
                    ftol=1e-10,
                    gtol=1e-6,
                    maxiter=1000,
                    verbose=1,
                    ),
                  )
end

