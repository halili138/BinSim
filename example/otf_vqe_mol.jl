include("../binsim.jl")


function run_vqe(mole::Mole; 
    x0::Vector{Float64}=Float64[], options::VQE_OPTIONS=VQE_OPTIONS()
)
    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    psi_space = basis.dim * 8 / (1 << 30)
    @printf("Num symmetry allowed elements: %d    %.4f GB\n\n", basis.dim, psi_space)

    ham = JW_hamiltonian(mole)

    ret = @timed ham_otf = OTF(basis, ham)
    println("Successifully Generate OTF in $(ret.time) seconds\n")

    orbs = Orbitals(); kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)
    println("Operator pool size: $(length(pool))")
    ret = @timed pool_otf = OTF(basis, pool)
    println("Successifully Generate Pool NET in $(ret.time) seconds")
    
    v0 = get_hf(basis, mole.nelec)
    lv = zeros(Float64, basis.dim)
    rv = zeros(Float64, basis.dim)
    idxs = [i for i in eachindex(pool)]

    if !isempty(x0)
        @assert length(x0) == length(idxs)
    else
        x0 = zeros(Float64, length(pool))
    end
    
    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        lv .= v0
        result = @timed energy_objective(basis, ham_otf, pool_otf, idxs, x, lv, rv)
        energy, grad, δ²H = result.value
        norm_g  = norm(grad)
        error   = energy - mole.e_scale
        options.verbose > 0 && show_optimze(energy, norm_g, δ²H, error)
        options.verbose > 1 && show_time(result)

        return energy, grad
    end

    return @time optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
end


if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    # x0 = load_x("save_temp.jld2")

    run_vqe(mole,
            # x0=x0,
            options=VQE_OPTIONS(
                ftol=1e-8,
                gtol=1e-6,
                maxiter=1000,
                verbose=1,
                # save_path="save_temp.jld2",
            ),
        )
end

