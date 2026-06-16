include("../jl/binsim.jl")

function run_vqe2(
    basis::BasisManager, 
    ham::BinaryQubitAABB{Ti,Tv,K,V}, 
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}, 
    lv::T1, rv::T2,
    v0_idxs::Vector{Int}, v0_vals::Vector{Tv}, 
    e_scale::Float64;
    x0::Vector{Float64}=Float64[], 
    options::VQE_OPTIONS=VQE_OPTIONS(),
) where {Ti,Tv,K,V,T1<:AbstractArray{Tv},T2<:AbstractArray{Tv}}
    funcs = OTF_Functions(basis, ham, pool)
    idxs  = [i for i in eachindex(pool)]

    if !isempty(x0)
        @assert length(x0) == length(idxs)
    else
        x0 = zeros(Float64, length(pool))
    end

    energy = Ref(0.0)
    gnorm  = Ref(0.0)
    δ²H    = Ref(0.0)
    error  = Ref(0.0)

    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        fill!(lv, 0.0)
        lv[v0_idxs] .= v0_vals

        result = @timed energy_objective(funcs.hvec, funcs.expm, funcs.backgrad, idxs, x, lv, rv)
        energy[], grads, δ²H[] = result.value
        gnorm[] = norm(grads)
        error[] = abs(energy[] - e_scale)
        options.verbose > 1 && show_optimze(energy[], gnorm[], δ²H[], error[])
        options.verbose > 2 && show_time(result)

        return energy[], grads
    end

    println("Performing VQE optimization ... ")
    time_ops = @elapsed e_opt, x_opt = optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
    @printf("Converged in %.4f seconds with: f = %.14f  |g| = %.3e  δ²H = %.3e  err = %.3e\n",
            time_ops, energy[], gnorm[], δ²H[], error[])
    println("\n")

    fill!(lv, 0.0)
    lv[v0_idxs] .= v0_vals

    for i in eachindex(idxs)
        funcs.expm(idxs[i], x_opt[i], lv)
    end

    return e_opt, lv, x_opt
end

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    mole.orbsym = Int64.(mole.orbsym .% 10)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)
    orbs  = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool  = FEB(orbs)

    lvec = get_hf(basis, mole.nelec)
    rvec = zeros(eltype(lvec), length(lvec))

    hf_idxs = findall(x -> x != 0, lvec) 
    hf_vals = lvec[hf_idxs]

    run_vqe2(basis, ham, pool, lvec, rvec, hf_idxs, hf_vals, mole.e_scale,
        options = VQE_OPTIONS(
            ftol      = 1e-8, 
            gtol      = 1e-6, 
            maxiter   = 9999, 
            verbose   = 3, 
            save_path = joinpath(@__DIR__, "callback/vqe_uccsd_amplitudes_$(ARGS[1])_$(ARGS[2])_$(ARGS[3]).jld2")
        )
    )
end
