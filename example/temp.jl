function run_vqe(pbc::Pbc; x0::Vector{Float64}=Float64[], options::VQE_OPTIONS=VQE_OPTIONS())
    basis = BasisManager(pbc.norb, pbc.nelec, pbc.orbsym)
    println("Num symmetry allowed elements: $(basis.dim)\n")

    ham = JW_hamiltonian(pbc)
    ret = @timed ham_net = AGG(basis, ham, pbc.orbsym)
    println("Successifully Generate Ham AGG in $(ret.time) seconds")
    print_info(ham_net)

    orbs = Orbitals(); kernel(pbc, orbs, generalize=false)
    pool = FEB(orbs)
    println("Operator pool size: $(length(pool))")
    ret = @timed pool_net = NET(basis, pool, pbc.orbsym)
    println("Successifully Generate Pool NET in $(ret.time) seconds")
    
    v0 = get_hf(basis, pbc.nelec, pbc.orbsym)
    lv = zeros(ComplexF64, basis.dim)
    rv = zeros(ComplexF64, basis.dim)
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
        result = @timed energy_objective(basis, ham_net, pool_net, idxs, x, lv, rv)
        energy, grad, δ²H = result.value
        norm_g  = norm(grad)
        error   = energy - pbc.e_scale
        options.verbose > 0 && show_optimze(energy, norm_g, δ²H, error)
        options.verbose > 1 && show_time(result)

        return energy, grad
    end

    return @time optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
end