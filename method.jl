function run_fci(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    v0::Vector{Tv};
    net::String="agg"
) where {Ti,Tv,K,V}
    psi_space = basis.dim * 8 / (1 << 30)
    @printf("Num symmetry allowed elements: %d    %.4f GB\n\n", basis.dim, psi_space)
    diags = get_diags(basis, ham)

    if net == "agg"
        ret = @timed agg = AGG(basis, ham)
        println("Successifully Generate Ham AGG in $(ret.time) seconds\n")
        print_info(agg)
        aop! = (src, dst) -> @time hvec_direct_agg!(basis, agg, src, dst)
    elseif net == "otf"
        ret = @timed otf = OTF(basis, ham)
        println("Successifully Generate Ham OTF in $(ret.time) seconds\n")
        aop! = (src, dst) -> @time hvec_otf!(basis, otf, src, dst)
    else
        error("Undefined NET name $(net)")
    end

    # @time ham_sp = to_sparse_matrix(ham, pbc.norb, pbc.nelec)
    # @time λ, ϕ = eigs(ham_sp, nev=1, which=:SR)
    # println(λ)
    
    return @time davidson(aop!, v0, diags, tol=1e-5)
end


function run_vqe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0::Vector{Tv},
    e_scale::Float64;
    net::String="agg",
    x0::Vector{Float64}=Float64[],
    options::VQE_OPTIONS=VQE_OPTIONS()
) where {Ti,Tv,K,V}
    println("Num symmetry allowed elements: $(basis.dim)\n")
    println("Operator pool size: $(length(pool))\n")

    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]

    if net == "agg"
        ret = @timed ham_agg = AGG(basis, ham)
        println("Successifully Generate Ham AGG in $(ret.time) seconds")
        print_info(ham_agg)
        ret = @timed pool_net = NET(basis, pool)
        println("Successifully Generate Pool AGG in $(ret.time) seconds")
        f_hvec = (lvec, rvec) -> hvec_direct_agg!(basis, ham_agg, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_net, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_net, idx, x, lvec, rvec)
    elseif net == "otf"
        ret = @timed ham_otf  = OTF(basis, ham)
        println("Successifully Generate Ham OTF in $(ret.time) seconds\n")
        ret = @timed pool_otf = OTF(basis, pool)
        println("Successifully Generate Pool OTF in $(ret.time) seconds\n")
        f_hvec = (lvec, rvec) -> hvec_otf!(basis, ham_otf, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_otf, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_otf, idx, x, lvec, rvec)
    else
        error("Undefined NET name $(net)")
    end

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
        result = @timed energy_objective(f_hvec, f_tvec, f_grad, idxs, x, lv, rv)
        energy, grad, δ²H = result.value
        norm_g = norm(grad)
        error = energy - e_scale
        options.verbose > 0 && show_optimze(energy, norm_g, δ²H, error)
        options.verbose > 1 && show_time(result)

        return energy, grad
    end

    return @time optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
end


function run_adapt_vqe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0::Vector{Tv},
    e_scale::Float64;
    net::String="agg",
    amplitudes::Vector{Float64}=Float64[], 
    selec_idxs::Vector{Int64}=Int64[],
    adapt_options::ADAPT_OPTIONS=ADAPT_OPTIONS(),
    vqe_options::VQE_OPTIONS=VQE_OPTIONS(ftol=1.0e-10, maxiter=1000, verbose=1),
) where {Ti,Tv,K,V}

    println("Num symmetry allowed elements: $(basis.dim)\n")
    println("Operator pool size: $(length(pool))\n")

    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]

    if net == "agg"
        ret = @timed ham_agg = AGG(basis, ham)
        println("Successifully Generate Ham AGG in $(ret.time) seconds")
        print_info(ham_agg)
        ret = @timed pool_net = NET(basis, pool)
        println("Successifully Generate Pool AGG in $(ret.time) seconds")
        f_hvec = (lvec, rvec) -> hvec_direct_agg!(basis, ham_agg, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_net, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_net, idx, x, lvec, rvec)
    elseif net == "otf"
        ret = @timed ham_otf  = OTF(basis, ham)
        println("Successifully Generate Ham OTF in $(ret.time) seconds\n")
        ret = @timed pool_otf = OTF(basis, pool)
        println("Successifully Generate Pool OTF in $(ret.time) seconds\n")
        f_hvec = (lvec, rvec) -> hvec_otf!(basis, ham_otf, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_otf, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_otf, idx, x, lvec, rvec)
    else
        error("Undefined NET name $(net)")
    end

    if !isempty(amplitudes) && !isempty(selec_idxs)
        @assert length(amplitudes) == length(selec_idxs)
    else
        amplitudes = Float64[]
        selec_idxs = Int64[]
    end

    _adapt_vqe(
        f_hvec,
        f_tvec, 
        f_grad,  
        idxs, 
        v0, 
        lv, 
        rv, 
        e_scale, 
        amplitudes,
        selec_idxs,
        adapt_options,
        vqe_options,
    )
end


function run_rk4(
    basis::BasisManager, 
    ham::BinaryQubitAABB{Ti,Tv,K,V}, 
    v0::Vector{Tv}, 
    e_scale::Float64;
    dτ::Float64=0.02, 
    max_step::Int64=1000, 
    tol::Float64=1e-8,
    net::String="agg",
) where {Ti,Tv,K,V}
    """
    四阶 Runge-Kutta 虚时演化 (4 次 hvec/步):
    dpsi/dtau = -H|psi>
    k1 = -H * psi
    k2 = -H * (psi + dtau/2 * k1)
    k3 = -H * (psi + dtau/2 * k2)
    k4 = -H * (psi + dtau * k3)
    psi_new = psi + dtau/6 * (k1 + 2*k2 + 2*k3 + k4)
    """

    if net == "agg"
        ret = @timed agg = AGG(basis, ham)
        println("Successifully Generate Ham AGG in $(ret.time) seconds\n")
        print_info(agg)
        hvec! = (src, dst) -> hvec_direct_agg!(basis, agg, src, dst)
    elseif net == "otf"
        ret = @timed otf = OTF(basis, ham)
        println("Successifully Generate Ham OTF in $(ret.time) seconds\n")
        hvec! = (src, dst) -> hvec_otf!(basis, otf, src, dst)
    else
        error("Undefined NET name $(net)")
    end

    v = v0
    ws::Vector{Vector{Tv}} = [zeros(Tv, basis.dim) for _ in 1:5]

    E_hist   = Float64[]
    dH_hist  = Float64[]

    step = 0
    while step <= max_step
        step += 1
        
        hvec!(v, ws[1])

        ln = norm(v) ^ 2
        rn = norm(ws[1]) ^ 2
        E  = real(dot(v, ws[1])) / ln
        dH = max(0.0, rn / ln - E ^ 2)
        push!(E_hist,  E)
        push!(dH_hist, dH)
        
        dE = step > 1 ? E_hist[end] - E_hist[end-1] : E_hist[end]

        @printf("  Step %03d     E %.14f    Err %.3e    dE %.3e    δ²H %.3e   τ %.2f\n",
                step, E, abs(E-e_scale), dE, dH, step * dτ)

        abs(dE) < tol && break

        vt = ws[5]

        ws[1] .*= -1.0

        @. vt = v + dτ / 2 * ws[1]
        hvec!(vt, ws[2])
        ws[2] .*= -1.0

        @. vt = v + dτ / 2 * ws[2]
        hvec!(vt, ws[3])
        ws[3] .*= -1.0

        @. vt = v + dτ * ws[3]
        hvec!(vt, ws[4])
        ws[4] .*= -1.0

        @. v += dτ / 6 * (ws[1] + 2 * ws[2] + 2 * ws[3] + ws[4])

        normalize!(v)
    end 
    
    println("  Converged at step $step\n")

    return E_hist[end]
end
