function show_optimze(
    energy::Float64, 
    norm_g::Float64, 
    δ²H::Float64, 
    error::Float64,
)
    @printf("  f: % 15.10f    |g|: %9.3e    δ²H: %9.3e    err: %9.3e\n",
               energy,        norm_g,       δ²H,          error)
end


function show_time(ret)
    Base.time_print(stdout, 
                    ret.time*1e9, 
                    ret.gcstats.allocd, 
                    ret.gcstats.total_time, 
                    Base.gc_alloc_count(ret.gcstats), 
                    ret.lock_conflicts, 
                    ret.compile_time*1e9, 
                    ret.recompile_time*1e9, 
                    true)
    println("")
end


function optimze_fg!(
    x0::Array{Float64,1}, 
    obj_func::Function, 
    optimizer::Optim.AbstractOptimizer, 
    optim_options::Optim.Options,
    verbose::Int64,
)
    verbose > 0 && println("Classical optimizing...")

    function fg!(F, G, x)
        f, g = obj_func(x)
        !isnothing(G) && (G .= g)
        !isnothing(F) && return f
        nothing
    end

    result = optimize(
        only_fg!(fg!),
        x0,
        optimizer,
        optim_options,
    )

    if verbose > 0
        @printf("  fcalls: %d\n", Optim.f_calls(result))
        @printf("  gcalls: %d\n", Optim.g_calls(result))
    end

    return Optim.minimum(result), Optim.minimizer(result)
end

function optimze_f!(
    x0::Array{Float64,1}, 
    obj_func::Function, 
    optimizer::Optim.AbstractOptimizer, 
    optim_options::Optim.Options,
    verbose::Int64,
)
    verbose > 0 && println("Classical optimizing...")

    result = optimize(
        obj_func,
        x0,
        optimizer,
        optim_options,
    )

    return Optim.minimum(result), Optim.minimizer(result)
end


struct VQE_OPTIONS
    optimizer::Optim.AbstractOptimizer
    options::Optim.Options
    verbose::Int
    save_path::String
end


function VQE_OPTIONS(;
    optimizer::String="lbfgs",
    ftol::Float64=1e-8,
    gtol::Float64=1e-6,
    maxiter::Int =100000,
    verbose::Int = 2,
    save_path::String="",
)
    options::Optim.Options = Optim.Options(
        f_abstol   = ftol,
        g_abstol   = gtol,
        iterations = maxiter,
    )

    if optimizer == "bfgs"
        return VQE_OPTIONS(
            BFGS(linesearch=LineSearches.MoreThuente()), options, verbose, save_path) 
    elseif optimizer == "lbfgs"
        return VQE_OPTIONS(
            LBFGS(linesearch=LineSearches.MoreThuente(), m=10), options, verbose, save_path)
    else
        error("Undefined optimizer name: $(optimizer)")
    end
end


function load_x(read_path::String)
    @assert isfile(read_path)
    jldopen(read_path, "r") do file
        x::Array{Float64,1} = file["x"]
        println("Successfully load precalculated amplitudes from $(read_path)\n")
        return x
    end
end


struct ADAPT_OPTIONS
    maxiter::Int
    Gtol::Float64
    gtol::Float64
    htol::Float64
    Δtol::Float64
    verbose::Int
    save_path::String
end


function ADAPT_OPTIONS(;
    maxiter::Int  = 100000,
    Gtol::Float64 = 1e-3,
    gtol::Float64 = 1e-4,
    htol::Float64 = 1e-3,
    Δtol::Float64 = 1e-8,
    verbose::Int  = 3,
    save_path::String = "",
)
    ADAPT_OPTIONS(maxiter, Gtol, gtol, htol, Δtol, verbose, save_path)
end


function _adapt_vqe(
    f_hvec::Function,
    f_tvec::Function,
    f_grad::Function,
    idxs::Vector{Int64},
    v0::Vector{Tv},
    lv::Vector{Tv},
    rv::Vector{Tv},
    e_scale::Float64,
    amplitudes::Vector{Float64}, 
    selec_idxs::Vector{Int64}, 
    adapt_options::ADAPT_OPTIONS,
    vqe_options::VQE_OPTIONS,
) where Tv
    @assert length(amplitudes) == length(selec_idxs)
    
    if !isempty(amplitudes)
        lv .= v0
        for i in eachindex(amplitudes)
            f_tvec(selec_idxs[i], amplitudes[i], lv)
        end
    end

    iter::Int  = length(amplitudes); maxiter::Int  = adapt_options.maxiter 
    G::Float64               = 999.; Gtol::Float64 = adapt_options.Gtol
    gi_max::Float64          = 999.; gtol::Float64 = adapt_options.gtol
    δ²H::Float64             = 999.; htol::Float64 = adapt_options.htol
    e_hist::Array{Float64,1} = [];   Δtol::Float64 = adapt_options.Δtol

    zero_grads = Vector{Float64}(undef, length(idxs))
    converged::Bool = false
    @time while !converged
        iter += 1
        
        f_hvec(lv, rv)
        
        for i in eachindex(idxs)
            zero_grads[i] = real(f_grad(idxs[i], 0.0, lv, rv)) * 2
        end
        
        max_idx = sortperm(abs.(zero_grads), rev=true)[1]
        G       = norm(zero_grads)
        gi_max  = abs(zero_grads[max_idx])

        if length(selec_idxs) > 0 && max_idx == selec_idxs[end]
            println("Have selected same operator, ADAPT loop finished!")
            break
        end

        push!(amplitudes, 0.0)
        push!(selec_idxs, idxs[max_idx])

        obj_func = x -> begin
            lv .= v0
            result = @timed energy_objective(f_hvec, f_tvec, f_grad, selec_idxs, x, lv, rv)
            energy, gradient, δ²H = result.value
            vqe_options.verbose > 1 && show_optimze(energy, norm(gradient), δ²H, energy-e_scale)
            vqe_options.verbose > 2 && show_time(result)

            return energy, gradient
        end

        e_opt, amplitudes = optimze_fg!(
            amplitudes, obj_func, vqe_options.optimizer, vqe_options.options, vqe_options.verbose)

        if !isempty(adapt_options.save_path)
            jldopen(adapt_options.save_path, "w") do file
                file["amplitudes"] = amplitudes
                file["selec_idxs"] = selec_idxs
            end
        end

        push!(e_hist, e_opt)

        lv .= v0
        for i in eachindex(amplitudes)
            f_tvec(selec_idxs[i], amplitudes[i], lv)
        end

        cond1::Bool = iter > maxiter
        cond2::Bool = (G < Gtol && gi_max < gtol && δ²H < htol)
        cond3::Bool = false

        if length(e_hist) > 5
            Δe_max = maximum(abs.(diff(e_hist[end-4:end])))
            if Δe_max < Δtol
                @printf("  ΔE: %9.3e < %.1e, ADAPT loop finished!\n", Δe_max, Δtol)
                cond3 = true
            end
        end

        converged = cond1 || cond2 || cond3

        if adapt_options.verbose > 0
            @printf("\nIteration: %d\n",                            iter)
            @printf("   E0: %.14f\n",                               e_opt)
            @printf("  err: %9.3e\n",                               e_opt-e_scale)
            @printf("  |G|: %9.3e    gmax: %9.3e     δ²H: %9.3e\n", G, gi_max, δ²H)
            println("============================================================================")
        end
    end
end


function load_idxs(read_path::String)
    @assert isfile(read_path)
    jldopen(read_path, "r") do file
        amplitudes::Array{Float64,1} = file["amplitudes"]
        selec_idxs::Array{Int64,1} = file["selec_idxs"]
        println("Successfully load precalculated amplitudes and selec_idxs from $(read_path)\n")
        return amplitudes, selec_idxs
    end
end  


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

