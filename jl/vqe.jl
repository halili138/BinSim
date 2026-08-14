function energy_objective(f_hvec::Function, f_expm::Function, f_backgrad::Function, idxs::Vector{Int64}, x::Vector{Float64}, lv::T1, rv::T2) where {Tv,T1<:AbstractArray{Tv,1},T2<:AbstractArray{Tv,1}}
    nparas = length(x)

    for i in 1:nparas
        f_expm(idxs[i], x[i], lv)
    end

    f_hvec(lv, rv)

    lnorm = norm(lv) ^ 2
    rnorm = norm(rv) ^ 2
    energy = real(dot(lv, rv)) / lnorm
    δ²H = max(0.0, rnorm / lnorm - energy^2)
    grad = Vector{Float64}(undef, nparas)

    for i in nparas:-1:1
        grad[i] = real(f_backgrad(idxs[i], x[i], lv, rv)) * 2 / lnorm
    end

    return energy, grad, δ²H
end

function show_optimze(energy::Float64, norm_g::Float64, δ²H::Float64, error::Float64)
    @printf("  f: % 15.10f    |g|: %9.3e    δ²H: %9.3e    err: %9.3e\n", energy, norm_g, δ²H, error)
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
    # verbose > 0 && println("Classical optimizing...")

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
        println("Successfully load x from $(read_path)\n")
        return x
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

function run_vqe(funcs, lv, rv, v0_idxs, v0_vals, e_scale::Float64, x0::Vector{Float64}, idxs::Vector{Int64}, options::VQE_OPTIONS)
    energy  = Ref(0.0)
    gnorm   = Ref(0.0)
    δ²H     = Ref(0.0)
    error   = Ref(0.0)

    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        fill!(lv, 0.0)
        lv[v0_idxs] .= v0_vals

        result = @timed energy_objective(
            funcs.hvec, 
            funcs.expm, 
            funcs.backgrad, 
            idxs, x, lv, rv
            )
            
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

    for (i, t) in zip(idxs, x_opt)
        funcs.expm(i, t, lv)
    end

    return e_opt, lv, x_opt
end

function run_adapt_vqe(funcs, lv, rv, v0_idxs, v0_vals, e_scale::Float64, nparams::Int64, amplitudes::Vector{Float64}, selec_idxs::Vector{Int64}, adapt_options::ADAPT_OPTIONS, vqe_options::VQE_OPTIONS)
    println("Performing ADAPT-VQE ... ")
    
    if !isempty(amplitudes)
        fill!(lv, 0.0)
        lv[v0_idxs] .= v0_vals
        for (i, t) in zip(selec_idxs, amplitudes)
            funcs.expm(i, t, lv)
        end
    end

    zero_grads  = similar(lv, nparams)
    _zero_grads = zeros(Float64, nparams)
    zero_amp    = similar(lv, Float64, nparams)
    zero_amp   .= 0.0
    iter        = length(amplitudes)
    gnorm       = 999.
    gmax        = 999.
    e_hist      = Float64[]  
    converged   = false

    @time while !converged
        iter += 1
        
        funcs.hvec(lv, rv)
        funcs.batchgrad(lv, rv, zero_grads, zero_amp)
        copyto!(_zero_grads, zero_grads)
        @. _zero_grads = real(_zero_grads) * 2

        permarray = sortperm(abs.(_zero_grads), rev=true)
        gnorm     = norm(_zero_grads)
        max_idx   = 0

        if length(selec_idxs) == 0
            max_idx = permarray[1]
        else
            for idx in permarray
                if idx != selec_idxs[end]
                    max_idx = idx
                    break
                end
            end
        end

        if max_idx == 0
            println("Cannot add new operator, ADAPT loop finished!")
            break
        end

        gmax = abs(_zero_grads[max_idx])

        push!(amplitudes, 0.0)
        push!(selec_idxs, max_idx)

        cur_energy   = Ref(0.0)
        cur_gnorm    = Ref(0.0)
        cur_variance = Ref(0.0)
        cur_error    = Ref(0.0)

        obj_func = x -> begin
            fill!(lv, 0.0)
            lv[v0_idxs] .= v0_vals

            result = @timed energy_objective(funcs.hvec, funcs.expm, funcs.backgrad, selec_idxs, x, lv, rv)

            cur_energy[], gradient, cur_variance[] = result.value
            cur_gnorm[] = norm(gradient)
            cur_error[] = abs(cur_energy[] - e_scale)

            vqe_options.verbose > 1 && show_optimze(cur_energy[], cur_gnorm[], cur_variance[], cur_error[])
            vqe_options.verbose > 2 && show_time(result)

            return cur_energy[], gradient
        end

        vqe_options.verbose >= 1 && println("Performing VQE optimization ... ")

        time_ops = @elapsed e_opt, amplitudes = optimze_fg!(amplitudes, obj_func, vqe_options.optimizer, vqe_options.options, vqe_options.verbose)

        vqe_options.verbose == 1 && @printf("Converged in %.4f seconds with: f = %.14f  |g| = %.3e  δ²H = %.3e  err = %.3e\n", 
                                             time_ops, cur_energy[], cur_gnorm[], cur_variance[], cur_error[])
        vqe_options.verbose >= 2 && @printf("Converged in %.4f seconds\n", time_ops)

        if !isempty(adapt_options.save_path)
            jldopen(adapt_options.save_path, "w") do file
                file["amplitudes"] = amplitudes
                file["selec_idxs"] = selec_idxs
            end
        end

        push!(e_hist, e_opt)

        fill!(lv, 0.0)
        lv[v0_idxs] .= v0_vals

        for (i, t) in zip(selec_idxs, amplitudes)
            funcs.expm(i, t, lv)
        end

        cond1::Bool = iter > adapt_options.maxiter
        cond2::Bool = (gnorm < adapt_options.Gtol && gmax < adapt_options.gtol && cur_variance[] < adapt_options.htol)
        cond3::Bool = false

        if length(e_hist) > 5
            delta_e = maximum(abs.(diff(e_hist[end-4:end])))
            cond3 = delta_e < adapt_options.Δtol
            cond3 && @printf("  \nΔE: %9.3e < %.1e, ADAPT loop finished!\n", delta_e, adapt_options.Δtol)
        end

        converged = cond1 || cond2 || cond3

        if adapt_options.verbose > 0
            @printf("\nIteration: %d\n",                            iter)
            @printf("   E0: %.14f\n",                               e_opt)
            @printf("  err: %9.3e\n",                               e_opt - e_scale)
            @printf("  |G|: %9.3e    gmax: %9.3e     δ²H: %9.3e\n", gnorm, gmax, cur_variance[])
            println("============================================================================")
        end
    end

    return amplitudes, selec_idxs
end
