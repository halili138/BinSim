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

function energy_objective(funcs, idxs::Vector{Int64}, x::Vector{Float64}, lv::T1, rv::T2) where {Tv,T1<:AbstractArray{Tv,1},T2<:AbstractArray{Tv,1}}
    nparas = length(x)

    for i in 1:nparas
        funcs.expm(idxs[i], x[i], lv)
    end

    funcs.hvec(lv, rv)

    lnorm = real(funcs.inner(lv, lv))
    rnorm = real(funcs.inner(rv, rv))
    energy = real(funcs.inner(lv, rv)) / lnorm
    δ²H = max(0.0, rnorm / lnorm - energy^2)
    grad = Vector{Float64}(undef, nparas)

    for i in nparas:-1:1
        grad[i] = real(funcs.backgrad(idxs[i], x[i], lv, rv)) * 2 / lnorm
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
    f_expm::Function,
    f_backgrad::Function,
    f_batchgrad::Function,
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
            f_expm(selec_idxs[i], amplitudes[i], lv)
        end
    end

    iter::Int  = length(amplitudes); maxiter::Int  = adapt_options.maxiter 
    G::Float64               = 999.; Gtol::Float64 = adapt_options.Gtol
    gi_max::Float64          = 999.; gtol::Float64 = adapt_options.gtol
    δ²H::Float64             = 999.; htol::Float64 = adapt_options.htol
    e_hist::Array{Float64,1} = [];   Δtol::Float64 = adapt_options.Δtol

    zero_grads = Vector{Float64}(undef, length(idxs))
    zero_amp = zeros(length(idxs))
    converged::Bool = false
    while !converged
        iter += 1
        
        f_hvec(lv, rv)
        
        f_batchgrad(lv, rv, zero_grads, zero_amp)
        @. zero_grads = real(zero_grads) * 2
        
        max_idx = sortperm(abs.(zero_grads), rev=true)[1]
        G       = norm(zero_grads)
        gi_max  = abs(zero_grads[max_idx])

        if length(selec_idxs) > 0 && max_idx == selec_idxs[end]
            println("Have selected same operator, ADAPT loop finished!")
            break
        end

        push!(amplitudes, 0.0)
        push!(selec_idxs, idxs[max_idx])

        e_l   = Ref(0.0)
        ng_l  = Ref(0.0)
        δ²H_l = Ref(0.0)
        err_l = Ref(0.0)

        obj_func = x -> begin
            lv .= v0
            result = @timed energy_objective(f_hvec, f_expm, f_backgrad, selec_idxs, x, lv, rv)
            e_l[], gradient, δ²H_l[] = result.value

            ng_l[]  = norm(gradient)
            err_l[] = abs(e_l[] - e_scale)
            vqe_options.verbose > 1 && show_optimze(e_l[], ng_l[], δ²H_l[], err_l[])
            vqe_options.verbose > 2 && show_time(result)

            return e_l[], gradient
        end

        vqe_options.verbose > 0 && println("Performing VQE optimization ... ")
        time_ops = @elapsed e_opt, amplitudes = optimze_fg!(
            amplitudes, obj_func, vqe_options.optimizer, vqe_options.options, vqe_options.verbose)

        if vqe_options.verbose == 1
            @printf("Converged in %.4f seconds with: f = %.14f  |g| = %.3e  δ²H = %.3e  err = %.3e\n", 
                    time_ops, e_l[], ng_l[], δ²H_l[], err_l[])
        elseif vqe_options.verbose >= 2
            @printf("Converged in %.4f seconds\n", time_ops)
        end

        if !isempty(adapt_options.save_path)
            jldopen(adapt_options.save_path, "w") do file
                file["amplitudes"] = amplitudes
                file["selec_idxs"] = selec_idxs
            end
        end

        push!(e_hist, e_opt)

        lv .= v0
        for i in eachindex(amplitudes)
            f_expm(selec_idxs[i], amplitudes[i], lv)
        end

        cond1::Bool = iter > maxiter
        cond2::Bool = (G < Gtol && gi_max < gtol && δ²H_l[] < htol)
        cond3::Bool = false

        if length(e_hist) > 5
            Δe_max = maximum(abs.(diff(e_hist[end-4:end])))
            if Δe_max < Δtol
                @printf("  \nΔE: %9.3e < %.1e, ADAPT loop finished!\n", Δe_max, Δtol)
                cond3 = true
            end
        end

        converged = cond1 || cond2 || cond3

        if adapt_options.verbose > 0
            @printf("\nIteration: %d\n",                            iter)
            @printf("   E0: %.14f\n",                               e_opt)
            @printf("  err: %9.3e\n",                               e_opt-e_scale)
            @printf("  |G|: %9.3e    gmax: %9.3e     δ²H: %9.3e\n", G, gi_max, δ²H_l[])
            println("============================================================================")
        end
    end

    return amplitudes, selec_idxs 
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
