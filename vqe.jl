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
    @time while !converged
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


function show_ssvqe_optimze(state_metrics)
    df_states   = [@sprintf("%d", m[1]) for m in state_metrics]
    df_energies = [@sprintf("%.14f", m[2]) for m in state_metrics]
    df_grads    = [@sprintf("%.3e", m[3]) for m in state_metrics]
    df_vars     = [@sprintf("%.3e", m[4]) for m in state_metrics]
    df_errs     = [@sprintf("%.3e", m[5]) for m in state_metrics]

    df_step = DataFrame(
        "State" => df_states,
        "f (Energy)" => df_energies,
        "|g|" => df_grads,
        "δ²H" => df_vars,
        "err" => df_errs
    )

    println("-----------------------------------------------------------")
    show(stdout, df_step, summary=false, eltypes=false, show_row_number=false)
    
    println("\n")
end


function _adapt_ssvqe(
    f_hvec::Function, 
    f_expm::Function, 
    f_grad::Function, 
    idxs::Vector{Int64},
    v0s::Vector{Vector{Tv}}, 
    weights::Vector{Float64}, 
    lv::Vector{Tv}, 
    rv::Vector{Tv},
    e_scales::Vector{Float64}, 
    amplitudes::Vector{Float64}, 
    selec_idxs::Vector{Int64}, 
    adapt_options::ADAPT_OPTIONS, 
    vqe_options::VQE_OPTIONS,
) where Tv

    K_states = length(v0s)
    @assert length(amplitudes) == length(selec_idxs)
    
    iter::Int = length(amplitudes)
    maxiter::Int = adapt_options.maxiter 
    Gtol::Float64 = adapt_options.Gtol
    gtol::Float64 = adapt_options.gtol
    htol::Float64 = adapt_options.htol
    Δtol::Float64 = adapt_options.Δtol

    G, gi_max, δ²H_max = 999.0, 999.0, 999.0
    L_hist::Array{Float64,1} = []
    
    zero_grads = Vector{Float64}(undef, length(idxs))
    converged::Bool = false
    target_L = sum(weights .* e_scales)

    @time while !converged
        iter += 1
        fill!(zero_grads, 0.0)
        δ²H_max = 0.0

        # =======================================================
        # 1. 计算算符池的加权零参数梯度
        # =======================================================
        for k in 1:K_states
            lv .= v0s[k]
            
            # 演化当前波函数
            for i in eachindex(amplitudes)
                f_expm(selec_idxs[i], amplitudes[i], lv)
            end
            
            # 计算 H|ψ_k>
            f_hvec(lv, rv)
            
            # 评估收敛情况的方差 (当前态)
            E_k = real(dot(lv, rv))
            δ²H_k = max(0.0, norm(rv)^2 - E_k^2)
            δ²H_max = max(δ²H_max, δ²H_k)

            # 累加池算符梯度
            for i in eachindex(idxs)
                g_k = real(f_grad(idxs[i], 0.0, lv, rv)) * 2
                zero_grads[i] += weights[k] * g_k
            end
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

        # =======================================================
        # 2. VQE 优化步骤 (加权代价函数)
        # =======================================================
        step_counter = Ref(0)
        obj_func = x -> begin
            total_L = 0.0
            total_grad = zeros(Float64, length(x))
            max_δ²H = 0.0
            state_metrics = []

            time_ops = @elapsed for k in 1:K_states
                lv .= v0s[k]
                e_k, g_k, δ²H_k = energy_objective(f_hvec, f_expm, f_grad, selec_idxs, x, lv, rv)                
                total_L += weights[k] * e_k
                total_grad .+= weights[k] .* g_k
                max_δ²H = max(max_δ²H, δ²H_k)

                norm_gk = norm(g_k)
                err_k = abs(e_k - e_scales[k])
                push!(state_metrics, (k, e_k, norm_gk, δ²H_k, err_k))
            end

            if vqe_options.verbose > 1
                step_counter[] += 1
                norm_g = norm(total_grad)
                target_L = sum(weights .* e_scales)
                error = total_L - target_L
                
                @printf(" SSVQE Eval %04d\n", step_counter[])
                @printf(" f: %.14f   |g|: %.3e   err: %.3e   time: %.3fs\n", 
                        total_L, norm_g, error, time_ops)
                show_ssvqe_optimze(state_metrics)
            end

            return total_L, total_grad
        end

        L_opt, amplitudes = optimze_fg!(
            amplitudes, obj_func, vqe_options.optimizer, vqe_options.options, vqe_options.verbose)

        if !isempty(adapt_options.save_path)
            jldopen(adapt_options.save_path, "w") do file
                file["amplitudes"] = amplitudes
                file["selec_idxs"] = selec_idxs
            end
        end

        push!(L_hist, L_opt)

        # 检查收敛条件
        cond1::Bool = iter > maxiter
        cond2::Bool = (G < Gtol && gi_max < gtol && δ²H_max < htol)
        cond3::Bool = false

        if length(L_hist) > 5
            ΔL_max = maximum(abs.(diff(L_hist[end-4:end])))
            if ΔL_max < Δtol
                @printf("  ΔL: %9.3e < %.1e, ADAPT loop finished!\n", ΔL_max, Δtol)
                cond3 = true
            end
        end

        converged = cond1 || cond2 || cond3

        if adapt_options.verbose > 0
            @printf("\nIteration: %d\n",                            iter)
            @printf("   Weighted L: %.14f\n",                       L_opt)
            @printf("  err (Loss): %9.3e\n",                        L_opt - target_L)
            @printf("  |G|: %9.3e    gmax: %9.3e     max_δ²H: %9.3e\n", G, gi_max, δ²H_max)
            println("============================================================================")
        end
    end
end

