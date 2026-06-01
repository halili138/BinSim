function rk4_step!(
    f_hvec::Function, v::T, vt::T, ws::Vector{T},
    shift::Number, dt::Float64,
) where {Tv,T<:AbstractArray{Tv,1}}
    # k1 = -i * H * v_exact
    f_hvec(v, ws[1])
    ws[1] .*= shift

    # k2 = -i * H * (v + dt/2 * k1)
    @. vt = v + ws[1] * dt / 2
    f_hvec(vt, ws[2])
    ws[2] .*= shift

    # k3 = -i * H * (v + dt/2 * k2)
    @. vt = v + ws[2] * dt / 2
    f_hvec(vt, ws[3])
    ws[3] .*= shift

    # k4 = -i * H * (v + dt * k3)
    @. vt = v + ws[3] * dt
    f_hvec(vt, ws[4])
    ws[4] .*= shift

    # ψ(t + dt) = ψ(t) + dt/6 * (k1 + 2k2 + 2k3 + k4)
    @. v += (ws[1] + 2 * ws[2] + 2 * ws[3] + ws[4]) * dt / 6

    normalize!(v)
end


function run_vqite_tfim_adjoint(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}},
    v0::Vector{Tv},
    e_scale::Float64;
    dt::Float64=0.01,
    max_step::Int=500,
    epsilon::Float64=1e-4,
    per_print::Int=10,
) where {Ti,Tv,TK,TV}
    println("============================================================================")
    println("--- VQITE with Strict Trajectory Tracking (RK4 Engine + adjoint method) ---")

    f_hvec = get_hvec(basis, ham, is_time=false)
    print("Pre-compiling Pool OTF ... ")
    time_ops = @elapsed pool_otf = OTF(basis, pool)
    @printf("Done in %.4f seconds\n", time_ops)

    f_expm = (idx, θ, vec) -> expm_svd!(basis, pool_otf, idx, θ, vec)
    f_backtran = (idx, θ, lvec, rvec, bvec) -> return backtran_svd!(basis, pool_otf, idx, θ, lvec, rvec, bvec)

    N = length(pool)
    x = zeros(Float64, N)
    x_hist = [copy(x)]
    e_hist = Float64[]
    fid_hist = Float64[]
    cond_hist = Float64[]

    M = zeros(Float64, N, N)
    V = zeros(Float64, N)

    ws = [zeros(Float64, basis.dim) for _ in 1:5]
    v_exact = copy(v0)
    v = ws[1]
    Hv = ws[2]
    τv = ws[3]
    bv = ws[4]
    vt = ws[5]

    @printf("  Step          Energy         Error      Trj Fid      cond(M)      |θ_dot|     Time\n")

    time_ops = @elapsed for step in 1:max_step
        rk4_step!(f_hvec, v_exact, vt, ws, -1.0, dt)

        v .= v0
        for k in 1:N
            f_expm(k, x[k], v)
        end

        vnorm = norm(v)^2

        f_hvec(v, Hv)
        e_curr = dot(v, Hv) / vnorm
        push!(e_hist, e_curr)
        fid = abs2(dot(v, v_exact)) / vnorm
        push!(fid_hist, fid)

        @. Hv = Hv - e_curr * v
        for j in N:-1:1
            f_backtran(j, -x[j], v, Hv, bv)
            M[j, j] = real(dot(bv, bv))
            V[j] = real(dot(bv, Hv))

            vt .= v
            for i in (j-1):-1:1
                f_backtran(i, -x[i], vt, bv, τv)
                val = real(dot(τv, bv))
                M[i, j] = val
                M[j, i] = val
            end
        end

        cond_M = cond(M)
        push!(cond_hist, cond_M)

        x_dot = (M + epsilon * I) \ V
        @. x -= dt * x_dot
        push!(x_hist, copy(x))

        if step % per_print == 0 || step == 1
            @printf("  %04d    % 15.10f    %.3e    %.6f    %9.3e    %.3e    %.4g\n",
                step, e_curr, abs(e_curr - e_scale), fid, cond_M, norm(x_dot), step * dt)
        end
    end

    @printf("\nFast VQTE completed in %.4f seconds.\n", time_ops)
    println("============================================================================\n")

    return (x=x_hist, e=e_hist, fid=fid_hist, cond=cond_hist)
end


function run_vqite_tfim_forward(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}},
    v0::Vector{Float64},
    e_scale::Float64;
    dt::Float64=0.01,
    max_step::Int=500,
    epsilon::Float64=1e-4,
    per_print::Int=10,
) where {Ti,Tv,TK,TV}
    println("============================================================================")
    println("--- VQITE with Strict Trajectory Tracking (RK4 Engine + forward method) ---")

    f_hvec = get_hvec(basis, ham, is_time=false)
    f_expm, f_tvec, _, _, _, _ = get_tvec(basis, pool, expm=true, tvec=true)

    N = length(pool)
    x = zeros(Float64, N)

    x_hist = [copy(x)]
    e_hist = Float64[]
    fid_hist = Float64[]
    cond_hist = Float64[]

    vs = [zeros(Float64, basis.dim) for _ in 1:N]
    dvs = [zeros(Float64, basis.dim) for _ in 1:N]
    ws = [zeros(Float64, basis.dim) for _ in 1:5]

    v_exact = copy(v0)
    v = ws[1]
    Hv = ws[2]
    vt = ws[5]

    @printf("  Step          Energy         Error      Trj Fid      cond(M)      |θ_dot|     Time\n")

    time_ops = @elapsed for step in 1:max_step
        rk4_step!(f_hvec, v_exact, vt, ws, -1.0, dt)

        v .= v0
        for k in 1:N
            vs[k] .= v
            f_expm(k, x[k], v)
        end
        vnorm = norm(v)^2

        f_hvec(v, Hv)
        e_curr = dot(v, Hv) / vnorm
        push!(e_hist, e_curr)

        fid = abs2(dot(v, v_exact)) / vnorm
        push!(fid_hist, fid)

        @. Hv = Hv - e_curr * v
        for k in 1:N
            f_tvec(k, vs[k], dvs[k])
            for j in k:N
                f_expm(j, x[j], dvs[k])
            end
        end

        D = reduce(hcat, dvs)
        V = zeros(Float64, N)
        M = zeros(Float64, N, N)

        V .= D' * Hv
        M .= D' * D

        cond_M = cond(M)
        push!(cond_hist, cond_M)

        x_dot = (M + epsilon * I) \ V
        @. x -= dt * x_dot
        push!(x_hist, copy(x))

        if step % per_print == 0 || step == 1
            @printf("  %04d    % 15.10f    %.3e    %.6f    %9.3e    %.3e    %.4g\n",
                step, e_curr, abs(e_curr - e_scale), fid, cond_M, norm(x_dot), step * dt)
        end
    end

    @printf("\nFast VQTE completed in %.4f seconds.\n", time_ops)
    println("============================================================================\n")

    return (x=x_hist, e=e_hist, fid=fid_hist, cond=cond_hist)
end


function run_vqrte_tfim_adjoint(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}},
    v0::Vector{Tv},
    e_scale::Float64;
    dt::Float64=0.01,
    max_step::Int=500,
    sv_tol::Float64=1e-4,
    per_print::Int=10,
) where {Ti,Tv,TK,TV}
    @assert Tv == ComplexF64 "VQRTE requires Tv=ComplexF64 for Hamiltonian, Pool, and initial state!"

    println("============================================================================")
    println("--- VQRTE with Strict Trajectory Tracking (RK4 Engine + adjoint method) ---")

    f_hvec = get_hvec(basis, ham, is_time=false)

    print("Pre-compiling Pool OTF ... ")
    time_ops = @elapsed pool_otf = OTF(basis, pool)
    @printf("Done in %.4f seconds\n", time_ops)

    f_expm = (idx, θ, vec) -> expm_svd!(basis, pool_otf, idx, θ, vec)
    f_backtran = (idx, θ, lvec, rvec, bvec) -> return backtran_svd!(basis, pool_otf, idx, θ, lvec, rvec, bvec)

    N = length(pool)
    e_hist = Float64[]
    fid_hist = Float64[]
    cond_hist = Float64[]

    M = zeros(Float64, N, N)
    V = zeros(Float64, N)

    x = zeros(Float64, N)
    x_hist = [copy(x)]
    xs = [zeros(Float64, N) for _ in 1:5]
    xt = xs[5]

    ws = [zeros(ComplexF64, basis.dim) for _ in 1:5]
    v_exact = copy(v0)
    v = ws[1]
    Hv = ws[2]
    τv = ws[3]
    bv = ws[4]
    vt = ws[5]

    function compute_xdot_and_energy!(x_in, dx_out)
        v .= v0
        for k in 1:N
            f_expm(k, x_in[k], v)
        end

        f_hvec(v, Hv)
        E_val = real(dot(v, Hv)) / norm(v)^2

        @. Hv = -im * (Hv - E_val * v)
        for j in N:-1:1
            f_backtran(j, -x_in[j], v, Hv, bv)
            M[j, j] = real(dot(bv, bv))
            V[j] = real(dot(bv, Hv))

            vt .= v
            for i in (j-1):-1:1
                f_backtran(i, -x_in[i], vt, bv, τv)
                val = real(dot(τv, bv))
                M[i, j] = val
                M[j, i] = val
            end
        end

        if dx_out === xs[1]
            push!(cond_hist, cond(M))
        end

        F = svd(M)
        inv_S = [s > sv_tol ? 1.0 / s : 0.0 for s in F.S]
        M_pinv = F.V * Diagonal(inv_S) * F.U'

        dx_out .= M_pinv * V

        return E_val
    end

    @printf("  Step          Energy         Error      Trj Fid      cond(M)      |θ_dot|      Time\n")

    time_ops = @elapsed for step in 1:max_step
        e_curr = compute_xdot_and_energy!(x, xs[1])
        push!(e_hist, e_curr)

        @. xt = x + 0.5 * dt * xs[1]
        compute_xdot_and_energy!(xt, xs[2])

        @. xt = x + 0.5 * dt * xs[2]
        compute_xdot_and_energy!(xt, xs[3])

        @. xt = x + dt * xs[3]
        compute_xdot_and_energy!(xt, xs[4])

        @. x += dt / 6 * (xs[1] + 2 * xs[2] + 2 * xs[3] + xs[4])
        push!(x_hist, copy(x))

        rk4_step!(f_hvec, v_exact, vt, ws, -im, dt)

        v .= v0
        for k in 1:N
            f_expm(k, x[k], v)
        end

        fid = abs2(dot(v, v_exact)) / norm(v)^2
        push!(fid_hist, fid)

        if step % per_print == 0 || step == 1
            cond_M = isempty(cond_hist) ? NaN : cond_hist[end]
            @printf("  %04d    % 15.10f    %.3e    %.6f    %.3e    %.3e    %.4g\n",
                step, e_curr, abs(e_curr - e_scale), fid, cond_M, norm(xs[1]), step * dt)
        end
    end

    @printf("\nNative Complex VQRTE (RK4) completed in %.4f seconds.\n", time_ops)
    println("============================================================================\n")

    return (x=x_hist, e=e_hist, fid=fid_hist, cond=cond_hist)
end


function run_vqrte_tfim_forward(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}},
    v0::Vector{Tv},
    e_scale::Float64;
    dt::Float64=0.01,
    max_step::Int=500,
    sv_tol::Float64=1e-4,
    per_print::Int=10,
) where {Ti,Tv,TK,TV}
    @assert Tv == ComplexF64 "VQRTE requires Tv=ComplexF64 for Hamiltonian, Pool, and initial state!"

    println("============================================================================")
    println("--- VQRTE with Strict Trajectory Tracking (RK4 Engine + forward method) ---")

    f_hvec = get_hvec(basis, ham, is_time=false)
    f_expm, f_tvec, f_grad, f_backgrad, f_batchgrad, f_tran = get_tvec(basis, pool, expm=true, tvec=true)

    N = length(pool)

    e_hist = Float64[]
    fid_hist = Float64[]
    cond_hist = Float64[]

    x = zeros(Float64, N)
    x_hist = [copy(x)]
    xs = [zeros(Float64, N) for _ in 1:5]
    xt = xs[5]

    vs = [zeros(ComplexF64, basis.dim) for _ in 1:N]
    dvs = [zeros(ComplexF64, basis.dim) for _ in 1:N]
    ws = [zeros(ComplexF64, basis.dim) for _ in 1:5]

    v_exact = copy(v0)
    v = ws[1]
    Hv = ws[2]
    vt = ws[5]


    function compute_xdot_and_energy!(x_in, dx_out)
        v .= v0
        for k in 1:N
            vs[k] .= v
            f_expm(k, x_in[k], v)
        end

        f_hvec(v, Hv)
        E_val = real(dot(v, Hv)) / norm(v)^2

        for k in 1:N
            f_tvec(k, vs[k], dvs[k])
            for j in k:N
                f_expm(j, x_in[j], dvs[k])
            end
        end

        @. Hv = -im * (Hv - E_val * v)

        D = reduce(hcat, dvs)
        V = real.(D' * Hv)
        M = real.(D' * D)

        if dx_out === xs[1]
            push!(cond_hist, cond(M))
        end

        F = svd(M)
        inv_S = [s > sv_tol ? 1.0 / s : 0.0 for s in F.S]
        M_pinv = F.V * Diagonal(inv_S) * F.U'

        dx_out .= M_pinv * V

        return E_val
    end

    @printf("  Step          Energy         Error      Trj Fid      cond(M)      |θ_dot|      Time\n")

    time_ops = @elapsed for step in 1:max_step
        e_curr = compute_xdot_and_energy!(x, xs[1])
        push!(e_hist, e_curr)

        @. xt = x + 0.5 * dt * xs[1]
        compute_xdot_and_energy!(xt, xs[2])

        @. xt = x + 0.5 * dt * xs[2]
        compute_xdot_and_energy!(xt, xs[3])

        @. xt = x + dt * xs[3]
        compute_xdot_and_energy!(xt, xs[4])

        @. x += dt / 6 * (xs[1] + 2 * xs[2] + 2 * xs[3] + xs[4])
        push!(x_hist, copy(x))

        rk4_step!(f_hvec, v_exact, vt, ws, -im, dt)

        v .= v0
        for k in 1:N
            f_expm(k, x[k], v)
        end

        fid = abs2(dot(v, v_exact)) / norm(v)^2
        push!(fid_hist, fid)

        if step % per_print == 0 || step == 1
            cond_M = isempty(cond_hist) ? NaN : cond_hist[end]
            @printf("  %04d    % 15.10f    %.3e    %.6f    %.3e    %.3e    %.4g\n",
                step, e_curr, abs(e_curr - e_scale), fid, cond_M, norm(xs[1]), step * dt)
        end
    end

    @printf("\nNative Complex VQRTE (RK4) completed in %.4f seconds.\n", time_ops)
    println("============================================================================\n")

    return (x=x_hist, e=e_hist, fid=fid_hist, cond=cond_hist)
end


function _run_vqite_tfim_adjoint(
    f_hvec::Function, f_expm::Function, f_backtran::Function,
    ws::Vector{Vector{Tv}}, v0::Vector{Tv}, ve::Vector{Tv}, 
    v::Vector{Tv}, Hv::Vector{Tv}, τv::Vector{Tv}, bv::Vector{Tv}, vt::Vector{Tv},
    active_idxs::Vector{Int64}, x::Vector{Float64}, e_scale::Float64, 
    dt::Float64=0.01, max_step::Int=500, xtol::Float64=1e-6, epsilon::Float64=1e-4, per_print::Int=10, verbose::Int=1,
) where {Tv}
    N   = length(active_idxs)
    M   = zeros(Float64, N, N)
    V   = zeros(Float64, N)
    ve .= v0

    fid = 0.0

    verbose > 0 && @printf("  Step          Energy         Error      Trj Fid      cond(M)      |θ_dot|     Time\n")

    time_ops = @elapsed for step in 1:max_step
        rk4_step!(f_hvec, ve, vt, ws, -1.0, dt)

        v .= v0
        for k in 1:N
            f_expm(k, x[k], v)
        end

        vnorm = norm(v) ^ 2

        f_hvec(v, Hv)
        e_curr = dot(v, Hv) / vnorm
        fid = abs2(dot(v, ve)) / vnorm

        @. Hv = Hv - e_curr * v
        for j in N:-1:1
            f_backtran(j, -x[j], v, Hv, bv)
            M[j, j] = real(dot(bv, bv))
            V[j] = real(dot(bv, Hv))

            vt .= v
            for i in (j-1):-1:1
                f_backtran(i, -x[i], vt, bv, τv)
                val = real(dot(τv, bv))
                M[i, j] = val
                M[j, i] = val
            end
        end

        x_dot = (M + epsilon * I) \ V
        @. x -= dt * x_dot
        
        θ_dot = norm(x_dot)
        if verbose > 0 && (step % per_print == 0 || step == 1)
            @printf("  %04d    % 15.10f    %.3e    %.6f    %9.3e    %.3e    %.4g\n",
                step, e_curr, abs(e_curr - e_scale), fid, cond(M), θ_dot, step * dt)
        end

        θ_dot < xtol && break
    end

    if verbose > 0 
        @printf("\nFast VQTE completed in %.4f seconds.\n", time_ops)
        println("============================================================================\n")
    end

    return fid
end


function run_adapt_vqite_tfim_adjoint(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}},
    v0::Vector{Float64},
    e_scale::Float64;
    dt::Float64=0.01,
    max_adapt_step::Int=50, 
    max_inner_step::Int=1000,
    Gtol::Float64=1e-3,
    xtol::Float64=1e-6,
    Δtol::Float64=1e-8,
    verbose::Int=1, 
    M_trunc::Float64=1e-4,
    inner_per_print::Int=10,
) where {Ti,Tv,TK,TV}
    println("============================================================================")
    println("--- Macro-Micro Adaptive VQITE (Outer: ADAPT Select | Inner: VQITE Relax) ---")

    f_hvec = get_hvec(basis, ham, is_time=false)

    print("Pre-compiling Full Candidate Pool OTF ... ")
    time_ops = @elapsed pool_otf = OTF(basis, pool)
    @printf("Done in %.4f seconds\n", time_ops)

    f_expm = (idx, θ, vec) -> expm_svd!(basis, pool_otf, idx, θ, vec)
    f_batchgrad = (lv, rv, g, x) -> batch_grad_svd(basis, pool_otf, x, lv, rv, g)
    f_backtran = (idx, θ, lvec, rvec, bvec) -> return backtran_svd!(basis, pool_otf, idx, θ, lvec, rvec, bvec)

    ws  = [zeros(Float64, basis.dim) for _ in 1:5]
    ve  = zeros(Float64, basis.dim)
    ve .= v0

    v   = ws[1]
    v  .= v0

    Hv  = ws[2]
    f_hvec(v, Hv)
    e_curr = dot(v, Hv)

    τv  = ws[3]
    bv  = ws[4]
    vt  = ws[5]

    zero_amp   = zeros(Float64, length(pool))
    zero_grads = zeros(Float64, length(pool))
    converged  = false
    iter = 0
    amplitudes = Float64[]
    selec_idxs = Int64[]
    e_hist     = Float64[]

    @time while !converged
        iter += 1
        @. Hv = -(Hv - e_curr * v)
        f_batchgrad(v, Hv, zero_grads, zero_amp)
        @. zero_grads = real(zero_grads) * 2
        sorted_idxs = sortperm(abs.(zero_grads), rev=true)
        Gnorm   = norm(zero_grads)
        max_idx = sorted_idxs[1]
        for i in 1:length(pool)
            if isempty(selec_idxs) || sorted_idxs[i] != selec_idxs[end]
                max_idx = sorted_idxs[i]
                break
            end
        end

        gi_max = abs(zero_grads[max_idx])

        if length(selec_idxs) > 0 && max_idx == selec_idxs[end]
            println("Have selected same operator, ADAPT loop finished!")
            break
        end

        push!(selec_idxs, max_idx)
        push!(amplitudes, 0.0)

        fid_opt = _run_vqite_tfim_adjoint(
            f_hvec, f_expm, f_backtran, 
            ws, v0, ve, v, Hv, τv, bv, vt, selec_idxs, amplitudes, e_scale, 
            dt, max_inner_step, xtol, M_trunc, inner_per_print, verbose-1)
        
        v .= v0
        for i in eachindex(selec_idxs)
            f_expm(selec_idxs[i], amplitudes[i], v)
        end

        f_hvec(v, Hv)
        e_opt = dot(v, Hv)

        push!(e_hist, e_opt)

        cond1::Bool = iter > max_adapt_step
        cond2::Bool = Gnorm < Gtol
        cond3::Bool = false

        if length(e_hist) > 5
            Δe_max = maximum(abs.(diff(e_hist[end-4:end])))
            if Δe_max < Δtol
                @printf("  \nΔE: %9.3e < %.1e, ADAPT loop finished!\n", Δe_max, Δtol)
                cond3 = true
            end
        end

        converged = cond1 || cond2 || cond3

        if verbose > 0
            @printf("\nIteration: %d\n",                            iter)
            @printf("   E0: %.14f\n",                               e_opt)
            @printf("  err: %9.3e\n",                               e_opt-e_scale)
            @printf("  |G|: %9.3e    gmax: %9.3e     fid: %9.3e\n", Gnorm, gi_max, fid_opt)
            println("============================================================================")
        end

    end

    return amplitudes, selec_idxs
end

