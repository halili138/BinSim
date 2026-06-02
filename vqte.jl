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

struct TimeEvolBuffer{Tv}
    vs::Vector{Vector{Tv}}
    dvs::Vector{Vector{Tv}}
    ws::Vector{Vector{Tv}}
    dx::Vector{Float64}
    xs::Vector{Vector{Float64}}
    M::Array{Float64,2}
    Md::Array{Float64,1}
    V::Array{Float64,1}
end

function TimeEvolBuffer(dim::Int, N::Int;
    method::String="ite", mode::String="adjoint", is_diag::Bool=false, Tv::DataType=Float64,
)
    @assert method in ["ite", "rte"]
    @assert mode in ["adjoint", "forward"]

    vs  = mode == "forward" ? [zeros(Tv, dim) for _ in 1:N] : Vector{Vector{Tv}}()
    dvs = mode == "forward" ? [zeros(Tv, dim) for _ in 1:N] : Vector{Vector{Tv}}()
    ws  = [zeros(Tv, dim) for _ in 1:5]

    dx  = zeros(Float64, N)
    xs  = method == "rte" ? [zeros(Float64, N) for _ in 1:5] : Vector{Vector{Float64}}()

    M   = is_diag ? zeros(Float64, 0, 0) : zeros(Float64, N, N)
    Md  = is_diag ? zeros(Float64, N) : Vector{Float64}()
    V   = zeros(Float64, N)

    return TimeEvolBuffer(vs, dvs, ws, dx, xs, M, Md, V)
end

struct OTF_Functions
    hvec::Function
    expm::Function
    tvec::Function
    grad::Function
    backgrad::Function
    backtran::Function
    batchgrad::Function 
    batchtran::Function
end

function OTF_Functions(
    basis::BasisManager, 
    ham::Union{Nothing,BinaryQubitAABB}, 
    pool::Union{Nothing,Vector{<:BinaryQubitAABB}};
    time_print::Bool=false,
)
    f_hvec      = (v, Hv)               -> nothing
    f_expm      = (idx, θ, v)           -> nothing
    f_tvec      = (idx, lv, rv)         -> nothing
    f_grad      = (idx, θ, lv, rv)      -> nothing
    f_backgrad  = (idx, θ, lv, rv)      -> nothing
    f_backtran  = (idx, θ, lv, rv, tlv) -> nothing
    f_batchgrad = (lv, rv, grads, x)    -> nothing 
    f_batchtran = (lv, rv, trans)       -> nothing

    if !isnothing(ham)
        print("Pre-compiling Ham OTF ... ")
        time_ops = @elapsed ham_otf = OTF(basis, ham)
        @printf("Done in %.4f seconds\n", time_ops)
        if time_print
            f_hvec = (v, Hv) -> @printf(
                "hvec time %.6f seconds", @elapsed hvec_otf!(basis, ham_otf, v, Hv))
        else
            f_hvec = (v, Hv) -> hvec_otf!(basis, ham_otf, v, Hv)
        end
    end
    if !isnothing(pool)
        print("Pre-compiling Pool OTF ... ")
        time_ops = @elapsed pool_otf = OTF(basis, pool)
        @printf("Done in %.4f seconds\n", time_ops)

        f_expm = (idx, θ, v) -> expm_svd!(basis, pool_otf, idx, θ, v)
        f_tvec = (idx, lv, rv) -> tvec_svd!(basis, pool_otf, idx, lv, rv)
        f_grad = (idx, θ, lv, rv) -> return grad_svd(basis, pool_otf, idx, θ, lv, rv)
        f_backgrad = (idx, θ, lv, rv) -> return backgrad_svd!(basis, pool_otf, idx, θ, lv, rv)
        f_backtran = (idx, θ, lv, rv, tlv) -> return backtran_svd!(basis, pool_otf, idx, θ, lv, rv, tlv)
        f_batchgrad = (lv, rv, grads, x) -> return batch_grad_svd(basis, pool_otf, x, lv, rv, grads)
        f_batchtran = (lv, rv, trans) -> return tran_svd(basis, pool_otf, lv, rv, trans)
    end

    return OTF_Functions(f_hvec, f_expm, f_tvec, f_grad, f_backgrad, f_backtran, f_batchgrad, f_batchtran)
end

struct TimeEvolOptions
    dt::Float64
    maxiter::Int64
    xtol::Float64
    mtol::Float64
    per_print::Int64
    verbose::Int64
    run_rk4::Bool
    method::String
    mode::String
    M_order::String
end

function TimeEvolOptions(;
    dt::Float64      = 1e-2,
    maxiter::Int64   = 9999,
    xtol::Float64    = 1e-6,
    mtol::Float64    = 1e-4,
    per_print::Int64 = 100,
    verbose::Int64   = 1,
    run_rk4::Bool    = true,
    method::String   = "ite",
    mode::String     = "forward",  
    M_order::String  = "exact",
)
    TimeEvolOptions(dt, maxiter, xtol, mtol, per_print, verbose, run_rk4, method, mode, M_order)
end

function process_1order_adjoint(f_backgrad, idxs, x, v, Hv, V, Md, N)
    for i in N:-1:1
        V[i]  = real(f_backgrad(idxs[i], x[i], v, Hv)) * 2
        Md[i] = real(dot(v, v))
    end

    return 0.0
end

function process_2order_adjoint(f_backtran, idxs, x, v, Hv, bv, τv, vt, V, M, N)
    for i in N:-1:1
        f_backtran(idxs[i], -x[i], v, Hv, bv)
        M[i, i] = real(dot(bv, bv))
        V[i]    = real(dot(bv, Hv))

        vt .= v
        for j in (i-1):-1:1
            f_backtran(idxs[j], -x[j], vt, bv, τv)
            val = real(dot(τv, bv))
            M[i, j] = val
            M[j, i] = val
        end
    end

    return cond(M)
end

function process_2order_forward(f_expm, f_tvec, idxs, x, vs, dvs, Hv, V, M, N)
    for i in 1:N
        f_tvec(idxs[i], vs[i], dvs[i])
        for j in i:N
            f_expm(idxs[j], x[j], dvs[i])
        end
    end

    D  = reduce(hcat, dvs)
    V .= real.(D' * Hv)
    M .= real.(D' * D)

    return cond(M)
end

function _run_vqte(
    idxs::Vector{Int64}, x::Vector{Float64}, e_scale::Float64,
    v0::Vector{Tv}, ve::Vector{Tv}, 
    otf_funcs::OTF_Functions, buffer::TimeEvolBuffer{Tv}, options::TimeEvolOptions,
) where {Tv}
    N  = length(idxs)

    @views M  = options.M_order == "diag" ? buffer.M[1:0, 1:0] : buffer.M[1:N, 1:N]
    @views Md = options.M_order == "diag" ? buffer.Md[1:N] : buffer.Md[1:0]
    @views V  = buffer.V[1:N]
    @views dx = buffer.dx[1:N]

    vs  = options.mode == "forward" ? buffer.vs[1:N] : buffer.vs[1:0]
    dvs = options.mode == "forward" ? buffer.dvs[1:N] : buffer.vs[1:0]
    ws  = buffer.ws
    v   = ws[1]
    Hv  = ws[2]
    τv  = ws[3]
    bv  = ws[4]
    vt  = ws[5]

    f_hvec = otf_funcs.hvec
    f_expm = otf_funcs.expm
    f_tvec = otf_funcs.tvec
    f_backgrad = otf_funcs.backgrad
    f_backtran = otf_funcs.backtran

    options.run_rk4 && (ve .= v0)
    fid   = 0.0
    t     = 0.0
    condM = 0.0

    method  = options.method
    mode    = options.mode  
    M_order = options.M_order
    dt      = options.dt
    mtol    = options.mtol

    options.verbose > 0 && @printf(
        "  Step          Energy         Error      Trj Fid      cond(M)       |dx|       Time\n")

    time_ops = @elapsed for step in 1:options.maxiter
        if method == "ite"
            shift = -1.0
        elseif method == "rte" 
            shift = -im
        else
            error("Unsupported method $(method)")
        end

        options.run_rk4 && rk4_step!(f_hvec, ve, vt, ws, shift, dt)

        v .= v0
        if mode == "adjoint"
            for k in 1:N
                f_expm(idxs[k], x[k], v)
            end
        elseif mode == "forward"
            for k in 1:N
                vs[k] .= v
                f_expm(idxs[k], x[k], v)
            end
        else
            error("Unsupported mode $(mode)")
        end

        vnorm = norm(v) ^ 2
        f_hvec(v, Hv)
        e = real(dot(v, Hv)) / vnorm
        fid = options.run_rk4 ? abs2(dot(v, ve)) / vnorm : 0.0

        if method == "ite"
            @. Hv = Hv - e * v
        elseif method == "rte" 
            @. Hv = -im * (Hv - e * v)
        end

        if (method, mode, M_order) == ("ite", "forward", "exact")
            condM = process_2order_forward(f_expm, f_tvec, idxs, x, vs, dvs, Hv, V, M, N)
            dx .= (M + mtol * I) \ V
        elseif (method, mode, M_order) == ("ite", "adjoint", "exact")
            condM = process_2order_adjoint(f_backtran, idxs, x, v, Hv, bv, τv, vt, V, M, N)
            dx .= (M + mtol * I) \ V
        elseif (method, mode, M_order) == ("ite", "adjoint", "diag")
            condM = process_1order_adjoint(f_backgrad, idxs, x, v, Hv, V, Md, N)
            @. dx = V / (Md + mtol)
        elseif (method, mode, M_order) == ("rte", "forward", "exact")
            condM = process_2order_forward(f_expm, f_tvec, idxs, x, vs, dvs, Hv, V, M, N)
            F      = svd(M)
            inv_S  = [s > mtol ? 1.0 / s : 0.0 for s in F.S]
            M_pinv = F.V * Diagonal(inv_S) * F.U'
            dx    .= M_pinv * V
        elseif (method, mode, M_order) == ("rte", "adjoint", "exact")
            condM = process_2order_adjoint(f_backtran, idxs, x, v, Hv, bv, τv, vt, V, M, N)
            F      = svd(M)
            inv_S  = [s > mtol ? 1.0 / s : 0.0 for s in F.S]
            M_pinv = F.V * Diagonal(inv_S) * F.U'
            dx    .= M_pinv * V
        else
            error("Unsupported combination when soler Mθ = V: $(method) $(mode) $(M_order)")
        end

        @. x -= dt * dx
        dx_norm = norm(dx)
        t += dt
        err = abs(e - e_scale)

        if options.verbose > 0 && (step % options.per_print == 0 || step == 1)
            @printf("  %04d    % 15.10f    %.3e    %.6f    %9.3e    %9.3e    %.4g\n",
                    step, e, err, fid, condM, dx_norm, t)
        end

        if dx_norm < options.xtol
            if options.verbose > 0
                @printf("  %04d    % 15.10f    %.3e    %.6f    %9.3e    %9.3e    %.4g\n",
                    step, e, err, fid, condM, dx_norm, t)
            end
            break
        end
    end

    if options.verbose > 0
        @printf("\nVQTE completed in %.4f seconds.\n", time_ops)
        println("============================================================================\n")
    end

    return fid, t
end

function run_vqte(
    basis::BasisManager, 
    ham::BinaryQubitAABB{Ti,Tv,TK,TV}, 
    pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}},
    v0::Vector{Tv},
    e_scale::Float64;
    options::TimeEvolOptions=TimeEvolOptions(),
) where {Ti,Tv,TK,TV}
    if options.method == "rte"
        @assert Tv <: Complex "RTE requires Tv <: Complex for Ham, Pool, and v0!"
    end

    dim    = basis.dim
    N      = length(pool)
    funcs  = OTF_Functions(basis, ham, pool)
    buffer = TimeEvolBuffer(
        dim, N, 
        method  = options.method,
        mode    = options.mode,
        is_diag = options.M_order == "diag" ? true : false, 
        Tv      = eltype(v0),
    )
    idxs = [i for i in 1:N]
    x    = zeros(Float64, N)
    ve   = options.run_rk4 ? copy(v0) : Tv[]
    _run_vqte(idxs, x, e_scale, v0, ve, funcs, buffer, options)
end

function run_adapt_vqte(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}},
    v0::Vector{Tv},
    e_scale::Float64;
    adapt_options::ADAPT_OPTIONS=ADAPT_OPTIONS(),
    vqte_options::TimeEvolOptions=TimeEvolOptions(),
) where {Ti,Tv,TK,TV}
    dim    = basis.dim
    N      = length(pool)
    funcs  = OTF_Functions(basis, ham, pool)

    adapt_maxiter = adapt_options.maxiter
    if vqte_options.mode == "forward"
        adapt_maxiter = min(adapt_maxiter, (10 << 30) ÷ (dim * sizeof(Tv)))
        println("Using forward mode, adapt maxiter is limited to $(adapt_maxiter)")
    end

    buffer = TimeEvolBuffer(
        dim, adapt_maxiter, 
        method  = vqte_options.method,
        mode    = vqte_options.mode,
        is_diag = vqte_options.M_order == "diag" ? true : false, 
        Tv      = eltype(v0),
    )

    zero_x     = zeros(Float64, N)
    zero_g     = zeros(Tv, N)
    converged  = false
    amplitudes = Float64[]
    selec_idxs = Int64[]
    e_hist     = Float64[]

    ve   = vqte_options.run_rk4 ? copy(v0) : Tv[] 
    v    = buffer.ws[1]
    Hv   = buffer.ws[2]
    v   .= v0
    funcs.hvec(v, Hv)
    e    = real(dot(v, Hv))
    t    = 0.0
    iter = 0
    
    @time while !converged
        iter += 1
        @. Hv = -(Hv - e * v)
        funcs.batchgrad(v, Hv, zero_g, zero_x)
        @. zero_g = real(zero_g) * 2
        sorted_idxs = sortperm(abs.(zero_g), rev=true)
        gnorm   = norm(zero_g)
        max_idx = sorted_idxs[1]
        for i in 1:length(pool)
            if isempty(selec_idxs) || sorted_idxs[i] != selec_idxs[end]
                max_idx = sorted_idxs[i]
                break
            end
        end

        gmax = abs(zero_g[max_idx])

        if length(selec_idxs) > 0 && max_idx == selec_idxs[end]
            println("Have selected same operator, ADAPT loop finished!")
            break
        end

        push!(selec_idxs, max_idx)
        push!(amplitudes, 0.0)

        fid_opt, t_opt = _run_vqte(selec_idxs, amplitudes, e_scale, v0, ve, funcs, buffer, vqte_options)

        v .= v0
        for i in eachindex(selec_idxs)
            funcs.expm(selec_idxs[i], amplitudes[i], v)
        end

        funcs.hvec(v, Hv)
        e   = real(dot(v, Hv))
        err = abs(e - e_scale)
        push!(e_hist, e)
        t += t_opt

        cond1::Bool = iter > adapt_maxiter
        cond2::Bool = (gnorm < adapt_options.Gtol && gmax < adapt_options.gtol)
        cond3::Bool = false

        if length(e_hist) > 5
            de = maximum(abs.(diff(e_hist[end-4:end])))
            if de < adapt_options.Δtol
                @printf("  \nΔE: %9.3e < %.1e, ADAPT loop finished!\n", de, adapt_options.Δtol)
                cond3 = true
            end
        end

        converged = cond1 || cond2 || cond3

        if adapt_options.verbose > 0
            @printf("\nIteration: %d\n", iter)
            @printf("   E0: %.14f\n",    e)
            @printf("  err: %9.3e\n",    err)
            @printf("  |G|: %9.3e    gmax: %9.3e     fid: %9.3e     Time: %.4f\n",
                gnorm, gmax, fid_opt, t)
            println("============================================================================")
        end

    end

    return amplitudes, selec_idxs
end

