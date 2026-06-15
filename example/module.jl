include("../jl/binsim.jl")

function run_vqrte_tdva(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,TK,TV}, pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}}, v0::Vector{Tv},
    obs_X::BinaryQubitAABB{Ti,Tv,TK,TV}, obs_Z::BinaryQubitAABB{Ti,Tv,TK,TV};
    dt::Float64=1e-2, max_step::Int=500, tikhonov_eps::Float64=1e-3, per_print::Int=10,
) where {Ti,Tv,TK,TV}
    @assert Tv == ComplexF64 "TDVA requires Tv=ComplexF64 for Hamiltonian, Pool, and initial state!"
    println("============================================================================")
    println("--- Time-Dependent Variational Algorithm (TDVA) ---")
    println("--- DOI: https://doi.org/10.1103/PhysRevX.7.021050 ---\n")

    funcs  = OTF_Functions(basis, ham, pool, time_print=false)
    xfuncs = OTF_Functions(basis, obs_X, eltype(pool)[], info_print=false, time_print=false)
    zfuncs = OTF_Functions(basis, obs_Z, eltype(pool)[], info_print=false, time_print=false)
    obs_X_hist = Float64[]
    obs_Z_hist = Float64[]
    exact_obs_X_hist = Float64[]
    exact_obs_Z_hist = Float64[]

    N = length(pool)
    e_hist = Float64[]

    x = zeros(Float64, N)
    xs = [zeros(Float64, N) for _ in 1:5]
    xt = xs[5]

    vs = [zeros(Tv, basis.dim) for _ in 1:N]
    ws = [zeros(Tv, basis.dim) for _ in 1:5]
    D  = zeros(Tv, N, basis.dim)

    ve = copy(v0)
    v  = ws[1]
    Hv = ws[2]
        
    function compute_xdot_and_energy!(x_in, dx_out)
        v .= v0
        for k in 1:N
            vs[k] .= v
            funcs.expm(k, x_in[k], v)
        end

        funcs.hvec(v, Hv)
        E_val = real(dot(v, Hv))

        for k in 1:N
            funcs.tvec(k, vs[k], ws[3]) 
            D[k, :] .= ws[3]
        end

        for j in 1:N
            funcs.batchexpm(j, x_in[j], D, N, j) 
        end

        @. Hv = conj(-im * (Hv - E_val * v))     
        V = real.(D * Hv)
        M = real.(D * D')       

        if any(isnan, M) || any(isinf, M)
            println("WARNING: M matrix contains NaN or Inf! Returning zero update.")
            dx_out .= 0.0
            return E_val, cond(M)
        end

        shift_val = 1e-12
        for i in 1:N
            M[i, i] += shift_val
        end

        F = eigen(Symmetric(M))
        inv_S = [1.0 / (abs(val) + tikhonov_eps) for val in F.values]
        M_pinv = F.vectors * Diagonal(inv_S) * F.vectors'
        dx_out .= M_pinv * V

        return E_val, cond(M) 
    end

    @printf("  Step        Energy           δE       Trj Fid     cond(M)       |dx|      Time\n")
    time_ops = @elapsed for step in 1:max_step
        e_curr, cond_M = compute_xdot_and_energy!(x, xs[1])
        push!(e_hist, e_curr)

        @. xt = x + 0.5 * dt * xs[1]
        compute_xdot_and_energy!(xt, xs[2])

        @. xt = x + 0.5 * dt * xs[2]
        compute_xdot_and_energy!(xt, xs[3])

        @. xt = x + dt * xs[3]
        compute_xdot_and_energy!(xt, xs[4])

        @. x += dt / 6 * (xs[1] + 2 * xs[2] + 2 * xs[3] + xs[4])

        rte_rk4_step!(funcs.hvec, ve, ws, dt)

        v .= v0
        for k in 1:N
            funcs.expm(k, x[k], v)
        end

        xfuncs.hvec(v,  ws[5]); push!(obs_X_hist, real(dot(v, ws[5])))
        zfuncs.hvec(v,  ws[5]); push!(obs_Z_hist, real(dot(v, ws[5])))
        xfuncs.hvec(ve, ws[5]); push!(exact_obs_X_hist, real(dot(ve, ws[5])))
        zfuncs.hvec(ve, ws[5]); push!(exact_obs_Z_hist, real(dot(ve, ws[5])))

        fid = abs2(dot(v, ve))

        if step % per_print == 0 || step == 1
            δe = step > 1 ? (e_hist[end] - e_hist[end-1]) : 0.0
            @printf("  %-4.d  % 15.10f    % 8.2e    %.6f    %8.2e    %8.2e    %-6.4g\n",
                step, e_curr, δe, fid, cond_M, norm(xs[1]), step * dt)
        end
    end

    @printf("\nTDVA (RK4) completed in %.4f seconds.\n", time_ops)
    println("============================================================================\n")
    
    for i in eachindex(obs_X_hist)
        @printf("  step: %04d  x : % 8.4f  xe : % 8.4f  z : % 8.4f  ze : % 8.4f\n",
                i, obs_X_hist[i], exact_obs_X_hist[i], obs_Z_hist[i], exact_obs_Z_hist[i])
    end

end

function pvqd_fig3_test(nq::Int, nL::Int)
    Ti = UInt32
    Tv = ComplexF64
    
    norb = nq ÷ 2 
    N = 1 << norb - 1
    astrs = [UInt32(i) for i in 0:N]
    bstrs = [UInt32(i) for i in 0:N]
    basis = BasisManager(norb, astrs, bstrs, zeros(Int64, norb))

    ref_astrs::Vector{UInt32} = [0]
    ref_bstrs::Vector{UInt32} = [0]
    ref_vals::Vector{Tv}      = [1]
    v0 = get_reference_state(basis, ref_astrs, ref_bstrs, ref_vals)
    normalize!(v0)

    J  = 0.25
    h  = 1.0
    ham = ising_module(nq, J, h, Tv=Tv, is_pbc=false)

    pool = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    for l in 1:nL
        alpha = (l % 2 != 0) ? "X" : "Y"
        for i in 0:nq-1
            push!(pool, QubitOperatorAABB([(i, alpha)], -im, Ti, Tv))
        end
        for i in 0:nq-2
            push!(pool, QubitOperatorAABB([(i, "Z"), (i+1, "Z")], -im, Ti, Tv))
        end
    end

    ops_mx = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    ops_mz = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    for i in 0:nq-1
        push!(ops_mx, QubitOperatorAABB([(i, "X")], 1.0 / nq, Ti, Tv))
        push!(ops_mz, QubitOperatorAABB([(i, "Z")], 1.0 / nq, Ti, Tv))
    end
    obs_X = linearcombine(ops_mx, ones(Tv, nq), 0.0, 1e-12)
    obs_Z = linearcombine(ops_mz, ones(Tv, nq), 0.0, 1e-12)

    run_vqrte_tdva(basis, ham, pool, v0, obs_X, obs_Z,
        dt=0.05,
        max_step=40,
        per_print=1
    )
end

pvqd_fig3_test(parse(Int, ARGS[1]), parse(Int, ARGS[2]))

