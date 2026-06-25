ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 8)
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")

function ham_inspired_feb(mole; 
    tol::Float64=1e-12, Ti::DataType=UInt32, Tv::DataType=Float64, complete::Bool=false,
)
    single_spatial_based = Tuple.(findall(x -> abs(x) > tol, mole.one_body_mo))
    double_spatial_based = Tuple.(findall(x -> abs(x) > tol, mole.two_body_mo))
    single_spin_based    = Vector{UInt16}[]
    double_spin_based    = Vector{UInt16}[]

    for (p, q) in single_spatial_based

        mole.orbsym[p] ⊻ mole.orbsym[q] != 0 && continue

        pa = 2 * p - 2
        pb = 2 * p - 1
        qa = 2 * q - 2
        qb = 2 * q - 1
        push!(single_spin_based, UInt16[pa, qa])
        push!(single_spin_based, UInt16[pb, qb])
    end

    for (p, q, r, s) in double_spatial_based

        mole.orbsym[p] ⊻ mole.orbsym[q] ⊻ mole.orbsym[r] ⊻ mole.orbsym[s] != 0 && continue

        pa = 2 * p - 2
        pb = 2 * p - 1
        qa = 2 * q - 2
        qb = 2 * q - 1
        ra = 2 * r - 2
        rb = 2 * r - 1
        sa = 2 * s - 2
        sb = 2 * s - 1
        push!(double_spin_based, UInt16[pa, qa, ra, sa])
        push!(double_spin_based, UInt16[pb, qb, rb, sb])
        push!(double_spin_based, UInt16[pa, qb, rb, sa])
        push!(double_spin_based, UInt16[pb, qa, ra, sb])
    end

    orbitals = Orbitals()
    f = a -> collect(UInt16, a)

    orbitals.spatial_based = vcat(f.(single_spatial_based), f.(double_spatial_based))
    orbitals.spin_based = vcat(single_spin_based, double_spin_based)

    return FEB(orbitals, Ti=Ti, Tv=Tv, complete=complete)
end

function mole_test(; dt=1e-3, nsteps=1000, per_print=10)
    Tv = ComplexF64 # 周期性或实时演化, 需使用ComplexF64

    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    mole.orbsym = Int64.(mole.orbsym .% 10) # 非阿贝尔点群(Dooh, Cooh), 需通过此修正为最近的阿贝尔点群(D2h, C2h)

    basis       = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham         = JW_hamiltonian(mole)
    ham         = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, Tv.(ham.cs))

    pool        = ham_inspired_feb(mole, tol=1e-12, Tv=Tv)
    v           = get_hf(basis, mole.nelec, Tv=Tv)
    w           = zeros(Tv, basis.dim)
    ve          = copy(v)
    ws          = [zeros(Tv, basis.dim) for _ in 1:5]
    funcs       = OTF_Functions(basis, ham, pool, time_print=false)
    amplitudes  = fill(dt, length(pool))
    e_hist      = Float64[]

    for step in 1:nsteps
        time_ops = @elapsed begin
            for (i, t) in enumerate(amplitudes)
                funcs.expm(i, t, v)
            end

            funcs.hvec(v, w)

            rte_rk4_step!(funcs.hvec, ve, ws, dt)

            e_curr = real(dot(v, w)) / norm(v) ^ 2
            push!(e_hist, e_curr)

            fid = abs2(dot(v, ve))
            δe  = step > 1 ? (e_hist[end] - e_hist[end-1]) : 0.0
            if step % per_print == 0 || step == 1
                @printf(" Step: %-4.d  Energy: % 15.10f    δE: % 8.2e    Fid: %.6f    dt: %-6.4g\n",
                            step, e_curr, δe, fid, step * dt)
            end
        end
    end
end

function ising_test(; nsites=10, dt=1e-3, nsteps=1000, per_print=10, is_pbc=false)
    Ti = UInt32
    Tv = ComplexF64
    
    norb = nsites ÷ 2 
    N = 1 << norb - 1
    astrs = [Ti(i) for i in 0:N]
    bstrs = [Ti(i) for i in 0:N]
    basis = BasisManager(norb, astrs, bstrs, zeros(Int64, norb))
    ref_astrs = Ti[0]
    ref_bstrs = Ti[0]
    ref_vals  = Tv[1]
    v = get_reference_state(basis, ref_astrs, ref_bstrs, ref_vals)
    normalize!(v)


    J   = 0.25
    h   = 1.0
    ops = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    cs  = Tv[]

    for i in 0:nsites-2
        push!(ops, QubitOperatorAABB([(i, "Z"), (i+1, "Z")], -J, Ti, Tv))
        push!(cs, 1)
    end

    if is_pbc
        push!(ops, QubitOperatorAABB([(nsites-1, "Z"), (0, "Z")], -J, Ti, Tv))
        push!(cs, 1)
    end

    for i in 0:nsites-1
        push!(ops, QubitOperatorAABB([(i, "X")], -h, Ti, Tv))
        push!(cs, 1)
    end

    ham    = linearcombine(ops, cs, 0.0, 1e-12)
    pool   = ops .* im


    ops_mx = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    ops_mz = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    for i in 0:nsites-1
        push!(ops_mx, QubitOperatorAABB([(i, "X")], 1.0 / nsites, Ti, Tv))
        push!(ops_mz, QubitOperatorAABB([(i, "Z")], 1.0 / nsites, Ti, Tv))
    end
    obs_X  = linearcombine(ops_mx, ones(Tv, nsites), 0.0, 1e-12)
    obs_Z  = linearcombine(ops_mz, ones(Tv, nsites), 0.0, 1e-12)
    xfuncs = OTF_Functions(basis, obs_X, eltype(pool)[], info_print=false, time_print=false)
    zfuncs = OTF_Functions(basis, obs_Z, eltype(pool)[], info_print=false, time_print=false)
    obs_X_hist = Float64[]
    obs_Z_hist = Float64[]
    exact_obs_X_hist = Float64[]
    exact_obs_Z_hist = Float64[]


    w           = zeros(Tv, basis.dim)
    ve          = copy(v)
    ws          = [zeros(Tv, basis.dim) for _ in 1:5]
    funcs       = OTF_Functions(basis, ham, pool, time_print=false)
    amplitudes  = fill(dt, length(pool))
    e_hist      = Float64[]

    for step in 1:nsteps
        time_ops = @elapsed begin
            for (i, t) in enumerate(amplitudes)
                funcs.expm(i, t, v)
            end

            funcs.hvec(v, w)

            rte_rk4_step!(funcs.hvec, ve, ws, dt)

            e_curr = real(dot(v, w)) / norm(v) ^ 2
            push!(e_hist, e_curr)

            fid = abs2(dot(v, ve))
            δe  = step > 1 ? (e_hist[end] - e_hist[end-1]) : 0.0

            xfuncs.hvec(v, ws[5])
            push!(obs_X_hist, real(dot(v, ws[5])))
            zfuncs.hvec(v, ws[5])
            push!(obs_Z_hist, real(dot(v, ws[5])))
            xfuncs.hvec(ve, ws[5])
            push!(exact_obs_X_hist, real(dot(ve, ws[5])))
            zfuncs.hvec(ve, ws[5])
            push!(exact_obs_Z_hist, real(dot(ve, ws[5])))

            if step % per_print == 0 || step == 1
                @printf(" Step: %-4.d  Energy: % 15.10f    δE: % 8.2e    Fid: %.6f    dt: %-6.4g\n",
                            step, e_curr, δe, fid, step * dt)
            end
        end
    end

    for i in eachindex(obs_X_hist)
        @printf("  step: %04d  x : % 8.4f  xe : % 8.4f  z : % 8.4f  ze : % 8.4f\n",
            i, obs_X_hist[i], exact_obs_X_hist[i], obs_Z_hist[i], exact_obs_Z_hist[i])
    end

end


# mole_test()
t  = 2
dt = 5e-2
nsteps = Int(t / dt)
@time ising_test(nsites=20, dt=dt, nsteps=nsteps)
