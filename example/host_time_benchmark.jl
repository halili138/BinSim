ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 8)
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")


function test_hvec(mole, nsteps)
    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)
    v     = get_hf(basis, mole.nelec)
    w     = zeros(Float64, basis.dim)
    funcs = OTF_Functions(basis, ham, typeof(ham)[])

    for _ in 1:nsteps
        funcs.hvec(v, w)
        println("")
        @. v -= 1e-2 * w
        normalize!(v)
    end
end


function test_cost_fun(mole, nsteps)
    basis       = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham         = JW_hamiltonian(mole)
    orbs        = Orbitals(); kernel(mole, orbs, generalize=false)
    pool        = FEB(orbs)
    v           = get_hf(basis, mole.nelec)
    w           = zeros(Float64, basis.dim)
    funcs       = OTF_Functions(basis, ham, pool, time_print=false)

    v0_idxs     = findall(x -> x != 0, v) 
    v0_vals     = v[v0_idxs]
    amplitudes  = rand(Float64, length(pool))

    for _ in 1:nsteps
        @time begin
            fill!(v, 0.0)
            v[v0_idxs] .= v0_vals

            for (i, t) in enumerate(amplitudes)
                funcs.expm(i, t, v)
            end

            funcs.hvec(v, w)

            real(dot(v, w)) / norm(v) ^ 2
        end
    end
end


function test_trotter(mole, nsteps)
    basis       = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    orbs        = Orbitals(); kernel(mole, orbs, generalize=false)
    pool        = FEB(orbs)
    v           = get_hf(basis, mole.nelec)
    funcs       = OTF_Functions(basis, eltype(pool)(), pool, time_print=false)

    v0_idxs     = findall(x -> x != 0, v) 
    v0_vals     = v[v0_idxs]
    amplitudes  = rand(Float64, length(pool))

    for _ in 1:nsteps
        @time begin
            fill!(v, 0.0)
            v[v0_idxs] .= v0_vals

            for (i, t) in enumerate(amplitudes)
                funcs.expm(i, t, v)
            end
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    mole.orbsym = Int64.(mole.orbsym .% 10)

    ARGS[3] == "1" && test_hvec(mole, 10)
    ARGS[3] == "2" && test_cost_fun(mole, 10)
    ARGS[3] == "3" && test_trotter(mole, 10)
end

