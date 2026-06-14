_nts   = length(ARGS) >= 1 ? ARGS[1] : 4
_alg   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
_name  = length(ARGS) >= 3 ? ARGS[3] : "h12"
_basis = length(ARGS) >= 4 ? ARGS[4] : "sto-3g"
_ratio = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.0

ENV["OMP_NUM_THREADS"] = _nts
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_PLACES"] = "cores"

include("../binsim.jl")


function test1(name, ratio, basis)
    mole = Mole()
    mole.name  = name
    mole.ratio = ratio
    mole.basis = basis

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)
    funcs = OTF_Functions(basis, ham, typeof(ham)[], time_print=true)
    v     = get_hf(basis, mole.nelec)
    w     = zeros(Float64, basis.dim)
    dτ    = 1e-1
    step  = 0
    while step <= 10
        step += 1
        funcs.hvec(v, w)
        
        @. v -= dτ * w

        @printf("  Norm: %.8f \n", norm(v))

        normalize!(v)
    end
end

function test2(name, ratio, basis)
    mole = Mole()
    mole.name  = name
    mole.ratio = ratio
    mole.basis = basis

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)
    orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
    pool  = FEB(orbs)
    v0    = get_hf(basis, mole.nelec)    
    run_vqe(basis, ham, pool, v0, mole.e_scale)
end

function test3(name, ratio, basis)
    mole = Mole()
    mole.name  = name
    mole.ratio = ratio
    mole.basis = basis

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
    pool  = FEB(orbs)
    funcs = OTF_Functions(basis, eltype(pool)(), pool)

    nparas = length(pool)
    amps   = rand(Float64, nparas)
    idxs   = [i for i in 1:nparas]
    lv     = rand(Float64, basis.dim)

    for _ in 1:10
        @time begin
            for i in 1:nparas
                funcs.expm(idxs[i], amps[i], lv)
            end
        end
    end
end

function test4(name, ratio, basis)
    mole = Mole()
    mole.name  = name
    mole.ratio = ratio
    mole.basis = basis

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)
    orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
    pool  = FEB(orbs)
    funcs = OTF_Functions(basis, ham, pool)

    nparas = length(pool)
    amps   = rand(Float64, nparas)
    idxs   = [i for i in 1:nparas]
    lv     = rand(Float64, basis.dim)
    rv     = zeros(Float64, basis.dim)

    for _ in 1:10
        @time begin
            for i in 1:nparas
                funcs.expm(idxs[i], amps[i], lv)
            end
            funcs.hvec(lv, rv)
            real(dot(lv, rv)) / norm(lv) ^ 2
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__    
    _alg == 1 && test1(_name, _ratio, _basis)
    _alg == 2 && test2(_name, _ratio, _basis)
    _alg == 3 && test3(_name, _ratio, _basis)
    _alg == 4 && test4(_name, _ratio, _basis)
end
