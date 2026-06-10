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
        println("")
        @. v -= dτ * w
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
    method = parse(Int, ARGS[4])
    method == 1 && test1(ARGS[1], parse(Float64, ARGS[2]), ARGS[3])
    method == 2 && test2(ARGS[1], parse(Float64, ARGS[2]), ARGS[3])
    method == 3 && test3(ARGS[1], parse(Float64, ARGS[2]), ARGS[3])
    method == 4 && test4(ARGS[1], parse(Float64, ARGS[2]), ARGS[3])
end
