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

Tv = Float64 # 周期性或实时演化, 需使用ComplexF64

mole = Mole()
mole.name  = ARGS[1]
mole.ratio = 1.0
mole.basis = ARGS[2]

build(mole)

mole.orbsym = Int64.(mole.orbsym .% 10) # 非阿贝尔点群(Dooh, Cooh), 需通过此修正为最近的阿贝尔点群(D2h, C2h)

basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
ham   = JW_hamiltonian(mole, spin="aabb")
ham   = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, Tv.(ham.cs))

# e_fci, v_fci = run_fci(basis, ham, get_hf(basis, mole.nelec, Tv=Tv))

pool = ham_inspired_feb(mole, tol=1e-12, Tv=Tv)

# orbs = Orbitals()
# kernel(mole, orbs, generalize=false)
# pool = FEB(orbs, Tv=Tv)

orbs = Orbitals()
kernel(mole, orbs, generalize=true)
pool = FEB(orbs, Tv=Tv)

# e_vqe, v_vqe, x_vqe = run_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci, options = VQE_OPTIONS(verbose=1))
# run_exact_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci)
# run_adapt_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci, vqe_options=VQE_OPTIONS(ftol=1e-8, gtol=1e-6))
