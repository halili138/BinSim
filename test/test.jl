ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 8)
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")

Tv = Float64 # 周期性或实时演化, 需使用ComplexF64

mole = Mole()
mole.name  = "h4"
mole.ratio = 1.0
mole.basis = "sto-3g"

build(mole)

mole.orbsym = Int64.(mole.orbsym .% 10) # 非阿贝尔点群(Dooh, Cooh), 需通过此修正为最近的阿贝尔点群(D2h, C2h)

basis = BasisManager(mole)
ham   = JW_hamiltonian(mole, spin="aabb")
ham   = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, Tv.(ham.cs))

# 对角化相关
e_fci, v_fci = run_fci(basis, ham, get_hf(basis, Tv=Tv))

# VQE相关
orbs = Orbitals()
kernel(mole, orbs, generalize=true)
pool = FEB(orbs, Tv=Tv)

v0      = get_hf(basis, Tv=Tv)
v0_idxs = findall(x -> x != 0, v0)
v0_vals = v0[v0_idxs]
lv      = zeros(Tv, basis.dim)
rv      = zeros(Tv, basis.dim)

funcs = OTF_Functions(basis, ham, pool)
x0    = zeros(Float64, length(pool))
idxs  = [i for i in eachindex(pool)]

e_vqe, v_vqe, x_vqe = run_vqe(funcs, lv, rv, v0_idxs, v0_vals, e_fci, x0, idxs,
    VQE_OPTIONS(ftol=1e-8, gtol=1e-6, maxiter=10000, verbose=1))

run_adapt_vqe(funcs, lv, rv, v0_idxs, v0_vals, e_fci, length(pool), Float64[], Int64[],
    ADAPT_OPTIONS(maxiter=100, verbose=1),
    VQE_OPTIONS(ftol=1e-8, gtol=1e-6, maxiter=10000, verbose=1))
