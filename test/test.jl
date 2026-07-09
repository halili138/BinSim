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
e_fcis, v_fcis = run_fci(basis, ham, k=3)

orbs = Orbitals()
kernel(mole, orbs, generalize=false)
pool = FEB(orbs, Tv=Tv)

# VQE相关
orbs = Orbitals()
kernel(mole, orbs, generalize=true)
pool = FEB(orbs, Tv=Tv)
e_vqe, v_vqe, x_vqe = run_vqe(basis, ham, pool, get_hf(basis), e_fci)
run_exact_vqe(basis, ham, pool, get_hf(basis), e_fci)
run_adapt_vqe(basis, ham, pool, get_hf(basis), e_fci, vqe_options=VQE_OPTIONS(ftol=1e-8, gtol=1e-6))

# 后处理相关
orbs = Orbitals();
kernel(mole, orbs, generalize=true);
pool = FEB(orbs)
run_enpt2(basis, ham, v_vqe, e_fci)
run_qse(basis, ham, pool, v_vqe, e_scales=e_fcis)
run_qeom(basis, ham, pool, v_vqe, e_scales=e_fcis)

# 虚时演化相关
run_rk4_ite(basis, ham, get_hf(basis), e_fci, dt=1e-1)
run_euler_ite(basis, ham, get_hf(basis), e_fci)
run_krylov_ite(basis, ham, get_hf(basis), e_fci, dt=5e-1)
