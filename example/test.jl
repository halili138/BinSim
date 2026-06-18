ENV["OMP_NUM_THREADS"] = 8
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_PLACES"] = "cores"

include("../jl/binsim.jl")

Tv = Float64 #周期性或实时演化, 需使用ComplexF64

mole = Mole()
mole.name  = ARGS[1]
mole.ratio = 1.0
mole.basis = ARGS[2]

build(mole)

# println(mole.orbsym)

# blocks, block_map = get_sym_blocks(mole.norb, mole.nelec, mole.orbsym, 0, UInt32)

# println(length(blocks))

# mole.orbsym = Int64.(mole.orbsym .% 10) # 非阿贝尔点群(Dooh, Cooh), 需通过此修正为最近的阿贝尔点群(D2h, C2h)

basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
ham   = JW_hamiltonian(mole, spin="aabb")
ham   = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, Tv.(ham.cs))

# # 对角化相关test
e_fci, v_fci = run_fci(basis, ham, get_hf(basis, mole.nelec, Tv=Tv))
# e_fcis, v_fcis = run_fci(basis, ham, k=3)

# VQE相关test
orbs = Orbitals()
kernel(mole, orbs, generalize=false)
pool = FEB(orbs, Tv=Tv)
# e_vqe, v_vqe, x_vqe = run_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci)
run_exact_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci)
# run_adapt_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci, vqe_options=VQE_OPTIONS(ftol=1e-8, gtol=1e-6))

# # 后处理相关test
# orbs = Orbitals();
# kernel(mole, orbs, generalize=true);
# pool = FEB(orbs)
# run_enpt2(basis, ham, v_vqe, e_fci)
# run_qse(basis, ham, pool, v_vqe, e_scales=e_fcis)
# run_qeom(basis, ham, pool, v_vqe, e_scales=e_fcis)

# # 虚时演化相关test
# run_rk4_ite(basis, ham, get_hf(basis, mole.nelec), e_fci, dt=1e-1)
# run_euler_ite(basis, ham, get_hf(basis, mole.nelec), e_fci)
# run_krylov_ite(basis, ham, get_hf(basis, mole.nelec), e_fci, dt=5e-1)

