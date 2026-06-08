include("../binsim.jl")

Tv = Float64

mole = Mole()
mole.name   = "h4"
mole.ratio  = 1.0
mole.basis  = "sto-3g"

build(mole)

basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
ham   = JW_hamiltonian(mole)
ham   = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, Tv.(ham.cs))
orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
pool  = FEB(orbs, Tv=Tv)

e_fci, v_fci = run_fci(basis, ham, get_hf(basis, mole.nelec, Tv=Tv))
e_fcis, v_fcis = run_fci(basis, ham, k=3)
e_vqe, v_vqe, x_vqe = run_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci)
run_exact_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci)
run_adapt_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci)
run_rk4_ite(basis, ham, get_hf(basis, mole.nelec), e_fci, per_print=100)
run_euler_ite(basis, ham, get_hf(basis, mole.nelec), e_fci, per_print=100)
run_krylov_ite(basis, ham, get_hf(basis, mole.nelec), e_fci)
run_enpt2(basis, ham, v_vqe, e_fci)
orbs  = Orbitals(); kernel(mole, orbs, generalize=true)
pool  = FEB(orbs)
run_qse(basis, ham, pool, v_vqe, e_scales=e_fcis)
run_qeom(basis, ham, pool, v_vqe, e_scales=e_fcis)
# run_qpe_ode(basis, ham, get_hf(basis, mole.nelec, Tv=Tv), e_scales=e_fcis)
# run_rk4_rte(basis, ham, get_hf(basis, mole.nelec, Tv=Tv))
# run_vqrte_tdva(basis, ham, pool, get_hf(basis, mole.nelec, Tv=Tv), per_print=50)
# run_vqrte_pvqd(basis, ham, pool, get_hf(basis, mole.nelec, Tv=Tv), per_print=50)
