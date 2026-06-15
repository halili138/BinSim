ENV["OMP_NUM_THREADS"] = 8
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_PLACES"] = "cores"

include("../binsim.jl")

Tv = Float64

mole = Mole()
mole.name   = ARGS[1]
mole.ratio  = 1.0
mole.basis  = ARGS[2]

build(mole)

# mole.orbsym = Int64.(mole.orbsym .% 10)

basis  = BasisManager(mole.norb, mole.nelec, mole.orbsym)
ham    = JW_hamiltonian(mole, spin = "abab")
orbsym = get_orbsym(ham, mole.norb)
println(orbsym)
blocks, block_map = get_sym_blocks(mole.norb, mole.nelec, orbsym, 0, UInt32)
println(length(blocks))

# println(sort(unique(mole.orbsym))) 
# println("count(orbsym >= 10) = ", count(x -> x >= 10, mole.orbsym))


# ham   = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, Tv.(ham.cs))
# orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
# pool  = FEB(orbs, Tv=Tv)

# e_fci, v_fci = run_fci(basis, ham, get_hf(basis, mole.nelec, Tv=Tv))
# e_fcis, v_fcis = run_fci(basis, ham, k=1)
# e_vqe, v_vqe, x_vqe = run_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci)
# run_exact_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci)
# run_adapt_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci, adapt_options=ADAPT_OPTIONS(), vqe_options=VQE_OPTIONS(ftol = 1e-8, gtol=1e-6))
# run_rk4_ite(basis, ham, get_hf(basis, mole.nelec), e_fci, per_print=100)
# run_euler_ite(basis, ham, get_hf(basis, mole.nelec), e_fci, per_print=100)
# run_krylov_ite(basis, ham, get_hf(basis, mole.nelec), e_fci)
# run_enpt2(basis, ham, v_vqe, e_fci)
# orbs  = Orbitals(); kernel(mole, orbs, generalize=true)
# pool  = FEB(orbs)
# run_qse(basis, ham, pool, v_vqe, e_scales=e_fcis)
# run_qeom(basis, ham, pool, v_vqe, e_scales=e_fcis)

    