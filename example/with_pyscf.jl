ENV["OMP_NUM_THREADS"] = 8
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_PLACES"] = "cores"

include("../binsim.jl")

using PyCall

@pyimport pyscf.gto as gto
@pyimport pyscf.scf as scf
@pyimport pyscf.fci as fci

pushfirst!(pyimport("sys")."path", joinpath(@__DIR__, "../py/"))
pyfun = pyimport("mole_pbc_int")

if abspath(PROGRAM_FILE) == @__FILE__
    ratio = 1.0
    a = 1.1 * ratio
    geo = "
    N 0.0 0.0 0.0;
    N 0.0 0.0 $(a);
    "
    mol = gto.M(atom=geo, basis="sto-3g", spin=0.0, symmetry=true)
    println("Use symmetry. Molecule point group: $(mol.groupname)")

    norb = mol.nao_nr()
    nelec::Tuple{Int64,Int64} = mol.nelec
    energy_nuc::Float64 = mol.energy_nuc()
    println("Norb: $(norb)   Ne: $(nelec)")

    mf = scf.RHF(mol)
    println("Running RHF...")
    mf.kernel()
    orbsym = mf.orbsym

    orbsym = hasproperty(mf, :orbsym) ? mf.orbsym : ones(Int64, norb)

    mf_fci = fci.FCI(mf)
    println("Running FCI ...")
    mf_fci.kernel()
    e_scale = mf_fci.e_tot
    println(@sprintf("E(FCI) = %.14f  E_corr = %.14f", e_scale, e_scale - mf.e_tot))

    one_body_mo::Array{Float64,2}, two_body_mo::Array{Float64,4} = pyfun.mol_int(mf)

    Tv   = Float64 # 波函数的数据类型, 根据需求自行选择 Float64 或 ComplexF64
    mole = Mole()
    mole.norb = norb
    mole.nelec = nelec
    mole.orbsym = orbsym
    mole.energy_nuc = energy_nuc
    mole.one_body_mo = one_body_mo
    mole.two_body_mo = two_body_mo
    mole.e_scale = e_scale

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym) # 默认构造CSF基, 如有需要, 请使用 BasisManager(norb, astrs, bstrs, orbsym) 自行定义 basis
    ham   = JW_hamiltonian(mole)
    ham   = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, Tv.(ham.cs))
    orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
    pool  = FEB(orbs, Tv=Tv) # 此处使用 UCCSD 算符池

    e_fci, v_fci = run_fci(basis, ham, get_hf(basis, mole.nelec, Tv=Tv))
    run_adapt_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci, vqe_options=VQE_OPTIONS(ftol = 1e-8, gtol=1e-6))
end

