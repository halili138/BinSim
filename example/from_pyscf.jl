ENV["OMP_NUM_THREADS"] = 4
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_PLACES"] = "cores"

include("../binsim.jl")

using PyCall

@pyimport pyscf.gto as gto
@pyimport pyscf.scf as scf
@pyimport pyscf.mcscf as mcscf
@pyimport pyscf.ao2mo as ao2mo
@pyimport pyscf.ci as ci
@pyimport pyscf.cc as cc
@pyimport pyscf.fci as fci
@pyimport pyscf.pbc as pbc
@pyimport pyscf.mp as mp

pushfirst!(pyimport("sys")."path", joinpath(@__DIR__, pypath))
pyfun = pyimport("mole_pbc_int")

function init_scf(
    name::String,
    ratio::Float64;
    basis::String="sto-3g",
    run_cisd::Bool=true,
    run_mp2::Bool=true,
    run_ccsd::Bool=true,
    run_fci::Bool=true,
)
    e_scf = 0.0
    e_cisd = 0.0
    e_mp2 = 0.0
    e_ccsd = 0.0
    e_fci = 0.0

    geo = molecule_geometry(name, ratio)

    mol = gto.M(atom=geo, basis=basis, spin=0.0, symmetry=true)
    println("Use symmetry. Molecule point group: $(mol.groupname)")

    norb = mol.nao_nr()
    nelec::Tuple{Int64,Int64} = mol.nelec
    energy_nuc::Float64 = mol.energy_nuc()
    println("Norb: $(norb)   Ne: $(nelec)")

    norb > 120 && return

    mf = scf.RHF(mol)
    println("Running RHF...")
    mf.kernel()
    orbsym = mf.orbsym
    e_scf = mf.e_tot
    e_scale = mf.e_tot

    orbsym = hasproperty(mf, :orbsym) ? mf.orbsym : ones(Int64, norb)

    if run_cisd
        mf_ci = ci.CISD(mf)
        println("Running CISD ...")
        mf_ci.kernel()
        e_cisd = mf_ci.e_tot
        e_scale = mf_ci.e_tot
    end

    if run_mp2
        mf_mp2 = mp.MP2(mf)
        println("Running MP2 ...")
        mf_mp2.kernel()
        e_mp2 = mf_mp2.e_tot
        e_scale = mf_mp2.e_tot
    end

    if run_ccsd
        mf_cc = cc.CCSD(mf)
        println("Running CCSD ...")
        mf_cc.kernel()
        e_ccsd = mf_cc.e_tot
        e_scale = mf_cc.e_tot
    end

    if run_fci
        mf_fci = fci.FCI(mf)
        println("Running FCI ...")
        mf_fci.kernel()
        e_fci = mf_fci.e_tot
        e_scale = mf_fci.e_tot
        println(@sprintf("E(FCI) = %.14f  E_corr = %.14f", e_scale, e_scale - mf.e_tot))
    end

    one_body_mo::Array{Float64,2}, two_body_mo::Array{Float64,4} = pyfun.mol_int(mf)

    return norb, nelec, orbsym, energy_nuc, one_body_mo, two_body_mo, e_scale
end


if abspath(PROGRAM_FILE) == @__FILE__
    norb, nelec, orbsym, energy_nuc, one_body_mo, two_body_mo, e_scale = init_scf("h8", 1.0, basis="sto-3g", run_fci = false)

    Tv = Float64 # 波函数的数据类型, 根据需求自行选择 Float64 或 ComplexF64

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
    e_fcis, v_fcis = run_fci(basis, ham, k=3)
    e_vqe, v_vqe, x_vqe = run_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci)
    # run_exact_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci)
    # run_adapt_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci)
    run_enpt2(basis, ham, v_vqe, e_fci)
    orbs  = Orbitals(); kernel(mole, orbs, generalize=true)
    pool  = FEB(orbs)
    run_qse(basis, ham, pool, v_vqe, e_scales=e_fcis)
    run_qeom(basis, ham, pool, v_vqe, e_scales=e_fcis)
end

