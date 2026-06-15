ENV["OMP_NUM_THREADS"] = 8
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

pushfirst!(pyimport("sys")."path", pypath)
pyfun = pyimport("mole_pbc_int")



function init_scf(
    name::String,
    ratio::Float64;
    basis::String="sto-3g",
    run_cisd::Bool=true,
    run_mp2::Bool=true,
    run_ccsd::Bool=true,
    run_fci::Bool=true,
    is_save::Bool=false,
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

    filepath = joinpath(jld2path, "$(basis)/$(name)-$(ratio)-$(basis).jld2")

    if is_save
        jldopen(filepath, "w") do file
            file["norb"] = norb
            file["nelec"] = nelec
            file["orbsym"] = orbsym
            file["energy_nuc"] = energy_nuc
            file["one_body_mo"] = one_body_mo
            file["two_body_mo"] = two_body_mo
            file["e_scale"] = e_scale
        end
        println("Saved to $(filepath)")
    end

    return e_scale, e_scf, e_cisd, e_mp2, e_ccsd, e_fci
end


function init_scf(
    phi_deg::Float64;
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

    geo = build_ethylene_geometry(1.0, phi_deg)
    mol = gto.M(atom=geo, basis="sto-3g", spin=0.0, symmetry=true)
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

    filepath = joinpath(jld2path, "c2h4-1.0-$(phi_deg)-sto-3g.jld2")

    jldopen(filepath, "w") do file
        file["norb"] = norb
        file["nelec"] = nelec
        file["orbsym"] = orbsym
        file["energy_nuc"] = energy_nuc
        file["one_body_mo"] = one_body_mo
        file["two_body_mo"] = two_body_mo
        file["e_scale"] = e_scale
    end
    println("Saved to $(filepath)")

    return e_scale, e_scf, e_cisd, e_mp2, e_ccsd, e_fci
end


if abspath(PROGRAM_FILE) == @__FILE__
    # for nh in 50:2:120
    #     init_scf(
    #         "h$(nh)", 1.0, 
    #         run_mp2=false, 
    #         run_ccsd=false, 
    #         run_fci=false, 
    #         is_save=true)
    # end
    # init_scf(
    #     ARGS[1], 1.0, basis=ARGS[2],
    #     run_cisd=false,
    #     run_mp2=false, 
    #     run_ccsd=false, 
    #     run_fci=false, 
    #     is_save=parse(Bool, ARGS[3]))

    # for n in ["lih", "beh2", "h2o", "nh3", "ch4", "n2", "co", "h2co", "c2h4", "c2h6", "co2", "hcn", "hf", "hcl", "c2", "o2", "sih4", "ch3oh", "c2h5oh", "hcooh", "ch3cooh", "c4h10"]
    #     for b in ["sto-3g", "6-31g", "cc-pvdz", "cc-pvtz"]
    #             init_scf(
    #                 n, 1.0, 
    #                 basis=b,
    #                 run_cisd=false,
    #                 run_mp2 =false, 
    #                 run_ccsd=false, 
    #                 run_fci =false, 
    #                 is_save =true)
    #     end
    # end

    # e_scfs = []
    # for r::Float64 in 0.5:0.1:2.5
    #     e_scale, e_scf, e_cisd, e_mp2, e_ccsd, e_fci = init_scf(ARGS[1], r, basis=ARGS[2], run_mp2=false, run_ccsd=false, run_fci=false, is_save=false)
    #     push!(e_scfs, e_scf)
    # end
    # for e in e_scfs
    #     println(e)
    # end
    # init_scf("c2h2", 1.0, basis="sto-3g", run_mp2=false, run_ccsd=false, run_fci=true, is_save=true)
    # init_scf("c2h2", 1.0, basis="6-31g", run_mp2=false, run_ccsd=false, run_fci=false, is_save=true)
    # init_scf("c2h2", 1.0, basis="cc-pvdz", run_mp2=false, run_ccsd=false, run_fci=false, is_save=true)
    init_scf(ARGS[1], 1.0, basis=ARGS[2], run_mp2=true, run_ccsd=true, run_fci=false, is_save=false)
end

