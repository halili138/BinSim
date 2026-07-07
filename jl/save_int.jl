function init_scf(pyfun, name::String, ratio::Float64, basis::String, save_path::String;
    run_cisd::Bool=false,
    run_mp2::Bool=false,
    run_ccsd::Bool=false,
    run_fci::Bool=true,
)
    gto = pyimport("pyscf.gto")
    scf = pyimport("pyscf.scf")
    ci  = pyimport("pyscf.ci")
    cc  = pyimport("pyscf.cc")
    fci = pyimport("pyscf.fci")
    mp  = pyimport("pyscf.mp")

    e_scf   = 0.0
    e_cisd  = 0.0
    e_mp2   = 0.0
    e_ccsd  = 0.0
    e_fci   = 0.0

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

    dir = dirname(save_path)
    mkpath(dir)
    jldopen(save_path, "w") do file
        file["norb"] = norb
        file["nelec"] = nelec
        file["orbsym"] = orbsym
        file["energy_nuc"] = energy_nuc
        file["one_body_mo"] = one_body_mo
        file["two_body_mo"] = two_body_mo
        file["e_scale"] = e_scale
    end

    println("Saved to $(save_path)\n")

    return norb, nelec, orbsym, energy_nuc, one_body_mo, two_body_mo, e_scale
end


function init_scf(phi_deg::Float64;
    run_cisd::Bool=true,
    run_mp2::Bool=true,
    run_ccsd::Bool=true,
    run_fci::Bool=true,
)
    gto = pyimport("pyscf.gto")
    scf = pyimport("pyscf.scf")
    ci  = pyimport("pyscf.ci")
    cc  = pyimport("pyscf.cc")
    fci = pyimport("pyscf.fci")
    mp  = pyimport("pyscf.mp")

    e_scf   = 0.0
    e_cisd  = 0.0
    e_mp2   = 0.0
    e_ccsd  = 0.0
    e_fci   = 0.0

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
