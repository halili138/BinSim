abstract type SysInfo end


mutable struct Mole <: SysInfo
    name::String
    ratio::Float64
    basis::String
    norb::Int
    nelec::Tuple{Int,Int}
    orbsym::Array{Int,1}
    energy_nuc::Float64
    one_body_mo::Array{Float64,2}
    two_body_mo::Array{Float64,4}
    e_scale::Float64
end


function Mole()
    name        = ""
    ratio       = 1.0
    basis       = ""
    norb        = 0
    nelec       = (0, 0)
    orbsym      = Int[]
    energy_nuc  = 0.0
    one_body_mo = Array{Float64,2}(undef, 0, 0)
    two_body_mo = Array{Float64,4}(undef, 0, 0, 0, 0)
    e_scale     = 0.0

    Mole(
        name,
        ratio,
        basis,
        norb,
        nelec,
        orbsym,
        energy_nuc,
        one_body_mo,
        two_body_mo,
        e_scale,
    )
end


function build(mole::Mole, filepath::String="")
    if isempty(filepath)
        name  = mole.name
        ratio = mole.ratio
        basis = mole.basis

        filename = "$(basis)/$(name)-$(ratio)-$(basis).jld2"
        filepath = joinpath(jld2path, filename)

        try
            jldopen(filepath, "r") do file
                mole.norb        = file["norb"]
                mole.nelec       = file["nelec"]
                mole.orbsym      = file["orbsym"]
                mole.energy_nuc  = file["energy_nuc"]
                mole.one_body_mo = file["one_body_mo"]
                mole.two_body_mo = file["two_body_mo"]
                mole.e_scale     = file["e_scale"]
            end

            if is_rank0_or_serial()
                println("Successfully read data from: $(abspath(filepath))")
                println("  name: $(name)")
                println("  ratio: $(ratio)")
                println("  basis: $(basis)")
            end
        catch
            pushfirst!(pyimport("sys")."path", pypath)
            pyfun = pyimport("mole_pbc_int")

            mole.norb, mole.nelec, mole.orbsym, mole.energy_nuc, 
            mole.one_body_mo, mole.two_body_mo, mole.e_scale = init_scf(
                pyfun, mole.name, mole.ratio, mole.basis, filepath, run_fci=false)
        end
    else
        try
            filepath = joinpath(jld2path, filepath)
            jldopen(filepath, "r") do file
                mole.norb        = file["norb"]
                mole.nelec       = file["nelec"]
                mole.orbsym      = file["orbsym"]
                mole.energy_nuc  = file["energy_nuc"]
                mole.one_body_mo = file["one_body_mo"]
                mole.two_body_mo = file["two_body_mo"]
                mole.e_scale     = file["e_scale"]
            end
            
            if is_rank0_or_serial()
                println("Successfully read data from: $(abspath(filepath))")
            end
        catch
            error("You need to run your algorithm to and save the system information into $(jld2file)")
        end
    end

    norb   = mole.norb
    na, nb = mole.nelec
    ne     = na + nb
    nq     = norb * 2

    if is_rank0_or_serial()
        @printf("  nα: %d, nβ: %d, ne: %d, norb: %d, nq: %d\n\n", 
        na, nb, ne, norb, nq)
    end
end

mutable struct Pbc <: SysInfo
    name::String
    ratio::Float64
    basis::String
    pseudo::String
    mesh::Array{Int,1}
    scaled_center::Array{Int,1}
    norb::Int
    nelec::Tuple{Int,Int}
    orbsym::Array{Int64,1}
    kconserv::Array{Int64,3}
    energy_nuc::Float64
    one_body_mo::Array{ComplexF64,2}
    two_body_mo::Array{ComplexF64,4}
    e_scale::Float64
end

function Pbc()
    name          = ""
    ratio         = 1.0
    basis         = ""
    pseudo        = ""
    mesh          = Int[]
    scaled_center = Int[]
    norb          = 0
    nelec         = (0, 0)
    orbsym        = Int[]
    kconserv      = Array{Int,3}(undef, 0, 0, 0)
    energy_nuc    = 0.0
    one_body_mo   = Array{ComplexF64,2}(undef, 0, 0)
    two_body_mo   = Array{ComplexF64,4}(undef, 0, 0, 0, 0)
    e_scale       = 0.0

    Pbc(
        name,
        ratio,
        basis,
        pseudo,
        mesh,
        scaled_center,
        norb, 
        nelec,
        orbsym,
        kconserv,
        energy_nuc,
        one_body_mo,
        two_body_mo,
        e_scale,
    )
end

function build(pbc::Pbc, filepath::String="")
    if isempty(filepath)
        name          = pbc.name
        ratio         = pbc.ratio
        basis         = pbc.basis
        pseudo        = pbc.pseudo
        mesh          = pbc.mesh
        scaled_center = pbc.scaled_center

        filename = "pbc/$(name)-$(ratio)-$(basis)-$(pseudo)-$(mesh)-$(scaled_center).jld2"
        filepath = joinpath(jld2path, filename)

        jldopen(filepath, "r") do file
            pbc.norb        = file["norb"]
            pbc.nelec       = file["nelec"]
            pbc.energy_nuc  = file["energy_nuc"]
            pbc.one_body_mo = file["one_body_mo"]
            pbc.two_body_mo = file["two_body_mo"]
            pbc.kconserv    = file["kconserv"]
            pbc.e_scale     = file["e_scale"]
        end

        if is_rank0_or_serial()
            println("Successfully read data from: $(abspath(filepath))")
            println("  name: $(name)")
            println("  ratio: $(ratio)")
            println("  basis: $(basis)")
            println("  pseudo: $(pseudo)")
            println("  mesh: $(mesh)")
            println("  scaled_center: $(scaled_center)")
        end
    else
        jldopen(filepath, "r") do file
            pbc.norb        = file["norb"]
            pbc.nelec       = file["nelec"]
            pbc.energy_nuc  = file["energy_nuc"]
            pbc.one_body_mo = file["one_body_mo"]
            pbc.two_body_mo = file["two_body_mo"]
            pbc.kconserv    = file["kconserv"]
            pbc.e_scale     = file["e_scale"]
        end

        if is_rank0_or_serial()
            println("Successfully read data from: $(abspath(filepath))")
        end
    end

    norb   = pbc.norb
    na, nb = pbc.nelec
    ne     = na + nb
    nq     = norb * 2

    if is_rank0_or_serial()
        @printf("  nα: %d, nβ: %d, ne: %d, norb: %d, nq: %d\n\n", 
        na, nb, ne, norb, nq)
    end
end
