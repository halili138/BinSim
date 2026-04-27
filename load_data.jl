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


function build(mole::Mole, filepath::Union{String, Nothing}=nothing)
    if isnothing(filepath)
        name  = mole.name
        ratio = mole.ratio
        basis = mole.basis

        filename = "$(basis)/$(name)-$(ratio)-$(basis).jld2"
        filepath = joinpath(jld2path, filename)

        jldopen(filepath, "r") do file
            mole.norb        = file["norb"]
            mole.nelec       = file["nelec"]
            mole.orbsym      = file["orbsym"]
            mole.energy_nuc  = file["energy_nuc"]
            mole.one_body_mo = file["one_body_mo"]
            mole.two_body_mo = file["two_body_mo"]
            mole.e_scale     = file["e_scale"]
        end

        println("Successfully read data from: $(abspath(filepath))")
        println("  name: $(name)")
        println("  ratio: $(ratio)")
        println("  basis: $(basis)")
    else
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
        
        println("Successfully read data from: $(abspath(filepath))")
    end

    norb   = mole.norb
    na, nb = mole.nelec
    ne     = na + nb
    nq     = norb * 2

    @printf("  nα: %d, nβ: %d, ne: %d, norb: %d, nq: %d\n\n", 
    na, nb, ne, norb, nq)
end
