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
