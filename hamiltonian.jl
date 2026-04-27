const LIB_HAM = joinpath(@__DIR__, "src/lib/libham.so")


struct HamResult_f64
    axs_ptr::Ptr{Cvoid}
    azs_ptr::Ptr{Cvoid}
    bxs_ptr::Ptr{Cvoid}
    bzs_ptr::Ptr{Cvoid}
    cs_ptr::Ptr{Float64}
    gs_ptr::Ptr{Int64}
    ncs::Csize_t
    ngs::Csize_t
end


function int2ham_ui64_f64(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{Float64,2},
    two_body_mo::Array{Float64,4},
    tol::Float64,
    based::Int64,
    verbose::Bool,
)
    norb > 31 && error("Maximum supported is (31o, 62q) for uint64 backend.")

    res = @ccall LIB_HAM.generate_hamiltonian_64_128_f64(
        energy_nuc::Cdouble,
        one_body_mo::Ptr{Cdouble},
        two_body_mo::Ptr{Cdouble},
        norb::Cint,
        tol::Cdouble,
        based::Cint,
        verbose::Bool,
    )::HamResult_f64

    axs = unsafe_wrap(Array, Ptr{UInt32}(res.axs_ptr), res.ncs; own=true)
    azs = unsafe_wrap(Array, Ptr{UInt32}(res.azs_ptr), res.ncs; own=true)
    bxs = unsafe_wrap(Array, Ptr{UInt32}(res.bxs_ptr), res.ncs; own=true)
    bzs = unsafe_wrap(Array, Ptr{UInt32}(res.bzs_ptr), res.ncs; own=true)
    cs = unsafe_wrap(Array, res.cs_ptr, res.ncs; own=true)
    gs = unsafe_wrap(Array, res.gs_ptr, res.ngs; own=true)


    return BinaryQubitAABB(axs, azs, bxs, bzs, cs, gs)
end


function int2ham_ui128_f64(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{Float64,2},
    two_body_mo::Array{Float64,4},
    tol::Float64,
    based::Int64,
    verbose::Bool,
)
    norb > 63 && error("Maximum supported is (63o, 126q) for uint128 backend.")

    res = @ccall LIB_HAM.generate_hamiltonian_128_256_f64(
        energy_nuc::Cdouble,
        one_body_mo::Ptr{Cdouble},
        two_body_mo::Ptr{Cdouble},
        norb::Cint,
        tol::Cdouble,
        based::Cint,
        verbose::Bool,
    )::HamResult_f64

    axs = unsafe_wrap(Array, Ptr{UInt64}(res.axs_ptr), res.ncs; own=true)
    azs = unsafe_wrap(Array, Ptr{UInt64}(res.azs_ptr), res.ncs; own=true)
    bxs = unsafe_wrap(Array, Ptr{UInt64}(res.bxs_ptr), res.ncs; own=true)
    bzs = unsafe_wrap(Array, Ptr{UInt64}(res.bzs_ptr), res.ncs; own=true)
    cs = unsafe_wrap(Array, res.cs_ptr, res.ncs; own=true)
    gs = unsafe_wrap(Array, res.gs_ptr, res.ngs; own=true)

    return BinaryQubitAABB(axs, azs, bxs, bzs, cs, gs)
end


function int2ham_ui256_f64(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{Float64,2},
    two_body_mo::Array{Float64,4},
    tol::Float64,
    based::Int64,
    verbose::Bool,
)
    norb > 127 && error("Maximum supported is (127o, 254q) for uint256 backend.")

    res = @ccall LIB_HAM.generate_hamiltonian_256_512_f64(
        energy_nuc::Cdouble,
        one_body_mo::Ptr{Cdouble},
        two_body_mo::Ptr{Cdouble},
        norb::Cint,
        tol::Cdouble,
        based::Cint,
        verbose::Bool,
    )::HamResult_f64

    axs = unsafe_wrap(Array, Ptr{UInt128}(res.axs_ptr), res.ncs; own=true)
    azs = unsafe_wrap(Array, Ptr{UInt128}(res.azs_ptr), res.ncs; own=true)
    bxs = unsafe_wrap(Array, Ptr{UInt128}(res.bxs_ptr), res.ncs; own=true)
    bzs = unsafe_wrap(Array, Ptr{UInt128}(res.bzs_ptr), res.ncs; own=true)
    cs = unsafe_wrap(Array, res.cs_ptr, res.ncs; own=true)
    gs = unsafe_wrap(Array, res.gs_ptr, res.ngs; own=true)

    return BinaryQubitAABB(axs, azs, bxs, bzs, cs, gs)
end


function JW_hamiltonian(mole::Mole; tol::Float64=1e-12, based::Int64=0, spin::String="aabb", verbose::Bool=false)
    if 0 <= mole.norb < 32
        Haabb = int2ham_ui64_f64(
            mole.norb, mole.energy_nuc, mole.one_body_mo, mole.two_body_mo, tol, based, verbose,
        )
    elseif 32 <= mole.norb < 64
        Haabb = int2ham_ui128_f64(
            mole.norb, mole.energy_nuc, mole.one_body_mo, mole.two_body_mo, tol, based, verbose,
        )
    elseif 64 <= mole.norb < 128
        return int2ham_ui256_f64(
            mole.norb, mole.energy_nuc, mole.one_body_mo, mole.two_body_mo, tol, based, verbose,
        )
    else
        error("Maximum supported is (127o, 254q)")
    end

    println("  ngs: $(length(Haabb.gs)-1)")
    println("  ncs: $(length(Haabb.cs))\n")

    if spin == "aabb"
        return Haabb
    elseif spin == "abab"  
        return BinaryQubit(Haabb)
    else
        throw(ArgumentError("Undefined spin: $(spin)"))
    end
end


function JW_hamiltonian(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{Float64,2},
    two_body_mo::Array{Float64,4};
    tol::Float64=1e-12,
    based::Int64=0,
    spin::String="aabb",
    verbose::Bool=false,
)
    if 0 <= norb < 32
        Haabb = int2ham_ui64_f64(
            norb, energy_nuc, one_body_mo, two_body_mo, tol, based, verbose,
        )
    elseif 32 <= norb < 64
        Haabb = int2ham_ui128_f64(
            norb, energy_nuc, one_body_mo, two_body_mo, tol, based, verbose,
        )
    elseif 64 <= norb < 128
        return int2ham_ui256_f64(
            norb, energy_nuc, one_body_mo, two_body_mo, tol, based, verbose,
        )
    else
        error("Maximum supported is (127o, 254q)")
    end

    # println("  ngs: $(length(Haabb.gs)-1)")
    # println("  ncs: $(length(Haabb.cs))\n")

    if spin == "aabb"
        return Haabb
    elseif spin == "abab"  
        return BinaryQubit(Haabb)
    else
        throw(ArgumentError("Undefined spin: $(spin)"))
    end
end


