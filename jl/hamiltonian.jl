struct HamResult_f64
    axs_ptr::Ptr{Cvoid}
    azs_ptr::Ptr{Cvoid}
    bxs_ptr::Ptr{Cvoid}
    bzs_ptr::Ptr{Cvoid}
    cs_ptr::Ptr{Float64}
    ncs::Csize_t
end


function int2ham_ui64_f64(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{Float64,2},
    two_body_mo::Array{Float64,4},
    tol::Float64,
    verbose::Bool,
)
    norb > 31 && error("Maximum supported is (31o, 62q) for uint64 backend.")

    res = @ccall LIB_HAM.generate_hamiltonian_64_128_f64(
        energy_nuc::Cdouble,
        one_body_mo::Ptr{Cdouble},
        two_body_mo::Ptr{Cdouble},
        norb::Cint,
        tol::Cdouble,
        verbose::Bool,
    )::HamResult_f64

    axs = unsafe_wrap(Array, Ptr{UInt32}(res.axs_ptr), res.ncs; own=true)
    azs = unsafe_wrap(Array, Ptr{UInt32}(res.azs_ptr), res.ncs; own=true)
    bxs = unsafe_wrap(Array, Ptr{UInt32}(res.bxs_ptr), res.ncs; own=true)
    bzs = unsafe_wrap(Array, Ptr{UInt32}(res.bzs_ptr), res.ncs; own=true)
    cs = unsafe_wrap(Array, res.cs_ptr, res.ncs; own=true)

    return BinaryQubitAABB(axs, bxs, azs, bzs, cs)
end


function int2ham_ui128_f64(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{Float64,2},
    two_body_mo::Array{Float64,4},
    tol::Float64,
    verbose::Bool,
)
    norb > 63 && error("Maximum supported is (63o, 126q) for uint128 backend.")

    res = @ccall LIB_HAM.generate_hamiltonian_128_256_f64(
        energy_nuc::Cdouble,
        one_body_mo::Ptr{Cdouble},
        two_body_mo::Ptr{Cdouble},
        norb::Cint,
        tol::Cdouble,
        verbose::Bool,
    )::HamResult_f64

    axs = unsafe_wrap(Array, Ptr{UInt64}(res.axs_ptr), res.ncs; own=true)
    azs = unsafe_wrap(Array, Ptr{UInt64}(res.azs_ptr), res.ncs; own=true)
    bxs = unsafe_wrap(Array, Ptr{UInt64}(res.bxs_ptr), res.ncs; own=true)
    bzs = unsafe_wrap(Array, Ptr{UInt64}(res.bzs_ptr), res.ncs; own=true)
    cs = unsafe_wrap(Array, res.cs_ptr, res.ncs; own=true)

    return BinaryQubitAABB(axs, bxs, azs, bzs, cs)
end


function int2ham_ui256_f64(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{Float64,2},
    two_body_mo::Array{Float64,4},
    tol::Float64,
    verbose::Bool,
)
    norb > 127 && error("Maximum supported is (127o, 254q) for uint256 backend.")

    res = @ccall LIB_HAM.generate_hamiltonian_256_512_f64(
        energy_nuc::Cdouble,
        one_body_mo::Ptr{Cdouble},
        two_body_mo::Ptr{Cdouble},
        norb::Cint,
        tol::Cdouble,
        verbose::Bool,
    )::HamResult_f64

    axs = unsafe_wrap(Array, Ptr{UInt128}(res.axs_ptr), res.ncs; own=true)
    azs = unsafe_wrap(Array, Ptr{UInt128}(res.azs_ptr), res.ncs; own=true)
    bxs = unsafe_wrap(Array, Ptr{UInt128}(res.bxs_ptr), res.ncs; own=true)
    bzs = unsafe_wrap(Array, Ptr{UInt128}(res.bzs_ptr), res.ncs; own=true)
    cs = unsafe_wrap(Array, res.cs_ptr, res.ncs; own=true)

    return BinaryQubitAABB(axs, bxs, azs, bzs, cs)
end


function JW_hamiltonian(
    mole::Mole; 
    tol::Float64=1e-12, 
    spin::String="aabb", 
    verbose::Bool=false,
)
    return _JW_hamiltonian(
        mole.norb, mole.energy_nuc, mole.one_body_mo, mole.two_body_mo,
        tol=tol,
        spin=spin,
        verbose=verbose,
    )
end


function _JW_hamiltonian(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{Float64,2},
    two_body_mo::Array{Float64,4};
    tol::Float64=1e-12,
    spin::String="aabb",
    verbose::Bool=false,
)
    if 0 <= norb < 32
        Haabb = int2ham_ui64_f64(norb, energy_nuc, one_body_mo, two_body_mo, tol, verbose)
    elseif 32 <= norb < 64
        Haabb = int2ham_ui128_f64(norb, energy_nuc, one_body_mo, two_body_mo, tol, verbose)
    elseif 64 <= norb < 128
        Haabb = int2ham_ui256_f64(norb, energy_nuc, one_body_mo, two_body_mo, tol, verbose)
    else
        error("Maximum supported is (127o, 254q)")
    end

    gs = get_bounds_1based(Haabb.axs, Haabb.bxs)
    
    println("  ngs: $(length(gs)-1)")
    println("  ncs: $(length(Haabb.cs))\n")

    if spin == "aabb"
        return Haabb
    elseif spin == "abab"
        xs = unzip_even_bit.(Haabb.axs) .| unzip_odd_bit.(Haabb.bxs)
        zs = unzip_even_bit.(Haabb.azs) .| unzip_odd_bit.(Haabb.bzs)  
        return BinaryQubitABAB(xs, zs, Haabb.cs)
    else
        throw(ArgumentError("Undefined spin: $(spin)"))
    end
end


struct HamResult_c64
    axs_ptr::Ptr{Cvoid}
    azs_ptr::Ptr{Cvoid}
    bxs_ptr::Ptr{Cvoid}
    bzs_ptr::Ptr{Cvoid}
    cs_ptr::Ptr{ComplexF64}
    ncs::Csize_t
end


function int2ham_ui64_c64(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{ComplexF64,2},
    two_body_mo::Array{ComplexF64,4},
    tol::Float64,
    verbose::Bool,
)
    norb > 31 && error("Maximum supported is (31o, 62q) for uint64 backend.")

    res = @ccall LIB_HAM.generate_hamiltonian_64_128_c64(
        energy_nuc::Cdouble,
        one_body_mo::Ptr{ComplexF64},
        two_body_mo::Ptr{ComplexF64},
        norb::Cint,
        tol::Cdouble,
        verbose::Bool,
    )::HamResult_c64

    axs = unsafe_wrap(Array, Ptr{UInt32}(res.axs_ptr), res.ncs; own=true)
    azs = unsafe_wrap(Array, Ptr{UInt32}(res.azs_ptr), res.ncs; own=true)
    bxs = unsafe_wrap(Array, Ptr{UInt32}(res.bxs_ptr), res.ncs; own=true)
    bzs = unsafe_wrap(Array, Ptr{UInt32}(res.bzs_ptr), res.ncs; own=true)
    cs = unsafe_wrap(Array, Ptr{ComplexF64}(res.cs_ptr), res.ncs; own=true)

    return BinaryQubitAABB(axs, bxs, azs, bzs, cs)
end


function int2ham_ui128_c64(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{ComplexF64,2},
    two_body_mo::Array{ComplexF64,4},
    tol::Float64,
    verbose::Bool,
)
    norb > 63 && error("Maximum supported is (63o, 126q) for uint128 backend.")

    res = @ccall LIB_HAM.generate_hamiltonian_128_256_c64(
        energy_nuc::Cdouble,
        one_body_mo::Ptr{ComplexF64},
        two_body_mo::Ptr{ComplexF64},
        norb::Cint,
        tol::Cdouble,
        verbose::Bool,
    )::HamResult_c64

    axs = unsafe_wrap(Array, Ptr{UInt64}(res.axs_ptr), res.ncs; own=true)
    azs = unsafe_wrap(Array, Ptr{UInt64}(res.azs_ptr), res.ncs; own=true)
    bxs = unsafe_wrap(Array, Ptr{UInt64}(res.bxs_ptr), res.ncs; own=true)
    bzs = unsafe_wrap(Array, Ptr{UInt64}(res.bzs_ptr), res.ncs; own=true)
    cs = unsafe_wrap(Array, Ptr{ComplexF64}(res.cs_ptr), res.ncs; own=true)

    return BinaryQubitAABB(axs, bxs, azs, bzs, cs)
end


function int2ham_ui256_c64(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{ComplexF64,2},
    two_body_mo::Array{ComplexF64,4},
    tol::Float64,
    verbose::Bool,
)
    norb > 127 && error("Maximum supported is (127o, 254q) for uint256 backend.")

    res = @ccall LIB_HAM.generate_hamiltonian_256_512_c64(
        energy_nuc::Cdouble,
        one_body_mo::Ptr{ComplexF64},
        two_body_mo::Ptr{ComplexF64},
        norb::Cint,
        tol::Cdouble,
        verbose::Bool,
    )::HamResult_c64

    axs = unsafe_wrap(Array, Ptr{UInt128}(res.axs_ptr), res.ncs; own=true)
    azs = unsafe_wrap(Array, Ptr{UInt128}(res.azs_ptr), res.ncs; own=true)
    bxs = unsafe_wrap(Array, Ptr{UInt128}(res.bxs_ptr), res.ncs; own=true)
    bzs = unsafe_wrap(Array, Ptr{UInt128}(res.bzs_ptr), res.ncs; own=true)
    cs = unsafe_wrap(Array, Ptr{ComplexF64}(res.cs_ptr), res.ncs; own=true)

    return BinaryQubitAABB(axs, bxs, azs, bzs, cs)
end


function JW_hamiltonian(
    pbc::Pbc; 
    tol::Float64=1e-12, 
    spin::String="aabb", 
    verbose::Bool=false,
)
    return _JW_hamiltonian(
        pbc.norb, pbc.energy_nuc, pbc.one_body_mo, pbc.two_body_mo,
        tol=tol,
        spin=spin,
        verbose=verbose,
    )
end


function _JW_hamiltonian(
    norb::Int64,
    energy_nuc::Float64,
    one_body_mo::Array{ComplexF64,2},
    two_body_mo::Array{ComplexF64,4};
    tol::Float64=1e-12,
    spin::String="aabb",
    verbose::Bool=false,
)
    if 0 <= norb < 32
        Haabb = int2ham_ui64_c64(norb, energy_nuc, one_body_mo, two_body_mo, tol, verbose)
    elseif 32 <= norb < 64
        Haabb = int2ham_ui128_c64(norb, energy_nuc, one_body_mo, two_body_mo, tol, verbose)
    elseif 64 <= norb < 128
        Haabb = int2ham_ui256_c64(norb, energy_nuc, one_body_mo, two_body_mo, tol, verbose)
    else
        error("Maximum supported is (127o, 254q)")
    end

    gs = get_bounds_1based(Haabb.axs, Haabb.bxs)

    println("  ngs: $(length(gs)-1)")
    println("  ncs: $(length(Haabb.cs))\n")

    if spin == "aabb"
        return Haabb
    elseif spin == "abab"
        xs = unzip_even_bit.(Haabb.axs) .| unzip_odd_bit.(Haabb.bxs)
        zs = unzip_even_bit.(Haabb.azs) .| unzip_odd_bit.(Haabb.bzs)  
        return BinaryQubitABAB(xs, zs, Haabb.cs)
    else
        throw(ArgumentError("Undefined spin: $(spin)"))
    end
end


function quantum_operator_aabb(norb::Int, Ti::Type, Tv::Type)
    nq::Int = norb * 2

    N  = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}()
    Sx = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}()
    Sy = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}()
    Sz = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}()

    for i in 0:nq-1
        N += FermionOperatorAABB([(i, 1), (i, 0)], 1.0, Ti, Tv)
    end
    for i in 0:norb-1
        ia = 2i
        ib = 2i + 1
        Sx += FermionOperatorAABB([(ia, 1), (ib, 0)], 0.5, Ti, Tv) + FermionOperatorAABB([(ib, 1), (ia, 0)], 0.5, Ti, Tv)
        Sy += FermionOperatorAABB([(ia, 1), (ib, 0)], 0.5, Ti, Tv) + FermionOperatorAABB([(ib, 1), (ia, 0)], -0.5, Ti, Tv)
        Sz += FermionOperatorAABB([(ia, 1), (ia, 0)], 0.5, Ti, Tv) + FermionOperatorAABB([(ib, 1), (ib, 0)], -0.5, Ti, Tv)
    end

    S2 = Sx^2 - Sy^2 + Sz^2

    return N, S2, Sz
end


function apply_constraint(
    H0b::BinaryQubitAABB, norb::Int, nelec::Tuple{Int,Int}, constr_c::NTuple{3,Float64}
)
    N, S2, Sz = quantum_operator_aabb(norb, eltype(H0b.axs), eltype(H0b.cs))

    return linearcombine([H0b, (N-sum(nelec))^2, S2, Sz], [1.0, constr_c...], 0.0, 1e-12)
end


function ising_module(nq::Int64, J::Float64=1.0, h::Float64=0.5; 
    Ti::DataType=UInt32, Tv::DataType=Float64, is_pbc::Bool=false)

    ops = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    cs  = Tv[]

    for i in 0:nq-2
        push!(ops, QubitOperatorAABB([(i, "Z"), (i+1, "Z")], -J, Ti, Tv))
        push!(cs, 1)
    end

    if is_pbc
        push!(ops, QubitOperatorAABB([(nq-1, "Z"), (0, "Z")], -J, Ti, Tv))
        push!(cs, 1)
    end

    for i in 0:nq-1
        push!(ops, QubitOperatorAABB([(i, "X")], -h, Ti, Tv))
        push!(cs, 1)
    end

    return linearcombine(ops, cs, 0.0, 1e-12)
end


function heisenberg_module(nq::Int, jx::Float64=1.0, jy::Float64=1.0, jz::Float64=1.0;
    Ti::DataType=UInt32, Tv::DataType=Float64, is_pbc::Bool=false)

    ops = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    cs  = Tv[]

    for i in 0:nq-2
        push!(ops, QubitOperatorAABB([(i, "X"), (i+1, "X")], jx, Ti, Tv))
        push!(ops, QubitOperatorAABB([(i, "Y"), (i+1, "Y")], jy, Ti, Tv))
        push!(ops, QubitOperatorAABB([(i, "Z"), (i+1, "Z")], jz, Ti, Tv))
        append!(cs, Tv[1, 1, 1])
    end

    if is_pbc
        push!(ops, QubitOperatorAABB([(nq-1, "X"), (0, "X")], jx, Ti, Tv))
        push!(ops, QubitOperatorAABB([(nq-1, "Y"), (0, "Y")], jy, Ti, Tv))
        push!(ops, QubitOperatorAABB([(nq-1, "Z"), (0, "Z")], jz, Ti, Tv))
        append!(cs, Tv[1, 1, 1])
    end

    return linearcombine(ops, cs, 0.0, 1e-12)
end


function ising_pool(nq::Int64; 
    Ti::DataType=UInt32, Tv::DataType=Float64, 
    is_pbc::Bool=false, pool_type::String="local")
    
    # 算符池是一个由单一 Pauli 字符串构成的数组，不需要 linearcombine
    pool = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    
    # ==========================================
    # 1. 单体算符 (Weight-1)
    # 必须包含 1 个 Y。由 [ZZ, X] 原始对易子产生。
    # ==========================================
    for i in 0:nq-1
        push!(pool, QubitOperatorAABB([(i, "Y")], 1.0, Ti, Tv))
    end

    # ==========================================
    # 2. 双体算符 (Weight-2)
    # 必须包含 1 个 Y 和 1 个非 Y (Z 或 X)，保证总 Y 数量为奇数
    # ==========================================
    
    # 根据用户选择，决定是只用近邻(local)还是全连接(all2all)
    pairs = Tuple{Int, Int}[]
    if pool_type == "local"
        for i in 0:nq-2
            push!(pairs, (i, i+1))
        end
        if is_pbc
            push!(pairs, (nq-1, 0))
        end
    elseif pool_type == "all2all"
        for i in 0:nq-1
            for j in i+1:nq-1
                push!(pairs, (i, j))
            end
        end
    end

    for (i, j) in pairs
        # ZY 和 YZ 组合：通常在 TFIM 中贡献最大的双体梯度
        push!(pool, QubitOperatorAABB([(i, "Z"), (j, "Y")], 1.0, Ti, Tv))
        push!(pool, QubitOperatorAABB([(i, "Y"), (j, "Z")], 1.0, Ti, Tv))
        
        # XY 和 YX 组合：为了进一步增加算符池的表达能力 (过完备性补充)
        push!(pool, QubitOperatorAABB([(i, "X"), (j, "Y")], 1.0, Ti, Tv))
        push!(pool, QubitOperatorAABB([(i, "Y"), (j, "X")], 1.0, Ti, Tv))
    end

    println("Size of $(pool_type) operator pool: $(length(pool))")

    return pool
end


function heisenberg_pool(nq::Int64; 
    Ti::DataType=UInt32, Tv::DataType=Float64, 
    is_pbc::Bool=false, pool_type::String="local")
    
    pool = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]

    if pool_type == "local"
        for i in 0:nq-1
            push!(pool, QubitOperatorAABB([(i, "X")], -im, Ti, Tv))
            push!(pool, QubitOperatorAABB([(i, "Y")], -im, Ti, Tv))
            push!(pool, QubitOperatorAABB([(i, "Z")], -im, Ti, Tv))
        end
        for i in 0:nq-2 
            push!(pool, QubitOperatorAABB([(i, "X"), (i+1, "X")], -im, Ti, Tv))
            push!(pool, QubitOperatorAABB([(i, "Y"), (i+1, "Y")], -im, Ti, Tv))
            push!(pool, QubitOperatorAABB([(i, "Z"), (i+1, "Z")], -im, Ti, Tv))
        end
        if is_pbc
            push!(pool, QubitOperatorAABB([(nq-1, "X"), (0, "X")], -im, Ti, Tv))
            push!(pool, QubitOperatorAABB([(nq-1, "Y"), (0, "Y")], -im, Ti, Tv))
            push!(pool, QubitOperatorAABB([(nq-1, "Z"), (0, "Z")], -im, Ti, Tv))
        end
    else
        for i in 0:nq-1
            push!(pool, QubitOperatorAABB([(i, "X")], -im, Ti, Tv))
            push!(pool, QubitOperatorAABB([(i, "Y")], -im, Ti, Tv))
            push!(pool, QubitOperatorAABB([(i, "Z")], -im, Ti, Tv))
        end
        for i in 0:nq-2 
            for j in i+1:nq-1
                push!(pool, QubitOperatorAABB([(i, "X"), (j, "X")], -im, Ti, Tv))
                push!(pool, QubitOperatorAABB([(i, "Y"), (j, "Y")], -im, Ti, Tv))
                push!(pool, QubitOperatorAABB([(i, "Z"), (j, "Z")], -im, Ti, Tv))
            end
        end
    end

    println("Size of $(pool_type) operator pool: $(length(pool))")
    
    return pool
end

