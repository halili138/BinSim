struct PauliStingAABB{Ti,Tv}
    ax::Ti
    bx::Ti
    az::Ti
    bz::Ti
    c::Tv
end

@inline encode(p::PauliStingAABB) = encode(unzip_even_bit(p.ax) | unzip_odd_bit(p.bx), unzip_even_bit(p.az) | unzip_odd_bit(p.bz))

Base.:isless(p1::PauliStingAABB, p2::PauliStingAABB) = isless(encode(p1), encode(p2))

struct BinaryQubitAABB{Ti,Tv,K<:AbstractArray{Ti,1},V<:AbstractArray{Tv,1}} <: AbstractArray{PauliStingAABB{Ti,Tv},1}
    axs::K
    bxs::K
    azs::K
    bzs::K
    cs::V
end

Base.length(A::BinaryQubitAABB) = length(A.cs)
Base.size(A::BinaryQubitAABB) = size(A.cs)

function Base.:getindex(A::BinaryQubitAABB, i...)
    PauliStingAABB(
        getindex(A.axs, i...), 
        getindex(A.bxs, i...), 
        getindex(A.azs, i...), 
        getindex(A.bzs, i...), 
        getindex(A.cs, i...)
    )
end

function Base.:setindex!(A::BinaryQubitAABB, p::PauliStingAABB, i...)
    setindex!(A.axs, p.ax, i...)
    setindex!(A.bxs, p.bx, i...)
    setindex!(A.azs, p.az, i...)
    setindex!(A.bzs, p.bz, i...)
    setindex!(A.cs, p.c, i...)
    A
end

function Base.view(A::BinaryQubitAABB, inds...)
    BinaryQubitAABB(
        view(A.axs, inds...), 
        view(A.bxs, inds...),
        view(A.azs, inds...), 
        view(A.bzs, inds...),
        view(A.cs, inds...),
    )
end

function Base.similar(A::BinaryQubitAABB)
    BinaryQubitAABB(
        similar(A.axs), 
        similar(A.bxs),
        similar(A.azs),
        similar(A.bzs),
        similar(A.cs),
    )
end

function Base.similar(A::BinaryQubitAABB, dim::Dims)
    BinaryQubitAABB(
        similar(A.axs, dim), 
        similar(A.bxs, dim),
        similar(A.azs, dim),
        similar(A.bzs, dim),
        similar(A.cs, dim),
    )
end

function Base.similar(A::BinaryQubitAABB, dims::Integer...)
    BinaryQubitAABB(
        similar(A.axs, dims), 
        similar(A.bxs, dims),
        similar(A.azs, dims),
        similar(A.bzs, dims),
        similar(A.cs, dims),
    )
end

function Base.resize!(A::BinaryQubitAABB, n::Integer)
    resize!(A.axs, n)
    resize!(A.bxs, n)
    resize!(A.azs, n)
    resize!(A.bzs, n)
    resize!(A.cs, n)
    A
end

function BinaryQubitAABB{Ti,Tv,K,V}() where {Ti,Tv,K,V}
    BinaryQubitAABB(K(), K(), K(), K(), V())
end

function Base.zero(::Type{BinaryQubitAABB{Ti,Tv,K,V}}) where {Ti,Tv,K,V}
    BinaryQubitAABB{Ti,Tv,K,V}()
end

Base.zero(A::BinaryQubitAABB) = zero(typeof(A))

function Base.one(::Type{BinaryQubitAABB{Ti,Tv,K,V}}) where {Ti,Tv,K,V}
    BinaryQubitAABB(K([0]), K([0]), K([0]), K([0]), V([1]))
end

Base.one(A::BinaryQubitAABB) = one(typeof(A))

function trunc_by_cs!(A::BinaryQubitAABB, tol::Float64)
    axs = A.axs
    bxs = A.bxs
    azs = A.azs
    bzs = A.bzs
    cs = A.cs

    count = 0
    for (i, c) in enumerate(cs)
        if abs(c) > tol
            count += 1
            axs[count] = axs[i]
            bxs[count] = bxs[i]
            azs[count] = azs[i]
            bzs[count] = bzs[i]
            cs[count] = c
        end
    end

    return count
end

function reduce_by_xz!(A::BinaryQubitAABB, tol::Float64)
    len = length(A)

    iszero(len) && return 0 

    axs = A.axs
    bxs = A.bxs
    azs = A.azs
    bzs = A.bzs
    cs = A.cs

    l = 1
    for r in 2:len
        axl = axs[l]
        bxl = bxs[l]
        azl = azs[l]
        bzl = bzs[l]
        cl  = cs[l]
        
        axr = axs[r]
        bxr = bxs[r]
        azr = azs[r]
        bzr = bzs[r]
        cr  = cs[r]

        ql = encode(unzip_even_bit(axl) | unzip_odd_bit(bxl), unzip_even_bit(azl) | unzip_odd_bit(bzl))
        qr = encode(unzip_even_bit(axr) | unzip_odd_bit(bxr), unzip_even_bit(azr) | unzip_odd_bit(bzr))
        
        if qr == ql
            cs[l] = cl + cr
        else
            if abs(cl) > tol
                l += 1
            end
            
            if l < r
                axs[l] = axr
                bxs[l] = bxr
                azs[l] = azr
                bzs[l] = bzr
                cs[l]  = cr
            end
        end
    end

    if abs(cs[l]) < tol
        l -= 1
    end

    return l
end

function Base.:*(A::BinaryQubitAABB, c::Number)
    res = deepcopy(A)
    res.cs .*= c
    res
end

Base.:*(c::Number, A::BinaryQubitAABB) = A * c

Base.:-(A::BinaryQubitAABB) = A * (-1)

function Base.:adjoint(A::BinaryQubitAABB)
    res = deepcopy(A)
    res.cs .= conj.(A.cs)
    res.cs .*= ((-1) .^ (count_ones.(A.axs .& A.azs) .+ count_ones.(A.bxs .& A.bzs)))
    res
end

function Base.rand(::Type{BinaryQubitAABB{Ti,Tv,K,V}}, nq::Int, nterms::Int) where {Ti,Tv,K,V}
    dim = Ti(1) << (nq ÷ 2)

    axs::Vector{Ti} = rand(0:dim-1, nterms)
    azs::Vector{Ti} = rand(0:dim-1, nterms)
    bxs::Vector{Ti} = rand(0:dim-1, nterms)
    bzs::Vector{Ti} = rand(0:dim-1, nterms)
    cs::Vector{Tv} = rand(Tv, nterms)

    res = BinaryQubitAABB(axs, bxs, azs, bzs, cs)

    sort!(res)

    resize!(res, reduce_by_xz!(res, 1e-12))

    return res
end

function Base.:show(io::IO, p::PauliStingAABB)
    x = unzip_even_bit(p.ax) | unzip_odd_bit(p.bx)
    z = unzip_even_bit(p.az) | unzip_odd_bit(p.bz)
    str = bin_to_pstr(x, z, p.c)
    println(io, str)
end

function Base.:(==)(A::BinaryQubitAABB, B::BinaryQubitAABB)
    A.axs ≠ B.axs && return false
    A.azs ≠ B.azs && return false
    A.bxs ≠ B.bxs && return false
    A.bzs ≠ B.bzs && return false

    is_zero1 = all(iszero.(A.cs))
    is_zero2 = all(iszero.(B.cs))

    is_zero1 && return is_zero2

    ratio = A.cs ./ B.cs

    c = first(ratio)

    return all(x -> isapprox(x, c; rtol=1e-12), ratio)
end

function Base.hash(A::BinaryQubitAABB, h::UInt)
    if all(iszero.(A.cs))
        return hash(0, h)
    end

    first_nz = findfirst(!iszero, A.cs)
    normalized = A.cs ./ A.cs[first_nz]

    return hash((unzip_even_bit.(A.axs) .| unzip_odd_bit.(A.bxs), normalized), h)
end

function linearcombine(
    As::Array{BinaryQubitAABB{Ti,Tv,K,V},1}, 
    Cs::Array{<:Number,1}, 
    C::Number, 
    tol::Float64,
) where {Ti,Tv,K,V}

    @assert length(As) == length(Cs)

    length(As) == 0 && return one(eltype(As)) * C

    ncs = 0
    for A in As
        ncs += length(A.cs)
    end

    !iszero(C) && (ncs += 1)

    res_axs = Vector{Ti}(undef, ncs)
    res_bxs = Vector{Ti}(undef, ncs)
    res_azs = Vector{Ti}(undef, ncs)
    res_bzs = Vector{Ti}(undef, ncs)
    res_cs = Vector{Tv}(undef, ncs)

    res = BinaryQubitAABB(res_axs, res_bxs, res_azs, res_bzs, res_cs)

    count = 0
    if !iszero(C)
        count += 1
        res_axs[count] = 0
        res_bxs[count] = 0
        res_azs[count] = 0
        res_bzs[count] = 0
        res_cs[count] = C
    end

    for (i, A) in enumerate(As)
        coeff = Cs[i]
        axs = A.axs
        bxs = A.bxs
        azs = A.azs
        bzs = A.bzs
        cs = A.cs
        for j in eachindex(cs)
            count += 1
            res_axs[count] = axs[j]
            res_bxs[count] = bxs[j]
            res_azs[count] = azs[j]
            res_bzs[count] = bzs[j]
            res_cs[count] = cs[j] * coeff
        end
    end

    sort!(res)

    resize!(res, reduce_by_xz!(res, tol))

    return res
end

function multiply(
    As::Array{BinaryQubitAABB{Ti,Tv,K,V},1}, tol1::Float64, tol2::Float64,
) where {Ti,Tv,K,V}

    length(As) == 0 && return zero(eltype(As))

    axs1 = Ti[0]
    bxs1 = Ti[0]
    azs1 = Ti[0]
    bzs1 = Ti[0]
    cs1 = Tv[1]
    
    for A in As
        axs2 = A.axs
        bxs2 = A.bxs
        azs2 = A.azs
        bzs2 = A.bzs
        cs2 = A.cs

        ncs = length(cs1) * length(cs2)

        iszero(ncs) && return zero(eltype(As))

        temp_axs = Vector{Ti}(undef, ncs)
        temp_bxs = Vector{Ti}(undef, ncs)
        temp_azs = Vector{Ti}(undef, ncs)
        temp_bzs = Vector{Ti}(undef, ncs)
        temp_cs = Vector{Tv}(undef, ncs)
        temp = BinaryQubitAABB(temp_axs, temp_bxs, temp_azs, temp_bzs, temp_cs)

        count = 0
        for (ax1, bx1, az1, bz1, c1) in zip(axs1, bxs1, azs1, bzs1, cs1)
            for (ax2, bx2, az2, bz2, c2) in zip(axs2, bxs2, azs2, bzs2, cs2)
                c3 = c1 * c2
                if abs(c3) > tol2
                    count += 1
                    temp_axs[count] = ax1 ⊻ ax2
                    temp_bxs[count] = bx1 ⊻ bx2
                    temp_azs[count] = az1 ⊻ az2
                    temp_bzs[count] = bz1 ⊻ bz2
                    temp_cs[count] = c3 * (-1) ^ (count_ones(az1 & ax2) + count_ones(bz1 & bx2))
                end
            end
        end

        iszero(count) && return zero(eltype(As))

        resize!(temp, count)
        sort!(temp)
        resize!(temp, reduce_by_xz!(temp, tol1))

        axs1 = temp.axs
        bxs1 = temp.bxs
        azs1 = temp.azs
        bzs1 = temp.bzs
        cs1 = temp.cs
    end

    isempty(cs1) && return zero(eltype(As))

    return BinaryQubitAABB(axs1, bxs1, azs1, bzs1, cs1)
end

Base.:+(A::BinaryQubitAABB, c::Number) = linearcombine([A], [1.0], c, eps2)
Base.:+(c::Number, A::BinaryQubitAABB) = linearcombine([A], [1.0], c, eps2)
Base.:-(A::BinaryQubitAABB, c::Number) = linearcombine([A], [1.0], -c, eps2)
Base.:-(c::Number, A::BinaryQubitAABB) = linearcombine([A], [-1.0], c, eps2)
Base.:+(A::BinaryQubitAABB, B::BinaryQubitAABB) = linearcombine([A, B], [1.0, 1.0], 0.0, eps2)
Base.:-(A::BinaryQubitAABB, B::BinaryQubitAABB) = linearcombine([A, B], [1.0, -1.0], 0.0, eps2)
Base.:*(A::BinaryQubitAABB, B::BinaryQubitAABB) = multiply([A, B], eps2, eps2)
Base.:^(A::BinaryQubitAABB, power::Int) = multiply(fill(A, power), eps2, eps2)

function aabb_a(pos::Int, is_creation::Bool, Ti::Type, Tv::Type)
    Td::DataType = double_width(Ti)
    x  = Td(1) << pos
    ax = zip_even_bit(x)
    bx = zip_odd_bit(x)

    z1 = Td(1) << pos - 1
    az1 = zip_even_bit(z1)
    bz1 = zip_odd_bit(z1)

    z2 = Td(1) << (pos + 1) - 1
    az2 = zip_even_bit(z2)
    bz2 = zip_odd_bit(z2)

    axs = Ti[ax, ax]
    bxs = Ti[bx, bx]
    azs = Ti[az1, az2]
    bzs = Ti[bz1, bz2]
    cs = Tv[0.5, is_creation ? 0.5 : -0.5]

    return BinaryQubitAABB(axs, bxs, azs, bzs, cs)
end

function aabb_Q(pos::Int, is_creation::Bool, Ti::Type, Tv::Type)
    Td::DataType = double_width(Ti)

    x  = Td(1) << pos
    ax = zip_even_bit(x)
    bx = zip_odd_bit(x)

    axs = Ti[ax, ax]
    bxs = Ti[bx, bx]
    azs = Ti[0, ax]
    bzs = Ti[0, bx]
    cs = Tv[0.5, is_creation ? 0.5 : -0.5]

    return BinaryQubitAABB(axs, bxs, azs, bzs, cs)
end

function FermionOperatorAABB(terms::Array{Tuple{Int,Int},1}, coeff::Number, Ti::Type, Tv::Type)
    multiply(
        [aabb_a(pos, Bool(is_creation), Ti, Tv) for (pos, is_creation) in terms], eps2, eps2
        ) * coeff
end

function QebOperatorAABB(terms::Array{Tuple{Int,Int},1}, coeff::Number, Ti::Type, Tv::Type)
    multiply(
        [aabb_Q(pos, Bool(is_creation), Ti, Tv) for (pos, is_creation) in terms], eps2, eps2
        ) * coeff
end

function QubitOperatorAABB(terms::Array{Tuple{Int,String},1}, coeff::Number, Ti::Type, Tv::Type)
    Td::DataType = double_width(Ti)

    x = Td(0)
    z = Td(0)

    for (pos, pauli) in terms
        site = Td(1) << pos
        if pauli == "X"
            x |= site
        elseif pauli == "Y"
            x |= site
            z |= site
        elseif pauli == "Z"
            z |= site
        else
            throw(DomainError("Invalid Pauli gate name: $(pauli)"))
        end
    end

    c = im ^ count_ones(x & z) * coeff

    ax = zip_even_bit(x)
    bx = zip_odd_bit(x)
    az = zip_even_bit(z)
    bz = zip_odd_bit(z)

    return BinaryQubitAABB(Ti[ax], Ti[bx], Ti[az], Ti[bz], Tv[c])
end

function to_sparse_matrix(
    A::BinaryQubitAABB{Ti,Tv,K,V}, 
    norb::Int,
    nelec::Tuple{Int,Int}, 
    tol::Float64=eps2
) where {Ti,Tv,K,V}

    na, nb = nelec

    a_combos = combinations([i for i in 0:norb-1], na)
    b_combos = combinations([i for i in 0:norb-1], nb)

    astrs = Vector{Ti}(undef, length(a_combos))
    bstrs = Vector{Ti}(undef, length(b_combos))

    count = 0
    for a in a_combos
        count += 1
        bin = Ti(0)
        for pos in a
            bin |= Ti(1) << pos
        end
        astrs[count] = bin
    end

    count = 0
    for b in b_combos
        count += 1
        bin = Ti(0)
        for pos in b
            bin |= Ti(1) << pos
        end
        bstrs[count] = bin
    end

    axs = A.axs
    bxs = A.bxs
    azs = A.azs
    bzs = A.bzs
    cs = A.cs

    bounds = get_bounds_1based(axs, bxs)
    ngs    = length(bounds) - 1
    dim    = length(astrs) * length(bstrs)
    nnz    = dim * ngs

    Td::DataType = double_width(Ti)

    nzrow  = Array{Td,1}(undef, nnz)
    nzcol  = Array{Td,1}(undef, nnz)
    nzval  = Array{Tv,1}(undef, nnz)

    count = 0
    for i in 1:ngs
        lb = bounds[i]
        rb = bounds[i+1] - 1
        ax = axs[lb]
        bx = bxs[lb]
        pa = Vector{Int64}(undef, rb-lb+1)

        
        for src_astr in astrs
            _src_astr = unzip_even_bit(src_astr)
            _dst_astr = unzip_even_bit(src_astr ⊻ ax)
            for (ia, ka) in enumerate(lb:rb)
                pa[ia] = 1 - 2 * (count_ones(azs[ka] & src_astr) & 1)
            end
            for src_bstr in bstrs
                _src_bstr = unzip_odd_bit(src_bstr)
                _dst_bstr = unzip_odd_bit(src_bstr ⊻ bx)

                val::Tv = 0
                for (ib, kb) in enumerate(lb:rb)
                    val += cs[kb] * pa[ib] * (1 - 2 * (count_ones(bzs[kb] & src_bstr) & 1))
                end

                (abs(val) <= tol) && continue

                count += 1
                nzrow[count] = (_dst_astr | _dst_bstr) + 1
                nzcol[count] = (_src_astr | _src_bstr) + 1
                nzval[count] = val
            end
        end
    end

    resize!(nzrow, count)
    resize!(nzcol, count)
    resize!(nzval, count)

    return sparse(nzrow, nzcol, nzval, 1 << (norb*2), 1 << (norb*2))
end
