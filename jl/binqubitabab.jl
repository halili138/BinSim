struct PauliStingABAB{Ti,Tv}
    x::Ti
    z::Ti
    c::Tv
end

@inline encode(p::PauliStingABAB) = encode(p.x, p.z)

Base.:isless(p1::PauliStingABAB, p2::PauliStingABAB) = isless(encode(p1), encode(p2))

struct BinaryQubitABAB{Ti,Tv,K<:AbstractArray{Ti,1},V<:AbstractArray{Tv,1}} <: AbstractArray{PauliStingABAB{Ti,Tv},1}
    xs::K
    zs::K
    cs::V
end

Base.length(A::BinaryQubitABAB) = length(A.cs)
Base.size(A::BinaryQubitABAB) = size(A.cs)

function Base.:getindex(A::BinaryQubitABAB, i...)
    PauliStingABAB(
        getindex(A.xs, i...), 
        getindex(A.zs, i...), 
        getindex(A.cs, i...)
    )
end

function Base.:setindex!(A::BinaryQubitABAB, p::PauliStingABAB, i...)
    setindex!(A.xs, p.x, i...)
    setindex!(A.zs, p.z, i...)
    setindex!(A.cs, p.c, i...)
    A
end

function Base.view(A::BinaryQubitABAB, inds...)
    BinaryQubitABAB(
        view(A.xs, inds...), 
        view(A.zs, inds...), 
        view(A.cs, inds...),
    )
end

function Base.similar(A::BinaryQubitABAB)
    BinaryQubitABAB(
        similar(A.xs), 
        similar(A.zs),
        similar(A.cs),
    )
end

function Base.similar(A::BinaryQubitABAB, dim::Dims)
    BinaryQubitABAB(
        similar(A.xs, dim), 
        similar(A.zs, dim),
        similar(A.cs, dim),
    )
end

function Base.similar(A::BinaryQubitABAB, dims::Integer...)
    BinaryQubitABAB(
        similar(A.xs, dims), 
        similar(A.zs, dims),
        similar(A.cs, dims),
    )
end

function Base.resize!(A::BinaryQubitABAB, n::Integer)
    resize!(A.xs, n)
    resize!(A.zs, n)
    resize!(A.cs, n)
    A
end

function BinaryQubitABAB{Ti,Tv,K,V}() where {Ti,Tv,K,V}
    BinaryQubitABAB(K(), K(), V())
end

function Base.zero(::Type{BinaryQubitABAB{Ti,Tv,K,V}}) where {Ti,Tv,K,V}
    BinaryQubit{Ti,Tv,K,V}()
end

Base.zero(A::BinaryQubitABAB) = zero(typeof(A))

function Base.one(::Type{BinaryQubitABAB{Ti,Tv,K,V}}) where {Ti,Tv,K,V}
    BinaryQubit(K([0]), K([0]), V([1]))
end

Base.one(A::BinaryQubitABAB) = one(typeof(A))

function trunc_by_cs!(A::BinaryQubitABAB, tol::Float64)
    xs = A.xs
    zs = A.zs
    cs = A.cs

    count = 0
    for (i, c) in enumerate(cs)
        if abs(c) > tol
            count += 1
            xs[count] = xs[i]
            zs[count] = zs[i]
            cs[count] = c
        end
    end

    return count
end

function reduce_by_xz!(A::BinaryQubitABAB, tol::Float64)
    len = length(A)

    iszero(len) && return 0 

    xs = A.xs
    zs = A.zs
    cs = A.cs

    l = 1
    for r in 2:len
        xl = xs[l]
        zl = zs[l]
        cl = cs[l]
        
        xr = xs[r]
        zr = zs[r]
        cr = cs[r]

        ql = encode(xl, zl)
        qr = encode(xr, zr)
        
        if qr == ql
            cs[l] = cl + cr
        else
            if abs(cl) > tol
                l += 1
            end
            
            if l < r
                xs[l] = xr
                zs[l] = zr
                cs[l] = cr
            end
        end
    end

    if abs(cs[l]) < tol
        l -= 1
    end

    return l
end

function Base.:*(A::BinaryQubitABAB, c::Number)
    res = deepcopy(A)
    res.cs .*= c
    res
end

Base.:*(c::Number, A::BinaryQubitABAB) = A * c

Base.:-(A::BinaryQubitABAB) = A * (-1)

function Base.:adjoint(A::BinaryQubitABAB)
    res = deepcopy(A)
    res.cs .= conj.(A.cs)
    res.cs .*= ((-1) .^ count_ones.(A.xs .& A.zs))
    res
end

function Base.rand(::Type{BinaryQubitABAB{Ti,Tv,K,V}}, nq::Int, nterms::Int) where {Ti,Tv,K,V}
    dim = Ti(1) << nq

    xs::Vector{Ti} = rand(0:dim-1, nterms)
    zs::Vector{Ti} = rand(0:dim-1, nterms)
    cs::Vector{Tv} = rand(Tv, nterms)

    res = BinaryQubitABAB(xs, zs, cs)

    sort!(res)

    resize!(res, reduce_by_xz!(res, 1e-12))

    return res
end

function bin_to_pstr(bin, pauli_name::String)
    pauli_num = count_ones(bin)
    pstr = Vector{Tuple{Int,String}}(undef, pauli_num)

    pos = 0
    count = 0
    while bin != 0
        if bin & 1 == 1
            count += 1
            pstr[count] = (pos, pauli_name)
        end
        bin >>= 1
        pos += 1
    end

    resize!(pstr, count)

    return pstr
end

function bin_to_pstr(x, z, c)
    Y = x & z
    X = x ⊻ Y
    Z = z ⊻ Y

    c /= im ^ count_ones(Y)

    X_pstr = bin_to_pstr(X, "X")
    Y_pstr = bin_to_pstr(Y, "Y")
    Z_pstr = bin_to_pstr(Z, "Z")

    pstr = vcat(X_pstr, Y_pstr, Z_pstr)

    sort!(pstr)

    pstr1 = "($(c))"
    pstr2 = ""
    for (pos, gate) in pstr
        pstr2 *= "$(gate)$(pos) "
    end
    pstr2 = strip(pstr2)

    return pstr1 * " [" * pstr2 * "]"
end

function Base.:show(io::IO, p::PauliStingABAB)
    str = bin_to_pstr(p.x, p.z, p.c)
    println(io, str)
end

function Base.:(==)(A::BinaryQubitABAB, B::BinaryQubitABAB)
    A.xs ≠ B.xs && return false
    A.zs ≠ B.zs && return false

    is_zero1 = all(iszero.(A.cs))
    is_zero2 = all(iszero.(B.cs))

    is_zero1 && return is_zero2

    ratio = A.cs ./ B.cs

    c = first(ratio)

    return all(x -> isapprox(x, c; rtol=1e-12), ratio)
end

function Base.hash(A::BinaryQubitABAB, h::UInt)
    if all(iszero.(A.cs))
        return hash(0, h)
    end

    first_nz = findfirst(!iszero, A.cs)
    normalized = A.cs ./ A.cs[first_nz]

    return hash((A.xs, normalized), h)
end

function linearcombine(
    As::Array{BinaryQubitABAB{Ti,Tv,K,V},1}, 
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

    res_xs = Vector{Ti}(undef, ncs)
    res_zs = Vector{Ti}(undef, ncs)
    res_cs = Vector{Tv}(undef, ncs)

    res = BinaryQubitABAB(res_xs, res_zs, res_cs)

    count = 0
    if !iszero(C)
        count += 1
        res_xs[count] = 0
        res_zs[count] = 0
        res_cs[count] = C
    end

    for (i, A) in enumerate(As)
        coeff = Cs[i]
        xs = A.xs
        zs = A.zs
        cs = A.cs
        for j in eachindex(cs)
            count += 1
            res_xs[count] = xs[j]
            res_zs[count] = zs[j]
            res_cs[count] = cs[j] * coeff
        end
    end

    sort!(res)

    resize!(res, reduce_by_xz!(res, tol))

    return res
end

function multiply(
    As::Array{BinaryQubitABAB{Ti,Tv,K,V},1}, tol1::Float64, tol2::Float64,
) where {Ti,Tv,K,V}

    length(As) == 0 && return zero(eltype(As))

    xs1 = Ti[0]
    zs1 = Ti[0]
    cs1 = Tv[1]
    
    for A in As
        xs2 = A.xs
        zs2 = A.zs
        cs2 = A.cs

        ncs = length(cs1) * length(cs2)

        iszero(ncs) && return zero(eltype(As))

        temp_xs = Vector{Ti}(undef, ncs)
        temp_zs = Vector{Ti}(undef, ncs)
        temp_cs = Vector{Tv}(undef, ncs)
        temp = BinaryQubitABAB(temp_xs, temp_zs, temp_cs)

        count = 0
        for (x1, z1, c1) in zip(xs1, zs1, cs1)
            for (x2, z2, c2) in zip(xs2, zs2, cs2)
                c3 = c1 * c2
                if abs(c3) > tol2
                    count += 1
                    temp_xs[count] = x1 ⊻ x2
                    temp_zs[count] = z1 ⊻ z2
                    temp_cs[count] = c3 * (-1) ^ count_ones(z1 & x2)
                end
            end
        end

        iszero(count) && return zero(eltype(As))

        resize!(temp, count)
        sort!(temp)
        resize!(temp, reduce_by_xz!(temp, tol1))

        xs1 = temp.xs
        zs1 = temp.zs
        cs1 = temp.cs
    end

    isempty(cs1) && return zero(eltype(As))

    return BinaryQubitABAB(xs1, zs1, cs1)
end

Base.:+(A::BinaryQubitABAB, c::Number) = linearcombine([A], [1.0], c, eps2)
Base.:+(c::Number, A::BinaryQubitABAB) = linearcombine([A], [1.0], c, eps2)
Base.:-(A::BinaryQubitABAB, c::Number) = linearcombine([A], [1.0], -c, eps2)
Base.:-(c::Number, A::BinaryQubitABAB) = linearcombine([A], [-1.0], c, eps2)
Base.:+(A::BinaryQubitABAB, B::BinaryQubitABAB) = linearcombine([A, B], [1.0, 1.0], 0.0, eps2)
Base.:-(A::BinaryQubitABAB, B::BinaryQubitABAB) = linearcombine([A, B], [1.0, -1.0], 0.0, eps2)
Base.:*(A::BinaryQubitABAB, B::BinaryQubitABAB) = multiply([A, B], eps2, eps2)
Base.:^(A::BinaryQubitABAB, power::Int) = multiply(fill(A, power), eps2, eps2)

function abab_a(pos::Int, is_creation::Bool, Ti::Type, Tv::Type)
    x  = Ti(1) << pos
    z1 = Ti(1) << pos - 1
    z2 = Ti(1) << (pos + 1) - 1

    xs = Ti[x, x]
    zs = Ti[z1, z2]
    cs = Tv[0.5, is_creation ? 0.5 : -0.5]

    return BinaryQubitABAB(xs, zs, cs)
end

function abab_Q(pos::Int, is_creation::Bool, Ti::Type, Tv::Type)
    x  = Ti(1) << pos

    xs = Ti[x, x]
    zs = Ti[0, x]
    cs = Tv[0.5, is_creation ? 0.5 : -0.5]

    return BinaryQubitABAB(xs, zs, cs)
end

function FermionOperatorABAB(terms::Array{Tuple{Int,Int},1}, coeff::Number, Ti::Type, Tv::Type)
    multiply(
        [abab_a(pos, Bool(is_creation), Ti, Tv) for (pos, is_creation) in terms], eps2, eps2
        ) * coeff
end

function QebOperatorABAB(terms::Array{Tuple{Int,Int},1}, coeff::Number, Ti::Type, Tv::Type)
    multiply(
        [abab_Q(pos, Bool(is_creation), Ti, Tv) for (pos, is_creation) in terms], eps2, eps2
        ) * coeff
end

function QubitOperatorABAB(terms::Array{Tuple{Int,String},1}, coeff::Number, Ti::Type, Tv::Type)
    x = Ti(0)
    z = Ti(0)

    for (pos, pauli) in terms
        site = Ti(1) << pos
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

    return BinaryQubitABAB(Ti[x], Ti[z], Tv[c])
end
