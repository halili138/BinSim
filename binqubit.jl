abstract type BinaryOperator end
abstract type IBOP <: BinaryOperator end # Ideal BinaryOperator(Noiseless BinaryOperator)
abstract type NBOP <: BinaryOperator end # Noise BinaryOperator


struct BinaryQubit{Ti,Tv,Tq,Tg,K<:AbstractArray{Ti,1},V<:AbstractArray{Tv,1},Q<:AbstractArray{Tq,1},G<:AbstractArray{Tg,1}} <: IBOP
    xs::K
    zs::K
    cs::V
    qs::Q
    gs::G
end


@inline function BinaryQubit{Ti,Tv,Tq,Tg,K,V,Q,G}() where {Ti,Tv,Tq,Tg,K,V,Q,G}
    BinaryQubit(K(), K(), V(), Q(), G())
end


@inline function Base.zero(::Type{BinaryQubit{Ti,Tv,Tq,Tg,K,V,Q,G}}) where {Ti,Tv,Tq,Tg,K,V,Q,G}
    BinaryQubit{Ti,Tv,Tq,Tg,K,V,Q,G}()
end


@inline Base.zero(A::BinaryQubit) = zero(typeof(A))


@inline function Base.one(::Type{BinaryQubit{Ti,Tv,Tq,Tg,K,V,Q,G}}) where {Ti,Tv,Tq,Tg,K,V,Q,G}
    BinaryQubit(K([0]), K([0]), V([1]), Q([0]), G([1, 2]))
end


@inline Base.one(A::BinaryQubit) = one(typeof(A))


@inline function encode_q(x::Ti, z::Ti, shift::Int, Tq::Type) where {Ti}
    Tq(x) << shift | Tq(z)
end


@inline function decode_x(q::Tq, shift::Int, Ti::Type) where {Tq}
    unsafe_trunc(Ti, q >> shift)
end


@inline function decode_z(q::Tq, Ti::Type) where {Tq}
    unsafe_trunc(Ti, q)
end


@inline function Base.:*(A::BinaryQubit, c::Number)
    res = deepcopy(A)
    res.cs .*= c
    res
end


@inline Base.:*(c::Number, A::BinaryQubit) = A * c


@inline Base.:-(A::BinaryQubit) = A * (-1)


@inline function Base.:adjoint(A::BinaryQubit)
    res = deepcopy(A)
    res.cs .= conj.(A.cs)
    res.cs .*= ((-1) .^ count_ones.(A.xs .& A.zs))
    res
end


const HostBinaryQubit{Ti,Tv,Tq,Tg} = BinaryQubit{Ti,Tv,Tq,Tg,Array{Ti,1},Array{Tv,1},Array{Tq,1},Array{Tg,1}}


@inline function BinaryQubit{Ti,Tv,Tq,Tg}() where {Ti,Tv,Tq,Tg}
    HostBinaryQubit{Ti,Tv,Tq,Tg}()
end


function Base.rand(::Type{HostBinaryQubit{Ti,Tv,Tq,Tg}}, nq::Int, nterms::Int) where {Ti,Tv,Tq,Tg}
    dim = Ti(1) << nq

    xs::Array{Ti,1} = rand(0:dim-1, nterms)
    zs::Array{Ti,1} = rand(0:dim-1, nterms)
    cs::Array{Tv,1} = rand(Tv, nterms)

    shift = sizeof(Ti) * 8
    qs = encode_q.(xs, zs, shift, Tq)

    qs_buffer = Buffer(qs, cs)

    sort!(qs_buffer)
    resize!(qs_buffer, reduce_by_first!(qs_buffer, eps1, +))

    qs = qs_buffer.indexs
    cs = qs_buffer.values
    xs = decode_x.(qs, shift, Ti)
    zs = decode_z.(qs, Ti)
    gs = get_bounds_1based(xs)

    BinaryQubit(xs, zs, cs, qs, gs)
end


function rand_hostbinqubit(nq::Int, nterms::Int, Ti::Type, Tv::Type)
    Tq = double_width(Ti)
    rand(HostBinaryQubit{Ti,Tv,Tq,Int64}, nq, nterms)
end


function bin_to_pstr(bin, pauli_name::String)
    pauli_num = count_ones(bin)
    pstr = Array{Tuple{Int,String},1}(undef, pauli_num)

    pos = 0
    cp = 0
    while bin != 0
        if bin & 1 == 1
            cp += 1
            pstr[cp] = (pos, pauli_name)
        end
        bin >>= 1
        pos += 1
    end

    resize!(pstr, cp)

    return pstr
end


function bin_to_pstr(x, z, c)
    Y = x & z
    X = x ⊻ Y
    Z = z ⊻ Y

    c /= im^count_ones(Y)

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


function Base.:show(io::IO, A::HostBinaryQubit)
    xs = A.xs
    zs = A.zs
    cs = A.cs

    for i in eachindex(xs)
        str = bin_to_pstr(xs[i], zs[i], cs[i])
        println(io, str)
    end
end


@inline function Base.:(==)(A::HostBinaryQubit, B::HostBinaryQubit)
    A.qs ≠ B.qs && return false

    is_zero1 = all(iszero.(A.cs))
    is_zero2 = all(iszero.(B.cs))

    is_zero1 && return is_zero2

    ratio = A.cs ./ B.cs

    c = first(ratio)

    return all(x -> isapprox(x, c; rtol=1e-12), ratio)
end


function Base.hash(A::HostBinaryQubit, h::UInt)
    if all(iszero.(A.cs))
        return hash(0, h)
    end

    first_nz = findfirst(!iszero, A.cs)
    normalized = A.cs ./ A.cs[first_nz]

    return hash((A.xs, normalized), h)
end


@inline function linearcombine(
    bqops::Array{HostBinaryQubit{Ti,Tv,Tq,Tg},1}, 
    coeffs::Array{<:Number,1}, 
    constant::Number, 
    tol::Float64;
    based::Int=1,
) where {Ti,Tv,Tq,Tg}

    @assert length(bqops) == length(coeffs)

    length(bqops) == 0 && return one(eltype(bqops)) * constant

    nqs = 0
    for op in bqops
        nqs += length(op.xs)
    end

    !iszero(constant) && (nqs += 1)

    res_qs = Array{Tq,1}(undef, nqs)
    res_cs = Array{Tv,1}(undef, nqs)
    res_buffer = Buffer(res_qs, res_cs)

    cp = 0
    if !iszero(constant)
        cp += 1
        res_qs[cp] = 0
        res_cs[cp] = constant
    end

    for (i, op) in enumerate(bqops)
        coeff = coeffs[i]
        qs = op.qs
        cs = op.cs
        for j in eachindex(qs)
            cp += 1
            res_qs[cp] = qs[j]
            res_cs[cp] = cs[j] * coeff
        end
    end

    sort!(res_buffer)
    resize!(res_buffer, reduce_by_first!(res_buffer, tol, +))

    res_xs = decode_x.(res_qs, sizeof(Ti) * 8, Ti)
    res_zs = decode_z.(res_qs, Ti)
    res_gs = based == 1 ? get_bounds_1based(res_xs) : get_bounds_0based(res_xs)

    return BinaryQubit(res_xs, res_zs, res_cs, res_qs, res_gs)
end


@inline function multiply(
    bqops::Array{HostBinaryQubit{Ti,Tv,Tq,Tg},1}, 
    tol1::Float64, 
    tol2::Float64;
    based::Int=1,
) where {Ti,Tv,Tq,Tg}

    length(bqops) == 0 && return zero(eltype(bqops))

    shift = sizeof(Ti) * 8

    qs1 = Tq[0]
    cs1 = Tv[1]
    zs1 = Ti[0]

    for op in bqops
        qs2 = op.qs
        cs2 = op.cs

        nqs = length(qs1) * length(qs2)

        iszero(nqs) && return zero(eltype(bqops))

        temp_qs = Array{Tq,1}(undef, nqs)
        temp_cs = Array{Tv,1}(undef, nqs)
        temp_buffer = Buffer(temp_qs, temp_cs)

        cp = 0
        for i in eachindex(qs1)
            q1 = qs1[i]
            c1 = cs1[i]
            z1 = zs1[i]
            for j in eachindex(qs2)
                q2 = qs2[j]
                c2 = cs2[j]
                x2 = decode_x(q2, shift, Ti)

                c3 = c1 * c2

                if abs(c3) > tol2
                    cp += 1
                    temp_qs[cp] = q1 ⊻ q2
                    temp_cs[cp] = c3 * (-1) ^ count_ones(z1 & x2)
                end
            end
        end

        iszero(cp) && return zero(eltype(bqops))

        @views sort!(temp_buffer[1:cp])

        @views nqs = reduce_by_first!(temp_buffer[1:cp], tol1, +)

        qs1 = temp_qs[1:nqs]
        cs1 = temp_cs[1:nqs]
        zs1 = decode_z.(qs1, Ti)
    end

    nqs = length(qs1)

    iszero(nqs) && return zero(eltype(bqops))

    xs1 = decode_x.(qs1, shift, Ti)
    gs1 = based == 1 ? get_bounds_1based(xs1) : get_bounds_0based(xs1)

    return BinaryQubit(xs1, zs1, cs1, qs1, gs1)
end


function to_sparse_matrix(
    A::HostBinaryQubit{Ti,Tv,Tq,Tg}, 
    basis::Array{Ti,1}, 
    nq::Int; 
    tol::Float64=eps2
) where {Ti,Tv,Tq,Tg}

    xs  = A.xs
    zs  = A.zs
    cs  = A.cs
    gs  = A.gs
    ngs = length(gs) - 1

    dim   = length(basis)
    nnz   = dim * ngs
    nzrow = Array{Ti,1}(undef, nnz)
    nzcol = Array{Ti,1}(undef, nnz)
    nzval = Array{Tv,1}(undef, nnz)

    count = 0
    for i in 1:ngs
        lb = gs[i]
        rb = gs[i+1] - 1
        x = xs[lb]
        for bin in basis
            val::Tv = 0

            for k in lb:rb
                val += cs[k] * (1 - 2 * (count_ones(zs[k] & bin) & 1))
            end

            (abs(val) <= tol) && continue

            count += 1
            nzrow[count] = x ⊻ bin + 1
            nzcol[count] = bin + 1
            nzval[count] = val
        end
    end

    resize!(nzrow, count)
    resize!(nzcol, count)
    resize!(nzval, count)

    return sparse(nzrow, nzcol, nzval, 1 << nq, 1 << nq)
end


@inline Base.:+(A::HostBinaryQubit, c::Number) = linearcombine([A], [1.0], c, eps2)
@inline Base.:+(c::Number, A::HostBinaryQubit) = linearcombine([A], [1.0], c, eps2)
@inline Base.:-(A::HostBinaryQubit, c::Number) = linearcombine([A], [1.0], -c, eps2)
@inline Base.:-(c::Number, A::HostBinaryQubit) = linearcombine([A], [-1.0], c, eps2)
@inline Base.:+(A::HostBinaryQubit, B::HostBinaryQubit) = linearcombine([A, B], [1.0, 1.0], 0.0, eps2)
@inline Base.:-(A::HostBinaryQubit, B::HostBinaryQubit) = linearcombine([A, B], [1.0, -1.0], 0.0, eps2)
@inline Base.:*(A::HostBinaryQubit, B::HostBinaryQubit) = multiply([A, B], eps2, eps2)
@inline Base.:^(A::HostBinaryQubit, power::Int) = multiply(fill(A, power), eps2, eps2)


@inline function _a(pos::Int, is_creation::Bool, Ti::Type, Tv::Type)
    x  = Ti(1) << pos
    z1 = Ti(1) << pos - 1
    z2 = Ti(1) << (pos + 1) - 1

    xs = Ti[x, x]
    zs = Ti[z1, z2]
    cs = Tv[0.5, is_creation ? 0.5 : -0.5]
    qs = encode_q.(xs, zs, sizeof(Ti) * 8, double_width(Ti))
    gs = get_bounds_1based(xs)

    return BinaryQubit(xs, zs, cs, qs, gs)
end


@inline function _Q(pos::Int, is_creation::Bool, Ti::Type, Tv::Type)
    x  = Ti(1) << pos

    xs = Ti[x, x]
    zs = Ti[0, x]
    cs = Tv[0.5, is_creation ? 0.5 : -0.5]
    qs = encode_q.(xs, zs, sizeof(Ti) * 8, double_width(Ti))
    gs = get_bounds_1based(xs)

    return BinaryQubit(xs, zs, cs, qs, gs)
end


@inline function FermionOperator(terms::Array{Tuple{Int,Int},1}, coeff::Number, Ti::Type, Tv::Type)
    multiply(
        [_a(pos, Bool(is_creation), Ti, Tv) for (pos, is_creation) in terms], 
        eps2, 
        eps2,
        based=0,
    ) * coeff
end


@inline function QebOperator(terms::Array{Tuple{Int,Int},1}, coeff::Number, Ti::Type, Tv::Type)
    multiply(
        [_Q(pos, Bool(is_creation), Ti, Tv) for (pos, is_creation) in terms], 
        eps2, 
        eps2,
        based=0,
    ) * coeff
end


function QubitOperator(terms::Array{Tuple{Int,String},1}, coeff::Number, Ti::Type, Tv::Type)
    Tq::Type = double_width(Ti)
    shift = sizeof(Ti) * 8

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

    c = im^count_ones(x & z) * coeff
    q = Tq(x) << shift | Tq(z)

    return BinaryQubit(Ti[x], Ti[z], Tv[c], Tq[q], Int[1, 2])
end


struct BinaryQubitAABB{Ti,Tv,Tg,K<:AbstractArray{Ti,1},V<:AbstractArray{Tv,1},G<:AbstractArray{Tg,1}}
    axs::K
    azs::K
    bxs::K
    bzs::K
    cs::V
    gs::G
end


function BinaryQubitAABB(A::BinaryQubit)
    xs = A.xs
    zs = A.zs

    return BinaryQubitAABB(
        zip_even_bit.(xs),
        zip_even_bit.(zs),
        zip_odd_bit.(xs),
        zip_odd_bit.(zs),
        A.cs,
        A.gs,
    )
end


function BinaryQubit(A::BinaryQubitAABB)
    axs = A.axs
    bxs = A.bxs
    azs = A.azs
    bzs = A.bzs

    s = sizeof(eltype(axs)) * 8
    Ti::DataType = double_width(eltype(axs))
    Tq::DataType = double_width(Ti)

    xs = unzip_even_bit.(axs) .| unzip_odd_bit.(bxs)
    zs = unzip_even_bit.(azs) .| unzip_odd_bit.(bzs)
    qs = encode_q.(xs, zs, 2s, Tq)
    
    BinaryQubit(
        xs, 
        zs, 
        A.cs,
        qs,
        A.gs,
    )
end


