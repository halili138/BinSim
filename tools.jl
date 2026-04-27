struct BufferElement{Ti,Tv}
    k::Ti
    v::Tv
end


@inline Base.:isless(a::BufferElement, b::BufferElement) = isless(a.k, b.k)


struct Buffer{Ti,Tv,K<:AbstractArray{Ti,1},V<:AbstractArray{Tv,1}} <: AbstractArray{BufferElement{Ti,Tv},1}
    indexs::K
    values::V
end


@inline Base.length(buffer::Buffer) = length(buffer.indexs)


@inline Base.size(buffer::Buffer) = size(buffer.indexs)


@inline Base.:getindex(buffer::Buffer, i...) = BufferElement(getindex(buffer.indexs, i...), getindex(buffer.values, i...))


@inline function Base.:setindex!(buffer::Buffer, p::BufferElement, i...)
    setindex!(buffer.indexs, p.k, i...)
    setindex!(buffer.values, p.v, i...)
    buffer
end


@inline Base.view(buffer::Buffer, inds...) = Buffer(view(buffer.indexs, inds...), view(buffer.values, inds...))


@inline function Buffer{Ti,Tv,K,V}() where {Ti,Tv,K,V}
    Buffer(Ti[], Tv[])
end


@inline function Base.resize!(buffer::Buffer, n::Integer)
    resize!(buffer.indexs, n)
    resize!(buffer.values, n)
end


@inline function trunc_by_second!(buffer::Buffer{Ti,Tv,K,V}, tol::Float64) where {Ti,Tv,K,V}
    indexs = buffer.indexs
    values = buffer.values

    cp = 0
    for i in eachindex(indexs)
        if abs(values[i]) > tol
            cp += 1
            indexs[cp] = indexs[i]
            values[cp] = values[i]
        end
    end

    return cp
end


@inline function reduce_by_first!(buffer::Buffer{Ti,Tv,K,V}, tol::Float64, reduce_fun::Function) where {Ti,Tv,K,V}
    len = length(buffer)

    iszero(len) && return 0

    indexs = buffer.indexs
    values = buffer.values

    l = 1
    for r in 2:len
        if indexs[r] == indexs[l]
            values[l] = reduce_fun(values[l], values[r])
        else
            if abs(values[l]) > tol
                l += 1
            else
                values[l] = Tv(0)
            end

            if l < r
                indexs[l] = indexs[r]
                values[l] = values[r]
            end
        end
    end

    (abs(values[l]) < tol) && (l -= 1)

    return l
end


function get_bounds_0based(ordered_vec)
    len = length(ordered_vec)

    len == 0 && return Int[0, 0]
    len == 1 && return Int[0, 1]

    bounds = Array{Int,1}(undef, len + 1)
    bounds[1] = 0

    cp = 2
    for i in 2:len
        if ordered_vec[i] != ordered_vec[i-1]
            bounds[cp] = i - 1
            cp += 1
        end
    end

    bounds[cp] = len

    resize!(bounds, cp)

    return bounds
end


function get_bounds_1based(ordered_vec)
    len = length(ordered_vec)

    len == 0 && return Int[0, 0]
    len == 1 && return Int[1, 2]

    bounds = Array{Int,1}(undef, len + 1)
    bounds[1] = 1
    cp = 2
    for i in 2:len
        if ordered_vec[i] != ordered_vec[i-1]
            bounds[cp] = i
            cp += 1
        end
    end

    bounds[cp] = len + 1

    resize!(bounds, cp)

    return bounds
end


function hartree_fock_str(nelec::Tuple{Int,Int}, Ti::Type)
    na, nb = nelec

    bin = Ti(0)
    for i in 0:na-1
        bin |= (Ti(1) << 2i)
    end

    for i in 0:nb-1
        bin |= (Ti(1) << (2i + 1))
    end

    return bin
end
