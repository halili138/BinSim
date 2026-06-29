function get_bounds_0based(xs::Vector{Ti}) where Ti
    len = length(xs)

    len == 0 && return Int[0, 0]
    len == 1 && return Int[0, 1]

    bounds = Vector{Int}(undef, len + 1)
    bounds[1] = 0

    count = 2
    for i in 2:len
        if xs[i] != xs[i-1]
            bounds[count] = i - 1
            count += 1
        end
    end

    bounds[count] = len

    resize!(bounds, count)

    return bounds
end

function get_bounds_0based(axs::Vector{Ti}, bxs::Vector{Ti}) where Ti
    len = length(axs)

    len == 0 && return Int[0, 0]
    len == 1 && return Int[0, 1]

    bounds = Vector{Int}(undef, len + 1)
    bounds[1] = 0

    count = 2
    for i in 2:len
        xl = unzip_even_bit(axs[i-1]) | unzip_odd_bit(bxs[i-1])
        xr = unzip_even_bit(axs[i]) | unzip_odd_bit(bxs[i])
        if xr != xl
            bounds[count] = i - 1
            count += 1
        end
    end

    bounds[count] = len

    resize!(bounds, count)

    return bounds
end

function get_bounds_1based(xs::Vector{Ti}) where Ti
    len = length(xs)

    len == 0 && return Int[0, 0]
    len == 1 && return Int[1, 2]

    bounds = Vector{Int}(undef, len + 1)
    bounds[1] = 1

    count = 2
    for i in 2:len
        if xs[i] != xs[i-1]
            bounds[count] = i
            count += 1
        end
    end

    bounds[count] = len + 1

    resize!(bounds, count)

    return bounds
end

function get_bounds_1based(axs::Vector{Ti}, bxs::Vector{Ti}) where Ti
    len = length(axs)

    len == 0 && return Int[0, 0]
    len == 1 && return Int[1, 2]

    bounds = Vector{Int}(undef, len + 1)
    bounds[1] = 1

    count = 2
    for i in 2:len
        xl = unzip_even_bit(axs[i-1]) | unzip_odd_bit(bxs[i-1])
        xr = unzip_even_bit(axs[i]) | unzip_odd_bit(bxs[i])
        if xr != xl
            bounds[count] = i
            count += 1
        end
    end

    bounds[count] = len + 1

    resize!(bounds, count)

    return bounds
end

@inline function phase(x)
    return 1 - 2 * (count_ones(x) & 1)
end
