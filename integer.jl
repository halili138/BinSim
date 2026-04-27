@inline double_width(::Type{Int32})   = Int64
@inline double_width(::Type{Int64})   = Int128

@inline double_width(::Type{UInt32})  = UInt64
@inline double_width(::Type{UInt64})  = UInt128

@inline half_width(::Type{Int64})     = Int32
@inline half_width(::Type{Int128})    = Int64

@inline half_width(::Type{UInt64})    = UInt32
@inline half_width(::Type{UInt128})   = UInt64


@inline bitwidth(T::Type{<:Integer}) = sizeof(T) * 8


@inline function zip_even_bit(x::UInt64)
    x &= 0x5555555555555555
    x = (x | (x >> 1))  & 0x3333333333333333
    x = (x | (x >> 2))  & 0x0F0F0F0F0F0F0F0F
    x = (x | (x >> 4))  & 0x00FF00FF00FF00FF
    x = (x | (x >> 8))  & 0x0000FFFF0000FFFF
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF
    return UInt32(x)
end


@inline function zip_odd_bit(x::UInt64)
    zip_even_bit(x>>1)
end


@inline function zip_even_bit(x::UInt128)
    x &= 0x55555555555555555555555555555555
    x = (x | (x >> 1))  & 0x33333333333333333333333333333333
    x = (x | (x >> 2))  & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F
    x = (x | (x >> 4))  & 0x00FF00FF00FF00FF00FF00FF00FF00FF
    x = (x | (x >> 8))  & 0x0000FFFF0000FFFF0000FFFF0000FFFF
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF00000000FFFFFFFF
    x = (x | (x >> 32)) & 0x0000000000000000FFFFFFFFFFFFFFFF

    return UInt64(x)
end


@inline function zip_odd_bit(x::UInt128)
    zip_even_bit(x>>1)
end


@inline function unzip_even_bit(x::UInt32)
    x = UInt64(x)
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8))  & 0x00FF00FF00FF00FF
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2))  & 0x3333333333333333
    x = (x | (x << 1))  & 0x5555555555555555
    return x
end


@inline function unzip_odd_bit(x::UInt32)
    unzip_even_bit(x) << 1
end


@inline function unzip_even_bit(x::UInt64)
    x = UInt128(x)
    x = (x | (x << 32)) & 0x00000000FFFFFFFF00000000FFFFFFFF
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF0000FFFF0000FFFF
    x = (x | (x << 8))  & 0x00FF00FF00FF00FF00FF00FF00FF00FF
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F
    x = (x | (x << 2))  & 0x33333333333333333333333333333333
    x = (x | (x << 1))  & 0x55555555555555555555555555555555
    return x
end


@inline function unzip_odd_bit(x::UInt64)
    unzip_even_bit(x) << 1
end
