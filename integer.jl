@inline double_width(::Type{Int8})    = Int16
@inline double_width(::Type{Int16})   = Int32
@inline double_width(::Type{Int32})   = Int64
@inline double_width(::Type{Int64})   = Int128
@inline double_width(::Type{Int128})  = Int256
@inline double_width(::Type{Int256})  = Int512

@inline double_width(::Type{UInt8})   = UInt16
@inline double_width(::Type{UInt16})  = UInt32
@inline double_width(::Type{UInt32})  = UInt64
@inline double_width(::Type{UInt64})  = UInt128
@inline double_width(::Type{UInt128}) = UInt256
@inline double_width(::Type{UInt256}) = UInt512

@inline half_width(::Type{Int16})     = Int8
@inline half_width(::Type{Int32})     = Int16
@inline half_width(::Type{Int64})     = Int32
@inline half_width(::Type{Int128})    = Int64
@inline half_width(::Type{Int256})    = Int128
@inline half_width(::Type{Int512})    = Int256

@inline half_width(::Type{UInt16})    = UInt8
@inline half_width(::Type{UInt32})    = UInt16
@inline half_width(::Type{UInt64})    = UInt32
@inline half_width(::Type{UInt128})   = UInt64
@inline half_width(::Type{UInt256})   = UInt128
@inline half_width(::Type{UInt512})   = UInt256

@inline bitwidth(T::Type{<:Integer}) = sizeof(T) * 8



# --- 32-bit PEXT / PDEP ---
@inline function pext_u32(a::UInt32, mask::UInt32)
    ir = """
    declare i32 @llvm.x86.bmi.pext.32(i32, i32)
    define i32 @entry(i32 %a, i32 %mask) alwaysinline {
    top:
        %res = call i32 @llvm.x86.bmi.pext.32(i32 %a, i32 %mask)
        ret i32 %res
    }
    """
    Base.llvmcall((ir, "entry"), UInt32, Tuple{UInt32, UInt32}, a, mask)
end

@inline function pdep_u32(a::UInt32, mask::UInt32)
    ir = """
    declare i32 @llvm.x86.bmi.pdep.32(i32, i32)
    define i32 @entry(i32 %a, i32 %mask) alwaysinline {
    top:
        %res = call i32 @llvm.x86.bmi.pdep.32(i32 %a, i32 %mask)
        ret i32 %res
    }
    """
    Base.llvmcall((ir, "entry"), UInt32, Tuple{UInt32, UInt32}, a, mask)
end

# --- 64-bit PEXT / PDEP ---
@inline function pext_u64(a::UInt64, mask::UInt64)
    ir = """
    declare i64 @llvm.x86.bmi.pext.64(i64, i64)
    define i64 @entry(i64 %a, i64 %mask) alwaysinline {
    top:
        %res = call i64 @llvm.x86.bmi.pext.64(i64 %a, i64 %mask)
        ret i64 %res
    }
    """
    Base.llvmcall((ir, "entry"), UInt64, Tuple{UInt64, UInt64}, a, mask)
end

@inline function pdep_u64(a::UInt64, mask::UInt64)
    ir = """
    declare i64 @llvm.x86.bmi.pdep.64(i64, i64)
    define i64 @entry(i64 %a, i64 %mask) alwaysinline {
    top:
        %res = call i64 @llvm.x86.bmi.pdep.64(i64 %a, i64 %mask)
        ret i64 %res
    }
    """
    Base.llvmcall((ir, "entry"), UInt64, Tuple{UInt64, UInt64}, a, mask)
end

const EVEN_MASK32 = 0x55555555
const ODD_MASK32  = 0xAAAAAAAA
const EVEN_MASK64 = 0x5555555555555555
const ODD_MASK64  = 0xAAAAAAAAAAAAAAAA


@inline zip_even_bit(x::UInt16) = UInt8(pext_u32(UInt32(x), EVEN_MASK32))
@inline zip_odd_bit(x::UInt16)  = UInt8(pext_u32(UInt32(x), ODD_MASK32))
@inline zip_even_bit(x::UInt32) = UInt16(pext_u32(x, EVEN_MASK32))
@inline zip_odd_bit(x::UInt32)  = UInt16(pext_u32(x, ODD_MASK32))
@inline zip_even_bit(x::UInt64) = UInt32(pext_u64(x, EVEN_MASK64))
@inline zip_odd_bit(x::UInt64)  = UInt32(pext_u64(x, ODD_MASK64))

@inline function zip_even_bit(x::UInt128)
    lo = pext_u64(UInt64(x), EVEN_MASK64)
    hi = pext_u64(UInt64(x >> 64), EVEN_MASK64)
    return (UInt64(hi) << 32) | UInt64(lo)
end

@inline function zip_odd_bit(x::UInt128)
    lo = pext_u64(UInt64(x), ODD_MASK64)
    hi = pext_u64(UInt64(x >> 64), ODD_MASK64)
    return (UInt64(hi) << 32) | UInt64(lo)
end

@inline function zip_even_bit(x::UInt256)
    c0 = pext_u64(UInt64(x), EVEN_MASK64)
    c1 = pext_u64(UInt64(x >> 64), EVEN_MASK64)
    c2 = pext_u64(UInt64(x >> 128), EVEN_MASK64)
    c3 = pext_u64(UInt64(x >> 192), EVEN_MASK64)
    return UInt128(c0) | (UInt128(c1) << 32) | (UInt128(c2) << 64) | (UInt128(c3) << 96)
end

@inline function zip_odd_bit(x::UInt256)
    c0 = pext_u64(UInt64(x), ODD_MASK64)
    c1 = pext_u64(UInt64(x >> 64), ODD_MASK64)
    c2 = pext_u64(UInt64(x >> 128), ODD_MASK64)
    c3 = pext_u64(UInt64(x >> 192), ODD_MASK64)
    return UInt128(c0) | (UInt128(c1) << 32) | (UInt128(c2) << 64) | (UInt128(c3) << 96)
end

@inline function zip_even_bit(x::UInt512)
    c0 = pext_u64(UInt64(x), EVEN_MASK64)
    c1 = pext_u64(UInt64(x >> 64), EVEN_MASK64)
    c2 = pext_u64(UInt64(x >> 128), EVEN_MASK64)
    c3 = pext_u64(UInt64(x >> 192), EVEN_MASK64)
    c4 = pext_u64(UInt64(x >> 256), EVEN_MASK64)
    c5 = pext_u64(UInt64(x >> 320), EVEN_MASK64)
    c6 = pext_u64(UInt64(x >> 384), EVEN_MASK64)
    c7 = pext_u64(UInt64(x >> 448), EVEN_MASK64)
    
    return UInt256(c0) | (UInt256(c1) << 32) | (UInt256(c2) << 64) | (UInt256(c3) << 96) | 
           (UInt256(c4) << 128) | (UInt256(c5) << 160) | (UInt256(c6) << 192) | (UInt256(c7) << 224)
end

@inline function zip_odd_bit(x::UInt512)
    c0 = pext_u64(UInt64(x), ODD_MASK64)
    c1 = pext_u64(UInt64(x >> 64), ODD_MASK64)
    c2 = pext_u64(UInt64(x >> 128), ODD_MASK64)
    c3 = pext_u64(UInt64(x >> 192), ODD_MASK64)
    c4 = pext_u64(UInt64(x >> 256), ODD_MASK64)
    c5 = pext_u64(UInt64(x >> 320), ODD_MASK64)
    c6 = pext_u64(UInt64(x >> 384), ODD_MASK64)
    c7 = pext_u64(UInt64(x >> 448), ODD_MASK64)
    
    return UInt256(c0) | (UInt256(c1) << 32) | (UInt256(c2) << 64) | (UInt256(c3) << 96) | 
           (UInt256(c4) << 128) | (UInt256(c5) << 160) | (UInt256(c6) << 192) | (UInt256(c7) << 224)
end

@inline unzip_even_bit(x::UInt8)  = UInt16(pdep_u32(UInt32(x), EVEN_MASK32))
@inline unzip_odd_bit(x::UInt8)   = UInt16(pdep_u32(UInt32(x), ODD_MASK32))
@inline unzip_even_bit(x::UInt16) = UInt32(pdep_u32(UInt32(x), EVEN_MASK32))
@inline unzip_odd_bit(x::UInt16)  = UInt32(pdep_u32(UInt32(x), ODD_MASK32))
@inline unzip_even_bit(x::UInt32) = pdep_u64(UInt64(x), EVEN_MASK64)
@inline unzip_odd_bit(x::UInt32)  = pdep_u64(UInt64(x), ODD_MASK64)

@inline function unzip_even_bit(x::UInt64)
    lo = pdep_u64(x & 0xFFFFFFFF, EVEN_MASK64)
    hi = pdep_u64(x >> 32, EVEN_MASK64)
    return (UInt128(hi) << 64) | UInt128(lo)
end

@inline function unzip_odd_bit(x::UInt64)
    lo = pdep_u64(x & 0xFFFFFFFF, ODD_MASK64)
    hi = pdep_u64(x >> 32, ODD_MASK64)
    return (UInt128(hi) << 64) | UInt128(lo)
end

@inline function unzip_even_bit(x::UInt128)
    c0 = pdep_u64(UInt64(x) & 0xFFFFFFFF, EVEN_MASK64)
    c1 = pdep_u64(UInt64(x >> 32) & 0xFFFFFFFF, EVEN_MASK64)
    c2 = pdep_u64(UInt64(x >> 64) & 0xFFFFFFFF, EVEN_MASK64)
    c3 = pdep_u64(UInt64(x >> 96) & 0xFFFFFFFF, EVEN_MASK64)
    return UInt256(c0) | (UInt256(c1) << 64) | (UInt256(c2) << 128) | (UInt256(c3) << 192)
end

@inline function unzip_odd_bit(x::UInt128)
    c0 = pdep_u64(UInt64(x) & 0xFFFFFFFF, ODD_MASK64)
    c1 = pdep_u64(UInt64(x >> 32) & 0xFFFFFFFF, ODD_MASK64)
    c2 = pdep_u64(UInt64(x >> 64) & 0xFFFFFFFF, ODD_MASK64)
    c3 = pdep_u64(UInt64(x >> 96) & 0xFFFFFFFF, ODD_MASK64)
    return UInt256(c0) | (UInt256(c1) << 64) | (UInt256(c2) << 128) | (UInt256(c3) << 192)
end

@inline function unzip_even_bit(x::UInt256)
    c0 = pdep_u64(UInt64(x) & 0xFFFFFFFF, EVEN_MASK64)
    c1 = pdep_u64(UInt64(x >> 32) & 0xFFFFFFFF, EVEN_MASK64)
    c2 = pdep_u64(UInt64(x >> 64) & 0xFFFFFFFF, EVEN_MASK64)
    c3 = pdep_u64(UInt64(x >> 96) & 0xFFFFFFFF, EVEN_MASK64)
    c4 = pdep_u64(UInt64(x >> 128) & 0xFFFFFFFF, EVEN_MASK64)
    c5 = pdep_u64(UInt64(x >> 160) & 0xFFFFFFFF, EVEN_MASK64)
    c6 = pdep_u64(UInt64(x >> 192) & 0xFFFFFFFF, EVEN_MASK64)
    c7 = pdep_u64(UInt64(x >> 224) & 0xFFFFFFFF, EVEN_MASK64)
    
    return UInt512(c0) | (UInt512(c1) << 64) | (UInt512(c2) << 128) | (UInt512(c3) << 192) |
           (UInt512(c4) << 256) | (UInt512(c5) << 320) | (UInt512(c6) << 384) | (UInt512(c7) << 448)
end

@inline function unzip_odd_bit(x::UInt256)
    c0 = pdep_u64(UInt64(x) & 0xFFFFFFFF, ODD_MASK64)
    c1 = pdep_u64(UInt64(x >> 32) & 0xFFFFFFFF, ODD_MASK64)
    c2 = pdep_u64(UInt64(x >> 64) & 0xFFFFFFFF, ODD_MASK64)
    c3 = pdep_u64(UInt64(x >> 96) & 0xFFFFFFFF, ODD_MASK64)
    c4 = pdep_u64(UInt64(x >> 128) & 0xFFFFFFFF, ODD_MASK64)
    c5 = pdep_u64(UInt64(x >> 160) & 0xFFFFFFFF, ODD_MASK64)
    c6 = pdep_u64(UInt64(x >> 192) & 0xFFFFFFFF, ODD_MASK64)
    c7 = pdep_u64(UInt64(x >> 224) & 0xFFFFFFFF, ODD_MASK64)
    
    return UInt512(c0) | (UInt512(c1) << 64) | (UInt512(c2) << 128) | (UInt512(c3) << 192) |
           (UInt512(c4) << 256) | (UInt512(c5) << 320) | (UInt512(c6) << 384) | (UInt512(c7) << 448)
end



@inline encode(x::UInt8, z::UInt8) = UInt16(x) << 8 | UInt16(z)
@inline encode(x::UInt16, z::UInt16) = UInt32(x) << 16 | UInt32(z)
@inline encode(x::UInt32, z::UInt32) = UInt64(x) << 32 | UInt64(z)
@inline encode(x::UInt64, z::UInt64) = UInt128(x) << 64 | UInt128(z)
@inline encode(x::UInt128, z::UInt128) = UInt256(x) << 128 | UInt256(z)
@inline encode(x::UInt256, z::UInt256) = UInt512(x) << 256 | UInt512(z)

@inline function decode(q::UInt16)
    return unsafe_trunc(UInt8, q >> 8), unsafe_trunc(UInt8, q)
end

@inline function decode(q::UInt32)
    return unsafe_trunc(UInt16, q >> 16), unsafe_trunc(UInt16, q)
end

@inline function decode(q::UInt64)
    return unsafe_trunc(UInt32, q >> 32), unsafe_trunc(UInt32, q)
end

@inline function decode(q::UInt128)
    return unsafe_trunc(UInt64, q >> 64), unsafe_trunc(UInt64, q)
end

@inline function decode(q::UInt256)
    return unsafe_trunc(UInt128, q >> 128), unsafe_trunc(UInt128, q)
end

@inline function decode(q::UInt512)
    return unsafe_trunc(UInt256, q >> 256), unsafe_trunc(UInt256, q)
end



function cal_binomial_cache(n::Int)
    a = Int[]
    for i in 1:n
        append!(a, [binomial(i, j) for j in 0:i-1])
    end

    return a
end

const BINOMIAL_CACHE::Vector{Int} = cal_binomial_cache(32)

@inline function cached_binomial(n::Int, k::Int)
    k > n   && return 0
    k == n  && return 1
    n <= 32 && return BINOMIAL_CACHE[n * (n - 1) ÷ 2 + k + 1]
    return binomial(n, k)
end

