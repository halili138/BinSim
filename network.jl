const LIB_BASIS = joinpath(@__DIR__, "src/lib/libbasis.so")
const LIB_NET = joinpath(@__DIR__, "src/lib/libnet.so")
const LIB_AGG = joinpath(@__DIR__, "src/lib/libagg.so")

mutable struct BasisManager
    ptr::Ptr{Cvoid}
    dim::Int64

    function BasisManager(norb::Int64, nelec::Tuple{Int64,Int64}, orbsym::Vector{Int64})
        total_sym = 0
        num_irreps = 16
        na, nb = nelec

        ptr = @ccall LIB_BASIS.create_basis_manager(
            norb::Int64, 
            na::Int64, 
            nb::Int64, 
            total_sym::Int64,
            orbsym::Ptr{Int64}, 
            num_irreps::Int64,
        )::Ptr{Cvoid}

        ptr == C_NULL && error("Failed to create C++ BasisManager.")

        dim = @ccall LIB_BASIS.get_subspace_dim(ptr::Ptr{Cvoid})::Int64

        obj = new(ptr, dim)

        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_BASIS.destroy_basis_manager(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end

        return obj
    end
end

function get_hf(
    basis::BasisManager,
    nelec::Tuple{Int, Int},
    orbsym::Vector{Int},
)
    hf = zeros(Float64, basis.dim)

    na, nb = nelec

    hf_astr = UInt32(0)
    for i in 0:na-1
        hf_astr |= (UInt32(1) << i)
    end

    hf_bstr = UInt32(0)
    for i in 0:nb-1
        hf_bstr |= (UInt32(1) << i)
    end

    hf_val = 1.0

    @ccall LIB_BASIS.set_det_coeff(
        basis.ptr::Ptr{Cvoid},
        hf_astr::UInt32,
        hf_bstr::UInt32,
        hf_val::Cdouble,
        orbsym::Ptr{Int64},
        hf::Ptr{Cdouble},
    )::Cvoid

    return hf
end

function get_xs_groups(H0b::BinaryQubitAABB)
    ngs = length(H0b.gs) - 1
    axs = Vector{UInt32}(undef, ngs)
    bxs = Vector{UInt32}(undef, ngs)
    for g in 1:ngs
        lb = H0b.gs[g] + 1
        axs[g] = H0b.axs[lb]
        bxs[g] = H0b.bxs[lb]
    end
    return axs, bxs
end

function get_xs_groups(H0b::T, pool_1b::Array{Array{T,1},1}) where T<:HostBinaryQubit{UInt64,Float64,UInt128,Int64}
    ngs = length(H0b.gs) - 1
    hxs = Vector{UInt64}(undef, ngs)
    for g in 1:ngs
        lb = H0b.gs[g] + 1
        hxs[g] = H0b.xs[lb]
    end

    tol_xs = copy(hxs) # Use copy to avoid aliasing
    idxs  = Int64[]
    count = ngs
    
    # Hash map for O(1) exact lookups
    xs_dict = Dict{UInt64, Int64}()
    for (i, x) in enumerate(tol_xs)
        xs_dict[x] = i
    end

    # 为了使用expN2和 grad2 来产生exp和梯度算符，pool里面是[op, op^2]，第一项对应原来的单个feimion算符，具有唯一的 x;
    # I + sinθ * op + (1-cosθ) * op^2，ferimon算符，即 τ = a†a†aa - conj，其平方由常数和纯 Z 构成，不会产生新的 x，
    # 这也是为什么其 expm 只有diag演化和唯一的off演化的原因。
    for t in pool_1b
        @assert (length(t[1].gs) - 1) == 1 
        x = t[1].xs[1]
        if haskey(xs_dict, x)
            push!(idxs, xs_dict[x])
        else
            count += 1
            push!(tol_xs, x)
            push!(idxs, count)
            xs_dict[x] = count
        end
    end

    tol_axs = zip_even_bit.(tol_xs)
    tol_bxs = zip_odd_bit.(tol_xs)

    return tol_axs, tol_bxs, idxs .- 1
end

struct SVDGroup{Ti, Tv}
    ax::Ti
    bx::Ti
    rank::Int
    azs::Vector{Ti}
    bzs::Vector{Ti}
    wa::Matrix{Tv}
    wb::Matrix{Tv}
    ncs::Int
end

function compress_by_svd(A0b::BinaryQubitAABB{Ti,Tv,Tg}, tol::Float64=1e-12) where {Ti, Tv, Tg}
    ngs = length(A0b.gs) - 1

    svd_groups = Vector{SVDGroup{Ti, Tv}}(undef, ngs)

    @threads for g in 1:ngs
        lb = A0b.gs[g] + 1
        rb = A0b.gs[g + 1]
        
        ax = A0b.axs[lb]
        bx = A0b.bxs[lb]
        
        sub_azs = A0b.azs[lb:rb]
        sub_bzs = A0b.bzs[lb:rb]
        sub_cs  = A0b.cs[lb:rb]
        
        unique_azs = unique(sub_azs)
        unique_bzs = unique(sub_bzs)

        sort!(unique_azs)
        sort!(unique_bzs)

        Na = length(unique_azs)
        Nb = length(unique_bzs)
        
        az_map = Dict(z => i for (i, z) in enumerate(unique_azs))
        bz_map = Dict(z => i for (i, z) in enumerate(unique_bzs))
        
        M = zeros(Tv, Na, Nb)
        for (k, c) in enumerate(sub_cs)
            i = az_map[sub_azs[k]]
            j = bz_map[sub_bzs[k]]
            M[i, j] += c
        end
        
        F = svd(M)
        
        valid_idx = findall(x -> abs(x) > tol, F.S)
        rank = length(valid_idx)
        
        wa = zeros(Tv, Na, rank)
        wb = zeros(Tv, Nb, rank)
        
        for (ri, r) in enumerate(valid_idx)
            sqrt_s = sqrt(F.S[r])
            @. wa[:, ri] = F.U[:, r] * sqrt_s
            @. wb[:, ri] = F.Vt[r, :] * sqrt_s
        end

        wa[abs.(wa) .< 1e-12] .= 0
        wb[abs.(wb) .< 1e-12] .= 0

        svd_groups[g] = SVDGroup{Ti, Tv}(ax, bx, rank, unique_azs, unique_bzs, wa, wb, length(sub_cs))
    end
    
    return svd_groups
end

function compress_by_svd(pool_0b::Array{BinaryQubitAABB{Ti,Tv,Tg,K,V,G}, 1}, tol::Float64=1e-12) where {Ti,Tv,Tg,K,V,G}
    ngs = length(pool_0b)
    svd_groups = Vector{SVDGroup{Ti, Tv}}(undef, ngs)

    for (idx, op) in enumerate(pool_0b)
        gs = op.gs
        @assert (length(gs) - 1) <= 1 "only support excitation operator with ngs <= 1"
        ax  = op.axs[end]
        bx  = op.bxs[end]
        azs = op.azs
        bzs = op.bzs
        cs  = op.cs
        
        unique_azs = unique(azs)
        unique_bzs = unique(bzs)
        sort!(unique_azs)
        sort!(unique_bzs)

        Na = length(unique_azs)
        Nb = length(unique_bzs)
        
        az_map = Dict(z => i for (i, z) in enumerate(unique_azs))
        bz_map = Dict(z => i for (i, z) in enumerate(unique_bzs))
        
        M = zeros(Tv, Na, Nb)
        for (k, c) in enumerate(cs)
            i = az_map[azs[k]]
            j = bz_map[bzs[k]]
            M[i, j] += c
        end
        
        F = svd(M)
        
        valid_idx = findall(x -> abs(x) > tol, F.S)
        rank = length(valid_idx)
        
        wa = zeros(Tv, Na, rank)
        wb = zeros(Tv, Nb, rank)
        
        for (ri, r) in enumerate(valid_idx)
            sqrt_s = sqrt(F.S[r])
            @. wa[:, ri] = F.U[:, r] * sqrt_s
            @. wb[:, ri] = F.Vt[r, :] * sqrt_s
        end
        
        wa[abs.(wa) .< 1e-12] .= 0
        wb[abs.(wb) .< 1e-12] .= 0

        svd_groups[idx] = SVDGroup{Ti, Tv}(ax, bx, rank, unique_azs, unique_bzs, wa, wb, length(cs))
    end
    
    return svd_groups
end

# mutable struct NET
#     ptr::Ptr{Cvoid}
#     dim::Int64

#     function NET(
#         basis::BasisManager, 
#         groups::Vector{SVDGroup{UInt32,Float64}}, 
#         orbsym::Vector{Int64}
#     )
#         ngs    = length(groups)
#         axs    = Vector{UInt32}(undef, ngs)
#         bxs    = Vector{UInt32}(undef, ngs)
#         ranks  = Vector{Int64}(undef,  ngs)
#         num_as = Vector{Int64}(undef,  ngs)
#         num_bs = Vector{Int64}(undef,  ngs)
        
#         flat_azs = UInt32[]
#         flat_bzs = UInt32[]
#         flat_wa  = Float64[]
#         flat_wb  = Float64[]
        
#         for (g, group) in enumerate(groups)
#             axs[g]    = group.ax
#             bxs[g]    = group.bx
#             ranks[g]  = group.rank
            
#             na = length(group.azs)
#             nb = length(group.bzs)
#             num_as[g] = na
#             num_bs[g] = nb
            
#             append!(flat_azs, group.azs)
#             append!(flat_bzs, group.bzs)
            
#             append!(flat_wa, vec(group.wa))
#             append!(flat_wb, vec(group.wb))
#         end
        
#         ptr = @ccall LIB_NET.create_svd_network(
#             basis.ptr::Ptr{Cvoid},
#             ngs::Int64,
#             axs::Ptr{UInt32},
#             bxs::Ptr{UInt32},
#             ranks::Ptr{Int64},
#             num_as::Ptr{Int64},
#             num_bs::Ptr{Int64},
#             flat_azs::Ptr{UInt32},
#             flat_bzs::Ptr{UInt32},
#             flat_wa::Ptr{Cdouble},
#             flat_wb::Ptr{Cdouble},
#             orbsym::Ptr{Int64},
#         )::Ptr{Cvoid}
        
#         ptr == C_NULL && error("Failed to create C++ SVDNetwork.")
        
#         obj = new(ptr, basis.dim)
#         finalizer(obj) do o
#             if o.ptr != C_NULL
#                 @ccall LIB_NET.destroy_svd_network(o.ptr::Ptr{Cvoid})::Cvoid
#                 o.ptr = C_NULL
#             end
#         end
        
#         return obj
#     end
# end

mutable struct NET
    ptr::Ptr{Cvoid}
    dim::Int64

    function NET(
        basis::BasisManager, 
        A0b::BinaryQubitAABB, 
        orbsym::Vector{Int64}, 
        tol::Float64=1e-12,
    )
        groups = compress_by_svd(A0b, tol)
        ngs    = length(groups)
        ncs    = length(A0b.cs)
        axs    = Vector{UInt32}(undef, ngs)
        bxs    = Vector{UInt32}(undef, ngs)
        ranks  = Vector{Int64}(undef,  ngs)
        num_as = Vector{Int64}(undef,  ngs)
        num_bs = Vector{Int64}(undef,  ngs)
        
        flat_azs = UInt32[]
        flat_bzs = UInt32[]
        flat_wa  = Float64[]
        flat_wb  = Float64[]
        
        for (g, group) in enumerate(groups)
            axs[g]    = group.ax
            bxs[g]    = group.bx
            ranks[g]  = group.rank
            
            na = length(group.azs)
            nb = length(group.bzs)
            num_as[g] = na
            num_bs[g] = nb
            
            append!(flat_azs, group.azs)
            append!(flat_bzs, group.bzs)
            
            append!(flat_wa, vec(group.wa))
            append!(flat_wb, vec(group.wb))
        end
        
        ptr = @ccall LIB_NET.create_svd_network(
            basis.ptr::Ptr{Cvoid},
            ncs::Int64,
            ngs::Int64,
            axs::Ptr{UInt32},
            bxs::Ptr{UInt32},
            A0b.azs::Ptr{UInt32},
            A0b.bzs::Ptr{UInt32},
            A0b.cs::Ptr{Cdouble},
            A0b.gs::Ptr{Int64},
            ranks::Ptr{Int64},
            num_as::Ptr{Int64},
            num_bs::Ptr{Int64},
            flat_azs::Ptr{UInt32},
            flat_bzs::Ptr{UInt32},
            flat_wa::Ptr{Cdouble},
            flat_wb::Ptr{Cdouble},
            orbsym::Ptr{Int64},
        )::Ptr{Cvoid}
        
        ptr == C_NULL && error("Failed to create C++ SVDNetwork.")
        
        obj = new(ptr, basis.dim)
        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_NET.destroy_svd_network(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end
        
        return obj
    end
end


mutable struct AGG
    ptr::Ptr{Cvoid}
    dim::Int64

    function AGG(
        basis::BasisManager, 
        A0b::BinaryQubitAABB, 
        orbsym::Vector{Int64}, 
        tol::Float64=1e-12,
    )      
        groups = compress_by_svd(A0b, tol)
        ngs    = length(groups)
        ncs    = length(A0b.cs)
        axs    = Vector{UInt32}(undef, ngs)
        bxs    = Vector{UInt32}(undef, ngs)
        ranks  = Vector{Int64}(undef,  ngs)
        num_as = Vector{Int64}(undef,  ngs)
        num_bs = Vector{Int64}(undef,  ngs)
        
        flat_azs = UInt32[]
        flat_bzs = UInt32[]
        flat_wa  = Float64[]
        flat_wb  = Float64[]
        
        for (g, group) in enumerate(groups)
            axs[g] = group.ax
            bxs[g] = group.bx
            ranks[g] = group.rank
            num_as[g] = length(group.azs)
            num_bs[g] = length(group.bzs)
            append!(flat_azs, group.azs)
            append!(flat_bzs, group.bzs)
            append!(flat_wa, vec(group.wa))
            append!(flat_wb, vec(group.wb))
        end

        ptr = @ccall LIB_AGG.build_direct_agg_network(
            basis.ptr::Ptr{Cvoid}, 
            ncs::Int64,
            ngs::Int64,
            axs::Ptr{UInt32},
            bxs::Ptr{UInt32},
            A0b.azs::Ptr{UInt32},
            A0b.bzs::Ptr{UInt32},
            A0b.cs::Ptr{Cdouble},
            A0b.gs::Ptr{Int64},
            ranks::Ptr{Int64},
            num_as::Ptr{Int64}, 
            num_bs::Ptr{Int64},
            flat_azs::Ptr{UInt32}, 
            flat_bzs::Ptr{UInt32},
            flat_wa::Ptr{Cdouble}, 
            flat_wb::Ptr{Cdouble},
            orbsym::Ptr{Int64},
        )::Ptr{Cvoid}
        
        ptr == C_NULL && error("Failed to create C++ AGG.")
        
        obj = new(ptr, basis.dim)
        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_AGG.destroy_direct_agg_network(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end

        return obj
    end
end

function get_diags(basis::BasisManager, agg::AGG)
    dim = basis.dim
    diags = zeros(Float64, dim)

    @ccall LIB_AGG.get_diagonal_elements_agg(
        basis.ptr::Ptr{Cvoid},
        agg.ptr::Ptr{Cvoid},
        diags::Ptr{Cdouble},
    )::Cvoid

    return diags
end

function hvec_direct_agg!(
    basis::BasisManager, 
    agg::AGG,
    src::Vector{Float64},
    dst::Vector{Float64},
)
    @ccall LIB_AGG.hvec_direct_agg_network(
        basis.ptr::Ptr{Cvoid},
        agg.ptr::Ptr{Cvoid},
        src::Ptr{Cdouble},
        dst::Ptr{Cdouble},
    )::Cvoid
end

function print_info(agg::AGG)
    @ccall LIB_AGG.print_agg_network_info(agg.ptr::Ptr{Cvoid})::Cvoid
end

function hvec_direct_agg_benchmark!(
    basis::BasisManager, 
    agg::AGG,
    src::Vector{Float64},
    dst::Vector{Float64},
    measure::Bool,
)
    @ccall LIB_AGG.hvec_direct_agg_network_benchmark(
        basis.ptr::Ptr{Cvoid},
        agg.ptr::Ptr{Cvoid},
        src::Ptr{Cdouble},
        dst::Ptr{Cdouble},
        measure::Cint,
    )::Cvoid
end
