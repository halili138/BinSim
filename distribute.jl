using MPI

const LIB_DIST = joinpath(@__DIR__, "src/lib/libdist.so")

# ==========================================
# 1. 全局内存图 (GlobalMemMap)
# ==========================================
mutable struct GlobalMemMap
    ptr::Ptr{Cvoid}
    local_dim::Int64
    mpi_rank::Int
    mpi_size::Int
end

function GlobalMemMap(basis::BasisManager, comm::MPI.Comm; Tv::DataType=Float64)
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    ptr = @ccall LIB_DIST.build_global_map_otf_f64(basis.ptr::Ptr{Cvoid}, rank::Cint, size::Cint)::Ptr{Cvoid}
    local_dim = @ccall LIB_DIST.get_local_dim_otf_gmap(ptr::Ptr{Cvoid})::Int64

    obj = GlobalMemMap(ptr, local_dim, rank, size)
    finalizer(obj) do o
        o.ptr != C_NULL && @ccall LIB_DIST.destroy_global_map_otf(o.ptr::Ptr{Cvoid})::Cvoid
    end
    return obj
end

function GlobalMemMap(basis::BasisManager; rank::Int=0, size::Int=1)
    ptr = @ccall LIB_DIST.build_global_map_otf_f64(basis.ptr::Ptr{Cvoid}, rank::Cint, size::Cint)::Ptr{Cvoid}
    local_dim = @ccall LIB_DIST.get_local_dim_otf_gmap(ptr::Ptr{Cvoid})::Int64

    obj = GlobalMemMap(ptr, local_dim, rank, size)
    finalizer(obj) do o
        o.ptr != C_NULL && @ccall LIB_DIST.destroy_global_map_otf(o.ptr::Ptr{Cvoid})::Cvoid
    end
    return obj
end

# ==========================================
# 2. 分段通信账本 (SubTopology)
# ==========================================
mutable struct SubTopology
    ptr::Ptr{Cvoid}
    send_dim::Int64
    recv_dim::Int64
    send_counts::Vector{Cint}
    recv_counts::Vector{Cint}
end

function SubTopology(basis::BasisManager, subnet::OTF, gmap::GlobalMemMap; num_phases::Int=1, phase_idx::Int=0)
    ptr = @ccall LIB_DIST.build_sub_topology_otf_f64(
        basis.ptr::Ptr{Cvoid}, subnet.ptr::Ptr{Cvoid}, gmap.ptr::Ptr{Cvoid}, num_phases::Cint, phase_idx::Cint,
    )::Ptr{Cvoid}

    dims = zeros(Int64, 2)
    @ccall LIB_DIST.get_sub_topology_info_otf(
        ptr::Ptr{Cvoid}, pointer(dims, 1)::Ptr{Int64}, pointer(dims, 2)::Ptr{Int64}
    )::Cvoid

    send_counts = zeros(Cint, gmap.mpi_size)
    recv_counts = zeros(Cint, gmap.mpi_size)

    @ccall LIB_DIST.get_sub_topology_mpi_counts_otf(ptr::Ptr{Cvoid}, send_counts::Ptr{Cint}, recv_counts::Ptr{Cint})::Cvoid

    obj = SubTopology(ptr, dims[1], dims[2], send_counts, recv_counts)
    finalizer(obj) do o
        o.ptr != C_NULL && @ccall LIB_DIST.destroy_sub_topology_otf(o.ptr::Ptr{Cvoid})::Cvoid
    end
    return obj
end

# ==========================================
# 3. 对称性切割子网络生成器
# ==========================================
function OTF_from_groups(basis::BasisManager, groups::Vector{SVDGroup{Ti,Tv}}) where {Ti,Tv}
    ngs = length(groups)
    axs = Vector{Ti}(undef, ngs)
    bxs = Vector{Ti}(undef, ngs)
    ranks = Vector{Int64}(undef, ngs)
    num_as = Vector{Int64}(undef, ngs)
    num_bs = Vector{Int64}(undef, ngs)
    flat_azs = Ti[]
    flat_bzs = Ti[]
    flat_wa = Tv[]
    flat_wb = Tv[]

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

    ptr = @ccall LIB_OTF.build_network_otf_f64(
        basis.ptr::Ptr{Cvoid}, basis.norb::Int64, ngs::Int64,
        axs::Ptr{Ti}, bxs::Ptr{Ti}, ranks::Ptr{Int64}, num_as::Ptr{Int64}, num_bs::Ptr{Int64},
        flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti}, flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv}
    )::Ptr{Cvoid}

    ptr == C_NULL && error("Failed to create sub OTFNET.")
    obj = OTF(ptr, basis.dim, ngs)
    finalizer(obj) do o
        o.ptr != C_NULL && @ccall LIB_OTF.destroy_network_otf_f64(o.ptr::Ptr{Cvoid})::Cvoid
    end
    return obj
end

function build_distributed_otfs(basis::BasisManager, A::BinaryQubitAABB{Ti,Tv,K,V}, tol::Float64=1e-12) where {Ti,Tv,K,V}
    # 借助 network.jl 中的原生 SVD 压缩 [cite: 43]
    groups = compress_by_svd(A, tol)

    # 按严格的 (asym, bsym) 物理对称性分桶
    dict = Dict{Tuple{Int,Int},Vector{SVDGroup{Ti,Tv}}}()
    for g in groups
        # 借助 symm.jl 中的对称性解析 [cite: 16]
        asym = get_symm(g.ax, basis.orbsym)
        bsym = get_symm(g.bx, basis.orbsym)
        sym_key = (asym, bsym)
        if !haskey(dict, sym_key)
            dict[sym_key] = SVDGroup{Ti,Tv}[]
        end
        push!(dict[sym_key], g)
    end

    # 转化成独立的 C++ 微型计算网络数组
    return [OTF_from_groups(basis, sub_groups) for sub_groups in values(dict)]
end

function set_local_reference_state!(
    gmap::GlobalMemMap, basis::BasisManager, local_v::AbstractVector{Tv},
    astrs::Vector{UInt32}, bstrs::Vector{UInt32}, vals::Vector{Tv}
) where Tv
    @assert length(astrs) == length(bstrs) == length(vals)
    fill!(local_v, zero(Tv))

    if Tv <: Complex
        for (astr, bstr, val) in zip(astrs, bstrs, vals)
            @ccall LIB_DIST.set_local_det_coeff_c64(
                gmap.ptr::Ptr{Cvoid}, basis.ptr::Ptr{Cvoid},
                astr::UInt32, bstr::UInt32, val::Cdouble, local_v::Ptr{ComplexF64}
            )::Cvoid
        end
    else
        for (astr, bstr, val) in zip(astrs, bstrs, vals)
            @ccall LIB_DIST.set_local_det_coeff_f64(
                gmap.ptr::Ptr{Cvoid}, basis.ptr::Ptr{Cvoid},
                astr::UInt32, bstr::UInt32, val::Cdouble, local_v::Ptr{Cdouble}
            )::Cvoid
        end
    end
end

function set_local_hf!(gmap::GlobalMemMap, basis::BasisManager, local_v::AbstractVector{Tv}, nelec::Tuple{Int,Int}) where Tv
    na, nb = nelec

    hf_astr = UInt32(0)
    for i in 0:na-1
        hf_astr |= (UInt32(1) << i)
    end

    hf_bstr = UInt32(0)
    for i in 0:nb-1
        hf_bstr |= (UInt32(1) << i)
    end

    set_local_reference_state!(gmap, basis, local_v, [hf_astr], [hf_bstr], [Tv(1.0)])
end
