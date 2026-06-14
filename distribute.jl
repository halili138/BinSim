# # dist.jl 补充逻辑

# # 从一批属于同一对称性的 SVDGroups 直接构造出 C++ 的小 OTF 网络
# function OTF(basis::BasisManager, groups::Vector{SVDGroup{Ti,Tv}}) where {Ti,Tv}
#     ngs = length(groups)
#     axs = Vector{Ti}(undef, ngs)
#     bxs = Vector{Ti}(undef, ngs)
#     ranks = Vector{Int64}(undef, ngs)
#     num_as = Vector{Int64}(undef, ngs)
#     num_bs = Vector{Int64}(undef, ngs)
#     flat_azs = Ti[]
#     flat_bzs = Ti[]
#     flat_wa = Tv[]
#     flat_wb = Tv[]

#     for (g, group) in enumerate(groups)
#         axs[g] = group.ax
#         bxs[g] = group.bx
#         ranks[g] = group.rank
#         num_as[g] = length(group.azs)
#         num_bs[g] = length(group.bzs)
#         append!(flat_azs, group.azs)
#         append!(flat_bzs, group.bzs)
#         append!(flat_wa, vec(group.wa))
#         append!(flat_wb, vec(group.wb))
#     end

#     ptr = Tv <: Complex ? (
#         @ccall LIB_OTF.build_network_otf_c64(basis.ptr::Ptr{Cvoid}, basis.norb::Int64, ngs::Int64,
#         axs::Ptr{Ti}, bxs::Ptr{Ti}, ranks::Ptr{Int64}, num_as::Ptr{Int64}, num_bs::Ptr{Int64},
#         flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti}, flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv})::Ptr{Cvoid}
#     ) : (
#         @ccall LIB_OTF.build_network_otf_f64(basis.ptr::Ptr{Cvoid}, basis.norb::Int64, ngs::Int64,
#         axs::Ptr{Ti}, bxs::Ptr{Ti}, ranks::Ptr{Int64}, num_as::Ptr{Int64}, num_bs::Ptr{Int64},
#         flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti}, flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv})::Ptr{Cvoid}
#     )

#     ptr == C_NULL && error("Failed to create C++ OTFNET.")

#     obj = OTF(ptr, basis.dim, ngs)

#     finalizer(obj) do o
#         if o.ptr != C_NULL
#             Tv <: Complex ?
#             (@ccall LIB_OTF.destroy_network_otf_c64(o.ptr::Ptr{Cvoid})::Cvoid) :
#             (@ccall LIB_OTF.destroy_network_otf_f64(o.ptr::Ptr{Cvoid})::Cvoid)
#             o.ptr = C_NULL
#         end
#     end

#     return obj
# end

# # 核心拆解函数：将哈密顿量切分为多个独立的子 OTF
# function build_distributed_otfs(basis::BasisManager, A::BinaryQubitAABB{Ti,Tv,K,V}, tol::Float64=1e-12) where {Ti,Tv,K,V}
#     groups = compress_by_svd(A, tol) # 借用你已有的 SVD 压缩 [cite: 46]

#     # 按照 (asym, bsym) 进行严密分类
#     dict = Dict{Tuple{Int,Int},Vector{SVDGroup{Ti,Tv}}}()
#     for g in groups
#         asym = get_symm(g.ax, basis.orbsym)
#         bsym = get_symm(g.bx, basis.orbsym)
#         sym_key = (asym, bsym)
#         if !haskey(dict, sym_key)
#             dict[sym_key] = SVDGroup{Ti,Tv}[]
#         end
#         push!(dict[sym_key], g)
#     end

#     # 将字典里的分类化作一个个 C++ 的微型计算网络
#     return [OTF(basis, sub_groups) for sub_groups in values(dict)]
# end

# function test_real_mpi_simulation(name, ratio, basis_name)
#     # ... MPI 初始化，构建 basis 和 mole [cite: 6]

#     # 1. 初始化一次全局地图 (Global Map)
#     gmap = GlobalMemMap(basis, comm)
#     local_dim = get_local_dim(gmap)

#     # 2. 生成几十个微小的子网络及其专属拓扑
#     sub_otfs = build_distributed_otfs(basis, ham)
#     sub_topos = [SubTopology(basis, sub_otf, gmap) for sub_otf in sub_otfs]

#     # 3. 找出所有片段中的最大通信量，开辟极限复用内存池
#     max_send_dim = maximum(t.send_dim for t in sub_topos)
#     max_recv_dim = maximum(t.recv_dim for t in sub_topos)

#     # 【神来之笔：连续缓存】 前半段永远安全地放 local_v，后半段是即用即毁的接收区
#     cache = zeros(Float64, local_dim + max_recv_dim)
#     local_v = @view cache[1:local_dim]
#     local_w = zeros(Float64, local_dim)
#     send_buffer = zeros(Float64, max_send_dim)

#     # 获取 HF 态并填充 local_v (同之前 scatter_global_v)
#     # ...

#     for step in 1:10
#         t0 = time_ns()
#         fill!(local_w, 0.0) # 累加目标必须清零

#         # 4. 对几十个分段轮流发起即用即毁的通信计算！
#         for i in eachindex(sub_otfs)
#             topo = sub_topos[i]
#             if topo.send_dim == 0 && topo.recv_dim == 0
#                 # 若纯本地，直接算
#                 @ccall LIB_DIST.compute_hvec_sub_chunk_f64(basis.ptr::Ptr{Cvoid}, sub_otfs[i].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, cache::Ptr{Float64}, local_w::Ptr{Float64})::Cvoid
#             else
#                 # A. 提取本片段该发的数据
#                 @ccall LIB_DIST.pack_send_buffer_f64(topo.ptr::Ptr{Cvoid}, cache::Ptr{Float64}, send_buffer::Ptr{Float64})::Cvoid

#                 # B. MPI 点对点交换 (仅交换本片段极小的数据量！)
#                 recv_view = @view cache[local_dim + 1 : local_dim + topo.recv_dim]
#                 MPI.Alltoallv!(MPI.VBuffer(send_buffer, topo.send_counts), MPI.VBuffer(recv_view, topo.recv_counts), comm)

#                 # C. 让 C++ 内核把这次算完，算完后 recv_view 的数据寿命终止
#                 @ccall LIB_DIST.compute_hvec_sub_chunk_f64(basis.ptr::Ptr{Cvoid}, sub_otfs[i].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, cache::Ptr{Float64}, local_w::Ptr{Float64})::Cvoid
#             end
#         end

#         @. local_v -= dτ * local_w
#         # ... 全局归一化逻辑 ...
#     end
# end


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

function SubTopology(basis::BasisManager, subnet::OTF, gmap::GlobalMemMap)
    ptr = @ccall LIB_DIST.build_sub_topology_otf_f64(basis.ptr::Ptr{Cvoid}, subnet.ptr::Ptr{Cvoid}, gmap.ptr::Ptr{Cvoid})::Ptr{Cvoid}

    dims = zeros(Int64, 2)
    @ccall LIB_DIST.get_sub_topology_info_otf(ptr::Ptr{Cvoid}, pointer(dims, 1)::Ptr{Int64}, pointer(dims, 2)::Ptr{Int64})::Cvoid

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
