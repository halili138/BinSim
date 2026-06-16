include("binsim.jl")

# ============================================================
# DistributedFunctions — 分布式计算的统一接口
# ============================================================
# 调用风格与 OTF_Functions 保持一致：
#   funcs = DistributedFunctions(basis, ham, comm)
#   v  = funcs.get_hf(nelec)
#   Hv = funcs.zeros()
#   funcs.hvec(v, Hv)
#   funcs.normalize(v)
# ============================================================

struct DistributedFunctions{Tv}
    # === MPI ======================
    comm::MPI.Comm
    rank::Int
    size::Int

    # === 几何 =====================
    basis::BasisManager
    gmap::GlobalMemMap
    local_dim::Int64

    # === 哈密顿量子网络 ===========
    ham_sub_otfs::Vector{OTF}
    ham_sub_topos::Vector{SubTopology}

    # === 预分配缓冲区 ==============
    cache::Vector{Tv}     # local_dim + max_recv_dim
    local_w::Vector{Tv}   # local_dim
    send_buf::Vector{Tv}  # max_send_dim

    # === 闭包 (已实现) ============
    hvec::Function
    normalize::Function
    zeros::Function
    get_hf::Function
    get_init::Function
    inner::Function

    # === 占位符 (待实现) ==========
    expm::Function
    tvec::Function
    grad::Function
    backgrad::Function
    pool_otf::OTF
end

function DistributedFunctions(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    comm::MPI.Comm;
    tol::Float64=1e-12,
) where {Ti,Tv,TK,TV}
    @assert Tv == Float64 "DistributedFunctions currently only supports Tv=Float64 (C++ backend is _f64)"

    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # 1. 全局内存图：贪心分配基函数块到各 rank
    gmap = GlobalMemMap(basis, comm; Tv=Tv)
    local_dim = gmap.local_dim

    # 2. 哈密顿量按 (asym, bsym) 对称性切片
    ham_sub_otfs = build_distributed_otfs(basis, ham, tol)
    n_subnets = length(ham_sub_otfs)

    # 3. 每个切片构建独立通信路由表
    ham_sub_topos = Vector{SubTopology}(undef, n_subnets)
    for i in 1:n_subnets
        ham_sub_topos[i] = SubTopology(basis, ham_sub_otfs[i], gmap)
    end

    max_send_dim = n_subnets == 0 ? 0 : maximum(t.send_dim for t in ham_sub_topos)
    max_recv_dim = n_subnets == 0 ? 0 : maximum(t.recv_dim for t in ham_sub_topos)

    # 4. 预分配可复用缓冲区
    cache    = zeros(Tv, local_dim + max_recv_dim)
    local_w  = zeros(Tv, local_dim)
    send_buf = zeros(Tv, max_send_dim)

    # ====================================================
    # 闭包
    # ====================================================

    _inner = (lv::AbstractVector{Tv}, rv::AbstractVector{Tv}) -> begin
        local_dot = real(dot(lv, rv))
        return MPI.Allreduce(local_dot, +, comm)
    end

    _hvec = (v::AbstractVector{Tv}, Hv::AbstractVector{Tv}) -> begin
        fill!(local_w, zero(Tv))
        cache[1:local_dim] .= v

        for i in 1:n_subnets
            topo = ham_sub_topos[i]
            otf  = ham_sub_otfs[i]

            # pack (only if there is data to send)
            if topo.send_dim > 0
                @ccall LIB_DIST.pack_send_buffer_f64_sub(
                    topo.ptr::Ptr{Cvoid},
                    cache::Ptr{Float64}, send_buf::Ptr{Float64},
                )::Cvoid
            end

            # Alltoallv is collective — ALL ranks MUST call it,
            # even empty ones with zero-size buffers
            recv_view = @view cache[local_dim+1 : local_dim+topo.recv_dim]
            send_vbuf = MPI.VBuffer(send_buf, topo.send_counts)
            recv_vbuf = MPI.VBuffer(recv_view, topo.recv_counts)
            MPI.Alltoallv!(send_vbuf, recv_vbuf, comm)

            @ccall LIB_DIST.compute_hvec_sub_chunk_f64(
                basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
                topo.ptr::Ptr{Cvoid},
                cache::Ptr{Float64}, local_w::Ptr{Float64},
            )::Cvoid
        end

        Hv .= local_w
    end

    _normalize = (v::AbstractVector{Tv}) -> begin
        local_n2 = sum(abs2, v)
        global_n = sqrt(MPI.Allreduce(local_n2, +, comm))
        v ./= global_n
    end

    _zeros  = () -> zeros(Tv, local_dim)

    _get_hf = (nelec::Tuple{Int,Int}) -> begin
        v = zeros(Tv, local_dim)
        set_local_hf!(gmap, basis, v, nelec)
        return v
    end

    _get_init = (arefs::Vector{UInt32}, brefs::Vector{UInt32}, vals::Vector{Tv}) -> begin
        v = zeros(Tv, local_dim)
        set_local_reference_state!(gmap, basis, v, arefs, brefs, vals)
        return v
    end

    _expm     = (idx, θ, v)           -> error("DistributedFunctions.expm: not yet implemented")
    _tvec     = (idx, lv, rv)         -> error("DistributedFunctions.tvec: not yet implemented")
    _grad     = (idx, θ, lv, rv)      -> error("DistributedFunctions.grad: not yet implemented")
    _backgrad = (idx, θ, lv, rv)      -> error("DistributedFunctions.backgrad: not yet implemented")

    _pool_otf = OTF(C_NULL, 0, 0)

    if rank == 0
        println("\nDistributedFunctions built:")
        println("  MPI ranks:        $(size)")
        println("  Local dim (rank0): $(local_dim)")
        println("  Symmetry fragments: $(n_subnets)")
        println("  Max send/recv:    $(max_send_dim) / $(max_recv_dim)\n")
    end

    return DistributedFunctions{Tv}(
        comm, rank, size, basis, gmap, local_dim,
        ham_sub_otfs, ham_sub_topos,
        cache, local_w, send_buf,
        _hvec, _normalize, _zeros, _get_hf, _get_init, _inner,
        _expm, _tvec, _grad, _backgrad, _pool_otf,
    )
end
