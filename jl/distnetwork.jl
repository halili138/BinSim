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


function rank_block_counts(gmap::GlobalMemMap)
    counts = zeros(Cint, gmap.mpi_size)
    @ccall LIB_DIST.get_rank_block_counts_otf_gmap(gmap.ptr::Ptr{Cvoid}, counts::Ptr{Cint})::Cvoid
    return Int.(counts)
end


function get_max_rank_num_blocks(basis::BasisManager, gmap::GlobalMemMap)
    return Int(@ccall LIB_DIST.get_max_rank_num_blocks_otf_gmap(
        basis.ptr::Ptr{Cvoid}, gmap.ptr::Ptr{Cvoid}
    )::Cint)
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

function build_distributed_otfs(basis::BasisManager, pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}, tol::Float64=1e-12) where {Ti,Tv,K,V}
    groups = compress_by_svd(pool, tol)
    return [OTF_from_groups(basis, SVDGroup{Ti,Tv}[group]) for group in groups]
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
    ham_sub_topos::Matrix{SubTopology}
    pool_sub_otfs::Vector{OTF}
    pool_sub_topos::Matrix{SubTopology}
    pool_global_to_fragment::Vector{Int}
    pool_fragment_to_global::Vector{Int}

    # === 预分配缓冲区 ==============
    cache::Vector{Tv}       # local_dim + max_recv_dim
    back_cache::Vector{Tv}  # local_dim + max_recv_dim, second vector ghost workspace for backgrad
    local_w::Vector{Tv}     # local_dim
    send_buf::Vector{Tv}    # max_send_dim

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
end


function DistributedFunctions(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}},
    comm::MPI.Comm;
    tol::Float64=1e-12,
    num_phases::Int=1,
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

    # 3. 每个切片、每个通信相位构建独立通信路由表
    @assert num_phases >= 1 "DistributedFunctions requires num_phases >= 1"
    block_counts = rank_block_counts(gmap)
    max_rank_num_blocks = maximum(block_counts)
    effective_num_phases = min(num_phases, max_rank_num_blocks)
    ham_sub_topos = Matrix{SubTopology}(undef, n_subnets, effective_num_phases)
    for i in 1:n_subnets, p in 1:effective_num_phases
        ham_sub_topos[i, p] = SubTopology(
            basis, ham_sub_otfs[i], gmap;
            num_phases=effective_num_phases, phase_idx=p - 1,
        )
    end

    pool_sub_otfs = build_distributed_otfs(basis, pool, tol)
    n_pool = length(pool_sub_otfs)
    pool_global_to_fragment = collect(1:n_pool)
    pool_fragment_to_global = collect(1:n_pool)
    # Exponential rotations must see a simultaneous snapshot of all paired
    # amplitudes.  Keep pool operators in a single communication phase even
    # when hvec uses phased accumulation.
    pool_num_phases = 1
    pool_sub_topos = Matrix{SubTopology}(undef, n_pool, pool_num_phases)
    for i in 1:n_pool, p in 1:pool_num_phases
        pool_sub_topos[i, p] = SubTopology(
            basis, pool_sub_otfs[i], gmap;
            num_phases=pool_num_phases, phase_idx=p - 1,
        )
    end

    max_send_dim = max(
        n_subnets == 0 ? 0 : maximum(t.send_dim for t in ham_sub_topos),
        n_pool == 0 ? 0 : maximum(t.send_dim for t in pool_sub_topos),
    )
    max_recv_dim = max(
        n_subnets == 0 ? 0 : maximum(t.recv_dim for t in ham_sub_topos),
        n_pool == 0 ? 0 : maximum(t.recv_dim for t in pool_sub_topos),
    )

    # 4. 预分配可复用缓冲区
    cache      = zeros(Tv, local_dim + max_recv_dim)
    back_cache = zeros(Tv, local_dim + max_recv_dim)
    local_w    = zeros(Tv, local_dim)
    send_buf   = zeros(Tv, max_send_dim)

    local_peak_scalar_count = 2 * (local_dim + max_recv_dim) + local_dim + max_send_dim
    local_peak_bytes = local_peak_scalar_count * sizeof(Tv)
    global_max_local_dim = MPI.Allreduce(local_dim, max, comm)
    global_max_send_dim = MPI.Allreduce(max_send_dim, max, comm)
    global_max_recv_dim = MPI.Allreduce(max_recv_dim, max, comm)
    global_peak_hvec_buffer_bytes = MPI.Allreduce(local_peak_bytes, max, comm)
    global_total_hvec_buffer_bytes = MPI.Allreduce(local_peak_bytes, +, comm)

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

        for i in 1:n_subnets, p in 1:effective_num_phases
            topo = ham_sub_topos[i, p]
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

    _expm = (idx, θ, v) -> begin
        n_pool == 0 && error("DistributedFunctions.expm requires an operator pool; construct with DistributedFunctions(basis, ham, pool, comm; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "DistributedFunctions.expm: pool index out of bounds"
        cache[1:local_dim] .= v

        for p in 1:pool_num_phases
            topo = pool_sub_topos[idx, p]
            otf = pool_sub_otfs[idx]

            if topo.send_dim > 0
                @ccall LIB_DIST.pack_send_buffer_f64_sub(
                    topo.ptr::Ptr{Cvoid},
                    cache::Ptr{Float64}, send_buf::Ptr{Float64},
                )::Cvoid
            end

            recv_view = @view cache[local_dim+1 : local_dim+topo.recv_dim]
            send_vbuf = MPI.VBuffer(send_buf, topo.send_counts)
            recv_vbuf = MPI.VBuffer(recv_view, topo.recv_counts)
            MPI.Alltoallv!(send_vbuf, recv_vbuf, comm)

            @ccall LIB_DIST.compute_expm_sub_chunk_f64(
                basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
                topo.ptr::Ptr{Cvoid},
                Int64(0)::Int64, θ::Cdouble, cache::Ptr{Float64},
            )::Cvoid
        end

        v .= @view cache[1:local_dim]
        return v
    end
    _tvec     = (idx, lv, rv)         -> error("DistributedFunctions.tvec: not yet implemented")
    _grad     = (idx, θ, lv, rv)      -> error("DistributedFunctions.grad: not yet implemented")
    _backgrad = (idx, θ, lv, rv) -> begin
        n_pool == 0 && error("DistributedFunctions.backgrad requires an operator pool; construct with DistributedFunctions(basis, ham, pool, comm; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "DistributedFunctions.backgrad: pool index out of bounds"
        cache[1:local_dim] .= lv
        back_cache[1:local_dim] .= rv

        local_grad = zero(Tv)
        for p in 1:pool_num_phases
            topo = pool_sub_topos[idx, p]
            otf = pool_sub_otfs[idx]

            if topo.send_dim > 0
                @ccall LIB_DIST.pack_send_buffer_f64_sub(
                    topo.ptr::Ptr{Cvoid},
                    cache::Ptr{Float64}, send_buf::Ptr{Float64},
                )::Cvoid
            end
            recv_view_l = @view cache[local_dim+1 : local_dim+topo.recv_dim]
            send_vbuf_l = MPI.VBuffer(send_buf, topo.send_counts)
            recv_vbuf_l = MPI.VBuffer(recv_view_l, topo.recv_counts)
            MPI.Alltoallv!(send_vbuf_l, recv_vbuf_l, comm)

            if topo.send_dim > 0
                @ccall LIB_DIST.pack_send_buffer_f64_sub(
                    topo.ptr::Ptr{Cvoid},
                    back_cache::Ptr{Float64}, send_buf::Ptr{Float64},
                )::Cvoid
            end
            recv_view_r = @view back_cache[local_dim+1 : local_dim+topo.recv_dim]
            send_vbuf_r = MPI.VBuffer(send_buf, topo.send_counts)
            recv_vbuf_r = MPI.VBuffer(recv_view_r, topo.recv_counts)
            MPI.Alltoallv!(send_vbuf_r, recv_vbuf_r, comm)

            local_grad += @ccall LIB_DIST.compute_backgrad_sub_chunk_f64(
                basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid},
                θ::Cdouble, cache::Ptr{Float64}, back_cache::Ptr{Float64},
            )::Cdouble
        end

        lv .= @view cache[1:local_dim]
        rv .= @view back_cache[1:local_dim]
        return MPI.Allreduce(local_grad, +, comm)
    end

    if rank == 0
        println("\nDistributedFunctions built:")

        @printf("  MPI ranks:               %d\n", size)
        @printf("  Local dim (rank0):       %d\n", local_dim)
        @printf("  Max local dim:           %d\n", global_max_local_dim)
        @printf("  Ham symmetry fragments:  %d\n", n_subnets)
        @printf("  Pool operators:          %d\n", n_pool)
        @printf("  Pool expm phases:        %d\n", pool_num_phases)

        if num_phases != effective_num_phases
            @printf("  Requested communication phases: %d, clamped to %d because max rank-local wavefunction blocks is %d\n", num_phases, effective_num_phases, max_rank_num_blocks)
        end

        @printf("  Communication phases:    %d\n", effective_num_phases)
        @printf("  Max send/recv:           %d / %d \n\n", global_max_send_dim, global_max_recv_dim)
        @printf("  Peak hvec buffer memory per rank:        %.3f GB\n", global_peak_hvec_buffer_bytes / (1 << 30))
        @printf("  Total hvec buffer memory across ranks:   %.3f GB\n", global_total_hvec_buffer_bytes / (1 << 30))
    end

    return DistributedFunctions{Tv}(
        comm, rank, size, basis, gmap, local_dim,
        ham_sub_otfs, ham_sub_topos, pool_sub_otfs, pool_sub_topos,
        pool_global_to_fragment, pool_fragment_to_global,
        cache, back_cache, local_w, send_buf,
        _hvec, _normalize, _zeros, _get_hf, _get_init, _inner,
        _expm, _tvec, _grad, _backgrad,
    )
end

function DistributedFunctions(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    comm::MPI.Comm;
    tol::Float64=1e-12,
    num_phases::Int=1,
) where {Ti,Tv,TK,TV}
    return DistributedFunctions(
        basis, ham, BinaryQubitAABB{Ti,Tv,TK,TV}[], comm;
        tol=tol, num_phases=num_phases,
    )
end

function DistributedFunctions(
    mole::Mole,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}},
    comm::MPI.Comm;
    virtual_k::Int=0,
    virtual_seed::Int=1234,
    virtual_orbsym::Vector{Int64}=Int64[],
    virtual_optimize::Bool=true,
    virtual_ntry::Int=64,
    tol::Float64=1e-12,
    num_phases::Int=1,
) where {Ti,Tv,TK,TV}
    basis = if virtual_k > 0 || !isempty(virtual_orbsym)
        k = virtual_k > 0 ? virtual_k : ceil(Int, log2(maximum(virtual_orbsym) + 1))
        partition = VirtualSymmetryPartition(
            mole.norb, k;
            seed=virtual_seed,
            orbsym=virtual_orbsym,
            nelec=mole.nelec,
            physical_orbsym=mole.orbsym,
            optimize=virtual_optimize && isempty(virtual_orbsym),
            ntry=virtual_ntry,
        )
        BasisManager(Int64(mole.norb), mole.nelec, mole.orbsym, partition)
    else
        BasisManager(Int64(mole.norb), mole.nelec, mole.orbsym)
    end

    return DistributedFunctions(basis, ham, pool, comm; tol=tol, num_phases=num_phases), basis
end


function DistributedFunctions(
    mole::Mole,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    comm::MPI.Comm;
    virtual_k::Int=0,
    virtual_seed::Int=1234,
    virtual_orbsym::Vector{Int64}=Int64[],
    virtual_optimize::Bool=true,
    virtual_ntry::Int=64,
    tol::Float64=1e-12,
    num_phases::Int=1,
) where {Ti,Tv,TK,TV}
    return DistributedFunctions(
        mole, ham, BinaryQubitAABB{Ti,Tv,TK,TV}[], comm;
        virtual_k=virtual_k,
        virtual_seed=virtual_seed,
        virtual_orbsym=virtual_orbsym,
        virtual_optimize=virtual_optimize,
        virtual_ntry=virtual_ntry,
        tol=tol,
        num_phases=num_phases,
    )
end
