include("cudistribute.jl")

# ============================================================
# CuDistributedFunctions — CUDA 分布式计算的统一接口
# ============================================================
# 三种模式，对外接口一致：
#   funcs = CuDistributedFunctions(ModeSerial, basis, ham; num_chunks=4)
#   funcs = CuDistributedFunctions(ModeNVLink, basis, ham, comm; num_phases=2)
#   funcs = CuDistributedFunctions(ModeHybrid, basis, ham, comm; num_chunks=16)
#
#   v  = funcs.get_hf(nelec)
#   Hv = funcs.zeros()
#   funcs.hvec(v, Hv)
#   funcs.normalize(v)
# ============================================================

abstract type CuDistMode end

struct ModeSerial <: CuDistMode end
struct ModeNVLink <: CuDistMode end
struct ModeHybrid <: CuDistMode end

struct CuDistributedFunctions{Mode<:CuDistMode}
    comm::MPI.Comm
    rank::Int
    size::Int
    local_dim::Int64

    hvec::Function
    normalize::Function
    zeros::Function
    get_hf::Function
    inner::Function

    expm::Function
    grad::Function
    backgrad::Function
end

function _num_wavefunction_symmetry_blocks(basis::BasisManager)
    if hasproperty(basis, :num_blocks)
        return Int(getproperty(basis, :num_blocks))
    end
    return Int(get_num_symmetry_blocks(basis.ptr))
end

_hvec_bytes(nscalars::Integer) = Int64(nscalars) * Int64(sizeof(Float64))
_hvec_gib(nbytes::Integer) = round(nbytes / 1024^3, digits=3)

function _sum_buffer_bytes(buffers)
    return Int64(sum(length, buffers; init=0)) * Int64(sizeof(Float64))
end

function _build_virtual_or_physical_basis(
    mole::Mole;
    virtual_k::Int=0,
    virtual_seed::Int=1234,
    virtual_orbsym::Vector{Int64}=Int64[],
    virtual_optimize::Bool=true,
    virtual_ntry::Int=64,
)
    if virtual_k > 0 || !isempty(virtual_orbsym)
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
        return BasisManager(Int64(mole.norb), mole.nelec, mole.orbsym, partition)
    end

    return BasisManager(Int64(mole.norb), mole.nelec, mole.orbsym)
end

# ============================================================
# SerialOOC: 单卡突破显存限制
# ============================================================
function CuDistributedFunctions(
    ::Type{ModeSerial},
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv_h,TK,TV},
    pool=nothing;
    num_chunks::Int=4,
    gpu_id::Int=0,
    tol::Float64=1e-12,
) where {Ti,Tv_h,TK,TV}
    CUDA.device!(gpu_id)

    cu_basis_dev = CuBasisManager(basis)

    @assert num_chunks >= 1 "num_chunks must be >= 1"
    num_blocks = _num_wavefunction_symmetry_blocks(basis)
    requested_num_chunks = num_chunks
    num_chunks = min(num_chunks, max(1, num_blocks))

    if num_chunks != requested_num_chunks
        println("Requested virtual chunks: $(requested_num_chunks); clamped to $(num_chunks) because basis has $(num_blocks) wavefunction blocks")
    end

    # 虚拟切片：用非MPI的GlobalMemMap构造各chunk的分块视图
    gmaps = [GlobalMemMap(basis; rank=r - 1, size=num_chunks) for r in 1:num_chunks]
    cpu_otfs, cu_otfs = build_distributed_cu_otfs(basis, ham, tol)
    n_otfs = length(cu_otfs)

    n_pool = pool === nothing ? 0 : length(pool)
    pool_cpu_otfs = Vector{Vector{OTF}}(undef, n_pool)
    pool_cu_otfs = Vector{Vector{CuOTF}}(undef, n_pool)
    for i in 1:n_pool
        pool_cpu_otfs[i], pool_cu_otfs[i] = build_distributed_cu_otfs(basis, pool[i], tol)
    end

    # 路由表: [chunk, otf]
    sub_topos = Matrix{CuSubTopology}(undef, num_chunks, n_otfs)
    for r in 1:num_chunks, i in 1:n_otfs
        sub_topos[r, i] = CuSubTopology(basis, cpu_otfs[i], gmaps[r])
    end

    # 各chunk的维度与缓冲区
    chunk_dims   = [gmaps[r].local_dim for r in 1:num_chunks]
    chunk_offsets = vcat(0, cumsum(chunk_dims))
    local_dim     = chunk_offsets[end]

    pool_sub_topos = [Matrix{CuSubTopology}(undef, num_chunks, length(pool_cpu_otfs[i])) for i in 1:n_pool]
    for i in 1:n_pool, r in 1:num_chunks, j in 1:length(pool_cpu_otfs[i])
        pool_sub_topos[i][r, j] = CuSubTopology(basis, pool_cpu_otfs[i][j], gmaps[r])
    end

    max_send_dims = [maximum(vcat([t.send_dim for t in sub_topos[r, :]], [topo[r, j].send_dim for topo in pool_sub_topos for j in 1:size(topo, 2)], [0])) for r in 1:num_chunks]
    max_recv_dims = [maximum(vcat([t.recv_dim for t in sub_topos[r, :]], [topo[r, j].recv_dim for topo in pool_sub_topos for j in 1:size(topo, 2)], [0])) for r in 1:num_chunks]

    host_v_chunks  = [Vector{Float64}(undef, chunk_dims[r]) for r in 1:num_chunks]
    host_w_chunks  = [Vector{Float64}(undef, chunk_dims[r]) for r in 1:num_chunks]
    host_send_bufs = [Vector{Float64}(undef, max_send_dims[r]) for r in 1:num_chunks]
    host_recv_bufs = [Vector{Float64}(undef, max_recv_dims[r]) for r in 1:num_chunks]
    host_l_chunks  = [Vector{Float64}(undef, chunk_dims[r]) for r in 1:num_chunks]
    host_r_chunks  = [Vector{Float64}(undef, chunk_dims[r]) for r in 1:num_chunks]
    host_send2_bufs = [Vector{Float64}(undef, max_send_dims[r]) for r in 1:num_chunks]
    host_recv2_bufs = [Vector{Float64}(undef, max_recv_dims[r]) for r in 1:num_chunks]

    # GPU 复用池
    global_max_local = maximum(chunk_dims)
    global_max_recv  = maximum(max_recv_dims)
    global_max_send  = maximum(max_send_dims)

    d_cache = CUDA.zeros(Float64, global_max_local + global_max_recv)
    d_send  = CUDA.zeros(Float64, global_max_send)
    d_w     = CUDA.zeros(Float64, global_max_local)
    d_back_cache = CUDA.zeros(Float64, global_max_local + global_max_recv)

    # ============================================================
    # CPU 内存路由器 (模拟 Alltoallv)
    # ============================================================
    function cpu_memory_router!(nchunks, topos_for_otf, send_bufs, recv_bufs)
        for r_recv in 1:nchunks
            recv_offset = 1
            for r_send in 1:nchunks
                sc = topos_for_otf[r_send].send_counts[r_recv]
                rc = topos_for_otf[r_recv].recv_counts[r_send]
                if sc > 0
                    so = 1 + sum(topos_for_otf[r_send].send_counts[1:r_recv-1]; init=0)
                    copyto!(recv_bufs[r_recv], recv_offset, send_bufs[r_send], so, sc)
                end
                recv_offset += rc
            end
        end
    end

    # ============================================================
    # 闭包
    # ============================================================
    _get_hf = (nelec::Tuple{Int,Int}) -> begin
        v = zeros(Float64, local_dim)
        offset = 1
        for r in 1:num_chunks
            ld = chunk_dims[r]
            if ld > 0
                d_view = @view d_cache[1:ld]
                set_local_hf_gpu!(gmaps[r], basis, d_view, nelec)
                copyto!(v, offset, Array(d_view), 1, ld)
            end
            offset += ld
        end
        return v
    end

    _zeros = () -> Vector{Float64}(undef, local_dim)

    _inner = (lv::Vector{Float64}, rv::Vector{Float64}) -> dot(lv, rv)

    _normalize = (v::Vector{Float64}) -> begin
        v ./= sqrt(sum(abs2, v))
    end

    _hvec = (v::Vector{Float64}, Hv::Vector{Float64}) -> begin
        # 1. 输入扁平向量 → 内部chunk缓冲区
        for r in 1:num_chunks
            ld = chunk_dims[r]
            ld > 0 && copyto!(host_v_chunks[r], 1, v, chunk_offsets[r] + 1, ld)
        end
        for w in host_w_chunks
            fill!(w, 0.0)
        end

        # 2. 逐个subnet: GPU打包 → CPU路由 → GPU计算
        for i in 1:n_otfs
            # 阶段A: 逐chunk GPU打包 → 拉回CPU
            for r in 1:num_chunks
                topo = sub_topos[r, i]
                if topo.send_dim > 0
                    ld = chunk_dims[r]
                    copyto!(d_cache, 1, host_v_chunks[r], 1, ld)
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(
                        topo.ptr::Ptr{Cvoid},
                        pointer(d_cache)::CuPtr{Float64},
                        pointer(d_send)::CuPtr{Float64},
                    )::Cvoid
                    copyto!(host_send_bufs[r], 1, d_send, 1, topo.send_dim)
                end
            end

            # 阶段B: CPU内存路由
            cpu_memory_router!(num_chunks, sub_topos[:, i], host_send_bufs, host_recv_bufs)

            # 阶段C: 逐chunk GPU计算 → 拉回CPU累加
            for r in 1:num_chunks
                topo = sub_topos[r, i]
                ld = chunk_dims[r]
                if ld == 0
                    continue
                end
                copyto!(d_cache, 1, host_v_chunks[r], 1, ld)
                if topo.recv_dim > 0
                    copyto!(d_cache, ld + 1, host_recv_bufs[r], 1, topo.recv_dim)
                end
                d_w .= 0.0
                @ccall LIB_CUDIST.compute_hvec_sub_chunk_gpu_f64(
                    cu_basis_dev.ptr::Ptr{Cvoid}, cu_otfs[i].ptr::Ptr{Cvoid},
                    topo.ptr::Ptr{Cvoid},
                    pointer(d_cache)::CuPtr{Float64}, pointer(d_w)::CuPtr{Float64},
                )::Cvoid
                host_w_chunks[r] .+= Array(d_w[1:ld])
            end
        end

        # 3. 内部chunk → 输出扁平向量
        for r in 1:num_chunks
            ld = chunk_dims[r]
            ld > 0 && copyto!(Hv, chunk_offsets[r] + 1, host_w_chunks[r], 1, ld)
        end
    end

    _expm = (idx, θ, v) -> begin
        n_pool == 0 && error("CuDistributedFunctions.expm requires an operator pool; construct with CuDistributedFunctions(ModeSerial, basis, ham, pool; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "CuDistributedFunctions.expm: pool index out of bounds"
        for r in 1:num_chunks
            ld = chunk_dims[r]
            ld > 0 && copyto!(host_v_chunks[r], 1, v, chunk_offsets[r] + 1, ld)
        end
        for j in 1:length(pool_cu_otfs[idx])
            topos = pool_sub_topos[idx][:, j]
            for r in 1:num_chunks
                topo = topos[r]
                if topo.send_dim > 0
                    ld = chunk_dims[r]
                    copyto!(d_cache, 1, host_v_chunks[r], 1, ld)
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(topo.ptr::Ptr{Cvoid}, pointer(d_cache)::CuPtr{Float64}, pointer(d_send)::CuPtr{Float64})::Cvoid
                    copyto!(host_send_bufs[r], 1, d_send, 1, topo.send_dim)
                end
            end
            cpu_memory_router!(num_chunks, topos, host_send_bufs, host_recv_bufs)
            for r in 1:num_chunks
                topo = topos[r]; ld = chunk_dims[r]
                ld == 0 && continue
                copyto!(d_cache, 1, host_v_chunks[r], 1, ld)
                topo.recv_dim > 0 && copyto!(d_cache, ld + 1, host_recv_bufs[r], 1, topo.recv_dim)
                @ccall LIB_CUDIST.compute_expm_sub_chunk_gpu_f64(cu_basis_dev.ptr::Ptr{Cvoid}, pool_cu_otfs[idx][j].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, Int64(0)::Int64, θ::Cdouble, pointer(d_cache)::CuPtr{Float64})::Cvoid
                copyto!(host_v_chunks[r], 1, d_cache, 1, ld)
            end
        end
        for r in 1:num_chunks
            ld = chunk_dims[r]
            ld > 0 && copyto!(v, chunk_offsets[r] + 1, host_v_chunks[r], 1, ld)
        end
        return v
    end
    _grad = (idx, θ, lv, rv) -> error("CuDistributedFunctions.grad: not yet implemented")
    _backgrad = (idx, θ, lv, rv) -> begin
        n_pool == 0 && error("CuDistributedFunctions.backgrad requires an operator pool; construct with CuDistributedFunctions(ModeSerial, basis, ham, pool; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "CuDistributedFunctions.backgrad: pool index out of bounds"
        for r in 1:num_chunks
            ld = chunk_dims[r]
            if ld > 0
                copyto!(host_l_chunks[r], 1, lv, chunk_offsets[r] + 1, ld)
                copyto!(host_r_chunks[r], 1, rv, chunk_offsets[r] + 1, ld)
            end
        end
        local_grad = 0.0
        for j in 1:length(pool_cu_otfs[idx])
            topos = pool_sub_topos[idx][:, j]
            for r in 1:num_chunks
                topo = topos[r]
                if topo.send_dim > 0
                    ld = chunk_dims[r]
                    copyto!(d_cache, 1, host_l_chunks[r], 1, ld)
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(topo.ptr::Ptr{Cvoid}, pointer(d_cache)::CuPtr{Float64}, pointer(d_send)::CuPtr{Float64})::Cvoid
                    copyto!(host_send_bufs[r], 1, d_send, 1, topo.send_dim)
                    copyto!(d_cache, 1, host_r_chunks[r], 1, ld)
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(topo.ptr::Ptr{Cvoid}, pointer(d_cache)::CuPtr{Float64}, pointer(d_send)::CuPtr{Float64})::Cvoid
                    copyto!(host_send2_bufs[r], 1, d_send, 1, topo.send_dim)
                end
            end
            cpu_memory_router!(num_chunks, topos, host_send_bufs, host_recv_bufs)
            cpu_memory_router!(num_chunks, topos, host_send2_bufs, host_recv2_bufs)
            for r in 1:num_chunks
                topo = topos[r]; ld = chunk_dims[r]
                ld == 0 && continue
                copyto!(d_cache, 1, host_l_chunks[r], 1, ld)
                copyto!(d_back_cache, 1, host_r_chunks[r], 1, ld)
                if topo.recv_dim > 0
                    copyto!(d_cache, ld + 1, host_recv_bufs[r], 1, topo.recv_dim)
                    copyto!(d_back_cache, ld + 1, host_recv2_bufs[r], 1, topo.recv_dim)
                end
                local_grad += @ccall LIB_CUDIST.compute_backgrad_sub_chunk_gpu_f64(cu_basis_dev.ptr::Ptr{Cvoid}, pool_cu_otfs[idx][j].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, θ::Cdouble, pointer(d_cache)::CuPtr{Float64}, pointer(d_back_cache)::CuPtr{Float64})::Cdouble
                copyto!(host_l_chunks[r], 1, d_cache, 1, ld)
                copyto!(host_r_chunks[r], 1, d_back_cache, 1, ld)
            end
        end
        for r in 1:num_chunks
            ld = chunk_dims[r]
            if ld > 0
                copyto!(lv, chunk_offsets[r] + 1, host_l_chunks[r], 1, ld)
                copyto!(rv, chunk_offsets[r] + 1, host_r_chunks[r], 1, ld)
            end
        end
        return local_grad
    end

    comm = MPI.COMM_SELF
    rank = 0
    nproc = 1

    println("\nCuDistributedFunctions (SerialOOC) built:")
    println("  GPU:                     $(CUDA.name(CUDA.device()))")
    println("  Virtual chunks requested: $(requested_num_chunks)")
    println("  Virtual chunks effective: $(num_chunks)")
    println("  Total local dim:          $(local_dim)")
    println("  Sub-networks:             $(n_otfs)")
    println("  Pool operators:           $(n_pool)")
    println("  Max local/chunk dim:      $(global_max_local)")
    println("  Max send dim:             $(global_max_send)")
    println("  Max recv dim:             $(global_max_recv)")
    d_cache_bytes = _hvec_bytes(global_max_local + global_max_recv)
    d_send_bytes = _hvec_bytes(global_max_send)
    d_w_bytes = _hvec_bytes(global_max_local)
    hvec_vram_bytes = d_cache_bytes + d_send_bytes + d_w_bytes
    host_v_chunks_bytes = _sum_buffer_bytes(host_v_chunks)
    host_w_chunks_bytes = _sum_buffer_bytes(host_w_chunks)
    host_send_bufs_bytes = _sum_buffer_bytes(host_send_bufs)
    host_recv_bufs_bytes = _sum_buffer_bytes(host_recv_bufs)
    println("  Hvec GPU buffers:         d_cache=$(_hvec_gib(d_cache_bytes)) GB ($(d_cache_bytes) bytes), d_send=$(_hvec_gib(d_send_bytes)) GB ($(d_send_bytes) bytes), d_w=$(_hvec_gib(d_w_bytes)) GB ($(d_w_bytes) bytes)")
    println("  Peak GPU hvec buffer/rank: $(_hvec_gib(hvec_vram_bytes)) GB ($(hvec_vram_bytes) bytes)")
    println("  Total GPU hvec buffers:   $(_hvec_gib(hvec_vram_bytes)) GB ($(hvec_vram_bytes) bytes)")
    println("  Host chunk buffers:       host_v_chunks=$(_hvec_gib(host_v_chunks_bytes)) GB ($(host_v_chunks_bytes) bytes), host_w_chunks=$(_hvec_gib(host_w_chunks_bytes)) GB ($(host_w_chunks_bytes) bytes)")
    println("  Host exchange buffers:    host_send_bufs=$(_hvec_gib(host_send_bufs_bytes)) GB ($(host_send_bufs_bytes) bytes), host_recv_bufs=$(_hvec_gib(host_recv_bufs_bytes)) GB ($(host_recv_bufs_bytes) bytes)\n")

    return CuDistributedFunctions{ModeSerial}(
        comm, rank, nproc, local_dim,
        _hvec, _normalize, _zeros, _get_hf, _inner,
        _expm, _grad, _backgrad,
    )
end

CuDistributedFunctions(::ModeSerial, basis::BasisManager, ham::BinaryQubitAABB, pool; num_chunks::Int=4, gpu_id::Int=0, tol::Float64=1e-12) =
    CuDistributedFunctions(ModeSerial, basis, ham, pool; num_chunks=num_chunks, gpu_id=gpu_id, tol=tol)

function CuDistributedFunctions(
    ::Type{ModeSerial},
    mole::Mole,
    ham::BinaryQubitAABB{Ti,Tv_h,TK,TV};
    virtual_k::Int=0,
    virtual_seed::Int=1234,
    virtual_orbsym::Vector{Int64}=Int64[],
    virtual_optimize::Bool=true,
    virtual_ntry::Int=64,
    num_chunks::Int=4,
    gpu_id::Int=0,
    tol::Float64=1e-12,
) where {Ti,Tv_h,TK,TV}
    basis = _build_virtual_or_physical_basis(
        mole;
        virtual_k=virtual_k,
        virtual_seed=virtual_seed,
        virtual_orbsym=virtual_orbsym,
        virtual_optimize=virtual_optimize,
        virtual_ntry=virtual_ntry,
    )

    return CuDistributedFunctions(
        ModeSerial, basis, ham;
        num_chunks=num_chunks,
        gpu_id=gpu_id,
        tol=tol,
    ), basis
end

# ============================================================
# NVLink: 多GPU + VRAM驻留
# ============================================================
function CuDistributedFunctions(
    ::Type{ModeNVLink},
    basis::BasisManager,
    ham::BinaryQubitAABB,
    comm::MPI.Comm;
    num_phases::Int=2,
    tol::Float64=1e-12,
)
    return CuDistributedFunctions(ModeNVLink, basis, ham, nothing, comm; num_phases=num_phases, tol=tol)
end

function CuDistributedFunctions(
    ::Type{ModeNVLink},
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv_h,TK,TV},
    pool,
    comm::MPI.Comm;
    num_phases::Int=2,
    tol::Float64=1e-12,
) where {Ti,Tv_h,TK,TV}
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    ngpus = length(CUDA.devices())
    CUDA.device!(rank % ngpus)

    cu_basis_dev = CuBasisManager(basis)
    gmap = GlobalMemMap(basis, comm)
    local_dim = gmap.local_dim

    @assert num_phases >= 1 "num_phases must be >= 1"
    requested_num_phases = num_phases
    max_rank_num_blocks = get_max_rank_num_blocks(basis, gmap)
    num_phases = min(num_phases, max_rank_num_blocks)

    if rank == 0 && num_phases != requested_num_phases
        println("CuDistributedFunctions (NVLink): clamping requested phases from $(requested_num_phases) to $(num_phases) because max rank block count is $(max_rank_num_blocks).")
    end

    cpu_otfs, cu_otfs = build_distributed_cu_otfs(basis, ham, tol)
    n_otfs = length(cu_otfs)

    n_pool = pool === nothing ? 0 : length(pool)
    pool_cpu_otfs = Vector{Vector{OTF}}(undef, n_pool)
    pool_cu_otfs = Vector{Vector{CuOTF}}(undef, n_pool)
    for i in 1:n_pool
        pool_cpu_otfs[i], pool_cu_otfs[i] = build_distributed_cu_otfs(basis, pool[i], tol)
    end

    # [otf_idx, phase]
    sub_topos = Matrix{CuSubTopology}(undef, n_otfs, num_phases)
    for i in 1:n_otfs, p in 1:num_phases
        sub_topos[i, p] = CuSubTopology(basis, cpu_otfs[i], gmap;
            num_phases=num_phases, phase_idx=p - 1)
    end

    # Pool topologies are indexed by [pool_idx][pool_otf_idx, phase].
    pool_sub_topos = [Matrix{CuSubTopology}(undef, length(pool_cpu_otfs[i]), num_phases) for i in 1:n_pool]
    for i in 1:n_pool, j in 1:length(pool_cpu_otfs[i]), p in 1:num_phases
        pool_sub_topos[i][j, p] = CuSubTopology(basis, pool_cpu_otfs[i][j], gmap;
            num_phases=num_phases, phase_idx=p - 1)
    end

    all_topos = CuSubTopology[]
    append!(all_topos, vec(sub_topos))
    for topos in pool_sub_topos
        append!(all_topos, vec(topos))
    end
    max_send_dim = isempty(all_topos) ? 0 : maximum(t.send_dim for t in all_topos)
    max_recv_dim = isempty(all_topos) ? 0 : maximum(t.recv_dim for t in all_topos)

    cache_scalars = local_dim + max_recv_dim
    send_scalars = max_send_dim
    recv_scalars = max_recv_dim
    w_scalars = local_dim
    hvec_scalar_count = cache_scalars + send_scalars + recv_scalars + w_scalars
    hvec_vram_bytes = Int64(hvec_scalar_count) * Int64(sizeof(Float64))

    max_local_dim_all = MPI.Allreduce(Int64(local_dim), max, comm)
    max_send_dim_all = MPI.Allreduce(Int64(max_send_dim), max, comm)
    max_recv_dim_all = MPI.Allreduce(Int64(max_recv_dim), max, comm)
    max_hvec_vram_bytes = MPI.Allreduce(hvec_vram_bytes, max, comm)
    total_hvec_vram_bytes = MPI.Allreduce(hvec_vram_bytes, +, comm)

    d_cache = CUDA.zeros(Float64, cache_scalars)
    d_send  = CUDA.zeros(Float64, send_scalars)
    d_recv  = CUDA.zeros(Float64, recv_scalars)
    d_w     = CUDA.zeros(Float64, w_scalars)
    d_left_cache = CUDA.zeros(Float64, cache_scalars)
    d_right_cache = CUDA.zeros(Float64, cache_scalars)
    d_send2 = CUDA.zeros(Float64, send_scalars)
    d_recv2 = CUDA.zeros(Float64, recv_scalars)

    d_local_v = @view d_cache[1:local_dim]

    function exchange_ghosts!(topo::CuSubTopology, cache, send, recv)
        if topo.send_dim > 0
            @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(
                topo.ptr::Ptr{Cvoid},
                pointer(cache)::CuPtr{Float64},
                pointer(send)::CuPtr{Float64},
            )::Cvoid
        end
        send_vbuf = MPI.VBuffer(send, topo.send_counts)
        recv_vbuf = MPI.VBuffer(recv, topo.recv_counts)
        MPI.Alltoallv!(send_vbuf, recv_vbuf, comm)
        if topo.recv_dim > 0
            recv_view = @view cache[local_dim + 1 : local_dim + topo.recv_dim]
            copyto!(recv_view, 1, recv, 1, topo.recv_dim)
        end
    end

    # ============================================================
    _get_hf = (nelec) -> begin
        set_local_hf_gpu!(gmap, basis, d_local_v, nelec)
        return copy(d_local_v)
    end

    _zeros = () -> CUDA.zeros(Float64, local_dim)

    _inner = (lv::CuVector{Float64}, rv::CuVector{Float64}) -> begin
        local_dot = real(sum(lv .* rv))
        return MPI.Allreduce(local_dot, +, comm)
    end

    _normalize = (v::CuVector{Float64}) -> begin
        n2 = sum(abs2, v)
        global_n = sqrt(MPI.Allreduce(n2, +, comm))
        v ./= global_n
    end

    _hvec = (v::CuVector{Float64}, Hv::CuVector{Float64}) -> begin
        copyto!(d_local_v, v)
        d_w .= 0.0

        for i in 1:n_otfs, p in 1:num_phases
            topo = sub_topos[i, p]
            exchange_ghosts!(topo, d_cache, d_send, d_recv)
            @ccall LIB_CUDIST.compute_hvec_sub_chunk_gpu_f64(
                cu_basis_dev.ptr::Ptr{Cvoid}, cu_otfs[i].ptr::Ptr{Cvoid},
                topo.ptr::Ptr{Cvoid},
                pointer(d_cache)::CuPtr{Float64}, pointer(d_w)::CuPtr{Float64},
            )::Cvoid
        end

        copyto!(Hv, d_w)
    end

    _expm = (idx, θ, v::CuVector{Float64}) -> begin
        n_pool == 0 && error("CuDistributedFunctions.expm requires an operator pool; construct with CuDistributedFunctions(ModeNVLink, basis, ham, pool, comm; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "CuDistributedFunctions.expm: pool index out of bounds"
        copyto!(d_local_v, v)
        for j in 1:length(pool_cu_otfs[idx]), p in 1:num_phases
            topo = pool_sub_topos[idx][j, p]
            exchange_ghosts!(topo, d_cache, d_send, d_recv)
            @ccall LIB_CUDIST.compute_expm_sub_chunk_gpu_f64(
                cu_basis_dev.ptr::Ptr{Cvoid}, pool_cu_otfs[idx][j].ptr::Ptr{Cvoid},
                topo.ptr::Ptr{Cvoid}, Int64(0)::Int64, θ::Cdouble,
                pointer(d_cache)::CuPtr{Float64},
            )::Cvoid
        end
        copyto!(v, d_local_v)
        return v
    end

    _grad = (idx, θ, lv, rv) -> error("CuDistributedFunctions.grad: not yet implemented")

    _backgrad = (idx, θ, lv::CuVector{Float64}, rv::CuVector{Float64}) -> begin
        n_pool == 0 && error("CuDistributedFunctions.backgrad requires an operator pool; construct with CuDistributedFunctions(ModeNVLink, basis, ham, pool, comm; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "CuDistributedFunctions.backgrad: pool index out of bounds"
        left_local = @view d_left_cache[1:local_dim]
        right_local = @view d_right_cache[1:local_dim]
        copyto!(left_local, lv)
        copyto!(right_local, rv)
        local_grad = 0.0
        for j in 1:length(pool_cu_otfs[idx]), p in 1:num_phases
            topo = pool_sub_topos[idx][j, p]
            exchange_ghosts!(topo, d_left_cache, d_send, d_recv)
            exchange_ghosts!(topo, d_right_cache, d_send2, d_recv2)
            local_grad += @ccall LIB_CUDIST.compute_backgrad_sub_chunk_gpu_f64(
                cu_basis_dev.ptr::Ptr{Cvoid}, pool_cu_otfs[idx][j].ptr::Ptr{Cvoid},
                topo.ptr::Ptr{Cvoid}, θ::Cdouble,
                pointer(d_left_cache)::CuPtr{Float64}, pointer(d_right_cache)::CuPtr{Float64},
            )::Cdouble
        end
        copyto!(lv, left_local)
        copyto!(rv, right_local)
        return MPI.Allreduce(local_grad, +, comm)
    end

    if rank == 0
        println("\nCuDistributedFunctions (NVLink) built:")
        println("  MPI ranks:              $(nproc)")
        println("  Local dim (r0):         $(local_dim)")
        println("  Max local/chunk dim:    $(max_local_dim_all)")
        println("  Sub-networks:           $(n_otfs)")
        println("  Pool operators:         $(n_pool)")
        println("  Phases requested:       $(requested_num_phases)")
        println("  Phases effective:       $(num_phases)")
        println("  Max send dim:           $(max_send_dim_all)")
        println("  Max recv dim:           $(max_recv_dim_all)")
        println("  Hvec buffers (r0):      d_cache=$(_hvec_gib(_hvec_bytes(cache_scalars))) GB, d_send=$(_hvec_gib(_hvec_bytes(send_scalars))) GB, d_recv=$(_hvec_gib(_hvec_bytes(recv_scalars))) GB, d_w=$(_hvec_gib(_hvec_bytes(w_scalars))) GB")
        println("  Peak GPU hvec buffer/rank: $(round(max_hvec_vram_bytes / 1024^3, digits=3)) GB ($(max_hvec_vram_bytes) bytes)")
        println("  Total GPU hvec buffers:    $(round(total_hvec_vram_bytes / 1024^3, digits=3)) GB ($(total_hvec_vram_bytes) bytes)")
        println()
    end

    return CuDistributedFunctions{ModeNVLink}(
        comm, rank, nproc, local_dim,
        _hvec, _normalize, _zeros, _get_hf, _inner,
        _expm, _grad, _backgrad,
    )
end

function CuDistributedFunctions(
    ::Type{ModeNVLink},
    mole::Mole,
    ham::BinaryQubitAABB{Ti,Tv_h,TK,TV},
    comm::MPI.Comm;
    virtual_k::Int=0,
    virtual_seed::Int=1234,
    virtual_orbsym::Vector{Int64}=Int64[],
    virtual_optimize::Bool=true,
    virtual_ntry::Int=64,
    tol::Float64=1e-12,
    num_phases::Int=2,
) where {Ti,Tv_h,TK,TV}
    basis = _build_virtual_or_physical_basis(
        mole;
        virtual_k=virtual_k,
        virtual_seed=virtual_seed,
        virtual_orbsym=virtual_orbsym,
        virtual_optimize=virtual_optimize,
        virtual_ntry=virtual_ntry,
    )

    return CuDistributedFunctions(ModeNVLink, basis, ham, comm; tol=tol, num_phases=num_phases), basis
end

# ============================================================
# HybridOOC: 多GPU + CPU驻留 + MPI
# ============================================================
function CuDistributedFunctions(
    ::Type{ModeHybrid},
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv_h,TK,TV},
    comm::MPI.Comm;
    num_chunks::Int=16,
    tol::Float64=1e-12,
) where {Ti,Tv_h,TK,TV}
    return CuDistributedFunctions(ModeHybrid, basis, ham, nothing, comm; num_chunks=num_chunks, tol=tol)
end

CuDistributedFunctions(
    ::ModeHybrid,
    basis::BasisManager,
    ham::BinaryQubitAABB,
    pool,
    comm::MPI.Comm;
    num_chunks::Int=16,
    tol::Float64=1e-12,
) = CuDistributedFunctions(ModeHybrid, basis, ham, pool, comm; num_chunks=num_chunks, tol=tol)

function CuDistributedFunctions(
    ::Type{ModeHybrid},
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv_h,TK,TV},
    pool,
    comm::MPI.Comm;
    num_chunks::Int=16,
    tol::Float64=1e-12,
) where {Ti,Tv_h,TK,TV}
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    ngpus = length(CUDA.devices())
    CUDA.device!(rank % ngpus)

    cu_basis_dev = CuBasisManager(basis)

    @assert num_chunks >= nproc "num_chunks must be >= MPI size"
    num_blocks = _num_wavefunction_symmetry_blocks(basis)
    requested_num_chunks = num_chunks
    num_chunks = min(num_chunks, max(nproc, num_blocks))

    if rank == 0 && num_chunks != requested_num_chunks
        println("CuDistributedFunctions (HybridOOC): clamping requested chunks from $(requested_num_chunks) to $(num_chunks) because basis has $(num_blocks) wavefunction blocks and MPI size is $(nproc).")
    end

    gmaps_all = [GlobalMemMap(basis; rank=r - 1, size=num_chunks) for r in 1:num_chunks]
    my_chunks = [c for c in 1:num_chunks if (c - 1) % nproc == rank]
    n_my = length(my_chunks)

    cpu_otfs, cu_otfs = build_distributed_cu_otfs(basis, ham, tol)
    n_otfs = length(cu_otfs)

    n_pool = pool === nothing ? 0 : length(pool)
    pool_cpu_otfs = Vector{Vector{OTF}}(undef, n_pool)
    pool_cu_otfs = Vector{Vector{CuOTF}}(undef, n_pool)
    for i in 1:n_pool
        pool_cpu_otfs[i], pool_cu_otfs[i] = build_distributed_cu_otfs(basis, pool[i], tol)
    end

    sub_topos_all = Matrix{CuSubTopology}(undef, num_chunks, n_otfs)
    for r in 1:num_chunks, i in 1:n_otfs
        sub_topos_all[r, i] = CuSubTopology(basis, cpu_otfs[i], gmaps_all[r])
    end

    pool_sub_topos_all = [Matrix{CuSubTopology}(undef, num_chunks, length(pool_cpu_otfs[i])) for i in 1:n_pool]
    for i in 1:n_pool, r in 1:num_chunks, j in 1:length(pool_cpu_otfs[i])
        pool_sub_topos_all[i][r, j] = CuSubTopology(basis, pool_cpu_otfs[i][j], gmaps_all[r])
    end

    chunk_dims    = [gmaps_all[r].local_dim for r in 1:num_chunks]
    my_chunk_dims = [chunk_dims[c] for c in my_chunks]
    local_dim     = sum(my_chunk_dims)
    chunk_offsets = vcat(0, cumsum(Int.(my_chunk_dims)))

    max_send_dims = [
        maximum(vcat(
            [t.send_dim for t in sub_topos_all[c, :]],
            [topo[c, j].send_dim for topo in pool_sub_topos_all for j in 1:size(topo, 2)],
            [0],
        )) for c in my_chunks
    ]
    max_recv_dims = [
        maximum(vcat(
            [t.recv_dim for t in sub_topos_all[c, :]],
            [topo[c, j].recv_dim for topo in pool_sub_topos_all for j in 1:size(topo, 2)],
            [0],
        )) for c in my_chunks
    ]

    host_v_chunks  = [Vector{Float64}(undef, my_chunk_dims[idx]) for idx in 1:n_my]
    host_w_chunks  = [Vector{Float64}(undef, my_chunk_dims[idx]) for idx in 1:n_my]
    host_send_bufs = [Vector{Float64}(undef, max_send_dims[idx]) for idx in 1:n_my]
    host_recv_bufs = [Vector{Float64}(undef, max_recv_dims[idx]) for idx in 1:n_my]
    host_l_chunks  = [Vector{Float64}(undef, my_chunk_dims[idx]) for idx in 1:n_my]
    host_r_chunks  = [Vector{Float64}(undef, my_chunk_dims[idx]) for idx in 1:n_my]
    host_send2_bufs = [Vector{Float64}(undef, max_send_dims[idx]) for idx in 1:n_my]
    host_recv2_bufs = [Vector{Float64}(undef, max_recv_dims[idx]) for idx in 1:n_my]

    my_max_local = n_my == 0 ? 0 : maximum(my_chunk_dims)
    my_max_recv = n_my == 0 ? 0 : maximum(max_recv_dims)
    my_max_send = n_my == 0 ? 0 : maximum(max_send_dims)

    d_cache = CUDA.zeros(Float64, my_max_local + my_max_recv)
    d_send  = CUDA.zeros(Float64, my_max_send)
    d_w     = CUDA.zeros(Float64, my_max_local)
    d_back_cache = CUDA.zeros(Float64, my_max_local + my_max_recv)

    # 混合路由器：同节点CPU拷贝 + 跨节点MPI
    function hybrid_memory_router!(my_cs, topos_for_otf, send_bufs, recv_bufs)
        mpi_send_counts = zeros(Cint, nproc)
        mpi_recv_counts = zeros(Cint, nproc)

        for src_c in my_cs
            for dest_c in 1:num_chunks
                # send_counts[dest_c] is indexed by destination chunk/rank.
                sc = topos_for_otf[src_c].send_counts[dest_c]
                dest_rank = (dest_c - 1) % nproc
                if dest_rank != rank
                    mpi_send_counts[dest_rank + 1] += sc
                end
            end
        end

        for dest_c in my_cs
            for src_c in 1:num_chunks
                # recv_counts[src_c] is indexed by source chunk/rank.
                rc = topos_for_otf[dest_c].recv_counts[src_c]
                src_rank = (src_c - 1) % nproc
                if src_rank != rank
                    mpi_recv_counts[src_rank + 1] += rc
                end
            end
        end

        mpi_send_buf = Vector{Float64}(undef, sum(mpi_send_counts))
        mpi_recv_buf = Vector{Float64}(undef, sum(mpi_recv_counts))
        send_off = vcat(0, cumsum(Int.(mpi_send_counts))[1:end-1])
        recv_off = vcat(0, cumsum(Int.(mpi_recv_counts))[1:end-1])
        write_pos = send_off .+ 1

        # 本地路由 + 打包MPI发送
        for (src_idx, src_c) in enumerate(my_cs)
            src_offset = 1
            for dest_c in 1:num_chunks
                # send_counts[dest_c] is indexed by destination chunk/rank.
                sc = topos_for_otf[src_c].send_counts[dest_c]
                if sc > 0
                    dest_rank = (dest_c - 1) % nproc
                    if dest_rank == rank
                        dest_idx = findfirst(==(dest_c), my_cs)
                        # recv_counts[src_c] is indexed by source chunk/rank.
                        recv_offset = 1 + sum(topos_for_otf[dest_c].recv_counts[1:src_c-1]; init=0)
                        copyto!(recv_bufs[dest_idx], recv_offset, send_bufs[src_idx], src_offset, sc)
                    else
                        wp = write_pos[dest_rank + 1]
                        copyto!(mpi_send_buf, wp, send_bufs[src_idx], src_offset, sc)
                        write_pos[dest_rank + 1] += sc
                    end
                end
                src_offset += sc
            end
        end

        sv = MPI.VBuffer(mpi_send_buf, mpi_send_counts)
        rv = MPI.VBuffer(mpi_recv_buf, mpi_recv_counts)
        MPI.Alltoallv!(sv, rv, comm)

        read_pos = recv_off .+ 1
        for (dest_idx, dest_c) in enumerate(my_cs)
            dest_offset = 1
            for src_c in 1:num_chunks
                # recv_counts[src_c] is indexed by source chunk/rank.
                rc = topos_for_otf[dest_c].recv_counts[src_c]
                if rc > 0
                    src_rank = (src_c - 1) % nproc
                    if src_rank != rank
                        rp = read_pos[src_rank + 1]
                        copyto!(recv_bufs[dest_idx], dest_offset, mpi_recv_buf, rp, rc)
                        read_pos[src_rank + 1] += rc
                    end
                end
                dest_offset += rc
            end
        end
    end

    # ============================================================
    _get_hf = (nelec) -> begin
        v = zeros(Float64, local_dim)
        offset = 1
        for (idx, c) in enumerate(my_chunks)
            ld = my_chunk_dims[idx]
            if ld > 0
                d_view = @view d_cache[1:ld]
                set_local_hf_gpu!(gmaps_all[c], basis, d_view, nelec)
                copyto!(v, offset, Array(d_view), 1, ld)
            end
            offset += ld
        end
        return v
    end

    _zeros = () -> Vector{Float64}(undef, local_dim)

    _inner = (lv, rv) -> begin
        local_dot = dot(lv, rv)
        return MPI.Allreduce(local_dot, +, comm)
    end

    _normalize = (v) -> begin
        n2 = sum(abs2, v)
        global_n = sqrt(MPI.Allreduce(n2, +, comm))
        v ./= global_n
    end

    _hvec = (v, Hv) -> begin
        # 输入 → 内部chunk
        for idx in 1:n_my
            ld = my_chunk_dims[idx]
            ld > 0 && copyto!(host_v_chunks[idx], 1, v, chunk_offsets[idx] + 1, ld)
        end
        for w in host_w_chunks; fill!(w, 0.0); end

        for i in 1:n_otfs
            # 阶段A: GPU打包
            for idx in 1:n_my
                c = my_chunks[idx]
                topo = sub_topos_all[c, i]
                if topo.send_dim > 0
                    ld = my_chunk_dims[idx]
                    copyto!(d_cache, 1, host_v_chunks[idx], 1, ld)
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(
                        topo.ptr::Ptr{Cvoid},
                        pointer(d_cache)::CuPtr{Float64},
                        pointer(d_send)::CuPtr{Float64},
                    )::Cvoid
                    copyto!(host_send_bufs[idx], 1, d_send, 1, topo.send_dim)
                end
            end

            # 阶段B: 混合路由
            hybrid_memory_router!(my_chunks, sub_topos_all[:, i], host_send_bufs, host_recv_bufs)

            # 阶段C: GPU计算
            for idx in 1:n_my
                c = my_chunks[idx]
                topo = sub_topos_all[c, i]
                ld = my_chunk_dims[idx]
                ld == 0 && continue
                copyto!(d_cache, 1, host_v_chunks[idx], 1, ld)
                if topo.recv_dim > 0
                    copyto!(d_cache, ld + 1, host_recv_bufs[idx], 1, topo.recv_dim)
                end
                d_w .= 0.0
                @ccall LIB_CUDIST.compute_hvec_sub_chunk_gpu_f64(
                    cu_basis_dev.ptr::Ptr{Cvoid}, cu_otfs[i].ptr::Ptr{Cvoid},
                    topo.ptr::Ptr{Cvoid},
                    pointer(d_cache)::CuPtr{Float64}, pointer(d_w)::CuPtr{Float64},
                )::Cvoid
                host_w_chunks[idx] .+= Array(d_w[1:ld])
            end
        end

        # 内部chunk → 输出
        for idx in 1:n_my
            ld = my_chunk_dims[idx]
            ld > 0 && copyto!(Hv, chunk_offsets[idx] + 1, host_w_chunks[idx], 1, ld)
        end
    end

    _expm = (idx, θ, v) -> begin
        n_pool == 0 && error("CuDistributedFunctions.expm requires an operator pool; construct with CuDistributedFunctions(ModeHybrid, basis, ham, pool, comm; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "CuDistributedFunctions.expm: pool index out of bounds"
        for chunk_idx in 1:n_my
            ld = my_chunk_dims[chunk_idx]
            ld > 0 && copyto!(host_v_chunks[chunk_idx], 1, v, chunk_offsets[chunk_idx] + 1, ld)
        end

        for j in 1:length(pool_cu_otfs[idx])
            topos = pool_sub_topos_all[idx][:, j]
            for chunk_idx in 1:n_my
                c = my_chunks[chunk_idx]
                topo = topos[c]
                if topo.send_dim > 0
                    ld = my_chunk_dims[chunk_idx]
                    copyto!(d_cache, 1, host_v_chunks[chunk_idx], 1, ld)
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(
                        topo.ptr::Ptr{Cvoid},
                        pointer(d_cache)::CuPtr{Float64},
                        pointer(d_send)::CuPtr{Float64},
                    )::Cvoid
                    copyto!(host_send_bufs[chunk_idx], 1, d_send, 1, topo.send_dim)
                end
            end

            hybrid_memory_router!(my_chunks, topos, host_send_bufs, host_recv_bufs)

            for chunk_idx in 1:n_my
                c = my_chunks[chunk_idx]
                topo = topos[c]
                ld = my_chunk_dims[chunk_idx]
                ld == 0 && continue
                copyto!(d_cache, 1, host_v_chunks[chunk_idx], 1, ld)
                if topo.recv_dim > 0
                    copyto!(d_cache, ld + 1, host_recv_bufs[chunk_idx], 1, topo.recv_dim)
                end
                @ccall LIB_CUDIST.compute_expm_sub_chunk_gpu_f64(
                    cu_basis_dev.ptr::Ptr{Cvoid}, pool_cu_otfs[idx][j].ptr::Ptr{Cvoid},
                    topo.ptr::Ptr{Cvoid}, Int64(0)::Int64, θ::Cdouble,
                    pointer(d_cache)::CuPtr{Float64},
                )::Cvoid
                copyto!(host_v_chunks[chunk_idx], 1, d_cache, 1, ld)
            end
        end

        for chunk_idx in 1:n_my
            ld = my_chunk_dims[chunk_idx]
            ld > 0 && copyto!(v, chunk_offsets[chunk_idx] + 1, host_v_chunks[chunk_idx], 1, ld)
        end
        return v
    end
    _grad = (idx, θ, lv, rv) -> error("CuDistributedFunctions.grad: not yet implemented")
    _backgrad = (idx, θ, lv, rv) -> begin
        n_pool == 0 && error("CuDistributedFunctions.backgrad requires an operator pool; construct with CuDistributedFunctions(ModeHybrid, basis, ham, pool, comm; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "CuDistributedFunctions.backgrad: pool index out of bounds"
        for chunk_idx in 1:n_my
            ld = my_chunk_dims[chunk_idx]
            if ld > 0
                copyto!(host_l_chunks[chunk_idx], 1, lv, chunk_offsets[chunk_idx] + 1, ld)
                copyto!(host_r_chunks[chunk_idx], 1, rv, chunk_offsets[chunk_idx] + 1, ld)
            end
        end

        local_grad = 0.0
        for j in 1:length(pool_cu_otfs[idx])
            topos = pool_sub_topos_all[idx][:, j]
            for chunk_idx in 1:n_my
                c = my_chunks[chunk_idx]
                topo = topos[c]
                if topo.send_dim > 0
                    ld = my_chunk_dims[chunk_idx]
                    copyto!(d_cache, 1, host_l_chunks[chunk_idx], 1, ld)
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(
                        topo.ptr::Ptr{Cvoid},
                        pointer(d_cache)::CuPtr{Float64},
                        pointer(d_send)::CuPtr{Float64},
                    )::Cvoid
                    copyto!(host_send_bufs[chunk_idx], 1, d_send, 1, topo.send_dim)

                    copyto!(d_cache, 1, host_r_chunks[chunk_idx], 1, ld)
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(
                        topo.ptr::Ptr{Cvoid},
                        pointer(d_cache)::CuPtr{Float64},
                        pointer(d_send)::CuPtr{Float64},
                    )::Cvoid
                    copyto!(host_send2_bufs[chunk_idx], 1, d_send, 1, topo.send_dim)
                end
            end

            hybrid_memory_router!(my_chunks, topos, host_send_bufs, host_recv_bufs)
            hybrid_memory_router!(my_chunks, topos, host_send2_bufs, host_recv2_bufs)

            for chunk_idx in 1:n_my
                c = my_chunks[chunk_idx]
                topo = topos[c]
                ld = my_chunk_dims[chunk_idx]
                ld == 0 && continue
                copyto!(d_cache, 1, host_l_chunks[chunk_idx], 1, ld)
                copyto!(d_back_cache, 1, host_r_chunks[chunk_idx], 1, ld)
                if topo.recv_dim > 0
                    copyto!(d_cache, ld + 1, host_recv_bufs[chunk_idx], 1, topo.recv_dim)
                    copyto!(d_back_cache, ld + 1, host_recv2_bufs[chunk_idx], 1, topo.recv_dim)
                end
                local_grad += @ccall LIB_CUDIST.compute_backgrad_sub_chunk_gpu_f64(
                    cu_basis_dev.ptr::Ptr{Cvoid}, pool_cu_otfs[idx][j].ptr::Ptr{Cvoid},
                    topo.ptr::Ptr{Cvoid}, θ::Cdouble,
                    pointer(d_cache)::CuPtr{Float64},
                    pointer(d_back_cache)::CuPtr{Float64},
                )::Cdouble
                copyto!(host_l_chunks[chunk_idx], 1, d_cache, 1, ld)
                copyto!(host_r_chunks[chunk_idx], 1, d_back_cache, 1, ld)
            end
        end

        for chunk_idx in 1:n_my
            ld = my_chunk_dims[chunk_idx]
            if ld > 0
                copyto!(lv, chunk_offsets[chunk_idx] + 1, host_l_chunks[chunk_idx], 1, ld)
                copyto!(rv, chunk_offsets[chunk_idx] + 1, host_r_chunks[chunk_idx], 1, ld)
            end
        end
        return MPI.Allreduce(local_grad, +, comm)
    end

    d_cache_bytes = _hvec_bytes(my_max_local + my_max_recv)
    d_send_bytes = _hvec_bytes(my_max_send)
    d_w_bytes = _hvec_bytes(my_max_local)
    hvec_vram_bytes = d_cache_bytes + d_send_bytes + d_w_bytes

    max_local_dim_all = MPI.Allreduce(Int64(my_max_local), max, comm)
    max_send_dim_all = MPI.Allreduce(Int64(my_max_send), max, comm)
    max_recv_dim_all = MPI.Allreduce(Int64(my_max_recv), max, comm)
    max_hvec_vram_bytes = MPI.Allreduce(hvec_vram_bytes, max, comm)
    total_hvec_vram_bytes = MPI.Allreduce(hvec_vram_bytes, +, comm)

    if rank == 0
        println("\nCuDistributedFunctions (HybridOOC) built:")
        println("  MPI ranks:              $(nproc)")
        println("  Virtual chunks requested: $(requested_num_chunks)")
        println("  Virtual chunks effective: $(num_chunks) (local r0: $(n_my))")
        println("  Local dim (r0):         $(local_dim)")
        println("  Max local/chunk dim:    $(max_local_dim_all)")
        println("  Sub-networks:           $(n_otfs)")
        println("  Pool operators:         $(n_pool)")
        println("  Max send dim:           $(max_send_dim_all)")
        println("  Max recv dim:           $(max_recv_dim_all)")
        println("  Hvec buffers (r0):      d_cache=$(_hvec_gib(d_cache_bytes)) GB, d_send=$(_hvec_gib(d_send_bytes)) GB, d_w=$(_hvec_gib(d_w_bytes)) GB")
        println("  Peak GPU hvec buffer/rank: $(round(max_hvec_vram_bytes / 1024^3, digits=3)) GB ($(max_hvec_vram_bytes) bytes)")
        println("  Total GPU hvec buffers:    $(round(total_hvec_vram_bytes / 1024^3, digits=3)) GB ($(total_hvec_vram_bytes) bytes)\n")
    end

    return CuDistributedFunctions{ModeHybrid}(
        comm, rank, nproc, local_dim,
        _hvec, _normalize, _zeros, _get_hf, _inner,
        _expm, _grad, _backgrad,
    )
end

function CuDistributedFunctions(
    ::Type{ModeHybrid},
    mole::Mole,
    ham::BinaryQubitAABB{Ti,Tv_h,TK,TV},
    comm::MPI.Comm;
    virtual_k::Int=0,
    virtual_seed::Int=1234,
    virtual_orbsym::Vector{Int64}=Int64[],
    virtual_optimize::Bool=true,
    virtual_ntry::Int=64,
    num_chunks::Int=16,
    tol::Float64=1e-12,
) where {Ti,Tv_h,TK,TV}
    basis = _build_virtual_or_physical_basis(
        mole;
        virtual_k=virtual_k,
        virtual_seed=virtual_seed,
        virtual_orbsym=virtual_orbsym,
        virtual_optimize=virtual_optimize,
        virtual_ntry=virtual_ntry,
    )

    return CuDistributedFunctions(ModeHybrid, basis, ham, comm; num_chunks=num_chunks, tol=tol), basis
end
