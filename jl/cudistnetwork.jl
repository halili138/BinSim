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
end

# ============================================================
# SerialOOC: 单卡突破显存限制
# ============================================================
function CuDistributedFunctions(
    ::Type{ModeSerial},
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv_h,TK,TV};
    num_chunks::Int=4,
    gpu_id::Int=0,
    tol::Float64=1e-12,
) where {Ti,Tv_h,TK,TV}
    CUDA.device!(gpu_id)

    cu_basis_dev = CuBasisManager(basis)

    # 虚拟切片：用非MPI的GlobalMemMap构造各chunk的分块视图
    gmaps = [GlobalMemMap(basis; rank=r - 1, size=num_chunks) for r in 1:num_chunks]
    cpu_otfs, cu_otfs = build_distributed_cu_otfs(basis, ham, tol)
    n_otfs = length(cu_otfs)

    # 路由表: [chunk, otf]
    sub_topos = Matrix{CuSubTopology}(undef, num_chunks, n_otfs)
    for r in 1:num_chunks, i in 1:n_otfs
        sub_topos[r, i] = CuSubTopology(basis, cpu_otfs[i], gmaps[r])
    end

    # 各chunk的维度与缓冲区
    chunk_dims   = [gmaps[r].local_dim for r in 1:num_chunks]
    chunk_offsets = vcat(0, cumsum(chunk_dims))
    local_dim     = chunk_offsets[end]

    max_send_dims = [maximum(t.send_dim for t in sub_topos[r, :]) for r in 1:num_chunks]
    max_recv_dims = [maximum(t.recv_dim for t in sub_topos[r, :]) for r in 1:num_chunks]

    host_v_chunks  = [Vector{Float64}(undef, chunk_dims[r]) for r in 1:num_chunks]
    host_w_chunks  = [Vector{Float64}(undef, chunk_dims[r]) for r in 1:num_chunks]
    host_send_bufs = [Vector{Float64}(undef, max_send_dims[r]) for r in 1:num_chunks]
    host_recv_bufs = [Vector{Float64}(undef, max_recv_dims[r]) for r in 1:num_chunks]

    # GPU 复用池
    global_max_local = maximum(chunk_dims)
    global_max_recv  = maximum(max_recv_dims)
    global_max_send  = maximum(max_send_dims)

    d_cache = CUDA.zeros(Float64, global_max_local + global_max_recv)
    d_send  = CUDA.zeros(Float64, global_max_send)
    d_w     = CUDA.zeros(Float64, global_max_local)

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

    _expm = (idx, θ, v) -> error("CuDistributedFunctions.expm: not yet implemented")
    _grad = (idx, θ, lv, rv) -> error("CuDistributedFunctions.grad: not yet implemented")

    comm = MPI.COMM_SELF
    rank = 0
    nproc = 1

    println("\nCuDistributedFunctions (SerialOOC) built:")
    println("  GPU:             $(CUDA.name(CUDA.device()))")
    println("  Virtual chunks:  $(num_chunks)")
    println("  Total local dim: $(local_dim)")
    println("  Sub-networks:    $(n_otfs)")
    vram_used = (global_max_local + global_max_recv + global_max_send) * 8 / (1024^3)
    println("  GPU VRAM peak:   $(round(vram_used, digits=3)) GB\n")

    return CuDistributedFunctions{ModeSerial}(
        comm, rank, nproc, local_dim,
        _hvec, _normalize, _zeros, _get_hf, _inner,
        _expm, _grad,
    )
end

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
    ham::BinaryQubitAABB{Ti,Tv_h,TK,TV},
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

    # [otf_idx, phase]
    sub_topos = Matrix{CuSubTopology}(undef, n_otfs, num_phases)
    for i in 1:n_otfs, p in 1:num_phases
        sub_topos[i, p] = CuSubTopology(basis, cpu_otfs[i], gmap;
            num_phases=num_phases, phase_idx=p - 1)
    end

    max_send_dim = n_otfs == 0 ? 0 : maximum(t.send_dim for t in sub_topos)
    max_recv_dim = n_otfs == 0 ? 0 : maximum(t.recv_dim for t in sub_topos)

    hvec_cache_scalars = local_dim + max_recv_dim
    hvec_send_scalars = max_send_dim
    hvec_recv_scalars = max_recv_dim
    hvec_w_scalars = local_dim
    hvec_scalar_count = hvec_cache_scalars + hvec_send_scalars + hvec_recv_scalars + hvec_w_scalars
    hvec_vram_bytes = Int64(hvec_scalar_count) * Int64(sizeof(Float64))

    max_local_dim_all = MPI.Allreduce(Int64(local_dim), max, comm)
    max_send_dim_all = MPI.Allreduce(Int64(max_send_dim), max, comm)
    max_recv_dim_all = MPI.Allreduce(Int64(max_recv_dim), max, comm)
    max_hvec_vram_bytes = MPI.Allreduce(hvec_vram_bytes, max, comm)
    total_hvec_vram_bytes = MPI.Allreduce(hvec_vram_bytes, +, comm)

    d_cache = CUDA.zeros(Float64, hvec_cache_scalars)
    d_send  = CUDA.zeros(Float64, hvec_send_scalars)
    d_recv  = CUDA.zeros(Float64, hvec_recv_scalars)
    d_w     = CUDA.zeros(Float64, hvec_w_scalars)

    d_local_v = @view d_cache[1:local_dim]

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

            if topo.send_dim > 0
                @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(
                    topo.ptr::Ptr{Cvoid},
                    pointer(d_cache)::CuPtr{Float64},
                    pointer(d_send)::CuPtr{Float64},
                )::Cvoid
            end

            # 集体通信：所有rank必须参与
            send_vbuf = MPI.VBuffer(d_send, topo.send_counts)
            recv_vbuf = MPI.VBuffer(d_recv, topo.recv_counts)
            MPI.Alltoallv!(send_vbuf, recv_vbuf, comm)

            if topo.recv_dim > 0
                recv_view = @view d_cache[local_dim + 1 : local_dim + topo.recv_dim]
                copyto!(recv_view, 1, d_recv, 1, topo.recv_dim)
            end

            @ccall LIB_CUDIST.compute_hvec_sub_chunk_gpu_f64(
                cu_basis_dev.ptr::Ptr{Cvoid}, cu_otfs[i].ptr::Ptr{Cvoid},
                topo.ptr::Ptr{Cvoid},
                pointer(d_cache)::CuPtr{Float64}, pointer(d_w)::CuPtr{Float64},
            )::Cvoid
        end

        copyto!(Hv, d_w)
    end

    _expm = (idx, θ, v) -> error("CuDistributedFunctions.expm: not yet implemented")
    _grad = (idx, θ, lv, rv) -> error("CuDistributedFunctions.grad: not yet implemented")

    if rank == 0
        println("\nCuDistributedFunctions (NVLink) built:")
        println("  MPI ranks:              $(nproc)")
        println("  Local dim (r0):         $(local_dim)")
        println("  Max local dim:          $(max_local_dim_all)")
        println("  Sub-networks:           $(n_otfs)")
        println("  Phases requested:       $(requested_num_phases)")
        println("  Phases effective:       $(num_phases)")
        println("  Max send dim:           $(max_send_dim_all)")
        println("  Max recv dim:           $(max_recv_dim_all)")
        println("  Hvec buffers (r0):      d_cache=$(hvec_cache_scalars), d_send=$(hvec_send_scalars), d_recv=$(hvec_recv_scalars), d_w=$(hvec_w_scalars) Float64 scalars")
        println("  Max hvec VRAM/rank:     $(round(max_hvec_vram_bytes / 1024^3, digits=3)) GB ($(max_hvec_vram_bytes) bytes)")
        println("  Total hvec VRAM/ranks:  $(round(total_hvec_vram_bytes / 1024^3, digits=3)) GB ($(total_hvec_vram_bytes) bytes)")
        println()
    end

    return CuDistributedFunctions{ModeNVLink}(
        comm, rank, nproc, local_dim,
        _hvec, _normalize, _zeros, _get_hf, _inner,
        _expm, _grad,
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
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    ngpus = length(CUDA.devices())
    CUDA.device!(rank % ngpus)

    cu_basis_dev = CuBasisManager(basis)

    @assert num_chunks >= nproc "num_chunks must be >= MPI size"

    gmaps_all = [GlobalMemMap(basis; rank=r - 1, size=num_chunks) for r in 1:num_chunks]
    my_chunks = [c for c in 1:num_chunks if (c - 1) % nproc == rank]
    n_my = length(my_chunks)

    cpu_otfs, cu_otfs = build_distributed_cu_otfs(basis, ham, tol)
    n_otfs = length(cu_otfs)

    sub_topos_all = Matrix{CuSubTopology}(undef, num_chunks, n_otfs)
    for r in 1:num_chunks, i in 1:n_otfs
        sub_topos_all[r, i] = CuSubTopology(basis, cpu_otfs[i], gmaps_all[r])
    end

    chunk_dims    = [gmaps_all[r].local_dim for r in 1:num_chunks]
    my_chunk_dims = [chunk_dims[c] for c in my_chunks]
    local_dim     = sum(my_chunk_dims)
    chunk_offsets = vcat(0, cumsum(Int.(my_chunk_dims)))

    max_send_dims = [maximum(t.send_dim for t in sub_topos_all[c, :]) for c in my_chunks]
    max_recv_dims = [maximum(t.recv_dim for t in sub_topos_all[c, :]) for c in my_chunks]

    host_v_chunks  = [Vector{Float64}(undef, my_chunk_dims[idx]) for idx in 1:n_my]
    host_w_chunks  = [Vector{Float64}(undef, my_chunk_dims[idx]) for idx in 1:n_my]
    host_send_bufs = [Vector{Float64}(undef, max_send_dims[idx]) for idx in 1:n_my]
    host_recv_bufs = [Vector{Float64}(undef, max_recv_dims[idx]) for idx in 1:n_my]

    my_max_local = n_my == 0 ? 0 : maximum(my_chunk_dims)
    my_max_recv = n_my == 0 ? 0 : maximum(max_recv_dims)
    my_max_send = n_my == 0 ? 0 : maximum(max_send_dims)

    d_cache = CUDA.zeros(Float64, my_max_local + my_max_recv)
    d_send  = CUDA.zeros(Float64, my_max_send)
    d_w     = CUDA.zeros(Float64, my_max_local)

    # 混合路由器：同节点CPU拷贝 + 跨节点MPI
    function hybrid_memory_router!(my_cs, topos_for_otf, send_bufs, recv_bufs)
        mpi_send_counts = zeros(Cint, nproc)
        mpi_recv_counts = zeros(Cint, nproc)

        for (idx, c) in enumerate(my_cs)
            for d in 1:num_chunks
                sc = topos_for_otf[d].send_counts[c]
                dest_rank = (d - 1) % nproc
                if dest_rank != rank
                    mpi_send_counts[dest_rank + 1] += sc
                end
                rc = topos_for_otf[d].recv_counts[c]
                src_rank = (d - 1) % nproc
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
        for (idx, c) in enumerate(my_cs)
            src_offset = 1
            for d in 1:num_chunks
                sc = topos_for_otf[d].send_counts[c]
                if sc > 0
                    dest_rank = (d - 1) % nproc
                    if dest_rank == rank
                        d_idx = findfirst(x -> x == d, my_cs)
                        rc_offset = 1 + sum(topos_for_otf[d].recv_counts[1:c-1]; init=0)
                        copyto!(recv_bufs[d_idx], rc_offset, send_bufs[idx], src_offset, sc)
                    else
                        wp = write_pos[dest_rank + 1]
                        copyto!(mpi_send_buf, wp, send_bufs[idx], src_offset, sc)
                        write_pos[dest_rank + 1] += sc
                    end
                end
                src_offset += sc
            end
        end

        if sum(mpi_send_counts) + sum(mpi_recv_counts) > 0
            sv = MPI.VBuffer(mpi_send_buf, mpi_send_counts)
            rv = MPI.VBuffer(mpi_recv_buf, mpi_recv_counts)
            MPI.Alltoallv!(sv, rv, comm)
        end

        read_pos = recv_off .+ 1
        for (idx, c) in enumerate(my_cs)
            dst_offset = 1
            for src_c in 1:num_chunks
                rc = topos_for_otf[src_c].recv_counts[c]
                if rc > 0
                    src_rank = (src_c - 1) % nproc
                    if src_rank != rank
                        rp = read_pos[src_rank + 1]
                        copyto!(recv_bufs[idx], dst_offset, mpi_recv_buf, rp, rc)
                        read_pos[src_rank + 1] += rc
                    end
                end
                dst_offset += rc
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

    _expm = (idx, θ, v) -> error("CuDistributedFunctions.expm: not yet implemented")
    _grad = (idx, θ, lv, rv) -> error("CuDistributedFunctions.grad: not yet implemented")

    if rank == 0
        vram_used = (my_max_local + max(my_max_recv, my_max_send)) * 8 / (1024^3)
        println("\nCuDistributedFunctions (HybridOOC) built:")
        println("  MPI ranks:       $(nproc)")
        println("  Virtual chunks:  $(num_chunks) (local: $(n_my))")
        println("  Local dim (r0):  $(local_dim)")
        println("  Sub-networks:    $(n_otfs)")
        println("  GPU VRAM peak:   $(round(vram_used, digits=3)) GB\n")
    end

    return CuDistributedFunctions{ModeHybrid}(
        comm, rank, nproc, local_dim,
        _hvec, _normalize, _zeros, _get_hf, _inner,
        _expm, _grad,
    )
end
