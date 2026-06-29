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
    d_back_cache_ref = Ref{Union{Nothing, CuVector{Float64}}}(nothing)

    function backgrad_cache!()
        if d_back_cache_ref[] === nothing
            d_back_cache_ref[] = CUDA.zeros(Float64, global_max_local + global_max_recv)
        end
        return d_back_cache_ref[]::CuVector{Float64}
    end

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
        d_back_cache = backgrad_cache!()
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

    _expm_2d = (idx, θ, v) -> begin
        n_pool == 0 && error("CuDistributedFunctions.expm_2d requires an operator pool; construct with CuDistributedFunctions(ModeSerial, basis, ham, pool; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "CuDistributedFunctions.expm_2d: pool index out of bounds"
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
                @ccall LIB_CUDIST.compute_expm_sub_chunk_gpu_2d_f64(cu_basis_dev.ptr::Ptr{Cvoid}, pool_cu_otfs[idx][j].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, Int64(0)::Int64, θ::Cdouble, pointer(d_cache)::CuPtr{Float64})::Cvoid
                copyto!(host_v_chunks[r], 1, d_cache, 1, ld + topo.recv_dim)
            end
        end
        for r in 1:num_chunks
            ld = chunk_dims[r]
            ld > 0 && copyto!(v, chunk_offsets[r] + 1, host_v_chunks[r], 1, ld)
        end
        return v
    end
    _grad_2d = (idx, θ, lv, rv) -> error("CuDistributedFunctions.grad_2d: not yet implemented")
    _backgrad_2d = (idx, θ, lv, rv) -> begin
        n_pool == 0 && error("CuDistributedFunctions.backgrad_2d requires an operator pool; construct with CuDistributedFunctions(ModeSerial, basis, ham, pool; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "CuDistributedFunctions.backgrad_2d: pool index out of bounds"
        d_back_cache = backgrad_cache!()
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
                local_grad += @ccall LIB_CUDIST.compute_backgrad_sub_chunk_gpu_2d_f64(cu_basis_dev.ptr::Ptr{Cvoid}, pool_cu_otfs[idx][j].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, θ::Cdouble, pointer(d_cache)::CuPtr{Float64}, pointer(d_back_cache)::CuPtr{Float64})::Cdouble
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
        _expm_2d, _backgrad_2d,
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
