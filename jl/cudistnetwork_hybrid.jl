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
    d_back_cache_ref = Ref{Union{Nothing, CuVector{Float64}}}(nothing)

    function backgrad_cache!()
        if d_back_cache_ref[] === nothing
            d_back_cache_ref[] = CUDA.zeros(Float64, my_max_local + my_max_recv)
        end
        return d_back_cache_ref[]::CuVector{Float64}
    end

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
        d_back_cache = backgrad_cache!()
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

    _expm_2d = (idx, θ, v) -> begin
        n_pool == 0 && error("CuDistributedFunctions.expm_2d requires an operator pool; construct with CuDistributedFunctions(ModeHybrid, basis, ham, pool, comm; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "CuDistributedFunctions.expm_2d: pool index out of bounds"
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
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(topo.ptr::Ptr{Cvoid}, pointer(d_cache)::CuPtr{Float64}, pointer(d_send)::CuPtr{Float64})::Cvoid
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
                @ccall LIB_CUDIST.compute_expm_sub_chunk_gpu_2d_f64(cu_basis_dev.ptr::Ptr{Cvoid}, pool_cu_otfs[idx][j].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, Int64(0)::Int64, θ::Cdouble, pointer(d_cache)::CuPtr{Float64})::Cvoid
                copyto!(host_v_chunks[chunk_idx], 1, d_cache, 1, ld)
            end
        end
        for chunk_idx in 1:n_my
            ld = my_chunk_dims[chunk_idx]
            ld > 0 && copyto!(v, chunk_offsets[chunk_idx] + 1, host_v_chunks[chunk_idx], 1, ld)
        end
        return v
    end
    _grad_2d = (idx, θ, lv, rv) -> error("CuDistributedFunctions.grad_2d: not yet implemented")
    _backgrad_2d = (idx, θ, lv, rv) -> begin
        n_pool == 0 && error("CuDistributedFunctions.backgrad_2d requires an operator pool; construct with CuDistributedFunctions(ModeHybrid, basis, ham, pool, comm; ...) for VQE usage")
        @assert 1 <= idx <= n_pool "CuDistributedFunctions.backgrad_2d: pool index out of bounds"
        d_back_cache = backgrad_cache!()
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
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(topo.ptr::Ptr{Cvoid}, pointer(d_cache)::CuPtr{Float64}, pointer(d_send)::CuPtr{Float64})::Cvoid
                    copyto!(host_send_bufs[chunk_idx], 1, d_send, 1, topo.send_dim)
                    copyto!(d_cache, 1, host_r_chunks[chunk_idx], 1, ld)
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(topo.ptr::Ptr{Cvoid}, pointer(d_cache)::CuPtr{Float64}, pointer(d_send)::CuPtr{Float64})::Cvoid
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
                local_grad += @ccall LIB_CUDIST.compute_backgrad_sub_chunk_gpu_2d_f64(cu_basis_dev.ptr::Ptr{Cvoid}, pool_cu_otfs[idx][j].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, θ::Cdouble, pointer(d_cache)::CuPtr{Float64}, pointer(d_back_cache)::CuPtr{Float64})::Cdouble
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
        _expm_2d, _backgrad_2d,
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
