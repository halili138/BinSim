# =====================================================================
# 混合核外架构 (Host RAM 驻留 + GPU 轮询流转 + MPI 通信)
# =====================================================================

ENV["OMP_NUM_THREADS"] = "1"
ENV["OMP_PROC_BIND"] = "false"

include("../jl/cudistribute.jl")

# 生成欺骗引擎的虚拟全局映射
function build_virtual_gmap(basis::BasisManager, rank::Int, size::Int)
    ptr = @ccall LIB_DIST.build_global_map_otf_f64(basis.ptr::Ptr{Cvoid}, Cint(rank), Cint(size))::Ptr{Cvoid}
    local_dim = @ccall LIB_DIST.get_local_dim_otf_gmap(ptr::Ptr{Cvoid})::Int64
    obj = GlobalMemMap(ptr, local_dim, rank, size)
    finalizer(obj) do o
        o.ptr != C_NULL && @ccall LIB_DIST.destroy_global_map_otf(o.ptr::Ptr{Cvoid})::Cvoid
    end
    return obj
end

# 双层混合路由器：同节点走内存，跨节点走 MPI 大巴
function hybrid_memory_router!(my_chunks, topos_for_otf, host_send_bufs, host_recv_bufs, mpi_size, mpi_rank, comm)
    total_chunks = size(topos_for_otf, 1)

    mpi_send_counts = zeros(Cint, mpi_size)
    mpi_recv_counts = zeros(Cint, mpi_size)

    for c in my_chunks, d in 1:total_chunks
        count_send = topos_for_otf[c].send_counts[d]
        dest_rank = (d - 1) % mpi_size
        if dest_rank != mpi_rank
            mpi_send_counts[dest_rank+1] += count_send
        end

        count_recv = topos_for_otf[c].recv_counts[d]
        src_rank = (d - 1) % mpi_size
        if src_rank != mpi_rank
            mpi_recv_counts[src_rank+1] += count_recv
        end
    end

    mpi_send_buf = zeros(Float64, sum(mpi_send_counts))
    mpi_recv_buf = zeros(Float64, sum(mpi_recv_counts))

    mpi_send_offsets = [0; cumsum(mpi_send_counts)[1:end-1]]
    mpi_recv_offsets = [0; cumsum(mpi_recv_counts)[1:end-1]]
    write_pos = copy(mpi_send_offsets) .+ 1

    for (idx, c) in enumerate(my_chunks)
        src_offset = 1
        for d in 1:total_chunks
            count = topos_for_otf[c].send_counts[d]
            if count > 0
                dest_rank = (d - 1) % mpi_size
                if dest_rank == mpi_rank
                    d_idx = findfirst(x -> x == d, my_chunks)
                    recv_offset = 1 + sum(topos_for_otf[d].recv_counts[1:c-1]; init=0)
                    copyto!(host_recv_bufs[d_idx], recv_offset, host_send_bufs[idx], src_offset, count)
                else
                    w_pos = write_pos[dest_rank+1]
                    copyto!(mpi_send_buf, w_pos, host_send_bufs[idx], src_offset, count)
                    write_pos[dest_rank+1] += count
                end
            end
            src_offset += count
        end
    end

    if sum(mpi_send_counts) > 0 || sum(mpi_recv_counts) > 0
        send_vbuf = MPI.VBuffer(mpi_send_buf, mpi_send_counts)
        recv_vbuf = MPI.VBuffer(mpi_recv_buf, mpi_recv_counts)
        MPI.Alltoallv!(send_vbuf, recv_vbuf, comm)
    end

    read_pos = copy(mpi_recv_offsets) .+ 1
    for (idx, c) in enumerate(my_chunks)
        dst_offset = 1
        for src_c in 1:total_chunks
            count = topos_for_otf[c].recv_counts[src_c]
            if count > 0
                src_rank = (src_c - 1) % mpi_size
                if src_rank != mpi_rank
                    r_pos = read_pos[src_rank+1]
                    copyto!(host_recv_bufs[idx], dst_offset, mpi_recv_buf, r_pos, count)
                    read_pos[src_rank+1] += count
                end
            end
            dst_offset += count
        end
    end
end

function test_multigpu_hybrid_ooc(name, ratio, basis_name; num_chunks::Int=16)
    MPI.Init()
    comm = MPI.COMM_WORLD
    mpi_rank = MPI.Comm_rank(comm)
    mpi_size = MPI.Comm_size(comm)

    num_gpus = length(CUDA.devices())
    CUDA.device!(mpi_rank % num_gpus)

    @assert num_chunks >= mpi_size "num_chunks (虚拟切片数) 必须 >= 物理进程数"

    mole = Mole()
    mole.name = name
    mole.ratio = ratio
    mole.basis = basis_name
    build(mole)
    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    cu_basis_dev = CuBasisManager(basis)
    ham = JW_hamiltonian(mole)

    if mpi_rank == 0
        println("="^65)
        @printf("=== 启动形态 B: 混合核外架构 (Host RAM OOC 破壁) ===\n")
        @printf("=== 物理进程: %d | 虚拟切片(Chunks): %d ===\n", mpi_size, num_chunks)
        println("="^65)
    end

    gmaps_global = [build_virtual_gmap(basis, r - 1, num_chunks) for r in 1:num_chunks]
    cpu_otfs, cu_otfs = build_distributed_cu_otfs(basis, ham)

    sub_topos_global = Matrix{CuSubTopology}(undef, num_chunks, length(cpu_otfs))
    # 注意 OOC 模式下不强制时间相位切分，因为它的 Chunk 切分本身就是时间切片
    for r in 1:num_chunks, i in 1:length(cpu_otfs)
        sub_topos_global[r, i] = CuSubTopology(basis, cpu_otfs[i], gmaps_global[r], 1, 0)
    end

    my_chunks = [c for c in 1:num_chunks if (c - 1) % mpi_size == mpi_rank]

    # 【海量主板内存：存储全量波函数片段】
    host_v_chunks = [zeros(Float64, gmaps_global[c].local_dim) for c in my_chunks]
    host_w_chunks = [zeros(Float64, gmaps_global[c].local_dim) for c in my_chunks]

    max_send_dims = [maximum(t.send_dim for t in sub_topos_global[c, :]) for c in my_chunks]
    max_recv_dims = [maximum(t.recv_dim for t in sub_topos_global[c, :]) for c in my_chunks]
    host_send_bufs = [zeros(Float64, max_send_dims[idx]) for idx in 1:length(my_chunks)]
    host_recv_bufs = [zeros(Float64, max_recv_dims[idx]) for idx in 1:length(my_chunks)]

    # 【极速显存：永远只保留 1 个 Chunk 的位置】
    my_max_local = isempty(my_chunks) ? 0 : maximum(gmaps_global[c].local_dim for c in my_chunks)
    my_max_recv = isempty(max_recv_dims) ? 0 : maximum(max_recv_dims)
    my_max_send = isempty(max_send_dims) ? 0 : maximum(max_send_dims)

    if mpi_rank == 0
        @printf("单卡显存被强行压缩至极小值: %.4f GB\n\n", (my_max_local + max(my_max_recv, my_max_send)) * 8 / (1024^3))
    end

    d_cache = CUDA.zeros(Float64, my_max_local + my_max_recv)
    d_send = CUDA.zeros(Float64, my_max_send)
    d_w = CUDA.zeros(Float64, my_max_local)

    for (idx, c) in enumerate(my_chunks)
        d_local_v_view = @view d_cache[1:gmaps_global[c].local_dim]
        set_local_hf_gpu!(gmaps_global[c], basis, d_local_v_view, mole.nelec)
        copyto!(host_v_chunks[idx], d_local_v_view)
    end

    local_sq = sum(sum(v .^ 2) for v in host_v_chunks)
    global_norm = sqrt(MPI.Allreduce(local_sq, +, comm))
    for v in host_v_chunks
        v ./= global_norm
    end

    dτ = 1e-1

    for step in 1:10
        t0 = time_ns()
        for w in host_w_chunks
            fill!(w, 0.0)
        end

        for i in 1:length(cu_otfs)
            # 阶段 1：串行显存打包并拷回主板
            for (idx, c) in enumerate(my_chunks)
                topo = sub_topos_global[c, i]
                if topo.send_dim > 0
                    copyto!(d_cache, 1, host_v_chunks[idx], 1, gmaps_global[c].local_dim)
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(
                        topo.ptr::Ptr{Cvoid}, pointer(d_cache)::CuPtr{Float64}, pointer(d_send)::CuPtr{Float64}
                    )::Cvoid
                    copyto!(host_send_bufs[idx], 1, d_send, 1, topo.send_dim)
                end
            end

            # 阶段 2：混合内存路由器 (主板 CPU-RAM 拷贝 + 跨网 MPI)
            hybrid_memory_router!(my_chunks, sub_topos_global[:, i], host_send_bufs, host_recv_bufs, mpi_size, mpi_rank, comm)

            # 阶段 3：拷入显存进行终极张量收缩，再拿回结果
            for (idx, c) in enumerate(my_chunks)
                topo = sub_topos_global[c, i]
                ldim = gmaps_global[c].local_dim

                copyto!(d_cache, 1, host_v_chunks[idx], 1, ldim)
                if topo.recv_dim > 0
                    copyto!(d_cache, ldim + 1, host_recv_bufs[idx], 1, topo.recv_dim)
                end

                d_w .= 0.0

                @ccall LIB_CUDIST.compute_hvec_sub_chunk_gpu_f64(
                    cu_basis_dev.ptr::Ptr{Cvoid}, cu_otfs[i].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid},
                    pointer(d_cache)::CuPtr{Float64}, pointer(d_w)::CuPtr{Float64}
                )::Cvoid

                temp_w_host = Array(d_w[1:ldim])
                host_w_chunks[idx] .+= temp_w_host
            end
        end

        for (idx, c) in enumerate(my_chunks)
            host_v_chunks[idx] .-= dτ .* host_w_chunks[idx]
        end

        local_sq = sum(sum(v .^ 2) for v in host_v_chunks)
        global_norm = sqrt(MPI.Allreduce(local_sq, +, comm))
        for v in host_v_chunks
            v ./= global_norm
        end

        CUDA.synchronize()
        if mpi_rank == 0
            @printf("Step %2d | OOC Time: %.4f s\n", step, (time_ns() - t0) / 1.0e9)
        end
    end
    MPI.Finalize()
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_multigpu_hybrid_ooc(ARGS[1], 1.0, ARGS[2]; num_chunks=16) # 将整个问题切分为 16 个超小碎片轮询
end
