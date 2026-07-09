_nts   = length(ARGS) >= 1 ? ARGS[1] : 4

ENV["OMP_NUM_THREADS"] = _nts
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")

using MPI

_name  = length(ARGS) >= 2 ? ARGS[2] : "h12"
_basis = length(ARGS) >= 3 ? ARGS[3] : "sto-3g"
_ratio = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0


include("../jl/binsim.jl")

function test_real_mpi_simulation(name, ratio, basis_name)
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    mpi_size = MPI.Comm_size(comm)

    mole = Mole()
    mole.name  = name; mole.ratio = ratio; mole.basis = basis_name
    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    
    if rank == 0
        println("="^60)
        println("=== 启动终极分布式并行测试 (片段滑窗复用架构) ===")
        @printf("MPI 总进程数: %d\n", mpi_size)
        println("="^60)
    end

    t_setup_start = time()
    
    # ---------------------------------------------------------
    # 1. 建立全局静态拓扑 (LPT) 与内存分配可视化
    # ---------------------------------------------------------
    if rank == 0; print("[1/3] 构建全局静态内存图 (GlobalMemMap) ... "); end
    t_tmp = time()
    gmap = GlobalMemMap(basis, comm)
    local_dim = gmap.local_dim
    
    # 将各节点的 local_dim 收集到主进程
    all_local_dims = MPI.Gather(local_dim, comm; root=0)
    if rank == 0
        @printf("Done (%.4f s)\n", time() - t_tmp)
        println("      -> [各节点波函数内存分布]:")
        for r in 0:mpi_size-1
            ldim = all_local_dims[r+1]
            @printf("         Rank %2d: %-10d 元素 (%.4f MB)\n", r, ldim, ldim * 8 / (1024^2))
        end
        total_dim = sum(all_local_dims)
        @printf("         总计分配: %-10d 元素 (%.4f MB)\n", total_dim, total_dim * 8 / (1024^2))
    end
    
    # ---------------------------------------------------------
    # 2. 强行拆分庞大的哈密顿量，化整为零
    # ---------------------------------------------------------
    if rank == 0; print("\n[2/3] 解析并切分物理对称性网络 (Sub OTFs) ... "); end
    t_tmp = time()
    sub_otfs = build_distributed_otfs(basis, ham)
    if rank == 0
        @printf("Done (%.4f s)\n", time() - t_tmp)
        @printf("      -> 成功将哈密顿量切分为 %d 个独立对称性片段!\n", length(sub_otfs))
    end

    # ---------------------------------------------------------
    # 3. 构建专属路由表与寻找峰值通信量
    # ---------------------------------------------------------
    if rank == 0; print("\n[3/3] 编译分段通信路由拓扑 (SubTopologies) ... "); end
    t_tmp = time()
    sub_topos = [SubTopology(basis, sub_otf, gmap) for sub_otf in sub_otfs]
    
    max_send_dim = isempty(sub_topos) ? 0 : maximum(t.send_dim for t in sub_topos)
    max_recv_dim = isempty(sub_topos) ? 0 : maximum(t.recv_dim for t in sub_topos)

    # 将各节点的峰值缓冲维度收集到主进程
    all_max_send = MPI.Gather(max_send_dim, comm; root=0)
    all_max_recv = MPI.Gather(max_recv_dim, comm; root=0)

    if rank == 0
        @printf("Done (%.4f s)\n", time() - t_tmp)
        println("      -> [各节点复用通信缓存峰值 (即用即毁)]:")
        for r in 0:mpi_size-1
            msend = all_max_send[r+1] * 8 / (1024^2)
            mrecv = all_max_recv[r+1] * 8 / (1024^2)
            @printf("         Rank %2d: Max Send = %8.4f MB | Max Recv = %8.4f MB\n", r, msend, mrecv)
        end
        println("="^60)
        @printf("全局配置与拓扑寻址完毕! 总耗时: %.4f s\n", time() - t_setup_start)
        println("="^60)
        println("\n--- 开始 Euler 迭代 ---")
    end

    # 【核心】：Cache = 本地永久态 + 即用即毁的远端临时态
    cache       = zeros(Float64, local_dim + max_recv_dim)
    local_w     = zeros(Float64, local_dim)
    send_buffer = zeros(Float64, max_send_dim)
    
    local_v     = @view cache[1:local_dim]

    # 直接在各节点内部生成属于自己的初态片段，彻底告别全局分配！
    set_local_hf!(gmap, basis, local_v, mole.nelec)

    # 局部归一化计算 (因为 HF 态只在某个特定节点非 0)
    local_sq_norm = sum(local_v .^ 2)
    global_norm = sqrt(MPI.Allreduce(local_sq_norm, +, comm))
    @. local_v /= global_norm

    dτ = 1e-1

    GC.gc()
    
    for step in 1:10
        t0 = time_ns()
        fill!(local_w, 0.0) 
        
        # 【极致复用主循环】
        for i in 1:length(sub_otfs)
            topo = sub_topos[i]
            
            if topo.send_dim == 0 && topo.recv_dim == 0
                @ccall LIB_DIST.compute_hvec_sub_chunk_f64(basis.ptr::Ptr{Cvoid}, sub_otfs[i].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, cache::Ptr{Float64}, local_w::Ptr{Float64})::Cvoid
                continue
            end
            
            @ccall LIB_DIST.pack_send_buffer_f64_sub(topo.ptr::Ptr{Cvoid}, cache::Ptr{Float64}, send_buffer::Ptr{Float64})::Cvoid
            
            recv_view = @view cache[local_dim + 1 : local_dim + topo.recv_dim]
            send_vbuf = MPI.VBuffer(send_buffer, topo.send_counts)
            recv_vbuf = MPI.VBuffer(recv_view, topo.recv_counts)
            MPI.Alltoallv!(send_vbuf, recv_vbuf, comm)
            
            @ccall LIB_DIST.compute_hvec_sub_chunk_f64(basis.ptr::Ptr{Cvoid}, sub_otfs[i].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, cache::Ptr{Float64}, local_w::Ptr{Float64})::Cvoid
        end
        
        @. local_v -= dτ * local_w
        
        local_sq_norm = sum(local_v .^ 2)
        global_norm = sqrt(MPI.Allreduce(local_sq_norm, +, comm))
        @. local_v /= global_norm

        t_hvec = (time_ns() - t0) / 1.0e9

        if rank == 0
            @printf("Step %2d | Global Norm: %.8f | MPI Time: %.4f s\n", step, global_norm, t_hvec)
        end
    end
    
    MPI.Finalize()
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_real_mpi_simulation(_name, _ratio, _basis)
end
