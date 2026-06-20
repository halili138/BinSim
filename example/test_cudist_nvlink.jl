# 禁用 OpenMP 抢占，因为在纯 GPU 模式下 CPU 只负责发号施令
ENV["OMP_NUM_THREADS"] = "1"
ENV["OMP_PROC_BIND"] = "false"
ENV["OMPI_MCA_btl"] = "^openib"
ENV["JULIA_CUDA_MEMORY_POOL"] = "none"

include("../jl/binsim.jl")
include("../jl/cuda_distributed.jl")
CUDADistributed.load!(@__MODULE__)

function test_multigpu_native(name, ratio, basis_name; num_phases::Int=2)
    # -------------------------------------------------------------
    # 1. MPI 初始化与 GPU 物理绑定
    # -------------------------------------------------------------
    MPI.Init()
    comm = MPI.COMM_WORLD
    mpi_rank = MPI.Comm_rank(comm)
    mpi_size = MPI.Comm_size(comm)

    # 【核心绑定】：1 个进程严格锁定 1 张显卡
    num_gpus = length(CUDA.devices())
    device_id = mpi_rank % num_gpus
    CUDA.device!(device_id)

    mole = Mole()
    mole.name = name
    mole.ratio = ratio
    mole.basis = basis_name
    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    cu_basis_dev = CuBasisManager(basis) # 物理字典上传至当前卡
    ham = JW_hamiltonian(mole)

    if mpi_rank == 0
        println("="^65)
        @printf("=== 启动形态 A: 纯多卡分布式计算 (NVLink 极速直通) ===\n")
        @printf("=== MPI 进程数: %d  |  使用物理显卡数: %d ===\n", mpi_size, num_gpus)
        println("="^65)
    end

    # -------------------------------------------------------------
    # 2. 控制面 (Control Plane): 在 CPU 计算全局与局部拓扑
    # -------------------------------------------------------------
    gmap = GlobalMemMap(basis, comm)
    local_dim = gmap.local_dim
    cpu_otfs, cu_otfs = build_distributed_cu_otfs(basis, ham)

    mpi_rank == 0 && println("Symm blocks块数: $(length(cpu_otfs)),  num_phases: $(num_phases)")

    sub_topos = Matrix{CuSubTopology}(undef, length(cpu_otfs), num_phases)
    for i in 1:length(cpu_otfs)
        for p in 1:num_phases
            sub_topos[i, p] = CuSubTopology(basis, cpu_otfs[i], gmap, num_phases=num_phases, phase_idx=p-1)
        end
    end
    
    max_send_dim = isempty(sub_topos) ? 0 : maximum(t.send_dim for t in sub_topos)
    max_recv_dim = isempty(sub_topos) ? 0 : maximum(t.recv_dim for t in sub_topos)

    if mpi_rank == 0
        @printf("拓扑建立完毕! VRAM 峰值接收缓存 (max_recv_dim): %.4f GB\n", max_recv_dim * 8 / (1 << 30))
    end

    d_cache         = CUDA.zeros(Float64, local_dim + max_recv_dim)
    d_local_w       = CUDA.zeros(Float64, local_dim)
    d_send_buffer   = CUDA.zeros(Float64, max_send_dim)
    d_recv_buffer   = CUDA.zeros(Float64, max_recv_dim)

    if mpi_rank == 0
        total_dim = length(d_cache) + length(d_local_w) + length(d_send_buffer) + length(d_recv_buffer)
        @printf("单卡显存峰值: %.4f GB\n", total_dim * 8 / (1 << 30))
    end

    d_local_v = @view d_cache[1:local_dim]

    # 初态定向注入显存
    set_local_hf_gpu!(gmap, basis, d_local_v, mole.nelec)

    # 显存内局部求和，跨 GPU 全局归一化
    local_sq_norm = sum(d_local_v .^ 2)
    global_norm = sqrt(MPI.Allreduce(local_sq_norm, +, comm))
    @. d_local_v /= global_norm

    dτ = 1e-1

    for step in 1:10
        t0 = time_ns()
        d_local_w .= 0.0
        for i in 1:length(cu_otfs)
            for p in 1:num_phases
                topo = sub_topos[i, p]
                
                if topo.send_dim == 0 && topo.recv_dim == 0
                    @ccall LIB_CUDIST.compute_hvec_sub_chunk_gpu_f64(
                        cu_basis_dev.ptr::Ptr{Cvoid}, cu_otfs[i].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, 
                        pointer(d_cache)::CuPtr{Float64}, pointer(d_local_w)::CuPtr{Float64}
                    )::Cvoid
                    continue
                end
                
                @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(
                    topo.ptr::Ptr{Cvoid}, pointer(d_cache)::CuPtr{Float64}, pointer(d_send_buffer)::CuPtr{Float64}
                )::Cvoid
                
                send_vbuf = MPI.VBuffer(d_send_buffer, topo.send_counts)
                recv_vbuf = MPI.VBuffer(d_recv_buffer, topo.recv_counts)
                MPI.Alltoallv!(send_vbuf, recv_vbuf, comm)
                
                if topo.recv_dim > 0
                    d_recv_view = @view d_cache[local_dim + 1 : local_dim + topo.recv_dim]
                    copyto!(d_recv_view, 1, d_recv_buffer, 1, topo.recv_dim)
                end
                
                @ccall LIB_CUDIST.compute_hvec_sub_chunk_gpu_f64(
                    cu_basis_dev.ptr::Ptr{Cvoid}, cu_otfs[i].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, 
                    pointer(d_cache)::CuPtr{Float64}, pointer(d_local_w)::CuPtr{Float64}
                )::Cvoid
            end
        end

        # 显存内原地演化
        @. d_local_v -= dτ * d_local_w

        local_sq_norm = sum(d_local_v .^ 2)
        global_norm = sqrt(MPI.Allreduce(local_sq_norm, +, comm))
        @. d_local_v /= global_norm

        CUDA.synchronize() # 确保计时精准
        t_hvec = (time_ns() - t0) / 1.0e9

        if mpi_rank == 0
            @printf("Step %2d | Global Norm: %.8f | Multi-GPU NVLink Time: %.4f s\n", step, global_norm, t_hvec)
        end
    end

    MPI.Finalize()
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_multigpu_native(ARGS[1], 1.0, ARGS[2], num_phases=parse(Int, ARGS[3]))
end
