ENV["OMP_NUM_THREADS"] = "1"
ENV["OMP_PROC_BIND"] = "false"

include("../jl/binsim.jl")
include("../jl/cuda_distributed.jl")
CUDADistributed.load!(@__MODULE__)

# =====================================================================
# 2. 纯 CPU 内存路由器 (完美平替底层的 MPI.Alltoallv!)
# =====================================================================
function cpu_memory_router!(num_chunks, topos_for_otf, host_send_bufs, host_recv_bufs)
    # 辅助函数：计算准确的内存偏移量
    get_offset = (counts, target) -> 1 + sum(counts[1:target-1]; init=0)

    for r_recv in 1:num_chunks
        recv_offset = 1
        for r_send in 1:num_chunks
            send_count = topos_for_otf[r_send].send_counts[r_recv]
            recv_count = topos_for_otf[r_recv].recv_counts[r_send]
            
            if send_count > 0
                send_offset = get_offset(topos_for_otf[r_send].send_counts, r_recv)
                # 极致的 CPU 连续内存拷贝
                copyto!(host_recv_bufs[r_recv], recv_offset, 
                        host_send_bufs[r_send], send_offset, send_count)
            end
            recv_offset += recv_count
        end
    end
end

# =====================================================================
# 3. 终极单卡 OOC 引擎
# =====================================================================
function test_single_gpu_ooc(name, ratio, basis_name, num_chunks::Int=4)
    CUDA.device!(0) # 绑定物理显卡

    mole = Mole()
    mole.name  = name; mole.ratio = ratio; mole.basis = basis_name
    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    cu_basis_dev = CuBasisManager(basis)
    ham = JW_hamiltonian(mole)
    
    println("="^65)
    @printf("=== 启动纯原生 Out-of-Core 演化 (单卡突破显存上限) ===\n")
    @printf("=== 虚拟切片数 (Chunks): %d ===\n", num_chunks)
    println("="^65)

    # A. 生成 N 个虚拟的全局拓扑映射
    gmaps = [GlobalMemMap(basis, rank = r-1, size = num_chunks) for r in 1:num_chunks]
    
    # B. 推送物理算子到 GPU
    cpu_otfs, cu_otfs = build_distributed_cu_otfs(basis, ham)
    
    # C. 为每个 Chunk 构建对应的 C++ 路由表 (Matrix: [Chunk, OTF])
    sub_topos = Matrix{CuSubTopology}(undef, num_chunks, length(cpu_otfs))
    for r in 1:num_chunks, i in 1:length(cpu_otfs)
        sub_topos[r, i] = CuSubTopology(basis, cpu_otfs[i], gmaps[r])
    end

    # -------------------------------------------------------------
    # 主板内存分配 (Host RAM - 廉价且管够)
    # -------------------------------------------------------------
    host_v_chunks  = [zeros(Float64, gmaps[r].local_dim) for r in 1:num_chunks]
    host_w_chunks  = [zeros(Float64, gmaps[r].local_dim) for r in 1:num_chunks]
    
    max_send_dims  = [maximum(t.send_dim for t in sub_topos[r, :]) for r in 1:num_chunks]
    max_recv_dims  = [maximum(t.recv_dim for t in sub_topos[r, :]) for r in 1:num_chunks]
    host_send_bufs = [zeros(Float64, max_send_dims[r]) for r in 1:num_chunks]
    host_recv_bufs = [zeros(Float64, max_recv_dims[r]) for r in 1:num_chunks]

    # -------------------------------------------------------------
    # 极速 VRAM 缓存池 (GPU 显存 - 整个程序唯一一次 cudaMalloc!)
    # -------------------------------------------------------------
    global_max_local = maximum(g.local_dim for g in gmaps)
    global_max_recv  = maximum(max_recv_dims)
    global_max_send  = maximum(max_send_dims)
    
    @printf("全量波函数大小: %.4f GB\n", basis.dim * 8 / (1024^3))
    @printf("GPU 显存驻留峰值被强行压缩至: %.4f GB\n\n", (global_max_local + max(global_max_recv, global_max_send)) * 8 / (1024^3))

    d_cache = CUDA.zeros(Float64, global_max_local + global_max_recv)
    d_send  = CUDA.zeros(Float64, global_max_send)
    d_w     = CUDA.zeros(Float64, global_max_local)

    # -------------------------------------------------------------
    # 初态定向注入
    # -------------------------------------------------------------
    for r in 1:num_chunks
        d_local_v_view = @view d_cache[1:gmaps[r].local_dim]
        set_local_hf_gpu!(gmaps[r], basis, d_local_v_view, mole.nelec)
        copyto!(host_v_chunks[r], d_local_v_view) # 从 GPU 拿回 CPU
    end

    # 全局归一化 (纯 CPU)
    global_norm = sqrt(sum(sum(v .^ 2) for v in host_v_chunks))
    for v in host_v_chunks; v ./= global_norm; end

    dτ = 1e-1

    for step in 1:10
        t0 = time_ns()
        for w in host_w_chunks; fill!(w, 0.0); end
        
        for i in 1:length(cu_otfs)
            # ==========================================
            # 阶段 1：GPU 打包 (OOC: CPU -> GPU -> CPU)
            # ==========================================
            for r in 1:num_chunks
                topo = sub_topos[r, i]
                if topo.send_dim > 0
                    # 将本次需要打包的数据推入 GPU
                    copyto!(d_cache, 1, host_v_chunks[r], 1, gmaps[r].local_dim)
                    
                    @ccall LIB_CUDIST.pack_send_buffer_gpu_f64(
                        topo.ptr::Ptr{Cvoid}, pointer(d_cache)::CuPtr{Float64}, pointer(d_send)::CuPtr{Float64}
                    )::Cvoid
                    
                    # 打包完毕，拉回 CPU 发送区
                    copyto!(host_send_bufs[r], 1, d_send, 1, topo.send_dim)
                end
            end
            
            # ==========================================
            # 阶段 2：CPU 极速内存路由 (平替 MPI)
            # ==========================================
            cpu_memory_router!(num_chunks, sub_topos[:, i], host_send_bufs, host_recv_bufs)

            # ==========================================
            # 阶段 3：GPU 算力狂飙 (OOC: CPU -> GPU -> CPU)
            # ==========================================
            for r in 1:num_chunks
                topo = sub_topos[r, i]
                ldim = gmaps[r].local_dim
                
                # 拼接完美视口：将 Local 和 Recv 按顺序拼接到 GPU Cache 池
                copyto!(d_cache, 1, host_v_chunks[r], 1, ldim)
                if topo.recv_dim > 0
                    copyto!(d_cache, ldim + 1, host_recv_bufs[r], 1, topo.recv_dim)
                end
                
                d_w .= 0.0 # 极速清零
                
                @ccall LIB_CUDIST.compute_hvec_sub_chunk_gpu_f64(
                    cu_basis_dev.ptr::Ptr{Cvoid}, cu_otfs[i].ptr::Ptr{Cvoid}, topo.ptr::Ptr{Cvoid}, 
                    pointer(d_cache)::CuPtr{Float64}, pointer(d_w)::CuPtr{Float64}
                )::Cvoid
                
                # 计算完毕，将导数拉回 CPU 主板内存累加
                # (注意：使用临时数组接管，防止隐式分配)
                temp_w_host = Array(d_w[1:ldim]) 
                host_w_chunks[r] .+= temp_w_host
            end
        end
        
        # -------------------------------------------------------------
        # 波函数演化与归一化 (纯 CPU)
        # -------------------------------------------------------------
        for r in 1:num_chunks
            host_v_chunks[r] .-= dτ .* host_w_chunks[r]
        end
        
        global_norm = sqrt(sum(sum(v .^ 2) for v in host_v_chunks))
        for v in host_v_chunks; v ./= global_norm; end

        CUDA.synchronize() 
        t_hvec = (time_ns() - t0) / 1.0e9

        @printf("Step %2d | Global Norm: %.8f | OOC Time: %.4f s\n", step, global_norm, t_hvec)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # 可以随意调节 num_chunks! 如果分子太大，把 num_chunks 设成 16 甚至 32！
    test_single_gpu_ooc("n2", 1.0, "6-31g", 4)
end
