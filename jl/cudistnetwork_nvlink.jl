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
    d_left_cache_ref = Ref{Union{Nothing, CuVector{Float64}}}(nothing)
    d_right_cache_ref = Ref{Union{Nothing, CuVector{Float64}}}(nothing)

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

    function backgrad_caches!()
        if d_left_cache_ref[] === nothing
            d_left_cache_ref[] = CUDA.zeros(Float64, cache_scalars)
        end
        if d_right_cache_ref[] === nothing
            d_right_cache_ref[] = CUDA.zeros(Float64, cache_scalars)
        end
        return d_left_cache_ref[]::CuVector{Float64}, d_right_cache_ref[]::CuVector{Float64}
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
        d_left_cache, d_right_cache = backgrad_caches!()
        left_local = @view d_left_cache[1:local_dim]
        right_local = @view d_right_cache[1:local_dim]
        copyto!(left_local, lv)
        copyto!(right_local, rv)
        local_grad = 0.0
        for j in 1:length(pool_cu_otfs[idx]), p in 1:num_phases
            topo = pool_sub_topos[idx][j, p]
            exchange_ghosts!(topo, d_left_cache, d_send, d_recv)
            # exchange_ghosts! is blocking and copies received ghosts into the
            # target cache before returning, so the same send/recv workspace can
            # be reused for the right-vector exchange.
            exchange_ghosts!(topo, d_right_cache, d_send, d_recv)
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
