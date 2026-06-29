mutable struct CuSubTopology
    ptr::Ptr{Cvoid}
    send_dim::Int64
    recv_dim::Int64
    send_counts::Vector{Cint}
    recv_counts::Vector{Cint}
end

function CuSubTopology(basis::BasisManager, subnet::OTF, gmap::GlobalMemMap; num_phases::Int=1, phase_idx::Int=0)
    ptr = @ccall LIB_CUDIST.build_sub_topology_gpu_f64(
        basis.ptr::Ptr{Cvoid}, subnet.ptr::Ptr{Cvoid}, gmap.ptr::Ptr{Cvoid}, num_phases::Cint, phase_idx::Cint,
    )::Ptr{Cvoid}

    dims = zeros(Int64, 2)
    @ccall LIB_CUDIST.get_sub_topology_info_gpu(ptr::Ptr{Cvoid}, pointer(dims, 1)::Ptr{Int64}, pointer(dims, 2)::Ptr{Int64})::Cvoid
    
    send_counts = zeros(Cint, gmap.mpi_size)
    recv_counts = zeros(Cint, gmap.mpi_size)
    @ccall LIB_CUDIST.get_sub_topology_mpi_counts_gpu(ptr::Ptr{Cvoid}, send_counts::Ptr{Cint}, recv_counts::Ptr{Cint})::Cvoid

    obj = CuSubTopology(ptr, dims[1], dims[2], send_counts, recv_counts)
    finalizer(obj) do o
        o.ptr != C_NULL && @ccall LIB_CUDIST.destroy_sub_topology_gpu(o.ptr::Ptr{Cvoid})::Cvoid
    end
    return obj
end

function build_distributed_cu_otfs(basis::BasisManager, A::BinaryQubitAABB{Ti,Tv,K,V}, tol::Float64=1e-12) where {Ti,Tv,K,V}
    groups = compress_by_svd(A, tol) 
    
    dict = Dict{Tuple{Int,Int}, Vector{SVDGroup{Ti,Tv}}}()
    for g in groups
        asym = get_symm(g.ax, basis.orbsym) 
        bsym = get_symm(g.bx, basis.orbsym)
        sym_key = (asym, bsym)
        if !haskey(dict, sym_key)
            dict[sym_key] = SVDGroup{Ti,Tv}[]
        end
        push!(dict[sym_key], g)
    end
    
    cu_otfs = CuOTF[]
    cpu_otfs = OTF[] 
    for sub_groups in values(dict)
        cpu_otf = OTF_from_groups(basis, sub_groups)
        push!(cpu_otfs, cpu_otf)
        push!(cu_otfs, CuOTF(cpu_otf)) 
    end
    return cpu_otfs, cu_otfs
end

function set_local_hf_gpu!(gmap::GlobalMemMap, basis::BasisManager, d_local_v::CuArray{Float64,1}, nelec::Tuple{Int,Int})
    na, nb = nelec
    hf_astr = UInt32(0); for i in 0:na-1; hf_astr |= (UInt32(1) << i); end
    hf_bstr = UInt32(0); for i in 0:nb-1; hf_bstr |= (UInt32(1) << i); end

    d_local_v .= 0.0

    @ccall LIB_CUDIST.set_local_det_coeff_gpu_f64(
        gmap.ptr::Ptr{Cvoid}, basis.ptr::Ptr{Cvoid},
        hf_astr::UInt32, hf_bstr::UInt32, 1.0::Cdouble, pointer(d_local_v)::CuPtr{Float64}
    )::Cvoid
end

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

    expm_2d::Function
    backgrad_2d::Function
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
