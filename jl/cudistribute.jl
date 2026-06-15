include("cunetwork.jl")

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
