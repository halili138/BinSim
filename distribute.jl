# ══════════════════════════════════════════════════════════════════════
# Distributed OTF (MPI-based)
# ══════════════════════════════════════════════════════════════════════
using MPI

const LIB_OTF_DIST = joinpath(@__DIR__, "src/lib/libdist.so")

mutable struct DistributedBasisManager
    ptr::Ptr{Cvoid}
    local_dim::Int64
    my_rank::Int64
    num_ranks::Int64

    function DistributedBasisManager(comm::MPI.Comm, basis::BasisManager)
        ptr = @ccall LIB_OTF_DIST.create_distributed_basis_f64(
            comm.val::Int32, basis.ptr::Ptr{Cvoid}, basis.norb::Int64,
        )::Ptr{Cvoid}
        ptr == C_NULL && error("Failed to create C++ DistributedBasisManager.")
        obj = new(ptr,
            @ccall(LIB_OTF_DIST.distributed_basis_local_dim_f64(ptr::Ptr{Cvoid})::Int64),
            @ccall(LIB_OTF_DIST.distributed_basis_my_rank_f64(ptr::Ptr{Cvoid})::Int64),
            @ccall(LIB_OTF_DIST.distributed_basis_num_ranks_f64(ptr::Ptr{Cvoid})::Int64))
        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_OTF_DIST.destroy_distributed_basis_f64(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end
        return obj
    end

    function DistributedBasisManager(comm::MPI.Comm, basis::BasisManager, ::Type{ComplexF64})
        ptr = @ccall LIB_OTF_DIST.create_distributed_basis_c64(
            comm.val::Int32, basis.ptr::Ptr{Cvoid}, basis.norb::Int64,
        )::Ptr{Cvoid}
        ptr == C_NULL && error("Failed to create C++ DistributedBasisManager (c64).")
        obj = new(ptr,
            @ccall(LIB_OTF_DIST.distributed_basis_local_dim_c64(ptr::Ptr{Cvoid})::Int64),
            @ccall(LIB_OTF_DIST.distributed_basis_my_rank_c64(ptr::Ptr{Cvoid})::Int64),
            @ccall(LIB_OTF_DIST.distributed_basis_num_ranks_c64(ptr::Ptr{Cvoid})::Int64))
        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_OTF_DIST.destroy_distributed_basis_c64(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct DistributedOTF
    ptr::Ptr{Cvoid}
    ngs::Int64

    function DistributedOTF(otf::OTF, orbsym::Vector{Int64})
        ptr = @ccall LIB_OTF_DIST.build_distributed_net_f64(
            otf.ptr::Ptr{Cvoid}, orbsym::Ptr{Int64})::Ptr{Cvoid}
        ptr == C_NULL && error("Failed to create C++ DistributedNetwork_OTF.")
        obj = new(ptr,
            @ccall(LIB_OTF_DIST.distributed_net_num_groups_f64(ptr::Ptr{Cvoid})::Int64))
        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_OTF_DIST.destroy_distributed_net_f64(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end
        return obj
    end

    function DistributedOTF(otf::OTF, orbsym::Vector{Int64}, ::Type{ComplexF64})
        ptr = @ccall LIB_OTF_DIST.build_distributed_net_c64(
            otf.ptr::Ptr{Cvoid}, orbsym::Ptr{Int64})::Ptr{Cvoid}
        ptr == C_NULL && error("Failed to create C++ DistributedNetwork_OTF (c64).")
        obj = new(ptr,
            @ccall(LIB_OTF_DIST.distributed_net_num_groups_c64(ptr::Ptr{Cvoid})::Int64))
        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_OTF_DIST.destroy_distributed_net_c64(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function hvec_otf_distributed!(dbasis::DistributedBasisManager, dnet::DistributedOTF,
                                src::T, dst::T) where {T<:AbstractArray{Float64,1}}
    @ccall LIB_OTF_DIST.hvec_gather_contract_otf_distributed_f64(
        dbasis.ptr::Ptr{Cvoid},
        dnet.ptr::Ptr{Cvoid},
        src::Ptr{Cdouble},
        dst::Ptr{Cdouble},
    )::Cvoid
end

function hvec_otf_distributed!(dbasis::DistributedBasisManager, dnet::DistributedOTF,
                                src::T, dst::T) where {T<:AbstractArray{ComplexF64,1}}
    @ccall LIB_OTF_DIST.hvec_gather_contract_otf_distributed_c64(
        dbasis.ptr::Ptr{Cvoid},
        dnet.ptr::Ptr{Cvoid},
        src::Ptr{ComplexF64},
        dst::Ptr{ComplexF64},
    )::Cvoid
end

function compute_local_diags!(
    dbasis::DistributedBasisManager,
    azs::Vector{UInt32}, bzs::Vector{UInt32},
    cs::Vector{Float64}, out::Vector{Float64})
    @ccall LIB_OTF_DIST.distributed_compute_local_diags_f64(
        dbasis.ptr::Ptr{Cvoid},
        azs::Ptr{UInt32}, bzs::Ptr{UInt32},
        cs::Ptr{Cdouble}, length(cs)::Int64,
        out::Ptr{Cdouble},
    )::Cvoid
end

function compute_local_diags!(
    dbasis::DistributedBasisManager,
    azs::Vector{UInt32}, bzs::Vector{UInt32},
    cs::Vector{ComplexF64}, out::Vector{ComplexF64})
    @ccall LIB_OTF_DIST.distributed_compute_local_diags_c64(
        dbasis.ptr::Ptr{Cvoid},
        azs::Ptr{UInt32}, bzs::Ptr{UInt32},
        cs::Ptr{ComplexF64}, length(cs)::Int64,
        out::Ptr{ComplexF64},
    )::Cvoid
end

function extract_local_vec!(
    dbasis::DistributedBasisManager,
    global_vec::Vector{Float64}, local_vec::Vector{Float64})
    @ccall LIB_OTF_DIST.distributed_extract_local_vec_f64(
        dbasis.ptr::Ptr{Cvoid},
        global_vec::Ptr{Cdouble},
        local_vec::Ptr{Cdouble},
    )::Cvoid
end

function extract_local_vec!(
    dbasis::DistributedBasisManager,
    global_vec::Vector{ComplexF64}, local_vec::Vector{ComplexF64})
    @ccall LIB_OTF_DIST.distributed_extract_local_vec_c64(
        dbasis.ptr::Ptr{Cvoid},
        global_vec::Ptr{ComplexF64},
        local_vec::Ptr{ComplexF64},
    )::Cvoid
end

