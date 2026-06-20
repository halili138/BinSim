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
