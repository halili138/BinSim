const LIB_BASIS = joinpath(@__DIR__, "src/lib/libbasis.so")
const LIB_OTF = joinpath(@__DIR__, "src/lib/libotf.so")

mutable struct BasisManager
    ptr::Ptr{Cvoid}
    dim::Int64
    norb::Int64
    nelec::Tuple{Int64,Int64}
    orbsym::Vector{Int64}

    function BasisManager(norb::Int64, nelec::Tuple{Int64,Int64}, orbsym::Vector{Int64})
        total_sym = 0
        num_irreps = 16
        na, nb = nelec

        ptr = @ccall LIB_BASIS.create_basis_manager(
            norb::Int64, na::Int64, nb::Int64, total_sym::Int64, orbsym::Ptr{Int64}, num_irreps::Int64,
        )::Ptr{Cvoid}

        ptr == C_NULL && error("Failed to create C++ BasisManager.")

        dim = @ccall LIB_BASIS.get_subspace_dim(ptr::Ptr{Cvoid})::Int64
        @printf("Num symmetry allowed elements: %d    %.4f GB\n\n", dim, dim * 8 / (1 << 30))

        obj = new(ptr, dim, norb, nelec, orbsym)

        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_BASIS.destroy_basis_manager(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end

        return obj
    end

    function BasisManager(norb::Int64, astrs::Vector{UInt32}, bstrs::Vector{UInt32}, orbsym::Vector{Int64})
        total_sym = 0
        num_irreps = 16
        num_astrs = length(astrs)
        num_bstrs = length(bstrs)

        ptr = @ccall LIB_BASIS.create_custom_basis_manager(
            norb::Int64,
            astrs::Ptr{UInt32}, num_astrs::Int64,
            bstrs::Ptr{UInt32}, num_bstrs::Int64,
            orbsym::Ptr{Int64}, total_sym::Int64, num_irreps::Int64,
        )::Ptr{Cvoid}

        ptr == C_NULL && error("Failed to create C++ BasisManager.")

        dim = @ccall LIB_BASIS.get_subspace_dim(ptr::Ptr{Cvoid})::Int64
        @printf("Num symmetry allowed elements: %d    %.4f GB\n\n", dim, dim * 8 / (1 << 30))

        obj = new(ptr, dim, norb, (0, 0), orbsym)

        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_BASIS.destroy_basis_manager(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end

        return obj
    end
end

function get_hf(basis::BasisManager, nelec::Tuple{Int,Int}; Tv::DataType=Float64)
    hf = zeros(Tv, basis.dim)
    na, nb = nelec

    hf_astr = UInt32(0)
    for i in 0:na-1
        hf_astr |= (UInt32(1) << i)
    end

    hf_bstr = UInt32(0)
    for i in 0:nb-1
        hf_bstr |= (UInt32(1) << i)
    end

    hf_val = Tv(1.0)

    if Tv <: Complex
        @ccall LIB_BASIS.set_det_coeff_c64(
            basis.ptr::Ptr{Cvoid},
            hf_astr::UInt32,
            hf_bstr::UInt32,
            hf_val::Cdouble,
            hf::Ptr{ComplexF64},
        )::Cvoid
    else
        @ccall LIB_BASIS.set_det_coeff_f64(
            basis.ptr::Ptr{Cvoid},
            hf_astr::UInt32,
            hf_bstr::UInt32,
            hf_val::Cdouble,
            hf::Ptr{Cdouble},
        )::Cvoid
    end

    return hf
end

function get_reference_state(basis::BasisManager, astrs::Vector{UInt32}, bstrs::Vector{UInt32}, vals::Vector{Tv}) where Tv
    @assert length(astrs) == length(bstrs) == length(vals)
    v0 = zeros(Tv, basis.dim)

    if Tv <: Complex
        for (astr::UInt32, bstr::UInt32, val::Tv) in zip(astrs, bstrs, vals)
            @ccall LIB_BASIS.set_det_coeff_c64(
                basis.ptr::Ptr{Cvoid},
                astr::UInt32,
                bstr::UInt32,
                val::Cdouble,
                hf::Ptr{ComplexF64},
            )::Cvoid
        end
    else
        for (astr::UInt32, bstr::UInt32, val::Tv) in zip(astrs, bstrs, vals)
            @ccall LIB_BASIS.set_det_coeff_f64(
                basis.ptr::Ptr{Cvoid},
                astr::UInt32,
                bstr::UInt32,
                val::Cdouble,
                hf::Ptr{Cdouble},
            )::Cvoid
        end
    end

    return v0
end

struct SVDGroup{Ti,Tv}
    ax::Ti
    bx::Ti
    rank::Int
    azs::Vector{Ti}
    bzs::Vector{Ti}
    wa::Matrix{Tv}
    wb::Matrix{Tv}
    ncs::Int
end

function compress_by_svd(A::BinaryQubitAABB{Ti,Tv,K,V}, tol::Float64=1e-12) where {Ti,Tv,K,V}
    gs = get_bounds_0based(A.axs, A.bxs)
    ngs = length(gs) - 1

    svd_groups = Vector{SVDGroup{Ti,Tv}}(undef, ngs)

    for g in 1:ngs
        lb = gs[g] + 1
        rb = gs[g+1]

        ax = A.axs[lb]
        bx = A.bxs[lb]

        sub_azs = A.azs[lb:rb]
        sub_bzs = A.bzs[lb:rb]
        sub_cs = A.cs[lb:rb]

        unique_azs = unique(sub_azs)
        unique_bzs = unique(sub_bzs)

        sort!(unique_azs)
        sort!(unique_bzs)

        Na = length(unique_azs)
        Nb = length(unique_bzs)

        az_map = Dict(z => i for (i, z) in enumerate(unique_azs))
        bz_map = Dict(z => i for (i, z) in enumerate(unique_bzs))

        M = zeros(Tv, Na, Nb)
        for (k, c) in enumerate(sub_cs)
            i = az_map[sub_azs[k]]
            j = bz_map[sub_bzs[k]]
            M[i, j] += c
        end

        F = svd(M)

        valid_idx = findall(x -> abs(x) > tol, F.S)
        rank = length(valid_idx)

        wa = zeros(Tv, Na, rank)
        wb = zeros(Tv, Nb, rank)

        for (ri, r) in enumerate(valid_idx)
            sqrt_s = sqrt(F.S[r])
            U_col = copy(F.U[:, r])
            Vt_row = copy(F.Vt[r, :])

            # ==== 消除复数域下的 SVD 寄生全局相位 ====
            # 这一步是为了配合 C++ 端反向边构建时不对 shared phases 进行深拷贝
            if Tv <: Complex
                if ax != 0 && bx == 0
                    # Pure A: B 弦算符是对角的 (Vt_row 必须严格为实数)
                    max_idx = argmax(abs.(Vt_row))
                    phase_angle = angle(Vt_row[max_idx])

                    U_col .*= exp(im * phase_angle)
                    Vt_row .*= exp(-im * phase_angle)

                    # 消除机器精度误差带来的微小虚部，并保持数据类型为 Tv (ComplexF64)
                    Vt_row = Tv.(real.(Vt_row))

                elseif ax == 0 && bx != 0
                    # Pure B: A 弦算符是对角的 (U_col 必须严格为实数)
                    max_idx = argmax(abs.(U_col))
                    phase_angle = angle(U_col[max_idx])

                    U_col .*= exp(-im * phase_angle)
                    Vt_row .*= exp(im * phase_angle)

                    # 消除机器精度误差带来的微小虚部，并保持数据类型为 Tv (ComplexF64)
                    U_col = Tv.(real.(U_col))
                end
            end
            # ==========================================

            @. wa[:, ri] = U_col * sqrt_s
            @. wb[:, ri] = Vt_row * sqrt_s
        end

        wa[abs.(wa).<1e-12] .= 0
        wb[abs.(wb).<1e-12] .= 0

        svd_groups[g] = SVDGroup{Ti,Tv}(ax, bx, rank, unique_azs, unique_bzs, wa, wb, length(sub_cs))
    end

    return svd_groups
end

function compress_by_svd(pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}, tol::Float64=1e-12) where {Ti,Tv,K,V}
    ngs = length(pool)

    svd_groups = Vector{SVDGroup{Ti,Tv}}(undef, ngs)

    for (idx, op) in enumerate(pool)
        gs = get_bounds_0based(op.axs, op.bxs)
        @assert (length(gs) - 1) <= 1 "only support excitation operator with ngs <= 1"
        ax = op.axs[end]
        bx = op.bxs[end]
        azs = op.azs
        bzs = op.bzs
        cs = op.cs

        unique_azs = unique(azs)
        unique_bzs = unique(bzs)
        sort!(unique_azs)
        sort!(unique_bzs)

        Na = length(unique_azs)
        Nb = length(unique_bzs)

        az_map = Dict(z => i for (i, z) in enumerate(unique_azs))
        bz_map = Dict(z => i for (i, z) in enumerate(unique_bzs))

        M = zeros(Tv, Na, Nb)
        for (k, c) in enumerate(cs)
            i = az_map[azs[k]]
            j = bz_map[bzs[k]]
            M[i, j] += c
        end

        F = svd(M)

        valid_idx = findall(x -> abs(x) > tol, F.S)
        rank = length(valid_idx)

        wa = zeros(Tv, Na, rank)
        wb = zeros(Tv, Nb, rank)

        for (ri, r) in enumerate(valid_idx)
            sqrt_s = sqrt(F.S[r])
            @. wa[:, ri] = F.U[:, r] * sqrt_s
            @. wb[:, ri] = F.Vt[r, :] * sqrt_s
        end

        wa[abs.(wa).<1e-12] .= 0
        wb[abs.(wb).<1e-12] .= 0

        svd_groups[idx] = SVDGroup{Ti,Tv}(ax, bx, rank, unique_azs, unique_bzs, wa, wb, length(cs))
    end

    return svd_groups
end

mutable struct OTF
    ptr::Ptr{Cvoid}
    dim::Int64
    ngs::Int64

    function OTF(basis::BasisManager, A::BinaryQubitAABB{Ti,Tv,K,V}, tol::Float64=1e-12) where {Ti,Tv,K,V}
        groups = compress_by_svd(A, tol)
        ngs = length(groups)
        axs = Vector{Ti}(undef, ngs)
        bxs = Vector{Ti}(undef, ngs)
        ranks = Vector{Int64}(undef, ngs)
        num_as = Vector{Int64}(undef, ngs)
        num_bs = Vector{Int64}(undef, ngs)

        flat_azs = Ti[]
        flat_bzs = Ti[]
        flat_wa = Tv[]
        flat_wb = Tv[]

        for (g, group) in enumerate(groups)
            axs[g] = group.ax
            bxs[g] = group.bx
            ranks[g] = group.rank

            na = length(group.azs)
            nb = length(group.bzs)
            num_as[g] = na
            num_bs[g] = nb

            append!(flat_azs, group.azs)
            append!(flat_bzs, group.bzs)

            append!(flat_wa, vec(group.wa))
            append!(flat_wb, vec(group.wb))
        end

        if Tv <: Complex
            ptr = @ccall LIB_OTF.build_network_otf_c64(
                basis.ptr::Ptr{Cvoid}, basis.norb::Int64, ngs::Int64,
                axs::Ptr{Ti}, bxs::Ptr{Ti}, ranks::Ptr{Int64}, num_as::Ptr{Int64}, num_bs::Ptr{Int64},
                flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti}, flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv},
            )::Ptr{Cvoid}
        else
            ptr = @ccall LIB_OTF.build_network_otf_f64(
                basis.ptr::Ptr{Cvoid}, basis.norb::Int64, ngs::Int64,
                axs::Ptr{Ti}, bxs::Ptr{Ti}, ranks::Ptr{Int64}, num_as::Ptr{Int64}, num_bs::Ptr{Int64},
                flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti}, flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv},
            )::Ptr{Cvoid}
        end

        ptr == C_NULL && error("Failed to create C++ OTFNET.")

        obj = new(ptr, basis.dim, ngs)

        if Tv <: Complex
            finalizer(obj) do o
                if o.ptr != C_NULL
                    @ccall LIB_OTF.destroy_network_otf_c64(o.ptr::Ptr{Cvoid})::Cvoid
                    o.ptr = C_NULL
                end
            end
        else
            finalizer(obj) do o
                if o.ptr != C_NULL
                    @ccall LIB_OTF.destroy_network_otf_f64(o.ptr::Ptr{Cvoid})::Cvoid
                    o.ptr = C_NULL
                end
            end
        end

        return obj
    end

    function OTF(
        basis::BasisManager,
        pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
        tol::Float64=1e-12,
    ) where {Ti,Tv,K,V}

        groups = compress_by_svd(pool, tol)
        ngs = length(groups)
        axs = Vector{Ti}(undef, ngs)
        bxs = Vector{Ti}(undef, ngs)

        ranks = Vector{Int64}(undef, ngs)
        num_as = Vector{Int64}(undef, ngs)
        num_bs = Vector{Int64}(undef, ngs)

        flat_azs = Ti[]
        flat_bzs = Ti[]
        flat_wa = Tv[]
        flat_wb = Tv[]

        for (g, group) in enumerate(groups)
            axs[g] = group.ax
            bxs[g] = group.bx
            ranks[g] = group.rank

            na = length(group.azs)
            nb = length(group.bzs)
            num_as[g] = na
            num_bs[g] = nb

            append!(flat_azs, group.azs)
            append!(flat_bzs, group.bzs)

            append!(flat_wa, vec(group.wa))
            append!(flat_wb, vec(group.wb))
        end

        if Tv <: Complex
            ptr = @ccall LIB_OTF.build_network_otf_c64(
                basis.ptr::Ptr{Cvoid}, basis.norb::Int64, ngs::Int64,
                axs::Ptr{Ti}, bxs::Ptr{Ti}, ranks::Ptr{Int64}, num_as::Ptr{Int64}, num_bs::Ptr{Int64},
                flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti}, flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv},
            )::Ptr{Cvoid}
        else
            ptr = @ccall LIB_OTF.build_network_otf_f64(
                basis.ptr::Ptr{Cvoid}, basis.norb::Int64, ngs::Int64,
                axs::Ptr{Ti}, bxs::Ptr{Ti}, ranks::Ptr{Int64}, num_as::Ptr{Int64}, num_bs::Ptr{Int64},
                flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti}, flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv},
            )::Ptr{Cvoid}
        end

        ptr == C_NULL && error("Failed to create C++ OTFNET.")

        obj = new(ptr, basis.dim, ngs)

        if Tv <: Complex
            finalizer(obj) do o
                if o.ptr != C_NULL
                    @ccall LIB_OTF.destroy_network_otf_c64(o.ptr::Ptr{Cvoid})::Cvoid
                    o.ptr = C_NULL
                end
            end
        else
            finalizer(obj) do o
                if o.ptr != C_NULL
                    @ccall LIB_OTF.destroy_network_otf_f64(o.ptr::Ptr{Cvoid})::Cvoid
                    o.ptr = C_NULL
                end
            end
        end

        return obj
    end
end

function get_diags(basis::BasisManager, otf::OTF, Tv::DataType)
    diags = zeros(Tv, basis.dim)

    if Tv <: Complex
        @ccall LIB_OTF.get_diags_elements_c64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{ComplexF64},
        )::Cvoid
    else
        @ccall LIB_OTF.get_diags_elements_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Cdouble},
        )::Cvoid
    end

    return diags
end

function hvec_otf!(basis::BasisManager, otf::OTF, src::T, dst::T) where {Tv,T<:AbstractArray{Tv,1}}
    Tv <: Complex ? (
        @ccall LIB_OTF.hvec_gather_contract_otf_c64(
        basis.ptr::Ptr{Cvoid},
        otf.ptr::Ptr{Cvoid},
        src::Ptr{Tv},
        dst::Ptr{Tv},
    )::Cvoid
    ) : (
        @ccall LIB_OTF.hvec_gather_contract_otf_f64(
        basis.ptr::Ptr{Cvoid},
        otf.ptr::Ptr{Cvoid},
        src::Ptr{Tv},
        dst::Ptr{Tv},
    )::Cvoid
    )
end

function expm_svd!(basis::BasisManager, otf::OTF, idx::Int64, θ::Float64, vec::T) where {Tv,T<:AbstractArray{Tv,1}}
    Tv <: Complex ? (
        @ccall LIB_OTF.expm_contract_otf_c64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        (idx - 1)::Int64, θ::Cdouble, vec::Ptr{Tv},
    )::Cvoid
    ) : (
        @ccall LIB_OTF.expm_contract_otf_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        (idx - 1)::Int64, θ::Cdouble, vec::Ptr{Tv},
    )::Cvoid
    )
end

function grad_svd(basis::BasisManager, otf::OTF, idx::Int64, θ::Float64, lv::T, rv::T) where {Tv,T<:AbstractArray{Tv,1}}
    Tv <: Complex ? (
        return @ccall LIB_OTF.grad_contract_otf_c64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
            (idx - 1)::Int64, θ::Cdouble, lv::Ptr{ComplexF64}, rv::Ptr{ComplexF64},
        )::ComplexF64
    ) : (
        return @ccall LIB_OTF.grad_contract_otf_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
            (idx - 1)::Int64, θ::Cdouble, lv::Ptr{Cdouble}, rv::Ptr{Cdouble},
        )::Cdouble
    )
end

function backgrad_svd!(basis::BasisManager, otf::OTF, idx::Int64, θ::Float64, lv::T, rv::T) where {Tv,T<:AbstractArray{Tv,1}}
    Tv <: Complex ? (
        return @ccall LIB_OTF.backgrad_contract_otf_c64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
            (idx - 1)::Int64, θ::Cdouble, lv::Ptr{ComplexF64}, rv::Ptr{ComplexF64},
        )::ComplexF64
    ) : (
        return @ccall LIB_OTF.backgrad_contract_otf_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
            (idx - 1)::Int64, θ::Cdouble, lv::Ptr{Cdouble}, rv::Ptr{Cdouble},
        )::Cdouble
    )
end

function backtran_svd!(basis::BasisManager, otf::OTF, idx::Int64, θ::Float64, lv::T, rv::T, bv::T) where {Tv,T<:AbstractArray{Tv,1}}
    Tv <: Complex ? (
        return @ccall LIB_OTF.backtran_contract_otf_c64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
            (idx - 1)::Int64, θ::Cdouble, lv::Ptr{ComplexF64}, rv::Ptr{ComplexF64}, bv::Ptr{ComplexF64},
        )::Cvoid
    ) : (
        return @ccall LIB_OTF.backtran_contract_otf_f64(
            basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
            (idx - 1)::Int64, θ::Cdouble, lv::Ptr{Cdouble}, rv::Ptr{Cdouble}, bv::Ptr{Cdouble},
        )::Cvoid
    )
end

function batch_grad_svd(basis::BasisManager, otf::OTF, x::Vector{Float64}, lv::T, rv::T, grads::T) where {Tv,T<:AbstractArray{Tv,1}}
    Tv <: Complex ? (
        @ccall LIB_OTF.batch_grad_contract_otf_c64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        x::Ptr{Cdouble}, lv::Ptr{Tv}, rv::Ptr{Tv}, grads::Ptr{Tv},
    )::Cvoid
    ) : (
        @ccall LIB_OTF.batch_grad_contract_otf_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        x::Ptr{Cdouble}, lv::Ptr{Tv}, rv::Ptr{Tv}, grads::Ptr{Tv},
    )::Cvoid
    )
end

function tvec_svd!(basis::BasisManager, otf::OTF, idx::Int64, src::T, dst::T) where {Tv,T<:AbstractArray{Tv,1}}
    Tv <: Complex ? (
        @ccall LIB_OTF.tvec_contract_otf_c64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        (idx - 1)::Int64, src::Ptr{Tv}, dst::Ptr{Tv},
    )::Cvoid
    ) : (
        @ccall LIB_OTF.tvec_contract_otf_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        (idx - 1)::Int64, src::Ptr{Tv}, dst::Ptr{Tv},
    )::Cvoid
    )
end

function tran_svd(basis::BasisManager, otf::OTF, lv::T, rv::T, trans::T) where {Tv,T<:AbstractArray{Tv,1}}
    Tv <: Complex ? (
        @ccall LIB_OTF.batch_tran_contract_otf_c64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        lv::Ptr{Tv}, rv::Ptr{Tv}, trans::Ptr{Tv},
    )::Cvoid
    ) : (
        @ccall LIB_OTF.batch_tran_contract_otf_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        lv::Ptr{Tv}, rv::Ptr{Tv}, trans::Ptr{Tv},
    )::Cvoid
    )
end


# ══════════════════════════════════════════════════════════════════════
# Distributed OTF (MPI-based)
# ══════════════════════════════════════════════════════════════════════

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

