mutable struct BasisManager
    ptr::Ptr{Cvoid}
    dim::Int64
    norb::Int64
    nelec::Tuple{Int64,Int64}
    orbsym::Vector{Int64}
end

function BasisManager()
    return BasisManager(C_NULL, 0, 0, (0, 0), Int64[])
end

function get_num_symmetry_blocks(ptr::Ptr{Cvoid})
    return @ccall LIB_BASIS.get_num_symmetry_blocks(ptr::Ptr{Cvoid})::Int64
end

function BasisManager(norb::Int64, nelec::Tuple{Int64,Int64}, orbsym::Vector{Int64})
    total_sym = 0
    num_irreps = 16
    na, nb = nelec

    ptr = @ccall LIB_BASIS.create_basis_manager(
        norb::Int64, na::Int64, nb::Int64, total_sym::Int64, orbsym::Ptr{Int64}, num_irreps::Int64,
    )::Ptr{Cvoid}

    ptr == C_NULL && error("Failed to create C++ BasisManager.")

    dim = @ccall LIB_BASIS.get_subspace_dim(ptr::Ptr{Cvoid})::Int64
    num_blocks = get_num_symmetry_blocks(ptr)
    if is_rank0_or_serial()
        @printf("Num symmetry allowed elements: %d    %.4f GB\n", dim, dim * 8 / (1 << 30))
        @printf("Num wavefunction symmetry blocks: %d\n\n", num_blocks)
    end

    obj = BasisManager(ptr, dim, norb, nelec, orbsym)

    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_BASIS.destroy_basis_manager(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end

    return obj
end

struct VirtualSymmetryPartition
    k::Int64
    orbsym::Vector{Int64}
    num_irreps::Int64
    seed::Int64
end

function make_virtual_orbsym(norb::Int, k::Int; seed::Int=1234)
    @assert k >= 0 "virtual symmetry rank k must be non-negative"
    k == 0 && return zeros(Int64, norb)
    @assert k <= 30 "k is too large for Int64 bit labels"

    rng = MersenneTwister(seed)
    mask = Int64(1 << k) - 1
    return [Int64(rand(rng, 0:mask)) for _ in 1:norb]
end

function VirtualSymmetryPartition(norb::Int, k::Int; seed::Int=1234, orbsym::Vector{Int64}=Int64[])
    virtual_orbsym = isempty(orbsym) ? make_virtual_orbsym(norb, k; seed=seed) : Int64.(orbsym)
    @assert length(virtual_orbsym) == norb
    @assert all(0 .<= virtual_orbsym .< (Int64(1) << k))
    return VirtualSymmetryPartition(Int64(k), virtual_orbsym, Int64(1) << k, Int64(seed))
end

function combine_orbsym(physical_orbsym::Vector{Int64}, virtual_orbsym::Vector{Int64}; physical_num_irreps::Int64=16)
    @assert length(physical_orbsym) == length(virtual_orbsym)
    physical_bits = 0
    while (Int64(1) << physical_bits) < physical_num_irreps
        physical_bits += 1
    end
    return Int64.(physical_orbsym) .| (Int64.(virtual_orbsym) .<< physical_bits)
end

function BasisManager(
    norb::Int64,
    nelec::Tuple{Int64,Int64},
    physical_orbsym::Vector{Int64},
    partition::VirtualSymmetryPartition;
    physical_total_sym::Int64=0,
    physical_num_irreps::Int64=16,
)
    @assert length(physical_orbsym) == norb
    @assert length(partition.orbsym) == norb
    na, nb = nelec

    ptr = @ccall LIB_BASIS.create_partitioned_basis_manager(
        norb::Int64, na::Int64, nb::Int64,
        physical_total_sym::Int64,
        physical_orbsym::Ptr{Int64},
        partition.orbsym::Ptr{Int64},
        physical_num_irreps::Int64,
        partition.num_irreps::Int64,
    )::Ptr{Cvoid}

    ptr == C_NULL && error("Failed to create partitioned C++ BasisManager.")

    dim = @ccall LIB_BASIS.get_subspace_dim(ptr::Ptr{Cvoid})::Int64
    num_blocks = get_num_symmetry_blocks(ptr)
    combined_orbsym = combine_orbsym(physical_orbsym, partition.orbsym; physical_num_irreps=physical_num_irreps)

    if is_rank0_or_serial()
        @printf("Num virtual-symmetry partitioned elements: %d    %.4f GB\n", dim, dim * 8 / (1 << 30))
        @printf("Num wavefunction symmetry blocks: %d\n", num_blocks)
        @printf("Virtual symmetry: Z2^%d (%d labels)\n\n", partition.k, partition.num_irreps)
    end

    obj = BasisManager(ptr, dim, norb, nelec, combined_orbsym)

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
    num_blocks = get_num_symmetry_blocks(ptr)
    if is_rank0_or_serial()
        @printf("Num symmetry allowed elements: %d    %.4f GB\n", dim, dim * 8 / (1 << 30))
        @printf("Num wavefunction symmetry blocks: %d\n\n", num_blocks)
    end

    obj = BasisManager(ptr, dim, norb, (0, 0), orbsym)

    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_BASIS.destroy_basis_manager(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end

    return obj
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
                v0::Ptr{ComplexF64},
            )::Cvoid
        end
    else
        for (astr::UInt32, bstr::UInt32, val::Tv) in zip(astrs, bstrs, vals)
            @ccall LIB_BASIS.set_det_coeff_f64(
                basis.ptr::Ptr{Cvoid},
                astr::UInt32,
                bstr::UInt32,
                val::Cdouble,
                v0::Ptr{Cdouble},
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

                    # 消除机器精度误差带来的微小虚部, 并保持数据类型为 Tv (ComplexF64)
                    Vt_row = Tv.(real.(Vt_row))

                elseif ax == 0 && bx != 0
                    # Pure B: A 弦算符是对角的 (U_col 必须严格为实数)
                    max_idx = argmax(abs.(U_col))
                    phase_angle = angle(U_col[max_idx])

                    U_col .*= exp(-im * phase_angle)
                    Vt_row .*= exp(im * phase_angle)

                    # 消除机器精度误差带来的微小虚部, 并保持数据类型为 Tv (ComplexF64)
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
end

function OTF()
    return OTF(C_NULL, 0, 0)
end

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

    obj = OTF(ptr, basis.dim, ngs)

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

function OTF(basis::BasisManager, pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}, tol::Float64=1e-12) where {Ti,Tv,K,V}
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

    obj = OTF(ptr, basis.dim, ngs)

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

function hvec_svd!(basis::BasisManager, otf::OTF, src::T1, dst::T2) where {Tv,T1<:AbstractArray{Tv,1},T2<:AbstractArray{Tv,1}}
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

function tvec_svd!(basis::BasisManager, otf::OTF, idx::Int64, src::T1, dst::T2) where {Tv,T1<:AbstractArray{Tv,1}, T2<:AbstractArray{Tv,1}}
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

function grad_svd(basis::BasisManager, otf::OTF, idx::Int64, θ::Float64, lv::T1, rv::T2) where {Tv,T1<:AbstractArray{Tv,1},T2<:AbstractArray{Tv,1}}
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

function back_grad_svd!(basis::BasisManager, otf::OTF, idx::Int64, θ::Float64, lv::T1, rv::T2) where {Tv,T1<:AbstractArray{Tv,1},T2<:AbstractArray{Tv,1}}
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

function back_tran_svd!(basis::BasisManager, otf::OTF, idx::Int64, θ::Float64, lv::T1, rv::T2, bv::T3) where {Tv,T1<:AbstractArray{Tv,1},T2<:AbstractArray{Tv,1},T3<:AbstractArray{Tv,1}}
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

function batch_expm_svd!(basis::BasisManager, otf::OTF, idx::Int64, θ::Float64, mat::T, ncols::Int64, valid_ncols::Int64) where {Tv,T<:AbstractArray{Tv,2}}
    Tv <: Complex ? (
        @ccall LIB_OTF.batch_expm_contract_otf_c64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        (idx - 1)::Int64, θ::Cdouble, mat::Ptr{Tv},
        ncols::Int64, valid_ncols::Int64
    )::Cvoid
    ) : (
        @ccall LIB_OTF.batch_expm_contract_otf_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        (idx - 1)::Int64, θ::Cdouble, mat::Ptr{Tv},
        ncols::Int64, valid_ncols::Int64
    )::Cvoid
    )
end

function batch_grad_svd(basis::BasisManager, otf::OTF, x::Vector{Float64}, lv::T1, rv::T2, grads::T3) where {Tv,T1<:AbstractArray{Tv,1},T2<:AbstractArray{Tv,1},T3<:AbstractArray{Tv,1}}
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

function batch_tran_svd(basis::BasisManager, otf::OTF, lv::T1, rv::T2, trans::T3) where {Tv,T1<:AbstractArray{Tv,1},T2<:AbstractArray{Tv,1},T3<:AbstractArray{Tv,1}}
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

"""
OTF_Functions 结构体包含以下操作函数：
# 基本操作
- hvec(v, Hv)::Function             |Hv⟩ = H|v⟩             : 哈密顿量作用
- expm(idx, θ, v)::Function         |v⟩ = exp(θT)|v⟩        : 原地演化 T 的指数算子, T 在 operator pool 中的索引为 idx
- tvec(idx, lv, rv)::Function       |rv⟩ = T|lv⟩            : 算子 T 作用
# 梯度计算
- grad(idx, θ, lv, rv)::Function    g = ⟨lv|Texp(θT)|rv⟩    : 计算单个梯度并返回
# 反向传播相关
- backgrad::Function : 按顺序组合操作：
  1. |lv⟩ = exp(-θT)|lv⟩    : 原地演化, 注意是-θ
  2. g = ⟨lv|T exp(θT)|rv⟩  : 返回 g
  3. |rv⟩ = exp(-θT)|rv⟩    : 原地演化
- backtran(idx, θ, lv, rv, tlv)::Function : 按顺序组合操作：
  1. |lv⟩ = exp(θT)|lv⟩     : 原地演化, 注意是θ
  2. |tlv⟩ = T|lv⟩
  3. |rv⟩ = exp(θT)|rv⟩     : 原地演化
# 批量操作
- batchexpm(idx, θ, mat, N, j)::Function    : 批量原地演化 expm, 作用于 (N × dim) 矩阵 mat 的前 j 列
- batchgrad(lv, rv, grads, x)::Function     : 批量计算算子池的梯度(输入 |lv⟩ 和 |rv⟩), 结果保存到 grads
- batchtran(lv, rv, trans)::Function        : 批量计算算子池的 ⟨lv|T|rv⟩, 结果保存到 trans
"""
struct OTF_Functions
    hvec::Function
    expm::Function
    tvec::Function
    grad::Function
    backgrad::Function
    backtran::Function
    batchexpm::Function
    batchgrad::Function 
    batchtran::Function
    ham::OTF
    pool::OTF
end

function OTF_Functions(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,TK,TV}, pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}}; 
    info_print::Bool=true, time_print::Bool=false
) where {Ti,Tv,TK,TV}
    f_hvec      = (v, Hv)               -> nothing
    f_expm      = (idx, θ, v)           -> nothing
    f_tvec      = (idx, lv, rv)         -> nothing
    f_grad      = (idx, θ, lv, rv)      -> nothing
    f_backgrad  = (idx, θ, lv, rv)      -> nothing
    f_backtran  = (idx, θ, lv, rv, tlv) -> nothing
    f_batchexpm = (idx, θ, mat, N, j)   -> nothing
    f_batchgrad = (lv, rv, grads, x)    -> nothing 
    f_batchtran = (lv, rv, trans)       -> nothing
    ham_otf     = OTF(C_NULL, 0, 0)
    pool_otf    = OTF(C_NULL, 0, 0)

    if !isempty(ham)
        info_print && print("Pre-compiling Ham OTF ... ")
        time_ops = @elapsed ham_otf = OTF(basis, ham)
        info_print && @printf("Done in %.4f seconds\n", time_ops)
        if time_print
            f_hvec = (v, Hv) -> @printf(
                "hvec time %.6f seconds", @elapsed hvec_svd!(basis, ham_otf, v, Hv))
        else
            f_hvec = (v, Hv) -> hvec_svd!(basis, ham_otf, v, Hv)
        end
    end
    if !isempty(pool)
        info_print && print("Pre-compiling Pool OTF ... ")
        time_ops = @elapsed pool_otf = OTF(basis, pool)
        info_print && @printf("Done in %.4f seconds\n", time_ops)

        f_expm = (idx, θ, v) -> expm_svd!(basis, pool_otf, idx, θ, v)
        f_tvec = (idx, lv, rv) -> tvec_svd!(basis, pool_otf, idx, lv, rv)
        f_grad = (idx, θ, lv, rv) -> return grad_svd(basis, pool_otf, idx, θ, lv, rv)
        f_backgrad = (idx, θ, lv, rv) -> return back_grad_svd!(basis, pool_otf, idx, θ, lv, rv)
        f_backtran = (idx, θ, lv, rv, tlv) -> return back_tran_svd!(basis, pool_otf, idx, θ, lv, rv, tlv)
        f_batchexpm = (idx, θ, mat, N, j) -> batch_expm_svd!(basis, pool_otf, idx, θ, mat, N, j)
        f_batchgrad = (lv, rv, grads, x) -> return batch_grad_svd(basis, pool_otf, x, lv, rv, grads)
        f_batchtran = (lv, rv, trans) -> return batch_tran_svd(basis, pool_otf, lv, rv, trans)
    end

    return OTF_Functions(f_hvec, f_expm, f_tvec, f_grad, f_backgrad, f_backtran, f_batchexpm, f_batchgrad, f_batchtran, ham_otf, pool_otf)
end
