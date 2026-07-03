const LIB_SCI_BITSTR = joinpath(libpath, "libsci_otf_bitstr.so")
# Keep the bitstring SCI symmetry layout aligned with BasisManager in jl/network.jl.
const SCI_BITSTR_NUM_IRREPS = Int64(16)

mutable struct SciBasisManagerBitstr
    ptr::Ptr{Cvoid}
    dim::Int64
    norb::Int64
    num_blocks::Int64
    na::Int64
    nb::Int64
    astrs::Vector{UInt32}
    bstrs::Vector{UInt32}
    a_idx_map::Dict{UInt32,Int}
    b_idx_map::Dict{UInt32,Int}
end

function SciBasisManagerBitstr()
    return SciBasisManagerBitstr(C_NULL, 0, 0, 0, -1, -1, UInt32[], UInt32[],
                                  Dict{UInt32,Int}(), Dict{UInt32,Int}())
end

function get_string_sym(str::UInt32, orbsym::Vector{Int64})::Int64
    sym = Int64(0)
    pos = 0
    while str > 0
        if (str & 1) != 0
            sym ⊻= orbsym[pos + 1]
        end
        str >>= 1
        pos += 1
    end
    return sym
end

function sort_by_sym(arr::Vector{UInt32}, orbsym::Vector{Int64}, num_irreps::Integer)::Vector{UInt32}
    num_irreps = Int64(num_irreps)
    by_sym = [Vector{UInt32}() for _ in 1:num_irreps]
    for a in arr
        sym = get_string_sym(a, orbsym)
        if 0 <= sym < num_irreps
            push!(by_sym[sym + 1], a)
        end
    end
    result = UInt32[]
    for s in 1:num_irreps
        sort!(by_sym[s])
        append!(result, by_sym[s])
    end
    return result
end

function extract_ax_bx(svd_groups::Vector{<:Any})
    axs = UInt32[]
    bxs = UInt32[]
    for g in svd_groups
        push!(axs, g.ax)
        push!(bxs, g.bx)
    end
    return axs, bxs
end

function build_ax_bx_group_bucket(svd_groups::Vector{<:Any})
    bucket = Dict{Tuple{UInt32,UInt32}, Vector{Int64}}()
    for (idx, g) in enumerate(svd_groups)
        push!(get!(bucket, (UInt32(g.ax), UInt32(g.bx)), Int64[]), Int64(idx - 1))
    end
    return bucket
end

function flatten_ax_bx_group_bucket(bucket::Dict{Tuple{UInt32,UInt32}, Vector{Int64}})
    keys_sorted = sort!(collect(keys(bucket)))
    bucket_axs = UInt32[first(key) for key in keys_sorted]
    bucket_bxs = UInt32[last(key) for key in keys_sorted]
    offsets = Vector{Int64}(undef, length(keys_sorted) + 1)
    group_ids = Int64[]
    offsets[1] = 0
    for (i, key) in enumerate(keys_sorted)
        append!(group_ids, bucket[key])
        offsets[i + 1] = length(group_ids)
    end
    return bucket_axs, bucket_bxs, offsets, group_ids
end

mutable struct ExternalLinkSelectContextBitstr
    ptr::Ptr{Cvoid}
end

function destroy_external_link_select_context_bitstr(ctx::ExternalLinkSelectContextBitstr)
    if ctx.ptr != C_NULL
        @ccall LIB_SCI_BITSTR.destroy_external_link_select_context_bitstr_f64(ctx.ptr::Ptr{Cvoid})::Cvoid
        ctx.ptr = C_NULL
    end
    return nothing
end

function SciBasisManagerBitstr(
    astrs::Vector{UInt32}, bstrs::Vector{UInt32},
    norb::Int64, total_sym::Int64, orbsym::Vector{Int64},
    na::Integer, nb::Integer; sorted::Bool=false, num_irreps::Integer=SCI_BITSTR_NUM_IRREPS)

    na64_expected = Int64(na)
    nb64_expected = Int64(nb)
    astrs = UInt32[a for a in astrs if count_ones(a) == na64_expected]
    bstrs = UInt32[b for b in bstrs if count_ones(b) == nb64_expected]
    @assert all(count_ones.(astrs) .== na64_expected)
    @assert all(count_ones.(bstrs) .== nb64_expected)

    if !sorted
        astrs = sort_by_sym(astrs, orbsym, num_irreps)
        bstrs = sort_by_sym(bstrs, orbsym, num_irreps)
    end

    num_a = length(astrs)
    num_b = length(bstrs)
    a_idx_map = Dict{UInt32,Int}(astrs[i] => i-1 for i in 1:num_a)
    b_idx_map = Dict{UInt32,Int}(bstrs[i] => i-1 for i in 1:num_b)

    num_a64 = Int64(num_a)
    num_b64 = Int64(num_b)
    nirp64 = Int64(num_irreps)
    ptr = @ccall LIB_SCI_BITSTR.create_sci_basis_manager_bitstr_f64(
        astrs::Ptr{UInt32}, num_a64::Int64,
        bstrs::Ptr{UInt32}, num_b64::Int64,
        norb::Int64, orbsym::Ptr{Int64},
        total_sym::Int64, nirp64::Int64
    )::Ptr{Cvoid}
    ptr == C_NULL && error("Failed to create bitstr basis manager.")
    dim = @ccall LIB_SCI_BITSTR.sci_basis_dim_bitstr(ptr::Ptr{Cvoid})::Int64
    nbk = @ccall LIB_SCI_BITSTR.sci_basis_num_blocks_bitstr(ptr::Ptr{Cvoid})::Int64

    obj = SciBasisManagerBitstr(ptr, dim, norb, nbk, na64_expected, nb64_expected, astrs, bstrs, a_idx_map, b_idx_map)
    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_SCI_BITSTR.destroy_sci_basis_manager_bitstr_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end
    return obj
end

function OTF_bitstr(orbsym::Vector{Int64}, norb::Int64, A::BinaryQubitAABB{Ti,Tv,K,V};
                    tol::Float64=1e-12) where {Ti,Tv,K,V}
    groups = compress_by_svd(A, tol)
    ngs = length(groups)
    axs    = Vector{Ti}(undef, ngs);    bxs    = Vector{Ti}(undef, ngs)
    ranks  = Vector{Int64}(undef, ngs); num_as = Vector{Int64}(undef, ngs)
    num_bs = Vector{Int64}(undef, ngs)
    flat_azs = Ti[]; flat_bzs = Ti[]; flat_wa = Tv[]; flat_wb = Tv[]
    for (g, group) in enumerate(groups)
        axs[g]=group.ax; bxs[g]=group.bx; ranks[g]=group.rank
        num_as[g]=length(group.azs); num_bs[g]=length(group.bzs)
        append!(flat_azs, group.azs); append!(flat_bzs, group.bzs)
        append!(flat_wa, vec(group.wa)); append!(flat_wb, vec(group.wb))
    end
    ptr = @ccall LIB_SCI_BITSTR.build_network_otf_sci_bitstr_f64(
        orbsym::Ptr{Int64}, norb::Int64, ngs::Int64,
        axs::Ptr{Ti}, bxs::Ptr{Ti}, ranks::Ptr{Int64},
        num_as::Ptr{Int64}, num_bs::Ptr{Int64},
        flat_azs::Ptr{Ti}, flat_bzs::Ptr{Ti},
        flat_wa::Ptr{Tv}, flat_wb::Ptr{Tv}
    )::Ptr{Cvoid}
    ptr == C_NULL && error("Failed to build bitstr OTF.")
    obj = OTF(ptr, 0, ngs)
    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_OTF.destroy_network_otf_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end
    return obj
end

function expand_bitstrings_bitstr(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32},
    axs::Vector{UInt32}, bxs::Vector{UInt32},
    na::Int, nb::Int, orbsym::Vector{Int64}, num_irreps::Integer=SCI_BITSTR_NUM_IRREPS)

    a_map = Dict{UInt32,Int}(a => 0 for a in src_astrs)
    b_map = Dict{UInt32,Int}(b => 0 for b in src_bstrs)

    for (ax, bx) in zip(axs, bxs)
        if ax != 0
            for a in src_astrs
                new_a = a ⊻ ax
                count_ones(new_a) == na && get!(a_map, new_a, 1)
            end
        end
        if bx != 0
            for b in src_bstrs
                new_b = b ⊻ bx
                count_ones(new_b) == nb && get!(b_map, new_b, 1)
            end
        end
    end

    dst_astrs = sort_by_sym(collect(keys(a_map)), orbsym, num_irreps)
    dst_bstrs = sort_by_sym(collect(keys(b_map)), orbsym, num_irreps)

    is_new_a = Bool[a_map[a] == 1 for a in dst_astrs]
    is_new_b = Bool[b_map[b] == 1 for b in dst_bstrs]

    return dst_astrs, dst_bstrs, is_new_a, is_new_b
end

function sci_hvec_select_for_block_bitstr!(
    tgt::SciBasisManagerBitstr, src::SciBasisManagerBitstr, otf::OTF,
    blk::Integer, psi::Vector{Float64}, candidate_diags::Vector{Float64},
    variational_energy::Float64, chunk_size::Int, eps::Float64,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    max_per_block = tgt.dim
    buf_a = Vector{UInt32}(undef, max_per_block)
    buf_b = Vector{UInt32}(undef, max_per_block)
    buf_v = Vector{Float64}(undef, max_per_block)
    mb64 = Int64(max_per_block)
    blk64 = Int64(blk)
    n = @ccall LIB_SCI_BITSTR.sci_hvec_select_for_block_bitstr_f64(
        tgt.ptr::Ptr{Cvoid}, src.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        blk64::Int64, psi::Ptr{Float64}, candidate_diags::Ptr{Float64},
        variational_energy::Cdouble, chunk_size::Cint, eps::Cdouble,
        buf_a::Ptr{UInt32}, buf_b::Ptr{UInt32}, buf_v::Ptr{Float64},
        mb64::Int64
    )::Int64
    append!(sel_a, view(buf_a, 1:n))
    append!(sel_b, view(buf_b, 1:n))
    append!(sel_v, view(buf_v, 1:n))
    return n
end

function sci_hvec_select_external_bitstr!(
    tgt::SciBasisManagerBitstr, src::SciBasisManagerBitstr, otf::OTF,
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool},
    blk::Integer, psi::Vector{Float64}, candidate_diags::Vector{Float64},
    variational_energy::Float64, chunk_size::Int, eps::Float64,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    max_per_block = tgt.dim
    buf_a = Vector{UInt32}(undef, max_per_block)
    buf_b = Vector{UInt32}(undef, max_per_block)
    buf_v = Vector{Float64}(undef, max_per_block)
    mb64 = Int64(max_per_block)
    blk64 = Int64(blk)
    n = @ccall LIB_SCI_BITSTR.sci_hvec_select_external_bitstr_f64(
        tgt.ptr::Ptr{Cvoid}, src.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        is_new_a::Ptr{Bool}, is_new_b::Ptr{Bool},
        blk64::Int64, psi::Ptr{Float64}, candidate_diags::Ptr{Float64},
        variational_energy::Cdouble, chunk_size::Cint, eps::Cdouble,
        buf_a::Ptr{UInt32}, buf_b::Ptr{UInt32}, buf_v::Ptr{Float64},
        mb64::Int64
    )::Int64
    append!(sel_a, view(buf_a, 1:n))
    append!(sel_b, view(buf_b, 1:n))
    append!(sel_v, view(buf_v, 1:n))
    return n
end


function ExternalLinkSelectContextBitstr(
    tgt::SciBasisManagerBitstr, src::SciBasisManagerBitstr, otf::OTF,
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool},
    unique_axs::Vector{UInt32}, unique_bxs::Vector{UInt32},
    group_bucket::Dict{Tuple{UInt32,UInt32}, Vector{Int64}})

    bucket_axs, bucket_bxs, bucket_offsets, bucket_group_ids = flatten_ax_bx_group_bucket(group_bucket)
    ptr = @ccall LIB_SCI_BITSTR.create_external_link_select_context_bitstr_f64(
        tgt.ptr::Ptr{Cvoid}, src.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        is_new_a::Ptr{Bool}, is_new_b::Ptr{Bool},
        unique_axs::Ptr{UInt32}, Int64(length(unique_axs))::Int64,
        unique_bxs::Ptr{UInt32}, Int64(length(unique_bxs))::Int64,
        bucket_axs::Ptr{UInt32}, bucket_bxs::Ptr{UInt32},
        bucket_offsets::Ptr{Int64}, bucket_group_ids::Ptr{Int64},
        Int64(length(bucket_axs))::Int64
    )::Ptr{Cvoid}
    ptr == C_NULL && error("Failed to create external-link select context.")
    ctx = ExternalLinkSelectContextBitstr(ptr)
    finalizer(destroy_external_link_select_context_bitstr, ctx)
    return ctx
end

function sci_hvec_select_external_link_all_blocks_bitstr!(
    ctx::ExternalLinkSelectContextBitstr, tgt::SciBasisManagerBitstr, src::SciBasisManagerBitstr, otf::OTF,
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool},
    psi::Vector{Float64}, candidate_diags::Vector{Float64},
    variational_energy::Float64, chunk_size::Int, eps::Float64,
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    max_entries = tgt.dim
    buf_a = Vector{UInt32}(undef, max_entries)
    buf_b = Vector{UInt32}(undef, max_entries)
    buf_v = Vector{Float64}(undef, max_entries)
    n = @ccall LIB_SCI_BITSTR.sci_hvec_select_external_link_all_blocks_with_context_bitstr_f64(
        ctx.ptr::Ptr{Cvoid}, tgt.ptr::Ptr{Cvoid}, src.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        is_new_a::Ptr{Bool}, is_new_b::Ptr{Bool},
        psi::Ptr{Float64}, candidate_diags::Ptr{Float64},
        variational_energy::Cdouble, chunk_size::Cint, eps::Cdouble,
        buf_a::Ptr{UInt32}, buf_b::Ptr{UInt32}, buf_v::Ptr{Float64},
        Int64(max_entries)::Int64
    )::Int64
    append!(sel_a, view(buf_a, 1:n))
    append!(sel_b, view(buf_b, 1:n))
    append!(sel_v, view(buf_v, 1:n))
    return n
end


function selected_pair_set(sel_a::Vector{UInt32}, sel_b::Vector{UInt32})
    return Set(zip(sel_a, sel_b))
end


function selected_pair_hpsi_map(sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})
    result = Dict{Tuple{UInt32,UInt32},Float64}()
    for i in eachindex(sel_a, sel_b, sel_v)
        key = (sel_a[i], sel_b[i])
        result[key] = get(result, key, 0.0) + sel_v[i]
    end
    return result
end

function selected_hpsi_max_abs_diff(left::Dict{Tuple{UInt32,UInt32},Float64},
                                    right::Dict{Tuple{UInt32,UInt32},Float64})
    common = intersect(keys(left), keys(right))
    isempty(common) && return 0.0
    return maximum(abs(left[key] - right[key]) for key in common)
end

function filter_new_selected(
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64},
    dst_a_idx_map::Dict{UInt32,Int}, dst_b_idx_map::Dict{UInt32,Int},
    is_new_a::Vector{Bool}, is_new_b::Vector{Bool})

    new_a = UInt32[]; new_b = UInt32[]; new_v = Float64[]
    for i in 1:length(sel_a)
        a_local = get(dst_a_idx_map, sel_a[i], -1)
        b_local = get(dst_b_idx_map, sel_b[i], -1)
        if a_local == -1 || b_local == -1
            continue
        end
        if is_new_a[a_local + 1] || is_new_b[b_local + 1]
            push!(new_a, sel_a[i])
            push!(new_b, sel_b[i])
            push!(new_v, sel_v[i])
        end
    end
    return new_a, new_b, new_v
end

function merge_bitstrings(
    src_astrs::Vector{UInt32}, src_bstrs::Vector{UInt32},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32},
    orbsym::Vector{Int64}, num_irreps::Integer=SCI_BITSTR_NUM_IRREPS)

    a_union = sort_by_sym(collect(union(Set(src_astrs), Set(sel_a))), orbsym, num_irreps)
    b_union = sort_by_sym(collect(union(Set(src_bstrs), Set(sel_b))), orbsym, num_irreps)
    return a_union, b_union
end

function fixed_electron_bitstrings(norb::Integer, nelec::Integer)::Vector{UInt32}
    norb64 = Int64(norb)
    nelec64 = Int64(nelec)
    nelec64 < 0 && error("nelec must be non-negative")
    nelec64 > norb64 && error("nelec must be <= norb")

    strings = UInt32[]
    for str in UInt32(0):(UInt32(1) << norb64) - UInt32(1)
        count_ones(str) == nelec64 && push!(strings, str)
    end
    return strings
end

function count_missing_bitstrings(reference::Vector{UInt32}, candidates::Vector{UInt32})::Int
    candidate_set = Set(candidates)
    return count(str -> !(str in candidate_set), reference)
end

function remap_wavefunction_bitstr!(
    old::SciBasisManagerBitstr, old_psi::Vector{Float64},
    new::SciBasisManagerBitstr, new_psi::Vector{Float64},
    sel_a::Vector{UInt32}, sel_b::Vector{UInt32}, sel_v::Vector{Float64})

    nn64 = Int64(length(sel_a))
    @ccall LIB_SCI_BITSTR.remap_wavefunction_sci_bitstr_f64(
        old.ptr::Ptr{Cvoid}, old_psi::Ptr{Float64},
        new.ptr::Ptr{Cvoid}, new_psi::Ptr{Float64},
        sel_a::Ptr{UInt32}, sel_b::Ptr{UInt32}, sel_v::Ptr{Float64},
        nn64::Int64
    )::Cvoid
end

function get_diags_bitstr!(basis::SciBasisManagerBitstr, otf::OTF, diags::Vector{Float64})
    @ccall LIB_SCI_BITSTR.get_diags_elements_sci_bitstr_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid}, diags::Ptr{Float64})::Cvoid
end

function hvec_sci_full_bitstr!(basis::SciBasisManagerBitstr, otf::OTF,
                                src::Vector{Float64}, dst::Vector{Float64})
    @ccall LIB_SCI_BITSTR.hvec_sci_full_bitstr_f64(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        src::Ptr{Float64}, dst::Ptr{Float64})::Cvoid
end

function destroy_sci_basis_manager_bitstr(sb::SciBasisManagerBitstr)
    if sb.ptr != C_NULL
        @ccall LIB_SCI_BITSTR.destroy_sci_basis_manager_bitstr_f64(sb.ptr::Ptr{Cvoid})::Cvoid
        sb.ptr = C_NULL
    end
end

function create_sci_basis_from_standard_bitstr(basis::BasisManager)
    ptr = @ccall LIB_SCI_BITSTR.create_sci_basis_from_standard_bitstr_f64(
        basis.ptr::Ptr{Cvoid})::Ptr{Cvoid}
    dim = @ccall LIB_SCI_BITSTR.sci_basis_dim_bitstr(ptr::Ptr{Cvoid})::Int64
    nb  = @ccall LIB_SCI_BITSTR.sci_basis_num_blocks_bitstr(ptr::Ptr{Cvoid})::Int64
    obj = SciBasisManagerBitstr(ptr, dim, basis.norb, nb, -1, -1, UInt32[], UInt32[],
                                 Dict{UInt32,Int}(), Dict{UInt32,Int}())
    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_SCI_BITSTR.destroy_sci_basis_manager_bitstr_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end
    return obj
end

function run_sci_bitstr(mole::Mole;
    max_iter::Int=20, max_size::Int=10000, eps::Float64=1e-6,
    chunk_size::Int=256, davidson_tol::Float64=1e-5, verbose::Bool=true,
    debug_compare_fci::Bool=false, total_sym::Int64=0,
    use_external_select::Union{Bool,Nothing}=nothing,
    use_link_external_select::Bool=false,
    select_mode::Symbol=:external_link, debug_external_select::Bool=false)

    if use_external_select !== nothing
        select_mode = use_external_select ? :external_block : :full
    end
    if use_link_external_select
        select_mode = :external_link
    end
    if select_mode == :external_links
        select_mode = :external_link
    end
    if !(select_mode in (:full, :external_block, :external_link, :auto))
        error("select_mode must be one of :full, :external_block, :external_link, or :auto")
    end
    check_link_select = debug_external_select || get(ENV, "BINSIM_SCI_BITSTR_CHECK_LINK_SELECT", "") != ""

    # SCI selection uses a first-order/CIPSI-style amplitude estimate for each
    # candidate determinant: abs(Hψ(candidate) / (E - Haa)) > eps, where E is
    # the current variational energy and Haa is the candidate diagonal matrix
    # element.  Thus eps thresholds an energy-aware estimated CI coefficient,
    # not the raw residual |Hψ(candidate)|.

    na, nb = mole.nelec
    num_irreps = SCI_BITSTR_NUM_IRREPS

    fci_basis = BasisManager()
    fci_astrs = UInt32[]
    fci_bstrs = UInt32[]
    fci_dim = Int64(-1)
    if debug_compare_fci
        fci_basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
        # create_sci_basis_from_standard_bitstr currently preserves the native
        # bitstring layout on the C++ side but does not expose the alpha/beta
        # string arrays back to Julia.  Generate the fixed-electron string sets
        # directly here and use the standard BasisManager dimension as the FCI
        # determinant-pair reference for the requested total symmetry.
        fci_astrs = sort_by_sym(fixed_electron_bitstrings(mole.norb, na), mole.orbsym, num_irreps)
        fci_bstrs = sort_by_sym(fixed_electron_bitstrings(mole.norb, nb), mole.orbsym, num_irreps)
        fci_dim = fci_basis.dim
        verbose && @printf("[SCI bitstr FCI debug] full strings: nα=%d nβ=%d dim=%d total_sym=%d\n",
                           length(fci_astrs), length(fci_bstrs), fci_dim, total_sym)
    end

    verbose && print("Building OTF ... ")
    t0 = @elapsed begin
        ham = JW_hamiltonian(mole; verbose=false)
        svd_groups = compress_by_svd(ham)
        all_axs, all_bxs = extract_ax_bx(svd_groups)
        ax_bx_group_bucket = build_ax_bx_group_bucket(svd_groups)
        unique_axs = unique(all_axs)
        unique_bxs = unique(all_bxs)
        ham_otf = OTF_bitstr(mole.orbsym, mole.norb, ham)
    end
    verbose && @printf("Done in %.4f s  (ngroups=%d)\n", t0, length(svd_groups))

    hf_astr = UInt32((1 << na) - 1)
    hf_bstr = UInt32((1 << nb) - 1)
    basis = SciBasisManagerBitstr([hf_astr], [hf_bstr], mole.norb, total_sym, mole.orbsym, na, nb; num_irreps=num_irreps)
    psi = Float64[1.0]
    diags = zeros(Float64, 1)
    get_diags_bitstr!(basis, ham_otf, diags)
    current_energy = diags[1]
    verbose && @printf("Initial basis: dim=%d  E0=%.10f\n", basis.dim, current_energy)

    for iter in 1:max_iter
        t_iter = @elapsed begin
            t1 = @elapsed dst_a, dst_b, is_new_a, is_new_b = expand_bitstrings_bitstr(
                basis.astrs, basis.bstrs, all_axs, all_bxs, na, nb, mole.orbsym, num_irreps)
            tgt = SciBasisManagerBitstr(dst_a, dst_b, mole.norb, total_sym, mole.orbsym, na, nb; sorted=true, num_irreps=num_irreps)
            tgt_diags = zeros(Float64, tgt.dim)
            get_diags_bitstr!(tgt, ham_otf, tgt_diags)

            if debug_compare_fci && verbose
                cur_missing_a = count_missing_bitstrings(fci_astrs, basis.astrs)
                cur_missing_b = count_missing_bitstrings(fci_bstrs, basis.bstrs)
                dst_missing_a = count_missing_bitstrings(fci_astrs, dst_a)
                dst_missing_b = count_missing_bitstrings(fci_bstrs, dst_b)
                @printf("  [%d FCI debug] current strings missing: α=%d/%d β=%d/%d dim=%d/%d\n",
                        iter, cur_missing_a, length(fci_astrs), cur_missing_b, length(fci_bstrs),
                        basis.dim, fci_dim)
                @printf("  [%d FCI debug] expanded strings missing: α=%d/%d β=%d/%d dim=%d/%d\n",
                        iter, dst_missing_a, length(fci_astrs), dst_missing_b, length(fci_bstrs),
                        tgt.dim, fci_dim)
            end

            sel_a = UInt32[]; sel_b = UInt32[]; sel_v = Float64[]
            raw_sel = 0
            if select_mode == :external_block
                t2 = @elapsed for blk in 0:tgt.num_blocks-1
                    sci_hvec_select_external_bitstr!(tgt, basis, ham_otf, is_new_a, is_new_b,
                                                      blk, psi, tgt_diags, current_energy,
                                                      chunk_size, eps, sel_a, sel_b, sel_v)
                end
                raw_sel = length(sel_v)
            elseif select_mode == :external_link || select_mode == :auto
                t2 = @elapsed begin
                    link_ctx = ExternalLinkSelectContextBitstr(
                        tgt, basis, ham_otf, is_new_a, is_new_b, unique_axs, unique_bxs, ax_bx_group_bucket)
                    try
                        sci_hvec_select_external_link_all_blocks_bitstr!(
                            link_ctx, tgt, basis, ham_otf, is_new_a, is_new_b, psi, tgt_diags,
                            current_energy, chunk_size, eps, sel_a, sel_b, sel_v)
                    finally
                        destroy_external_link_select_context_bitstr(link_ctx)
                    end
                end
                raw_sel = length(sel_v)
            else
                t2 = @elapsed for blk in 0:tgt.num_blocks-1
                    sci_hvec_select_for_block_bitstr!(tgt, basis, ham_otf, blk, psi,
                                                       tgt_diags, current_energy,
                                                       chunk_size, eps, sel_a, sel_b, sel_v)
                end
                raw_sel = length(sel_v)
                sel_a, sel_b, sel_v = filter_new_selected(
                    sel_a, sel_b, sel_v, tgt.a_idx_map, tgt.b_idx_map, is_new_a, is_new_b)
            end

            if check_link_select
                full_a = UInt32[]; full_b = UInt32[]; full_v = Float64[]
                for blk in 0:tgt.num_blocks-1
                    sci_hvec_select_for_block_bitstr!(tgt, basis, ham_otf, blk, psi,
                                                       tgt_diags, current_energy,
                                                       chunk_size, eps, full_a, full_b, full_v)
                end
                full_a, full_b, full_v = filter_new_selected(
                    full_a, full_b, full_v, tgt.a_idx_map, tgt.b_idx_map, is_new_a, is_new_b)

                block_a = UInt32[]; block_b = UInt32[]; block_v = Float64[]
                if select_mode == :external_block
                    append!(block_a, sel_a); append!(block_b, sel_b); append!(block_v, sel_v)
                else
                    for blk in 0:tgt.num_blocks-1
                        sci_hvec_select_external_bitstr!(tgt, basis, ham_otf, is_new_a, is_new_b,
                                                          blk, psi, tgt_diags, current_energy,
                                                          chunk_size, eps, block_a, block_b, block_v)
                    end
                end

                link_a = UInt32[]; link_b = UInt32[]; link_v = Float64[]
                if select_mode == :external_link || select_mode == :auto
                    append!(link_a, sel_a); append!(link_b, sel_b); append!(link_v, sel_v)
                else
                    link_ctx = ExternalLinkSelectContextBitstr(
                        tgt, basis, ham_otf, is_new_a, is_new_b, unique_axs, unique_bxs, ax_bx_group_bucket)
                    try
                        sci_hvec_select_external_link_all_blocks_bitstr!(
                            link_ctx, tgt, basis, ham_otf, is_new_a, is_new_b, psi, tgt_diags,
                            current_energy, chunk_size, eps, link_a, link_b, link_v)
                    finally
                        destroy_external_link_select_context_bitstr(link_ctx)
                    end
                end

                full_map = selected_pair_hpsi_map(full_a, full_b, full_v)
                block_map = selected_pair_hpsi_map(block_a, block_b, block_v)
                link_map = selected_pair_hpsi_map(link_a, link_b, link_v)
                full_set = Set(keys(full_map))
                block_set = Set(keys(block_map))
                link_set = Set(keys(link_map))
                full_block_missing = setdiff(full_set, block_set); full_block_extra = setdiff(block_set, full_set)
                full_link_missing = setdiff(full_set, link_set); full_link_extra = setdiff(link_set, full_set)
                block_link_missing = setdiff(block_set, link_set); block_link_extra = setdiff(link_set, block_set)
                full_block_match = isempty(full_block_missing) && isempty(full_block_extra)
                full_link_match = isempty(full_link_missing) && isempty(full_link_extra)
                block_link_match = isempty(block_link_missing) && isempty(block_link_extra)
                full_block_hpsi_diff = selected_hpsi_max_abs_diff(full_map, block_map)
                full_link_hpsi_diff = selected_hpsi_max_abs_diff(full_map, link_map)
                block_link_hpsi_diff = selected_hpsi_max_abs_diff(block_map, link_map)
                @printf("[sci_bitstr select check] iter=%d full_count=%d external_block_count=%d external_link_count=%d full_vs_block_match=%d full_vs_link_match=%d block_vs_link_match=%d full_block_hpsi_max_abs_diff=%.17g full_link_hpsi_max_abs_diff=%.17g block_link_hpsi_max_abs_diff=%.17g full_block_missing=%d full_block_extra=%d full_link_missing=%d full_link_extra=%d block_link_missing=%d block_link_extra=%d\n",
                                   iter, length(full_set), length(block_set), length(link_set),
                                   full_block_match ? 1 : 0, full_link_match ? 1 : 0, block_link_match ? 1 : 0,
                                   full_block_hpsi_diff, full_link_hpsi_diff, block_link_hpsi_diff,
                                   length(full_block_missing), length(full_block_extra),
                                   length(full_link_missing), length(full_link_extra),
                                   length(block_link_missing), length(block_link_extra))
                if debug_external_select && (!full_block_match || !full_link_match || !block_link_match)
                    error("external selection mismatch at iter $iter: full_block_missing=$(length(full_block_missing)) full_block_extra=$(length(full_block_extra)) full_link_missing=$(length(full_link_missing)) full_link_extra=$(length(full_link_extra)) block_link_missing=$(length(block_link_missing)) block_link_extra=$(length(block_link_extra))")
                end
            end

            verbose && @printf("  [%d] expand %d→%d  raw_sel=%d  ",
                               iter, basis.dim, tgt.dim, raw_sel)

            nsel = length(sel_v)
            verbose && @printf("new_sel=%d  ", nsel)

            if debug_compare_fci && verbose
                selected_pairs = Set(zip(sel_a, sel_b))
                @printf("\n  [%d FCI debug] selected new determinant pairs=%d; expanded determinant space %s FCI\n",
                        iter, length(selected_pairs), tgt.dim == fci_dim ? "matches" : "does not match")
                if tgt.dim == fci_dim && nsel == 0
                    @printf("  [%d FCI debug] FCI determinant pairs are present in the expanded basis; growth stopped at selection, not at the string-expansion level.\n",
                            iter)
                elseif tgt.dim < fci_dim && dst_a == basis.astrs && dst_b == basis.bstrs
                    @printf("  [%d FCI debug] expanded strings stopped growing before reaching FCI.\n", iter)
                end
            end

            if nsel == 0
                verbose && println("No new states, done.")
                destroy_sci_basis_manager_bitstr(tgt)
                break
            end

            t4 = @elapsed new_a, new_b = merge_bitstrings(basis.astrs, basis.bstrs,
                                             sel_a, sel_b, mole.orbsym, num_irreps)
            new_basis = SciBasisManagerBitstr(new_a, new_b, mole.norb, total_sym, mole.orbsym, na, nb; sorted=true, num_irreps=num_irreps)
            verbose && @printf("merged=%d  ", new_basis.dim)

            new_psi = zeros(Float64, new_basis.dim)
            t5 = @elapsed remap_wavefunction_bitstr!(basis, psi, new_basis, new_psi, sel_a, sel_b, sel_v)
            new_diags = zeros(Float64, new_basis.dim)
            t6 = @elapsed get_diags_bitstr!(new_basis, ham_otf, new_diags)

            n = new_basis.dim
            hvec_fn = (v, Hv) -> hvec_sci_full_bitstr!(new_basis, ham_otf, v, Hv)
            E, psi_new = davidson(hvec_fn, new_psi, new_diags;
                                  tol=davidson_tol, ncv=1, maxspace=min(max_size, n+20),
                                  verbose=false)
            verbose && @printf("E=%.10f  err=%.1e\n", E, abs(E - mole.e_scale))

            destroy_sci_basis_manager_bitstr(tgt)
            destroy_sci_basis_manager_bitstr(basis)
            basis, psi, diags = new_basis, psi_new, new_diags
            current_energy = E

            @printf("expand_bitstrings_bitstr:      %.2e seconds\n", t1)
            @printf("sci_hvec_select:               %.2e seconds\n", t2)
            @printf("merge_bitstrings:              %.2e seconds\n", t4)
            @printf("remap_wavefunction:            %.2e seconds\n", t5)
            @printf("get_diags:                     %.2e seconds\n", t6)
        end
        verbose && @printf("  iter %d time=%.2f s\n", iter, t_iter)
    end
    if debug_compare_fci && fci_basis.ptr != C_NULL
        @ccall LIB_BASIS.destroy_basis_manager(fci_basis.ptr::Ptr{Cvoid})::Cvoid
        fci_basis.ptr = C_NULL
    end
    return basis, psi, diags
end
