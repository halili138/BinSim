function cal_diff_excit_order_terms(
    occ_idxs::AbstractArray, 
    vir_idxs::AbstractArray, 
    kmax::Int,
    generator::Function, 
    T::Type,
)
    diffk_occ_terms = Array{Array{T,1},1}(undef, kmax+1)
    diffk_vir_terms = Array{Array{T,1},1}(undef, kmax+1)

    for k in 0:kmax
        diffk_occ_terms[k+1] = generator.(combinations(occ_idxs, k))
        diffk_vir_terms[k+1] = generator.(combinations(vir_idxs, k))
    end

    return diffk_occ_terms, diffk_vir_terms
end


function flatten_diff_excit_order_terms(
    kmax::Int,
    diffk_occ_terms::Array{Array{T,1},1},
    diffk_vir_terms::Array{Array{T,1},1},
) where {T}
    len = 0
    for k in 0:kmax
        len += length(diffk_occ_terms[k+1]) * length(diffk_vir_terms[k+1])
    end

    k_map = Array{Int,1}(undef, len)
    occ_terms_map = Array{T,1}(undef, len)
    vir_terms_map = Array{T,1}(undef, len)

    count = 0
    for k in 0:kmax
        occ_terms::Array{T,1} = diffk_occ_terms[k+1]
        vir_terms::Array{T,1} = diffk_vir_terms[k+1]
        for occ_term in occ_terms
            for vir_term in vir_terms
                count += 1
                k_map[count] = k
                occ_terms_map[count] = occ_term
                vir_terms_map[count] = vir_term
            end
        end
    end

    return k_map, occ_terms_map, vir_terms_map
end


function merge_results(results::Array{Array{T,1},1}) where {T}
    while length(results) > 1
        n = length(results)
        temp_results = Array{Array{T,1},1}(undef, (n+1)÷2)
        @threads for i in 1:2:n-1
            temp_results[(i+1)÷2] = merge_two(results[i], results[i+1])
        end
        (n%2 == 1) && (temp_results[end] = results[end])
        results = temp_results
    end

    return results[1]
end


function generate_ci_spin_orbitals(
    norb::Int, 
    nelec::Tuple{Int,Int}, 
    orbsym::Array{Int,1}; 
    kmax::Int=2, 
    generalize::Bool=false,
)
    function symm_generator(comb)
        sym::Int = 0
        for pos in comb
            sym ⊻= orbsym[pos+1]
        end

        return sym
    end

    na, nb = nelec

    if generalize
        occ_a_idxs = 0:norb-1
        vir_a_idxs = 0:norb-1
        occ_b_idxs = 0:norb-1
        vir_b_idxs = 0:norb-1
    else
        occ_a_idxs = 0:na-1
        vir_a_idxs = na:norb-1
        occ_b_idxs = 0:nb-1
        vir_b_idxs = nb:norb-1
    end

    To::Type = Array{UInt16,1}

    occ_aorbs_vecs, vir_aorbs_vecs = cal_diff_excit_order_terms(occ_a_idxs, vir_a_idxs, min(na, kmax), comb -> comb .* 2, To) 
    occ_asyms_vecs, vir_asyms_vecs = cal_diff_excit_order_terms(occ_a_idxs, vir_a_idxs, min(na, kmax), symm_generator, Int)
    occ_borbs_vecs, vir_borbs_vecs = cal_diff_excit_order_terms(occ_b_idxs, vir_b_idxs, min(nb, kmax), comb -> comb .* 2 .+ 1, To) 
    occ_bsyms_vecs, vir_bsyms_vecs = cal_diff_excit_order_terms(occ_b_idxs, vir_b_idxs, min(nb, kmax), symm_generator, Int) 

    ka_map, occ_aorb_map, vir_aorb_map = flatten_diff_excit_order_terms(min(na, kmax), occ_aorbs_vecs, vir_aorbs_vecs)
    ka_map, occ_asym_map, vir_asym_map = flatten_diff_excit_order_terms(min(na, kmax), occ_asyms_vecs, vir_asyms_vecs)
    
    len      = length(ka_map)
    BS       = nthreads()
    GS       = cld(len, BS)
    results  = Array{Array{To,1},1}(undef, BS)
    zsym     = 0

    @threads for tid in 1:BS
        lb = (tid - 1) * GS + 1
        rb = min(tid * GS, len)

        count = 0
        for pq in lb:rb
            ka::Int       = ka_map[pq]
            occ_asym::Int = occ_asym_map[pq]
            vir_asym::Int = vir_asym_map[pq]
            for kb in 0:min(nb, kmax - ka)
                idx = kb + 1
                occ_bsyms = occ_bsyms_vecs[idx]
                vir_bsyms = vir_bsyms_vecs[idx]
                for occ_bsym in occ_bsyms, vir_bsym in vir_bsyms
                    if occ_asym ⊻ vir_asym ⊻ occ_bsym ⊻ vir_bsym == zsym
                        count += 1
                    end
                end
            end
        end

        results_local = Array{To,1}(undef, count)

        count = 0
        for pq in lb:rb
            ka::Int       = ka_map[pq]
            occ_aorb::To  = occ_aorb_map[pq]
            vir_aorb::To  = vir_aorb_map[pq]
            occ_asym::Int = occ_asym_map[pq]
            vir_asym::Int = vir_asym_map[pq]
            for kb in 0:min(nb, kmax - ka)
                idx = kb + 1
                occ_borbs = occ_borbs_vecs[idx]
                vir_borbs = vir_borbs_vecs[idx]
                occ_bsyms = occ_bsyms_vecs[idx]
                vir_bsyms = vir_bsyms_vecs[idx]
                for (occ_borb, occ_bsym) in zip(occ_borbs, occ_bsyms)
                    for (vir_borb, vir_bsym) in zip(vir_borbs, vir_bsyms)
                        if occ_asym ⊻ vir_asym ⊻ occ_bsym ⊻ vir_bsym == zsym
                            count += 1
                            results_local[count] = vcat(vir_aorb, vir_borb, occ_aorb, occ_borb)
                        end
                    end
                end
            end
        end
        results[tid] = results_local
    end

    result = reduce(vcat, results)

    popfirst!(result)

    return result
end


mutable struct Orbitals
    spatial_based::Array{Array{UInt16,1},1}
    spin_based::Array{Array{UInt16,1},1}

    function Orbitals()
        new(Array{Array{UInt16,1},1}(), Array{Array{UInt16,1},1}())
    end
end


function kernel(
    info::SysInfo, orbitals::Orbitals; 
    excited_order::Int=2, symm_reduce::Bool=true, generalize::Bool=false,
)
    if symm_reduce
        spin_orbitals = generate_ci_spin_orbitals(
            info.norb, info.nelec, info.orbsym,
            kmax=excited_order, generalize=generalize,
        )
    else
        spin_orbitals = generate_ci_spin_orbitals(
            info.norb, info.nelec, ones(info.norb),
            kmax=excited_order, generalize=generalize,
        )
    end

    spin2spatial = x -> x .÷ 2
    spatial_orbitals = unique(spin2spatial.(spin_orbitals))

    orbitals.spatial_based = spatial_orbitals
    orbitals.spin_based = spin_orbitals
end


function unique_operator_pool(pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}) where {Ti,Tv,K,V}
    len = length(pool)
    index_map = Dict{eltype(pool),Int}()
    sizehint!(index_map, len)
    unique_indices = Array{Int,1}(undef, len)

    To = eltype(pool)

    count = 0
    for (i, op) in enumerate(pool)
        (op == zero(To) || op == one(To)) && continue
        if !haskey(index_map, op)
            count += 1
            index_map[op] = count
            unique_indices[count] = i
        end
    end

    resize!(unique_indices, count)

    return pool[unique_indices]
end


function spin_orbital_to_generator_terms(spin_orbital::Array{UInt16,1})
    norb = length(spin_orbital)
    mid = norb ÷ 2

    cre_terms = [(Int(i), 1) for i in spin_orbital[1:mid]]
    ann_terms = [(Int(i), 0) for i in spin_orbital[mid+1:end]]

    return vcat(cre_terms, ann_terms)
end


function ucc_like(
    orbitals::Orbitals, generator::Function, complete::Bool, Ti::DataType, Tv::DataType
)
    spin_orbitals = orbitals.spin_based
    len = length(spin_orbitals)

    f = term -> begin
        t = generator(term, 1.0, Ti, Tv)
        return t - t'
    end

    pool = Vector{BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}}(undef, len)
    for i in eachindex(spin_orbitals)
        pool[i] = f(spin_orbital_to_generator_terms(spin_orbitals[i]))
    end

    if complete
        f = term -> begin
            t = generator(term, 1.0, Ti, Tv)
            return im * (t + t')
        end

        c_pool = Vector{BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}}(undef, len)
        for i in eachindex(spin_orbitals)
            c_pool[i] = f(spin_orbital_to_generator_terms(spin_orbitals[i]))
        end

        pool = vcat(pool, c_pool)
    end

    pool = unique_operator_pool(pool)

    # println("Size of operator pool: $(length(pool))\n")

    return pool
end


function FEB(orbitals::Orbitals; 
    Ti::DataType=UInt32, Tv::DataType=Float64, complete::Bool=false,
)
    println("Generate Fermion Based Excitation Operator Pool")
    return ucc_like(orbitals, FermionOperatorAABB, complete, Ti, Tv)
end


function QEB(orbitals::Orbitals;
    Ti::DataType=UInt32, Tv::DataType=Float64, complete::Bool=false,
)
    println("Generate Qubit Based Excitation Operator Pool")
    return ucc_like(orbitals, QebOperatorAABB, complete, Ti, Tv)
end


