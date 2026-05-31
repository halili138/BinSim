function get_symm(str::Ti, orbsym::Vector{Int64}) where {Ti}
    i = 0
    sym = 0
    while str != 0 
        if str & 1 != 0
            sym ⊻= orbsym[i+1]
        end
        str >>= 1
        i += 1
    end

    return sym
end


function cal_strs_syms(norb::Int, ne::Int, orbsym::Vector{Int64}, Ti::Type)    
    f_str = (str, pos) -> str | (Ti(1) << pos)
    f_sym = (sym, pos) -> sym ⊻ orbsym[pos+1] # pos 是 0based 索引, orbsym 是 1based 向量

    strs = Array{Ti, 1}(undef, binomial(norb, ne))
    syms = Array{Int,1}(undef, binomial(norb, ne))

    count = 0
    for comb in combinations([i for i in 0:norb-1], ne)
        count += 1
        strs[count] = foldl(f_str, comb; init=Ti(0))
        syms[count] = foldl(f_sym, comb; init=0)
    end

    dict = Dict{Int, Array{Ti,1}}()
    for (sym, str) in zip(syms, strs)
        index = Base.ht_keyindex2!(dict, sym)
        if index > 0
            @inbounds push!(dict.vals[index], str)
        else
            @inbounds Base._setindex!(dict, Ti[str], sym, -index)
        end
    end

    return dict
end


struct BlockDesc{Ti,Tc,Tp}
    asym::Tc
    bsym::Tc
    num_a::Tc
    num_b::Tc
    astrs::Array{Ti,1}
    bstrs::Array{Ti,1}
    offset::Tp
end


function get_sym_blocks(norb::Int, nelec::Tuple{Int,Int}, orbsym::Vector{Int64}, total_sym::Int, Ti::Type)
    na, nb = nelec
    adict = cal_strs_syms(norb, na, orbsym, Ti)
    bdict = cal_strs_syms(norb, nb, orbsym, Ti)

    blocks = BlockDesc{Ti, Int, Int}[]
    current_offset = 0

    max_asym = 0
    max_bsym = 0
    current_offset = 0

    for (asym, astrs) in adict
        bsym = total_sym ⊻ asym
        if haskey(bdict, bsym)
            bstrs = bdict[bsym]

            sort!(astrs)
            sort!(bstrs)
            
            num_a = length(astrs)
            num_b = length(bstrs)
            
            push!(blocks, BlockDesc(asym, bsym, num_a, num_b, astrs, bstrs, current_offset))
            
            current_offset += num_a * num_b
            (asym > max_asym) && (max_asym = asym)
            (bsym > max_bsym) && (max_bsym = bsym)
        end
    end

    block_map = fill(-1, max_asym+1, max_bsym+1)

    for i in eachindex(blocks)
        B = blocks[i]
        asym = B.asym 
        bsym = B.bsym
        block_map[asym+1,bsym+1] = i - 1
    end

    return blocks, block_map
end

