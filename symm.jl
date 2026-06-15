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


function find_kernel!(E::Matrix{Bool})
    m, n = size(E)
    pivot_cols = zeros(Int, m)

    row = 1
    for col = 1:n
        pivot_row = findfirst(r -> E[r, col], row:m)

        if !isnothing(pivot_row)
            pivot_row += row - 1
            pivot_cols[row] = col

            E[[row, pivot_row], :] .= E[[pivot_row, row], :]

            for r in [1:row-1; row+1:m]
                if E[r, col]
                    E[r, :] .⊻= E[row, :]
                end
            end

            row += 1
        end
    end

    return E[1:row-1, :], pivot_cols[1:row-1]
end


function get_kernel_basis(A::BinaryQubitABAB{Ti,Tv,K,V}, nq::Int) where {Ti,Tv,K,V}
    xs  = A.xs
    zs  = A.zs
    ncs = length(A.xs)

    nblocks    = cld(ncs, 2048)
    chunk_size = cld(ncs, nblocks)
    Es = Vector{Matrix{Bool}}(undef, nblocks)

    @threads for i in 1:nblocks
        l = (i - 1) * chunk_size + 1
        r = min(i * chunk_size, ncs)
        ncs_local = r - l + 1
        E_local = Es[i] = zeros(Bool, ncs_local, 2 * nq)
        @views xs_local = xs[l:r]
        @views zs_local = zs[l:r]
        for j in eachindex(xs_local)
            x = xs_local[j]
            z = zs_local[j]
            for k in 0:nq-1
                mask = Ti(1) << k
                if x & mask != 0
                    E_local[j, nq+k+1] = true
                end
                if z & mask != 0
                    E_local[j, k+1] = true
                end
            end
        end
    end

    while length(Es) > 1
        n = length(Es)
        new_Es = Vector{Matrix{Bool}}(undef, (n + 1) ÷ 2)
        @threads for i in 1:2:n-1
            E1, _ = find_kernel!(Es[i])
            E2, _ = find_kernel!(Es[i+1])
            new_Es[(i+1)÷2] = vcat(E1, E2)
        end

        if n % 2 == 1
            Eend, _ = find_kernel!(Es[end])
            new_Es[end] = Eend
        end

        Es = new_Es
    end

    final_E, final_pivot_cols = find_kernel!(Es[1])
    m, n = size(final_E)
    free_vars = setdiff(1:n, final_pivot_cols)
    kernel_basis = Vector{Bool}[]

    for j in free_vars
        v = zeros(Bool, n)
        v[j] = true
        for i in 1:m
            if final_E[i, j]
                v[final_pivot_cols[i]] = true
            end
        end
        push!(kernel_basis, v)
    end

    return kernel_basis
end


function symplectic_gram_schmidt(kernel_basis::Vector{Vector{Bool}})
    generators = Vector{Bool}[]

    length(kernel_basis) == 0 && return generators

    nq = length(kernel_basis[1]) ÷ 2

    for i in eachindex(kernel_basis)
        vi = copy(kernel_basis[i])
        while true
            found_opposite = false
            for j in eachindex(generators)
                vj = generators[j]
                ip = false
                for k in 1:nq
                    ip ⊻= (vi[k] & vj[k+nq])
                    ip ⊻= (vi[k+nq] & vj[k])
                end
                if ip
                    vi .⊻= vj
                    found_opposite = true
                    break
                end
            end

            if !found_opposite && any(vi)
                push!(generators, vi)
                break
            end

            if all(.!vi)
                break
            end
        end
    end

    return generators
end


function to_BinaryQubitABAB(generator::Vector{Bool}, Ti::Type, Tv::Type)
    nq = length(generator) ÷ 2
    x  = Ti(0)
    z  = Ti(0)
    for i in 1:nq
        if generator[i]
            x |= Ti(1) << (i - 1)
        end
        if generator[i+nq]
            z |= Ti(1) << (i - 1)
        end
    end

    Y = x & z
    c = im ^ (count_ones(Y))

    return BinaryQubitABAB(Ti[x], Ti[z], Tv[c])
end


function get_orbsym(A::BinaryQubitABAB{Ti,Tv,K,V}, norb::Int) where {Ti,Tv,K,V}
    nq = norb * 2
    E_kernel = get_kernel_basis(A, nq)
    generators = symplectic_gram_schmidt(E_kernel)
    generator_bops = to_BinaryQubitABAB.(generators, Ti, Tv)

    orbsym = zeros(Int, norb)

    for i in 1:norb
        spin_up_qubit = 2 * (i - 1)
        mask = 0
        for j in eachindex(generator_bops)
            Gj = generator_bops[j]
            z2 = Gj.zs[1]
            if (z2 & (Ti(1) << spin_up_qubit)) != 0
                mask |= 1 << (j - 1)
            end
        end
        orbsym[i] = mask
    end

    return orbsym
end

