# ============================================================
# symm.jl —— 由哈密顿量 (BinaryQubitABAB) 发现轨道对称性标签。
#
# 从 binsim-v1.10/jl/symm.jl 精简而来, 只保留 Hamiltonian 对称性发现链路
#   get_symm (取自 v1.10 jl/basis.jl)
#   find_kernel! -> get_kernel_basis -> symplectic_gram_schmidt
#   -> to_BinaryQubitABAB -> get_orbsym
#
# 已去掉 (本项目不需要):
#   - 虚拟对称性分区机制 (VirtualSymmetryPartition / make_virtual_orbsym /
#     count_strings_by_combined_sym / get_block_string_count_extrema /
#     combine_orbsym / next_combination_uint): 属已删除的分布式 SCI 路径。
#   - 纯 Julia 基组块构造 (BlockDesc / cal_strs_syms / get_sym_blocks):
#     本项目基组由 src 的 libbasis.so 直接构建。
#   - liborbsym.so 的 C++ 后端 (get_orbsym_c 及 UInt64 专用方法):
#     本项目不构建 liborbsym.so, 统一走 Julia 实现。
#   - orbsym 到标准 D2h 标签的 GF2 仿射对齐 (learn_affine_gf2 /
#     apply_affine_gf2 / align_ham_orbsym 等): 原仅供上游对齐测试使用。
#
# 注: 字符串对称性 get_symm 在 v1.10 中由 jl/basis.jl 提供; 本项目未引入
# basis.jl, 故将其一并放入本文件, 作为最基础的 XOR 标签工具。
# ============================================================


# 字符串 (比特串) 的 orbsym 标签: 对每个占据轨道的标签做 XOR。
# 例: 闭壳层/行列式的 Γ = get_symm(stra, orbsym) ⊻ get_symm(strb, orbsym)。
function get_symm(str::Ti, orbsym::Vector{Int}) where {Ti}
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
        # 单遍消去与已有生成元的交换积。vj 加入时已与此前所有 vk (k<j) 正交,
        # 故与 vj 异或不会破坏已消去的 vk, 一遍即可收敛。
        # 注: 上游 binsim-v1.10 此处为 `while true` 反复重扫, 当 <vi,vj>=1 时
        #     异或 vj 后该条件仍成立 (因 <vj,vj>=0), 会无限循环 (已实测),
        #     故本副本改为单遍; 对能正常终止的输入结果与上游一致。
        for j in eachindex(generators)
            vj = generators[j]
            ip = false
            for k in 1:nq
                ip ⊻= (vi[k] & vj[k+nq])
                ip ⊻= (vi[k+nq] & vj[k])
            end
            if ip
                vi .⊻= vj
            end
        end

        any(vi) && push!(generators, vi)
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


function _get_orbsym_julia(A::BinaryQubitABAB{Ti,Tv,K,V}, norb::Int) where {Ti,Tv,K,V}
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


# 本项目无 liborbsym.so, get_orbsym 统一走 Julia 实现 (含 UInt64/UInt128 等所有 Ti)。
function get_orbsym(H::BinaryQubitABAB{Ti,Tv,K,V}, norb::Int) where {Ti,Tv,K,V}
    return _get_orbsym_julia(H, norb)
end
