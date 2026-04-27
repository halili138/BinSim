import numpy
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from openfermion import FermionOperator, QubitOperator, jordan_wigner, normal_ordered

def _process_1body_chunk(chunk_indices, _1_body_mo):
    ham_ferm_1 = FermionOperator()

    for (p, q) in chunk_indices:
        c = _1_body_mo[p, q]
        pa = 2 * int(p)
        qa = 2 * int(q)
        pb = 2 * int(p) + 1
        qb = 2 * int(q) + 1
        ham_ferm_1 += FermionOperator(((pa, 1), (qa, 0)), c)
        ham_ferm_1 += FermionOperator(((pb, 1), (qb, 0)), c)

    ham_ferm_1 = normal_ordered(ham_ferm_1)

    return jordan_wigner(ham_ferm_1)


def _process_2body_chunk(chunk_indices, _2_body_mo):
    ham_ferm_2 = FermionOperator()

    for (p, q, r, s) in chunk_indices:
        c = _2_body_mo[p, q, r, s] * 0.5
        pa = 2 * int(p)
        qa = 2 * int(q)
        ra = 2 * int(r)
        sa = 2 * int(s)
        pb = 2 * int(p) + 1
        qb = 2 * int(q) + 1
        rb = 2 * int(r) + 1
        sb = 2 * int(s) + 1
        ham_ferm_2 += FermionOperator(((pa, 1), (qa, 1), (ra, 0), (sa, 0)), c)
        ham_ferm_2 += FermionOperator(((pb, 1), (qb, 1), (rb, 0), (sb, 0)), c)
        ham_ferm_2 += FermionOperator(((pa, 1), (qb, 1), (rb, 0), (sa, 0)), c)
        ham_ferm_2 += FermionOperator(((pb, 1), (qa, 1), (ra, 0), (sb, 0)), c)
    
    ham_ferm_2 = normal_ordered(ham_ferm_2)

    return jordan_wigner(ham_ferm_2)


def tree_merge_operators(op_list):
    """分治（树状）归并 QubitOperator 列表，避免超级大字典的单点追加瓶颈"""
    if not op_list:
        return QubitOperator()
    
    # 只要列表中还有多于1个元素，就继续两两配对相加
    while len(op_list) > 1:
        next_level = []
        # 步长为2进行遍历，两两归并
        for i in range(0, len(op_list), 2):
            if i + 1 < len(op_list):
                next_level.append(op_list[i] + op_list[i+1])
            else:
                # 落单的直接放入下一轮
                next_level.append(op_list[i])
        op_list = next_level
        
    return op_list[0]


def int2ham_parallel(energy_nuc, _1_body_mo, _2_body_mo, eps=1e-12, n_processes=None):
    t1 = datetime.datetime.now()

    if n_processes is None:
        n_processes = cpu_count()
    
    indices_1body = list(zip(*((abs(_1_body_mo) > eps).nonzero())))
    indices_2body = list(zip(*((abs(_2_body_mo) > eps).nonzero())))
    
    chunks_1body = [indices_1body[i::n_processes] for i in range(n_processes)]
    chunks_2body = [indices_2body[i::n_processes] for i in range(n_processes)]
    
    # 过滤掉可能的空列表（当非零元素总数少于进程数时可能发生）
    chunks_1body = [c for c in chunks_1body if c]
    chunks_2body = [c for c in chunks_2body if c]

    t2 = datetime.datetime.now()

    with Pool(processes=n_processes) as pool:
        func_1body = partial(_process_1body_chunk, _1_body_mo=_1_body_mo)
        results_1body = pool.map(func_1body, chunks_1body)

        t3 = datetime.datetime.now()
        
        ham_qubit_1 = QubitOperator()
        for result in results_1body:
            ham_qubit_1 += result

        t4 = datetime.datetime.now()
        
    with Pool(processes=n_processes) as pool:
        func_2body = partial(_process_2body_chunk, _2_body_mo=_2_body_mo)
        results_2body = pool.map(func_2body, chunks_2body)

        t5 = datetime.datetime.now()
        
        ham_qubit_2 = QubitOperator()
        for result in results_2body:
            ham_qubit_2 += result

        t6 = datetime.datetime.now()
        
    ham_qubit  = energy_nuc
    ham_qubit += ham_qubit_1
    ham_qubit += ham_qubit_2

    t7 = datetime.datetime.now()
    t_tol  = (t7 - t1).total_seconds()
    
    print(f"n_orb:        {len(_1_body_mo)}")
    print(f"n_proces:     {n_processes}")
    print(f"Pre chunk:    {(t2 - t1).total_seconds()}")
    print(f"Map _1_mo:    {(t3 - t2).total_seconds()}")
    print(f"Sum _1_mo:    {(t4 - t3).total_seconds()}")
    print(f"Map _2_mo:    {(t5 - t4).total_seconds()}")
    print(f"Sum _2_mo:    {(t6 - t5).total_seconds()}")
    print(f"Sum:          {(t7 - t6).total_seconds()}")
    print(f"Total:        {t_tol}")
    
    return t_tol


def generate_physical_random_integrals_vectorized(norb, seed=42):
    if seed is not None:
        numpy.random.seed(seed)
        
    # ==========================================
    # 1. 生成单体积分 (1-body MO integrals)
    # ==========================================
    h1_random = numpy.random.randn(norb, norb)
    h1_mo = 0.5 * (h1_random + h1_random.T)
    numpy.fill_diagonal(h1_mo, h1_mo.diagonal() - numpy.abs(numpy.random.randn(norb)) * 5)

    # ==========================================
    # 2. 生成双体积分 (2-body MO integrals)
    # ==========================================
    n_pairs = norb * (norb + 1) // 2
    rank = max(1, n_pairs // 2) 
    L = numpy.random.randn(n_pairs, rank)
    V_IJ = L @ L.T  # 半正定实对称矩阵

    # === 向量化消除 4 重 for 循环 ===
    
    # 1. 生成 p 和 q 的坐标网格 (维度: norb x norb)
    p, q = numpy.indices((norb, norb))
    
    # 2. 计算 2D 索引映射矩阵 idx_map
    # 对应之前的 pair_idx 逻辑: max(p,q)*(max(p,q)+1)//2 + min(p,q)
    idx_map = numpy.maximum(p, q) * (numpy.maximum(p, q) + 1) // 2 + numpy.minimum(p, q)
    
    # 3. 利用 NumPy 广播机制，直接映射成 4D 化学家记号张量 V_chem
    # idx_map[:, :, None, None] 将形状扩展为 (norb, norb, 1, 1)
    # idx_map[None, None, :, :] 将形状扩展为 (1, 1, norb, norb)
    # 这样高级索引会自动广播生成 (norb, norb, norb, norb) 的结果
    V_chem = V_IJ[idx_map[:, :, None, None], idx_map[None, None, :, :]]
    
    # 4. 转换为物理学家记号 <pq|rs> = (pr|qs)
    # 原逻辑: h2_mo[p, q, r, s] = V_chem[p, r, q, s]
    # 在 NumPy 中，这等价于交换第 1 和第 2 轴（0-indexed）
    h2_mo = V_chem.transpose(0, 2, 1, 3)

    return h1_mo, h2_mo


if __name__ == "__main__":
    h1_mo, h2_mo = generate_physical_random_integrals_vectorized(20, seed=None)
    int2ham_parallel(
        1.0, h1_mo, h2_mo, 1e-12, 12
    )


