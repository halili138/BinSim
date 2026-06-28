from openfermion import (FermionOperator, jordan_wigner, hermitian_conjugated,
                         qubit_operator_sparse, QubitOperator,get_sparse_operator)
import numpy as np
from numpy.linalg import eig, inv
from cmath import sqrt
from scipy.linalg import expm
from scipy.sparse.linalg import expm as spexpm
from scipy.sparse import csc_array,csc_matrix
import json
import opt_einsum as oe
import time
from scipy import sparse

def LeftVacuum(N):
    """
    构建 LeftVacuum 态叠加向量。
    
    参数:
        N (int): 子系统粒子数/格点数
    
    返回:
        superposition (scipy.sparse.csc_matrix): 稀疏列向量，维度为 (2^(2N), 1)
    """
    total_states = 2 ** N
    total_dim = 2 ** (2 * N)
    
    # 预分配列表以存储行索引和数据值
    # 长度为 total_states，因为每个 i 对应一个非零元素
    row_indices = np.zeros(total_states, dtype=np.int64)
    values = np.zeros(total_states, dtype=np.complex128)
    
    for i in range(total_states):
        # 1. 计算相位 phase = (-1i) ^ (sum(dec2bin(i)) % 2)
        # MATLAB: sum(dec2bin(i)) 计算的是二进制字符串中 '1' 的个数 (因为 '0' 是 48, '1' 是 49, sum 后 mod 2 等价于 count('1') mod 2)
        # Python: bin(i).count('1') 直接统计 '1' 的个数
        ones_count = bin(i).count('1')
        
        if ones_count % 2 == 0:
            phase = 1.0 + 0.0j
        else:
            phase = -1.0j # (-1j)^1 = -1j
            
        # 2. 构建完整状态索引: (i << N) | i
        # MATLAB: bitshift(i, N) -> i << N
        # MATLAB: bitor(..., ...) -> |
        full_row = (i << N) | i
        
        # 3. 存储数据
        # MATLAB 索引从 1 开始，所以有 +1。Python 从 0 开始，直接用 full_row
        row_indices[i] = full_row
        values[i] = phase
        
    # 4. 构建稀疏向量
    # 格式：COO (Coordinate) -> 转换为 CSC (Compressed Sparse Column) 以便后续计算
    # 行索引: row_indices
    # 列索引: 全是 0 (因为是列向量)
    # 数据: values
    # 形状: (total_dim, 1)
    col_indices = np.zeros(total_states, dtype=np.int64)
    
    superposition = sparse.coo_matrix((values, (row_indices, col_indices)), 
                                      shape=(total_dim, 1)).tocsc()
    
    return superposition

def generate_same_hamming(N, diagonal_only=False):
    
    # 1. 生成所有单粒子态 0 到 2^N - 1
    all_states = np.arange(2 ** N, dtype=np.int64)
    
    # 2. 快速计算所有状态的汉明权重 (Python 3.8+ 支持 int.bit_count)
    # 如果 python 版本低，可用 bin(x).count('1') 代替，但稍慢
    weights = np.array([x.bit_count() for x in all_states], dtype=np.int8)
    
    row_indices_list = []
    
    # 3. 按权重分组处理 (桶排序)
    # 权重范围是 0 到 N
    for k in range(N + 1):
        # 找出所有权重为 k 的状态索引 (布尔掩码)
        mask = (weights == k)
        states_k = all_states[mask]
        
        if len(states_k) == 0:
            continue
            
        if diagonal_only:
            # 对角模式: m = n
            # 索引: (m << N) | m
            # 向量化操作
            idxs = (states_k << N) | states_k
            row_indices_list.append(idxs)
        else:
            # 块状模式: 所有 m 和所有 n 配对
            # 利用 NumPy 广播:
            # states_k[:, None] 形状 (M, 1)
            # states_k[None, :] 形状 (1, M)
            # 结果形状 (M, M)
            
            m_col = states_k[:, np.newaxis]  # 列向量
            n_row = states_k[np.newaxis, :]  # 行向量
            
            # 位运算构建索引矩阵: (m << N) | n
            idx_matrix = (m_col << N) | n_row
            
            # 展平为一维数组
            row_indices_list.append(idx_matrix.ravel())

    # 4. 合并所有索引
    if not row_indices_list:
        # 空情况 (理论上不会发生，除非 N<0)
        return sparse.csc_matrix((2**(2*N), 1), dtype=np.complex128)
        
    final_indices = np.concatenate(row_indices_list)
    
    # 5. 构建稀疏向量
    total_dim = 2 ** (2 * N)
    values = np.ones_like(final_indices, dtype=np.complex128)
    col_indices = np.zeros_like(final_indices, dtype=np.int64)
    
    # COO 格式构建后转为 CSC (适合线性代数运算)
    superposition = sparse.coo_matrix(
        (values, (final_indices, col_indices)), 
        shape=(total_dim, 1)
    ).tocsc()
    
    return superposition

# --- 1. 辅助函数定义 ---
def fd(E, mu, T):
    """费米 - 狄拉克分布"""
    if T == 0:
        return (E < mu).astype(float)
    # 使用 expit 或者手动计算，注意处理溢出
    arg = (E - mu) / T
    # 防止 exp 过大溢出
    arg = np.clip(arg, -700, 700) 
    return 1.0 / (np.exp(arg) + 1.0)
def solve_negf_selfconsistent(NS, NL, U_val, T_val, ESMAX, EKMAX, uc, max_iter=10000, tol=1e-8):
    """
    求解非平衡格林函数自洽问题 (基于提供的 MATLAB 逻辑转换)
    
    参数:
        NS (int): 中心区域格点数
        NL (int): 左右电极格点数
        U_val (float): 偏压大小 (U)
        T_val (float): 温度 (T)
        ESMAX (float): 中心区域能量范围 [-ESMAX, ESMAX]
        EKMAX (float): 电极能量范围 [-EKMAX, EKMAX]
        uc (float): 相互作用强度
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差
        
    返回:
        ni (np.ndarray): 收敛后的中心区域电子密度 (长度 NS)
        e (np.ndarray): 最终迭代的本征值
        v (np.ndarray): 最终迭代的右本征向量
        v2 (np.ndarray): 最终迭代的左本征向量 (v 的逆)
        info (dict): 包含是否收敛、迭代次数等信息的字典
    """
    
    

    # --- 2. 参数初始化 ---
    n = NS + 2 * NL
    
    UL = U_val / 2
    UR = -U_val / 2
    ts=EKMAX/(2**NS*1.0)
    DE = 2 * EKMAX
    TL = T_val
    TR = T_val
    
    # 避免除以零
    if NL <= 1:
        raise ValueError("NL 必须大于 1 以计算 de 和 yita")
        
    de = DE / (NL - 1)
    yita = (NL - 1) / DE
    gama = 2 * de
    t_val = np.sqrt(1 / (2 * np.pi * yita))

    # 生成能量网格
    ES = np.linspace(-ESMAX, ESMAX, NS)
    EK = np.linspace(-EKMAX, EKMAX, NL)

    # --- 3. 计算费米分布与 Gamma 矩阵 (这些在迭代中不变) ---
    fl2 = fd(EK, UL, TL)
    fl1 = 1 - fl2
    fr2 = fd(EK, UR, TR)
    fr1 = 1 - fr2

    zeros_NS = np.zeros(NS)
    
    # Gammap
    gammap_diag = gama * np.concatenate([zeros_NS, fl2, fr2])
    Gammap = np.diag(gammap_diag.astype(np.complex128)) # 显式转为复数

    # Gammam
    gammam_diag = gama * np.concatenate([zeros_NS, fl1, fr1])
    Gammam = np.diag(gammam_diag.astype(np.complex128))

    Omega = Gammam - Gammap

    # 预计算 H 的静态部分 (不含 U)
    H_top_left = np.diag(ES)+np.diag(-ts * np.ones(NS - 1), -1) +np.diag(-ts * np.ones(NS - 1), 1)
    H_top_right = -t_val * np.ones((NS, NL * 2))
    H_bottom_left = -t_val * np.ones((NL * 2, NS))
    H_bottom_right = np.diag(np.concatenate([EK, EK]))
    
    H_static = np.block([
        [H_top_left, H_top_right],
        [H_bottom_left, H_bottom_right]
    ]).astype(np.complex128)

    # --- 4. 初始化电子密度 ni ---
    ni = 0.1 * np.ones(NS, dtype=np.complex128)
    ni0 = ni.copy()
    
    # 预定义全零部分用于 U 矩阵构建 (L 和 R 部分始终为 0)
    diag_elements_L = np.zeros(NL, dtype=np.complex128)
    diag_elements_R = np.zeros(NL, dtype=np.complex128)

    converged = False
    final_iter = 0

    # --- 5. 自洽迭代循环 ---
    for itera in range(max_iter):
        # 构建当前的 U 矩阵
        if NS > 1:
            # ni[1:] (去掉第一个) + [0]
            term1 = np.concatenate([ni[1:], [0]])
            # [0] + ni[:-1] (去掉最后一个)
            term2 = np.concatenate([[0], ni[:-1]])
            
            diag_elements_S = uc * (term1 + term2)
        else:
            # 如果 NS=1, 逻辑可能需要调整，原代码 if NS>1 没覆盖 NS=1 的情况
            # 原代码中如果 NS=1, U 保持为 0 (标量)，但在矩阵加法中会广播或报错。
            # 这里假设 NS>=2 符合物理意义，若 NS=1 则 S 部分为 0
            diag_elements_S = np.zeros(1, dtype=np.complex128)
            
        full_diag = np.concatenate([diag_elements_S, diag_elements_L, diag_elements_R])
        U_mat = np.diag(full_diag)

        # 构建总矩阵 mt
        # mt = [ H + U - i*Omega,   2*Gammap ]
        #      [ -2*Gammam,         H + U + i*Omega ]
        
        H_U = H_static + U_mat
        
        top_left = H_U - 1j * Omega
        top_right = 2 * Gammap
        bottom_left = -2 * Gammam
        bottom_right = H_U + 1j * Omega

        mt = np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])

        # 对角化
        e, v = np.linalg.eig(mt)
        
        # 计算逆矩阵 v2 = v^-1
        # 注意：如果 v 接近奇异，inv 可能会不稳定，但在该物理模型中通常可逆
        v2 = np.linalg.inv(v)

        # 构建 des 矩阵 (投影算子)
        # des[i,i] = 1 if Im(e[i]) > 0 else 0
        des = np.zeros((2 * n, 2 * n), dtype=np.complex128)
        mask = e.imag > 0
        des[np.arange(2 * n)[mask], np.arange(2 * n)[mask]] = 1.0

        # 计算新的电子密度
        # nnn = real(diag(v2.T @ des @ v.T))
        # 注意：MATLAB 中的 ' 是共轭转置，Python 中 .T 是转置，.conj().T 是共轭转置
        # 原代码：v2.T @ des @ v.T 
        # 在 MATLAB 中，如果 v 是复数，v' 是共轭转置。
        # 你的 Python 代码之前写的是 v.T (非共轭)。
        # 检查原 MATLAB 片段：v2=inv(v); ... v2.T @ des @ v.T (如果是直接翻译)
        # 但通常物理公式里涉及正交归一化时用共轭。
        # 依据你提供的 Python 代码片段：v2.T @ des @ v.T (没有 .conj())
        # 我们严格遵循你提供的 Python 逻辑：
        
        temp_matrix = v2.T @ des @ v.T
        nnn = np.real(np.diag(temp_matrix))
        
        # 提取中心区域密度
        ni0_new = nnn[:NS]
        
        # 检查收敛
        # 原代码：nierror = sum(ni - ni0) -> 这里应该是比较新旧差异
        # 原代码逻辑有点奇怪：它比较的是 ni (旧) 和 ni0 (上一步计算的?)
        # 让我们修正为标准自洽逻辑：比较 current_ni 和 new_ni
        ni_error = np.sum(np.abs(ni - ni0_new))
        
        # 更新 ni
        ni = ni0_new
        
        if ni_error < tol:
            converged = True
            final_iter = itera + 1
            break
            
    if not converged:
        final_iter = max_iter
        
    info = {
        "converged": converged,
        "iterations": final_iter,
        "error": ni_error if 'ni_error' in locals() else None
    }
    ind = np.argsort(e)
    e = e[ind]
    v = v[:, ind]
    for i in range(0,2*n,2):
        ind = np.argsort(e[i:i+2].imag)+i
        v[:,i:i+2]=v[:, ind]
        e[i:i+2]=e[ind]
    v2 = np.linalg.inv(v)
    des = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    mask = e.imag > 0
    des[np.arange(2 * n)[mask], np.arange(2 * n)[mask]] = 1.0
    D   =  v2.T @ des @ v.T
    return ni, e, v, v2,  D,t_val, info

class SiteQuasiPartcle():
    size = None
    ni = None
    v = None
    v2 = None
    e = None
    E = None
    ck_operator = [None]
    cd_operator = [None]
    tck_operator = [None]
    tcd_operator = [None]
    f0 = None
    R0 = [None]
    fun = None
    I0 = None
    l0 = None
    c1_operator=[None]
    c2_operator=[None]
    mt=None
    tsingle=None
    tdouble=None
    ttriple=None
    op_pool=[]
    exit_pool=[]
    nops=None
    d2=None
    d4=None
    d2_pool=None
    d4_pool=None
    d_operator=None
    q_operator=None
    def __init__(self,NS, NL, U_val, T_val, ESMAX, EKMAX, uc):
        self.NS     =NS       
        self.NL     =NL
        self.size   =NS+NL*2       
        self.U_val  =U_val    
        self.T_val  =T_val    
        self.ESMAX  =ESMAX    
        self.EKMAX  =EKMAX    
        self.uc     =uc
        DE     =2*EKMAX
        de     =DE/(NL-1);
        yita   =(NL-1)/DE;
        self.gama   =2*de;
        self.t_val  =sqrt(1/(2*np.pi*yita));       
        # 波函数f0设置为complex128类型
        self.f0 = np.zeros((2 **(2 * self.size), 1), dtype=np.complex128)
        self.f0[0] = 1  # 1会自动转换为complex128类型的1+0j
        st=time.time()
        self.form_qpoperator()
        fi=time.time()
        print(f'form_qpoperator:{fi-st}')

        st=time.time()
        self.form_fun_qoperator()
        fi=time.time()
        print(f'cal fun:{fi-st}')

        st=time.time()
        self.form_simple_ccsd_operator()
        fi=time.time()
        print(f'form ccsd:{fi-st}')

    def form_zero_hamiltonia(self):
        NS      =   self.NS
        NL      =   self.NL
        U_val   =   self.U_val
        T_val   =   self.T_val
        ESMAX   =   self.ESMAX
        EKMAX   =   self.EKMAX
        uc      =   self.uc
        ni_final, e_final, v_final, v2_final, D,t_val, info =solve_negf_selfconsistent(NS, NL, U_val, T_val, ESMAX, EKMAX, uc, max_iter=10000, tol=1e-8)
        self.v=v_final
        self.v2=v2_final
        self.e=e_final
        self.D=D
        self.ni=ni_final
    def form_fun_qoperator(self):
        NS      =   self.NS
        NL      =   self.NL
        U_val   =   self.U_val
        T_val   =   self.T_val
        ESMAX   =   self.ESMAX
        EKMAX   =   self.EKMAX
        ts=EKMAX/(2**NS)
        uc      =   self.uc
        gama    =self.gama
        n       =self.size
        ES = np.linspace(-ESMAX, ESMAX, NS)
        EK = np.linspace(-EKMAX, EKMAX, NL)
        EK=np.concatenate([EK,EK])
        E=np.concatenate([ES,EK])
        t_val=self.t_val
        ak = [jordan_wigner(FermionOperator((x, 0))) for x in range(n)]
        tak = [jordan_wigner(FermionOperator((x + n, 0))) for x in range(n)]
        l1=[]
        l2=[]
        tl1=[]
        tl2=[]
        T_lead=T_val
        U_lead=U_val/2.0
        for i in range(2*NL):
            if i>=NL:
                U_lead=-U_val/2.0
                          
            l1.append(sqrt(gama*(1-fd(EK[i],U_lead,T_lead)))* ak[i+NS])
            l2.append(sqrt(gama*(fd(EK[i],U_lead,T_lead)))* hermitian_conjugated(ak[i+NS]))
            tl1.append(sqrt(gama*(1-fd(EK[i],U_lead,T_lead)))* tak[i+NS])
            tl2.append(sqrt(gama*(fd(EK[i],U_lead,T_lead)))* hermitian_conjugated(tak[i+NS]))
                
        l0 = QubitOperator()
        for i in range(2*NL):
            l0 = l0 - 1.j * (hermitian_conjugated(l1[i]) * l1[i] +
                           hermitian_conjugated(tl1[i]) * tl1[i] -
                           2 * (-1.j) * l1[i] * tl1[i])
            l0 = l0 - 1.j * (hermitian_conjugated(l2[i]) * l2[i] +
                           hermitian_conjugated(tl2[i]) * tl2[i] -
                           2 * (-1.j) * l2[i] * tl2[i])
        h1 = QubitOperator()
        h2 = QubitOperator()
        for i in range(n):
            h1+=E[i]*hermitian_conjugated(ak[i])*ak[i]
            h2+=E[i]*hermitian_conjugated(tak[i])*tak[i]
        for i in range(NS-1):
            h1-=ts*(hermitian_conjugated(ak[i+1])*ak[i]+hermitian_conjugated(ak[i])*ak[i+1])
            h2-=ts*(hermitian_conjugated(tak[i+1])*tak[i]+hermitian_conjugated(tak[i])*tak[i+1])
        for i in range(NS,n):
            for j in range(NS):
                c1=hermitian_conjugated(ak[i])*ak[j]
                c2=hermitian_conjugated(tak[i])*tak[j]
                h1=h1-t_val*(c1+hermitian_conjugated(c1))
                h2=h2-t_val*(c2+hermitian_conjugated(c2))
        for i in range(NS-1):
            h1+=uc*hermitian_conjugated(ak[i])*ak[i]*hermitian_conjugated(ak[i+1])*ak[i+1]
            h2+=uc*hermitian_conjugated(tak[i])*tak[i]*hermitian_conjugated(tak[i+1])*tak[i+1]
            
        self.fun = h1 - h2 + l0
        
    def form_qpoperator(self):
        self.form_zero_hamiltonia()
        print(self.ni)
        d_operator = []
        q_operator=[]
        for i in range(self.size):
            d_operator.append(jordan_wigner(FermionOperator((i, 0))))
            q_operator.append(0.5*(QubitOperator((i,'X'))+1j*QubitOperator((i,'Y'))))
        for i in range(self.size):
            d_operator.append(jordan_wigner(FermionOperator((i + self.size, 1))))
            q_operator.append(0.5*(QubitOperator((i+self.size,'X'))-1j*QubitOperator((i+self.size,'Y'))))
        self.d_operator=d_operator
        self.q_operator=q_operator

        self.ck_operator=[]
        self.cd_operator = []
        self.tck_operator = []
        self.tcd_operator = []
        self.E=[]
        ap_d=[]
        st1=time.time()
        for i in range(self.size):
            shck=QubitOperator()
            shcd=QubitOperator()
            shtck=QubitOperator()
            shtcd=QubitOperator()
            self.E.append(self.e[2*i].astype(np.complex128))
            for k in range(2*self.size):
                shcd=shcd+hermitian_conjugated(d_operator[k])*self.v[k,2*i]
                shck=shck+d_operator[k]*self.v2[2*i,k]
                shtcd=shtcd+d_operator[k]*self.v2[2*i+1,k]
                shtck=shtck+hermitian_conjugated(d_operator[k])*self.v[k,2*i+1]
            self.ck_operator.append(shck)
            self.cd_operator.append(shcd)
            self.tck_operator.append(shtck)
            self.tcd_operator.append(shtcd)
        fi1=time.time()
        print(f'form t_op:{fi1-st1}')

        l0 = QubitOperator('', 1)
        st2=time.time()
        self.R0 = [self.ck_operator[x] * self.tck_operator[x] for x in range(self.size)]
        fi2=time.time()
        print(f'form R0:{fi2-st2}')

    

    
    def form_simple_ccsd_operator(self):
        nops=0
        self.op_pool=[]
        self.exit_pool=[]

        for i in range(self.size):
            for j in range(self.size):
                qop=hermitian_conjugated(self.d_operator[i])*(self.d_operator[j+self.size])
                self.op_pool.append(qop-hermitian_conjugated(qop))
                self.exit_pool.append((i,j))
                nops=nops+1
                self.op_pool.append(1.j*(qop+hermitian_conjugated(qop)))
                self.exit_pool.append((i,j))
                nops=nops+1
        
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    for l in range(self.size):
                        if i !=j :
                            if k !=l:
                                qop=hermitian_conjugated(self.d_operator[i])*hermitian_conjugated(self.d_operator[j])*(self.d_operator[k+self.size])*(self.d_operator[l+self.size])
                                self.op_pool.append(qop-hermitian_conjugated(qop))
                                self.exit_pool.append((i,j,k,l))
                                nops=nops+1
                                self.op_pool.append(1.j*(qop+hermitian_conjugated(qop)))
                                self.exit_pool.append((i,j,k,l))
                                nops=nops+1

        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    for l in range(self.size):
                        if i !=j :
                            if k !=l:
                                qop=hermitian_conjugated(self.d_operator[i])*(self.d_operator[j])*(self.d_operator[k+self.size])*hermitian_conjugated(self.d_operator[l+self.size])
                                self.op_pool.append(qop-hermitian_conjugated(qop))
                                self.exit_pool.append((i,j,k,l))
                                nops=nops+1
                                self.op_pool.append(1.j*(qop+hermitian_conjugated(qop)))
                                self.exit_pool.append((i,j,k,l))
                                nops=nops+1
        self.nops=nops      
        
def filter_csc_complex_vector(csc_vec, threshold):
    """
    从CSC格式的复向量中移除绝对值小于阈值的元素
    
    参数:
    csc_vec: CSC格式的稀疏向量 (1D)
    threshold: 阈值，绝对值小于此值的元素将被移除
    
    返回:
    filtered_csc_vec: 过滤后的CSC稀疏向量
    """
    # 确保输入是CSC格式
    if not isinstance(csc_vec, csc_matrix):
        raise ValueError("输入必须是CSC格式的稀疏矩阵")
    
    # 将矩阵转换为CSR格式以便操作数据
    csr_vec = csc_vec.tocsr()
    
    # 获取非零元素的绝对值
    abs_data = np.abs(csr_vec.data)
    
    # 找出绝对值大于等于阈值的元素索引
    valid_indices = abs_data >= threshold
    
    # 更新数据和索引
    new_data = csr_vec.data[valid_indices]
    new_indices = csr_vec.indices[valid_indices]
    new_indptr = []
    
    # 重新构建indptr数组
    current_ptr = 0
    for i in range(csr_vec.shape[0] + 1):
        count = np.sum((csr_vec.indices >= current_ptr) & 
                      (csr_vec.indices < len(csr_vec.indices)) &
                      valid_indices[current_ptr:current_ptr+np.sum(csr_vec.indptr[i:i+1])])
        new_indptr.append(current_ptr)
        current_ptr += count
    
    # 修正indptr的计算方式
    new_indptr = [0]
    row_start = 0
    for i in range(csr_vec.shape[0]):
        row_end = csr_vec.indptr[i+1]
        valid_in_row = valid_indices[row_start:row_end]
        new_indptr.append(new_indptr[-1] + np.sum(valid_in_row))
        row_start = row_end
    
    # 创建新的CSR矩阵
    filtered_csr = csc_matrix((new_data, new_indices, new_indptr), 
                             shape=csr_vec.shape, dtype=complex)
    
    # 转回CSC格式
    return filtered_csr.tocsc()          
if __name__ == "__main__":
    # 初始化参数时显式确保为complex128

    params = {"NS": 3,"NL": 2,"U_val": 20,"T_val": 1,"ESMAX": 1,"EKMAX": 5,"uc": -10}
    size=params["NS"]+2*params["NL"]
    nq=2*size
    nqp = SiteQuasiPartcle(**params)
    I0 = LeftVacuum(size)
    r0= nqp.f0
    for x in nqp.R0:
        r0=get_sparse_operator(x,nq)@r0

    r0=r0/(I0.conj().T@r0).trace()
    r0=csc_array(r0)
    r0.data[np.abs(r0.data) < 1e-10] = 0
    r0.eliminate_zeros()

    e,v=sparse.linalg.eigs(get_sparse_operator(nqp.fun,nq),k=1,sigma=0)
    v=v/(I0.conj().T@v).trace()
    ns=[FermionOperator(((x,1),(x,0))) for x in range(params["NS"])]
    occ=[I0.conj().T@get_sparse_operator(x,nq)@v for x in ns]
    for x in occ:
        print(x)

    