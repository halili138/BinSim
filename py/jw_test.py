import numpy, math, datetime, opt_einsum
from multiprocessing import Pool, cpu_count
from functools import partial
from openfermion import FermionOperator, QubitOperator, jordan_wigner, normal_ordered
import pyscf.gto as gto
import pyscf.scf as scf


def _process_1body_chunk(chunk_indices, _1_body_mo):
    ham_ferm_1 = FermionOperator()

    for p, q in chunk_indices:
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

    for p, q, r, s in chunk_indices:
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


def int2ham_single(energy_nuc, _1_body_mo, _2_body_mo, eps=1e-12):
    indices_1body = list(zip(*((abs(_1_body_mo) > eps).nonzero())))
    indices_2body = list(zip(*((abs(_2_body_mo) > eps).nonzero())))

    ham_ferm_1 = FermionOperator()

    for p, q in indices_1body:
        c = _1_body_mo[p, q]
        pa = 2 * int(p)
        qa = 2 * int(q)
        pb = 2 * int(p) + 1
        qb = 2 * int(q) + 1
        ham_ferm_1 += FermionOperator(((pa, 1), (qa, 0)), c)
        ham_ferm_1 += FermionOperator(((pb, 1), (qb, 0)), c)

    ham_ferm_1 = normal_ordered(ham_ferm_1)

    ham_ferm_2 = FermionOperator()

    for p, q, r, s in indices_2body:
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

    ham_ferm = energy_nuc + ham_ferm_1 + ham_ferm_2
    ham_ferm = normal_ordered(ham_ferm)
    ham_qubit = jordan_wigner(ham_ferm)

    return ham_ferm, ham_qubit


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

    ham_qubit = energy_nuc
    ham_qubit += ham_qubit_1
    ham_qubit += ham_qubit_2

    t7 = datetime.datetime.now()
    t_tol = (t7 - t1).total_seconds()

    print(f"n_orb:        {len(_1_body_mo)}")
    print(f"n_proces:     {n_processes}")
    print(f"Pre chunk:    {(t2 - t1).total_seconds()}")
    print(f"Map _1_mo:    {(t3 - t2).total_seconds()}")
    print(f"Sum _1_mo:    {(t4 - t3).total_seconds()}")
    print(f"Map _2_mo:    {(t5 - t4).total_seconds()}")
    print(f"Sum _2_mo:    {(t6 - t5).total_seconds()}")
    print(f"Sum:          {(t7 - t6).total_seconds()}")
    print(f"Total:        {t_tol}")

    return ham_qubit


def generate_physical_random_integrals_vectorized(norb, seed=42):
    if seed is not None:
        numpy.random.seed(seed)

    h1_random = numpy.random.randn(norb, norb)
    h1_mo = 0.5 * (h1_random + h1_random.T)
    numpy.fill_diagonal(
        h1_mo, h1_mo.diagonal() - numpy.abs(numpy.random.randn(norb)) * 5
    )
    n_pairs = norb * (norb + 1) // 2
    rank = max(1, n_pairs // 2)
    L = numpy.random.randn(n_pairs, rank)
    V_IJ = L @ L.T  # 半正定实对称矩阵
    p, q = numpy.indices((norb, norb))
    idx_map = numpy.maximum(p, q) * (numpy.maximum(p, q) + 1) // 2 + numpy.minimum(p, q)
    V_chem = V_IJ[idx_map[:, :, None, None], idx_map[None, None, :, :]]
    h2_mo = V_chem.transpose(0, 2, 1, 3)

    return h1_mo, h2_mo


def mole_geo(name: str, ratio: float = 1.0) -> str:
    geo = ""

    if name == "lih":
        a = 1.595 * ratio
        geo = f"""
        Li 0.0 0.0 0.0;
        H  0.0 0.0 {a};
        """
    elif name == "h2":
        a = 0.76 * ratio
        geo = f"""
        H  0.0 0.0 0.0;
        H  0.0 0.0 {a};
        """
    elif name == "h4":
        a = 0.76 * ratio
        geo = f"""
        H  0.0 0.0 0.0;
        H  0.0 0.0 {a};
        H  0.0 0.0 {2 * a};
        H  0.0 0.0 {3 * a};
        """
    elif name == "beh2":
        a = 1.34 * ratio
        geo = f"""
        H  0.0 0.0 {-a};
        Be 0.0 0.0 0.0;
        H  0.0 0.0 {a};
        """
    elif name == "nh3":
        a = 1.01 * ratio
        theta = math.radians(107.3)
        w1 = math.sin(math.pi / 3)
        w2 = math.cos(math.pi / 3)
        s = a * math.sin(theta / 2)
        c = a * math.cos(theta / 2)
        geo = f"""
        N 0.0 0.0 0.0;
        H {s} {c} 0.0;
        H {-s*w2} {c} {s*w1};
        H {-s*w2} {c} {-s*w1};
        """
    elif name == "h2o":
        a = 0.958 * ratio
        theta = math.radians(104.5)
        s = a * math.sin(theta / 2)
        c = a * math.cos(theta / 2)
        geo = f"""
        H 0.0 {c} {-s};
        O 0.0 0.0 0.0;
        H 0.0 {c} {s};
        """
    elif name == "n2":
        a = 1.1 * ratio
        geo = f"""
        N 0.0 0.0 0.0;
        N 0.0 0.0 {a};
        """
    elif name == "hcn":
        a_ch = 1.06 * ratio
        a_cn = 1.16 * ratio
        geo = f"""
        H  0.0  0.0  {-a_ch};
        C  0.0  0.0  0.0;
        N  0.0  0.0  {a_cn};
        """
    elif name == "h2co":
        a1 = 1.21 * ratio
        a2 = 1.11 * ratio
        theta = math.radians(118)
        w1 = math.sin(theta / 2)
        w2 = math.cos(theta / 2)
        geo = f"""
        C 0.0 0.0 0.0;
        O 0.0 0.0 {a1};
        H {a2*w1} 0.0 {-a2*w2};
        H {-a2*w1} 0.0 {-a2*w2};
        """
    elif name == "co":
        a = 1.128 * ratio
        geo = f"""
        C 0.0 0.0 0.0;
        O 0.0 0.0 {a};
        """
    elif name == "co2":
        a = 1.16 * ratio
        geo = f"""
        O 0.0 0.0 {-a};
        C 0.0 0.0 0.0;
        O 0.0 0.0 {a};
        """
    elif name == "c2":
        a = 1.24 * ratio
        geo = f"""
        C 0.0 0.0 0.0;
        C 0.0 0.0 {a};
        """
    elif name == "o2":
        a = 1.21 * ratio
        geo = f"""
        O 0.0 0.0 0.0;
        O 0.0 0.0 {a};
        """
    elif name == "hf":
        a = 0.917 * ratio
        geo = f"""
        H 0.0 0.0 0.0;
        F 0.0 0.0 {a};
        """
    elif name == "hcl":
        a = 1.274 * ratio
        geo = f"""
        H 0.0 0.0 0.0;
        Cl 0.0 0.0 {a};
        """
    elif name == "ch4":
        a = 1.09 * ratio
        x = a / math.sqrt(3)
        geo = f"""
        C   0.000000000000   0.000000000000   0.000000000000;
        H   { x}  { x}  { x};
        H   { x}  {-x}  {-x};
        H   {-x}  { x}  {-x};
        H   {-x}  {-x}  { x};
        """
    elif name == "sih4":
        a = 1.48 * ratio
        x = a / math.sqrt(3)
        # 注意：这里将原先代码中的 C 改为了 Si
        geo = f"""
        Si  0.000000000000   0.000000000000   0.000000000000;
        H   { x}  { x}  { x};
        H   { x}  {-x}  {-x};
        H   {-x}  { x}  {-x};
        H   {-x}  {-x}  { x};
        """
    elif name == "c2h4":
        a1 = 1.33 * ratio
        a2 = 1.08 * ratio
        theta = math.radians(180 - 121.3)

        c1x, c1y, c1z = -a1 / 2, 0.0, 0.0
        c2x, c2y, c2z = a1 / 2, 0.0, 0.0

        h1x = c1x - a2 * math.cos(theta)
        h1y = c1y + a2 * math.sin(theta)
        h1z = 0.0

        h2y = c1y - a2 * math.sin(theta)
        h2x = c1x - a2 * math.cos(theta)
        h2z = 0.0

        h3x = c2x + a2 * math.cos(theta)
        h3y = c2y + a2 * math.sin(theta)
        h3z = 0.0

        h4x = c2x + a2 * math.cos(theta)
        h4y = c2y - a2 * math.sin(theta)
        h4z = 0.0

        geo = f"""
        C {c1x}  {c1y}  {c1z};
        C {c2x}  {c2y}  {c2z};
        H {h1x}  {h1y}  {h1z};
        H {h2x}  {h2y}  {h2z};
        H {h3x}  {h3y}  {h3z};
        H {h4x}  {h4y}  {h4z};
        """
    elif name == "c2h6":
        a1 = 1.54 * ratio
        a2 = 1.09 * ratio
        theta = math.radians(109.5)

        c1x, c1y, c1z = 0.0, 0.0, 0.0
        c2x, c2y, c2z = a1, 0.0, 0.0

        # C1上的氢原子（指向-X方向）
        h1x = a2 * math.cos(theta)
        h1y = a2 * math.sin(theta)
        h1z = 0.0
        h2x = a2 * math.cos(theta)
        h2y = a2 * math.sin(theta) * math.cos(math.radians(120))
        h2z = a2 * math.sin(theta) * math.sin(math.radians(120))
        h3x = a2 * math.cos(theta)
        h3y = a2 * math.sin(theta) * math.cos(math.radians(240))
        h3z = a2 * math.sin(theta) * math.sin(math.radians(240))

        # C2上的氢原子（指向+X方向，与C1上的氢交错60°）
        h4x = a1 - a2 * math.cos(theta)
        h4y = a2 * math.sin(theta) * math.cos(math.radians(60))
        h4z = a2 * math.sin(theta) * math.sin(math.radians(60))
        h5x = a1 - a2 * math.cos(theta)
        h5y = a2 * math.sin(theta) * math.cos(math.radians(180))
        h5z = a2 * math.sin(theta) * math.sin(math.radians(180))
        h6x = a1 - a2 * math.cos(theta)
        h6y = a2 * math.sin(theta) * math.cos(math.radians(300))
        h6z = a2 * math.sin(theta) * math.sin(math.radians(300))

        geo = f"""
        C {c1x} {c1y} {c1z};
        C {c2x} {c2y} {c2z};
        H {h1x} {h1y} {h1z};
        H {h2x} {h2y} {h2z};
        H {h3x} {h3y} {h3z};
        H {h4x} {h4y} {h4z};
        H {h5x} {h5y} {h5z};
        H {h6x} {h6y} {h6z};
        """
    elif name == "c6h6":
        cc_bond = 1.39 * ratio
        ch_bond = 1.08 * ratio
        ring_radius = cc_bond / (2 * math.sin(math.pi / 6))

        c_coords = []
        for i in range(6):
            angle = math.radians(60 * i)
            x = ring_radius * math.cos(angle)
            y = ring_radius * math.sin(angle)
            c_coords.append((x, y, 0.0))

        h_coords = []
        for i in range(6):
            angle = math.radians(60 * i + 30)
            x = c_coords[i][0] + ch_bond * math.cos(angle)
            y = c_coords[i][1] + ch_bond * math.sin(angle)
            h_coords.append((x, y, 0.0))

        geo = "\n"

        for i, (x, y, z) in enumerate(c_coords, start=1):
            geo += f"        C {x} {y} {z} {i};\n"
        for i, (x, y, z) in enumerate(h_coords, start=1):
            geo += f"        H {x} {y} {z} {i};\n"

    elif name == "cr2":
        a = 1.68 * ratio
        geo = f"""
        Cr 0.0 0.0 0.0;
        Cr 0.0 0.0 {a};
        """
    elif name == "ch3oh":
        geo = f"""
        C  {-0.046 * ratio}  { 0.662 * ratio}  { 0.000 * ratio};
        O  {-0.046 * ratio}  {-0.758 * ratio}  { 0.000 * ratio};
        H  {-1.085 * ratio}  { 1.030 * ratio}  { 0.000 * ratio};
        H  { 0.468 * ratio}  { 1.034 * ratio}  { 0.887 * ratio};
        H  { 0.468 * ratio}  { 1.034 * ratio}  {-0.887 * ratio};
        H  { 0.865 * ratio}  {-1.077 * ratio}  { 0.000 * ratio};
        """
    elif name == "c2h5oh":
        geo = f"""
        C  {-1.196 * ratio}  {-0.231 * ratio}  { 0.000 * ratio};
        C  { 0.117 * ratio}  { 0.525 * ratio}  { 0.000 * ratio};
        O  { 1.213 * ratio}  {-0.380 * ratio}  { 0.000 * ratio};
        H  {-1.258 * ratio}  {-0.871 * ratio}  { 0.886 * ratio};
        H  {-1.258 * ratio}  {-0.871 * ratio}  {-0.886 * ratio};
        H  {-2.053 * ratio}  { 0.446 * ratio}  { 0.000 * ratio};
        H  { 0.158 * ratio}  { 1.176 * ratio}  { 0.888 * ratio};
        H  { 0.158 * ratio}  { 1.176 * ratio}  {-0.888 * ratio};
        H  { 2.000 * ratio}  { 0.165 * ratio}  { 0.000 * ratio};
        """
    elif name == "hcooh":
        geo = f"""
        C  { 0.138 * ratio}  { 0.370 * ratio}  { 0.000 * ratio};
        O  {-0.957 * ratio}  {-0.347 * ratio}  { 0.000 * ratio};
        O  { 1.196 * ratio}  {-0.187 * ratio}  { 0.000 * ratio};
        H  {-1.745 * ratio}  { 0.218 * ratio}  { 0.000 * ratio};
        H  { 0.091 * ratio}  { 1.464 * ratio}  { 0.000 * ratio};
        """
    elif name == "ch3cooh":
        geo = f"""
        C  {-1.396 * ratio}  { 0.103 * ratio}  { 0.000 * ratio};
        C  { 0.061 * ratio}  { 0.125 * ratio}  { 0.000 * ratio};
        O  { 0.638 * ratio}  {-1.073 * ratio}  { 0.000 * ratio};
        O  { 0.730 * ratio}  { 1.139 * ratio}  { 0.000 * ratio};
        H  { 1.597 * ratio}  {-0.958 * ratio}  { 0.000 * ratio};
        H  {-1.761 * ratio}  { 0.627 * ratio}  { 0.888 * ratio};
        H  {-1.761 * ratio}  { 0.627 * ratio}  {-0.888 * ratio};
        H  {-1.803 * ratio}  {-0.906 * ratio}  { 0.000 * ratio};
        """
    elif name == "c4h10":
        geo = f"""
        C  {-1.921 * ratio}  { 0.313 * ratio}  { 0.000 * ratio};
        C  {-0.583 * ratio}  {-0.428 * ratio}  { 0.000 * ratio};
        C  { 0.583 * ratio}  { 0.428 * ratio}  { 0.000 * ratio};
        C  { 1.921 * ratio}  {-0.313 * ratio}  { 0.000 * ratio};
        H  {-1.961 * ratio}  { 1.399 * ratio}  { 0.000 * ratio};
        H  {-2.428 * ratio}  {-0.066 * ratio}  { 0.885 * ratio};
        H  {-2.428 * ratio}  {-0.066 * ratio}  {-0.885 * ratio};
        H  {-0.543 * ratio}  {-1.072 * ratio}  { 0.879 * ratio};
        H  {-0.543 * ratio}  {-1.072 * ratio}  {-0.879 * ratio};
        H  { 0.543 * ratio}  { 1.072 * ratio}  { 0.879 * ratio};
        H  { 0.543 * ratio}  { 1.072 * ratio}  {-0.879 * ratio};
        H  { 1.961 * ratio}  {-1.399 * ratio}  { 0.000 * ratio};
        H  { 2.428 * ratio}  { 0.066 * ratio}  { 0.885 * ratio};
        H  { 2.428 * ratio}  { 0.066 * ratio}  {-0.885 * ratio};
        """
    elif name == "c3h8":
        geo = f"""
        C   { 0.000 * ratio}  { 0.000 * ratio}  { 0.586 * ratio};
        C   { 0.000 * ratio}  { 1.276 * ratio}  {-0.259 * ratio};
        C   { 0.000 * ratio}  {-1.276 * ratio}  {-0.259 * ratio};
        H   { 0.885 * ratio}  { 0.000 * ratio}  { 1.229 * ratio};
        H   {-0.885 * ratio}  { 0.000 * ratio}  { 1.229 * ratio};
        H   { 0.000 * ratio}  { 2.164 * ratio}  { 0.379 * ratio};
        H   { 0.882 * ratio}  { 1.328 * ratio}  {-0.895 * ratio};
        H   {-0.882 * ratio}  { 1.328 * ratio}  {-0.895 * ratio};
        H   { 0.000 * ratio}  {-2.164 * ratio}  { 0.379 * ratio};
        H   { 0.882 * ratio}  {-1.328 * ratio}  {-0.895 * ratio};
        H   {-0.882 * ratio}  {-1.328 * ratio}  {-0.895 * ratio};
        """
    elif name == "c3h6":
        geo = f"""
        C  { 1.274 * ratio}  { 0.273 * ratio}  { 0.000 * ratio};
        C  { 0.000 * ratio}  {-0.188 * ratio}  { 0.000 * ratio};
        C  {-1.196 * ratio}  { 0.725 * ratio}  { 0.000 * ratio};
        H  { 1.442 * ratio}  { 1.343 * ratio}  { 0.000 * ratio};
        H  { 2.133 * ratio}  {-0.385 * ratio}  { 0.000 * ratio};
        H  {-0.134 * ratio}  {-1.268 * ratio}  { 0.000 * ratio};
        H  {-2.146 * ratio}  { 0.187 * ratio}  { 0.000 * ratio};
        H  {-1.157 * ratio}  { 1.365 * ratio}  { 0.880 * ratio};
        H  {-1.157 * ratio}  { 1.365 * ratio}  {-0.880 * ratio};
        """
    else:
        raise ValueError("No corresponding geometry name!")

    return geo


def mol_int(mf):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    hcore = mol.intor("int1e_nuc") + mol.intor("int1e_kin")
    eri = mol.intor("int2e")
    one_body_mo = mo_coeff.T @ hcore @ mo_coeff
    two_body_mo = opt_einsum.contract(
        "ijkl, ip, js, kq, lr -> pqrs", eri, mo_coeff, mo_coeff, mo_coeff, mo_coeff
    )

    return one_body_mo, two_body_mo


def init_scf(name, ratio, basis):

    geo = mole_geo(name, ratio)

    mol = gto.M(atom=geo, basis=basis, spin=0.0, symmetry=True)
    print(f"Use symmetry. Molecule point group: {mol.groupname}")

    norb = mol.nao_nr()
    nelec = mol.nelec
    print(f"Norb: {norb}   Ne: {nelec}")

    mf = scf.RHF(mol)
    print("Running RHF...")
    mf.kernel()

    one_body_mo, two_body_mo = mol_int(mf)

    ham_ferm, ham_qubit = int2ham_single(
        mol.energy_nuc(), one_body_mo, two_body_mo, 1e-12
    )
    # print(len(ham_ferm.terms))
    # print(len(ham_qubit.terms))
    print(ham_ferm)
    print(ham_qubit)


if __name__ == "__main__":
    # h1_mo, h2_mo = generate_physical_random_integrals_vectorized(10, seed=None)
    # int2ham_parallel(1.0, h1_mo, h2_mo, 1e-12, 12)
    init_scf("h4", 1.0, "sto-3g")
