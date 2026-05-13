import numpy, opt_einsum, itertools
from pyscf.pbc.tools import get_kconserv

def mol_int(mf):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    hcore = mol.intor("int1e_nuc") + mol.intor("int1e_kin")
    eri = mol.intor("int2e")
    one_body_mo = mo_coeff.T @ hcore @ mo_coeff
    two_body_mo = opt_einsum.contract("ijkl, ip, js, kq, lr -> pqrs", eri, mo_coeff, mo_coeff, mo_coeff, mo_coeff)

    return one_body_mo, two_body_mo

def pbc_int(kmf):
    cell        = kmf.cell
    n_orb_unit  = cell.nao_nr()
    kpts        = kmf.kpts
    n_kpts      = len(kpts)
    n_orb       = n_orb_unit * n_kpts
    mo_coeff    = kmf.mo_coeff
    hcore       = kmf.get_hcore(cell, kpts)
    kconserv    = get_kconserv(cell, kpts)

    one_body_mo = numpy.zeros([n_orb]*2, dtype=numpy.complex128)
    for k in range(n_kpts):
        one_body_mo_k = mo_coeff[k].T.conj() @ hcore[k] @ mo_coeff[k]
        for p, q in itertools.product(range(n_orb_unit), repeat=2):
            mp = k * n_orb_unit + p
            mq = k * n_orb_unit + q
            one_body_mo[mp, mq] = one_body_mo_k[p, q] / n_kpts

    two_body_mo = numpy.zeros([n_orb]*4, dtype=numpy.complex128)
    for kp, kq, kr in itertools.product(range(n_kpts), repeat=3):
        ks = kconserv[kp][kq][kr]
        two_body_mo_k = kmf.with_df.ao2mo(
            [mo_coeff[k] for k in (kp, kq, kr, ks)],
            [kpts[k] for k in (kp, kq, kr, ks)],
            compact=False
        )
        two_body_mo_k = two_body_mo_k.reshape([n_orb_unit]*4)
        for p, q, r, s in itertools.product(range(n_orb_unit), repeat=4):
            mp = kp * n_orb_unit + p
            mq = kq * n_orb_unit + q
            mr = kr * n_orb_unit + r
            ms = ks * n_orb_unit + s
            two_body_mo[mp, mq, mr, ms] = two_body_mo_k[p, q, r, s] / n_kpts**2

    return one_body_mo, two_body_mo, kconserv

def pbc_int_symm(kmf):
    cell            = kmf.cell
    n_orb_unit      = cell.nao_nr()
    kpts            = kmf.kpts

    kpts_bz         = kpts.kpts
    n_kpts_bz       = len(kpts_bz)
    n_orb_bz        = n_orb_unit * n_kpts_bz
    hcore_bz        = kmf.get_hcore(cell, kpts_bz)
    kconserv        = get_kconserv(cell, kpts_bz)
    bz2ibz          = kpts.bz2ibz
    mo_coeff_ibz    = kmf.mo_coeff
    
    one_body_mo = numpy.zeros([n_orb_bz]*2, dtype=numpy.complex128)
    for k in range(n_kpts_bz):
        one_body_mo_k = mo_coeff_ibz[bz2ibz[k]].T.conj() @ hcore_bz[k] @ mo_coeff_ibz[bz2ibz[k]]
        for p, q in itertools.product(range(n_orb_unit), repeat=2):
            mp = k * n_orb_unit + p
            mq = k * n_orb_unit + q
            one_body_mo[mp, mq] = one_body_mo_k[p, q] / n_kpts_bz

    two_body_mo = numpy.zeros([n_orb_bz]*4, dtype=numpy.complex128)
    for kp, kq, kr in itertools.product(range(n_kpts_bz), repeat=3):
        ks = kconserv[kp][kq][kr]
        two_body_mo_k = kmf.with_df.ao2mo(
            [mo_coeff_ibz[bz2ibz[k]] for k in (kp, kq, kr, ks)],
            [kpts_bz[k] for k in (kp, kq, kr, ks)],
            compact=False)
        two_body_mo_k = two_body_mo_k.reshape([n_orb_unit]*4)
        for p, q, r, s in itertools.product(range(n_orb_unit), repeat=4):
            mp = kp * n_orb_unit + p
            mq = kq * n_orb_unit + q
            mr = kr * n_orb_unit + r
            ms = ks * n_orb_unit + s
            two_body_mo[mp, mq, mr, ms] = two_body_mo_k[p, q, r, s]/ n_kpts_bz**2

    return one_body_mo, two_body_mo


if __name__ == "__main__":
    import openfermion
    from pyscf.pbc import gto, scf, df, cc
    import scipy.sparse.linalg    


    def int2ham(n_orb: int, one_body_mo: numpy.ndarray, two_body_mo: numpy.ndarray, eps=0.0):
        spin_orbital_one_body_mo = numpy.zeros([n_orb*2]*2, dtype=numpy.complex128)
        spin_orbital_two_body_mo = numpy.zeros([n_orb*2]*4, dtype=numpy.complex128)
        spin_orbital_one_body_mo[0::2, 0::2] = one_body_mo
        spin_orbital_one_body_mo[1::2, 1::2] = one_body_mo
        two_body_mo = numpy.moveaxis(two_body_mo, [0, 2, 3, 1], [0, 1, 2, 3])
        spin_orbital_two_body_mo[0::2, 0::2, 0::2, 0::2] = two_body_mo
        spin_orbital_two_body_mo[1::2, 1::2, 1::2, 1::2] = two_body_mo
        spin_orbital_two_body_mo[1::2, 0::2, 0::2, 1::2] = two_body_mo
        spin_orbital_two_body_mo[0::2, 1::2, 1::2, 0::2] = two_body_mo

        one_body_fermion = openfermion.FermionOperator()
        two_body_fermion = openfermion.FermionOperator()

        for (p, q) in zip(*((abs(spin_orbital_one_body_mo) > eps).nonzero())):
            p = int(p)
            q = int(q)
            one_body_fermion += openfermion.FermionOperator(((p,1), (q,0)), spin_orbital_one_body_mo[p][q])
        for (p, q, r, s) in zip(*((abs(spin_orbital_two_body_mo) > eps).nonzero())):
            p = int(p)
            q = int(q)
            r = int(r)
            s = int(s)
            two_body_fermion += openfermion.FermionOperator(((p,1), (q,1), (r,0), (s,0)), spin_orbital_two_body_mo[p][q][r][s]*0.5)

        one_body_fermion = openfermion.normal_ordered(one_body_fermion)
        two_body_fermion = openfermion.normal_ordered(two_body_fermion)

        ham_fermion = one_body_fermion + two_body_fermion

        return ham_fermion


    a         = 3.567*0.5
    cell      = gto.Cell()
    cell.atom = [["H", [0.0, 0.0, 0.0]], ["H", [1.0, 0.0, 0.0]]] 
    cell.a    = [[2.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]

    cell.basis  = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.spin = 0
    cell.exp_to_discard=0.1
    cell.build()
    mesh = [3,1,1]
    kpts = cell.make_kpts(mesh, scaled_center=[0,0,0], wrap_around=True)
    kmf = scf.KRHF(cell, kpts, exxdiv=None)
    kmf.with_df = df.MDF(cell, kpts=kpts)
    print("Running KRHF...")
    kmf.kernel()

    one_body_mo, two_body_mo = pbc_int(kmf)
    norb = len(one_body_mo)
    ham_fermion = int2ham(norb, one_body_mo, two_body_mo, eps=1e-12) + cell.energy_nuc()
    ham_spMat   = openfermion.get_sparse_operator(ham_fermion, norb * 2)
    e, v        = scipy.sparse.linalg.eigsh(ham_spMat, k=1, which="SA", tol=1e-10)
    print(f"Total FCI energy: {e}")

    # cell      = gto.Cell()
    # cell.atom = [["H", [0.0, 0.0, 0.0]], ["H", [1.0, 0.0, 0.0]]] 
    # cell.a    = [[2.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]

    # cell.basis = 'gth-szv'
    # cell.pseudo = 'gth-pade'
    # cell.spin = 0
    # cell.exp_to_discard=0.1
    # cell.space_group_symmetry = True
    # cell.symmorphic = True
    # cell.build()
    # cell.build_lattice_symmetry()
    # mesh = [3,1,1]
    # kpts = cell.make_kpts(mesh, scaled_center=[0,0,0], wrap_around=True, space_group_symmetry=True)
    # kmf = scf.KRHF(cell, kpts, exxdiv=None)
    # kmf.with_df = df.MDF(cell, kpts=kpts)
    # print("Running KRHF...")
    # kmf.kernel()

    
    # one_body_mo, two_body_mo = pbc_int_symm(kmf)
    # norb = len(one_body_mo)
    # ham_fermion = int2ham(norb, one_body_mo, two_body_mo, eps=1e-12) + cell.energy_nuc()
    # ham_spMat   = openfermion.get_sparse_operator(ham_fermion, norb * 2)
    # e, v        = scipy.sparse.linalg.eigsh(ham_spMat, k=1, which="SA", tol=1e-10)
    # print(f"Total FCI energy: {e}")
