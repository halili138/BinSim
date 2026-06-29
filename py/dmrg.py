import os

# # 在导入任何科学计算库之前设置
# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"
# os.environ["OPENBLAS_NUM_THREADS"] = "8"
# os.environ["NUMEXPR_NUM_THREADS"] = "8"

import sys, math
import shutil
import numpy as np
import opt_einsum
from pyscf import gto, scf
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import fci
import pyscf.ci as ci
import pyscf.cc as cc
from pyblock2._pyscf.ao2mo import integrals as itg


def mole_geo(name: str, ratio: float = 1.0) -> str:
    geo = ""
    if name == "lih":
        a = 1.595 * ratio
        geo = f"""
        Li 0.0 0.0 0.0;
        H  0.0 0.0 {a};
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
        geo = f"""
        Si   0.000000000000   0.000000000000   0.000000000000;
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
        C {c1x} {c1y} {c1z}
        C {c2x} {c2y} {c2z}
        H {h1x} {h1y} {h1z}
        H {h2x} {h2y} {h2z}
        H {h3x} {h3y} {h3z}
        H {h4x} {h4y} {h4z}
        H {h5x} {h5y} {h5z}
        H {h6x} {h6y} {h6z}
        """
    else:
        raise ValueError("No corresponding geometry name!")
    return geo


def run_block2_dmrg(
    h1e,
    g2e,
    ecore,
    n_elec,
    spin,
    scratch="./tmp_c2_block2",
    n_threads=4,
    bond_dims=None,
    noises=None,
    thrds=None,
):
    """
    h1e:   spatial orbital one-electron integral, shape (norb, norb)
    g2e:   spatial orbital two-electron integral in chemists' notation, (pq|rs)
    ecore: nuclear repulsion + possible frozen-core energy
    n_elec: total electron number
    spin: 2S. For singlet, spin=0.
    """

    if bond_dims is None:
        # 小例子用这个就行；cc-pVDZ 全空间想更准可以把 200/400 调大
        bond_dims = [100] * 4 + [200] * 4

    if noises is None:
        noises = [1e-4] * 4 + [1e-5] * 2 + [0.0] * 2

    if thrds is None:
        thrds = [1e-8] * len(bond_dims)

    norb = h1e.shape[0]

    if os.path.exists(scratch):
        shutil.rmtree(scratch)

    # RHF 闭壳层体系最推荐 SU2：spin-adapted，通常比 SZ/SGF 快
    driver = DMRGDriver(
        scratch=scratch,
        symm_type=SymmetryTypes.SU2,
        n_threads=n_threads,
        stack_mem=int(100 * 1024**3),  # 2 GB；大体系可调到 10-30 GB
    )

    try:
        # 这里不使用点群对称性，所以 orb_sym=None
        driver.initialize_system(
            n_sites=norb,
            n_elec=n_elec,
            spin=spin,
            orb_sym=None,
        )

        mpo = driver.get_qc_mpo(
            h1e=h1e,
            g2e=g2e,
            ecore=ecore,
            iprint=1,
        )

        ket = driver.get_random_mps(
            tag="GS",
            bond_dim=bond_dims[0],
            nroots=1,
        )

        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=len(bond_dims),
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
        )

        print(f"\nblock2 DMRG energy = {energy:.15f}")

        # 可选：用 1PDM/2PDM 回算能量，检查积分约定是否接对
        pdm1 = driver.get_1pdm(ket)
        pdm2 = driver.get_2pdm(ket).transpose(0, 3, 1, 2)
        g2e_full = driver.unpack_g2e(g2e, n_sites=norb)

        energy_from_pdm = (
            np.einsum("ij,ij->", pdm1, h1e)
            + 0.5 * np.einsum("ijkl,ijkl->", pdm2, g2e_full)
            + ecore
        )

        print(f"Energy from PDMs  = {energy_from_pdm:.15f}")

        return energy

    finally:
        driver.finalize()


def test():
    geo = mole_geo("c2", 1.0)

    mol = gto.M(
        atom=geo,
        basis="cc-pvdz",
        spin=0,
        symmetry="D2h",
    )

    print(f"Use symmetry. Molecule point group: {mol.groupname}")
    norb = mol.nao_nr()
    nelec = mol.nelec

    print(f"Norb: {norb}   Ne: {nelec}")

    mf = scf.RHF(mol)
    print("Running RHF...")
    mf.kernel()

    mol = mf.mol

    mf_ci = ci.CISD(mf)
    print("Running CISD ...")
    mf_ci.kernel()

    mf_cc = cc.CCSD(mf)
    print("Running CCSD ...")
    mf_cc.kernel()

    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
        mf,
        ncore=0,
        ncas=None,
        g2e_symm=8,
    )

    e_dmrg = run_block2_dmrg(
        h1e=h1e,
        g2e=g2e,
        ecore=ecore,
        n_elec=n_elec,
        spin=spin,
        scratch="./tmp_c2_ccpvdz_block2",
        n_threads=int(os.environ.get("OMP_NUM_THREADS", 4)),
        bond_dims=[200] * 4 + [400] * 4 + [800] * 4,
        noises=[1e-4] * 4 + [1e-5] * 4 + [1e-6] * 4,
        thrds=[1e-8] * 4 + [1e-9] * 4 + [1e-9] * 4,
    )

    # e_dmrg = run_block2_dmrg(
    #     h1e=h1e,
    #     g2e=g2e,
    #     ecore=mol.energy_nuc(),
    #     n_elec=n_elec,
    #     spin=mol.spin,  # singlet C2: 0
    #     scratch="./tmp_c2_block2",
    #     n_threads=int(os.environ.get("OMP_NUM_THREADS", 4)),
    # )

    print(f"\nFinal DMRG energy = {e_dmrg:.15f}")


if __name__ == "__main__":
    test()
