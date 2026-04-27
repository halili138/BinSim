import sys, math
from pyscf import gto, scf, fci
from pyscf.lib import logger

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

def init_scf(name:str, ratio:float, basis:str="sto-3g"):
    geo = mole_geo(name, ratio)
    mol = gto.M(atom=geo, basis=basis, spin=0.0, symmetry=True)
    print(f"Use symmetry. Molecule point group: {mol.topgroup}")
    norb = mol.nao_nr()
    nelec = mol.nelec
    print(f"Norb: {norb}   Ne: {nelec}")

    mf = scf.RHF(mol)
    print("Running RHF...")
    mf.kernel()

    mf_fci = fci.FCI(mf)
    mf_fci.max_memory = 768000
    mf_fci.verbose = logger.DEBUG1

    print("Running FCI ...")
    mf_fci.kernel()


if __name__ == "__main__":
    init_scf(sys.argv[1].lower(), 1.0, sys.argv[2].lower())

