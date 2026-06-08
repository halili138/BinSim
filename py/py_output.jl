include("../binsim.jl")

using PyCall


@pyimport pyscf.gto as gto
@pyimport pyscf.scf as scf
@pyimport pyscf.mcscf as mcscf
@pyimport pyscf.ao2mo as ao2mo
@pyimport pyscf.ci as ci
@pyimport pyscf.cc as cc
@pyimport pyscf.fci as fci
@pyimport pyscf.pbc as pbc
@pyimport pyscf.mp as mp

pushfirst!(pyimport("sys")."path", @__DIR__)
pyfun = pyimport("mole_pbc_int")


function h_chain(nh::Int, ratio::Float64=1.0)
    a = 0.74 * ratio
    geo = ""
    for i in 0:nh-1
        geo *= "H 0.0 0.0 $(i*a);"
    end

    return geo
end


function mole_geo(name::String, ratio::Float64=1.0)
    geo = ""

    if name == "lih"
        a = 1.595 * ratio
        geo = "
        Li 0.0 0.0 0.0;
        H  0.0 0.0 $(a);
        "
    elseif name == "beh2"
        a = 1.34 * ratio
        geo = "
        H  0.0 0.0 $(-a);
        Be 0.0 0.0 0.0;
        H  0.0 0.0 $(a);
        "
    elseif name == "nh3"
        a = 1.01 * ratio
        θ = deg2rad(107.3)
        ω1 = sin(pi / 3)
        ω2 = cos(pi / 3)
        s = a * sin(θ / 2)
        c = a * cos(θ / 2)
        geo = "
        N 0.0 0.0 0.0;
        H $(s) $(c) 0.0;
        H $(-s*ω2) $(c) $(s*ω1);
        H $(-s*ω2) $(c) $(-s*ω1);
        "
    elseif name == "h2o"
        a = 0.958 * ratio
        θ = deg2rad(104.5)
        s = a * sin(θ / 2)
        c = a * cos(θ / 2)
        geo = "
        H 0.0 $(c) $(-s);
        O 0.0 0.0 0.0;
        H 0.0 $(c) $(s);
        "
    elseif name == "n2"
        a = 1.1 * ratio
        geo = "
        N 0.0 0.0 0.0;
        N 0.0 0.0 $(a);
        "
    elseif name == "hcn"
        a_ch = 1.06 * ratio
        a_cn = 1.16 * ratio
        geo = "
        H  0.0  0.0  $(-a_ch);
        C  0.0  0.0  0.0;
        N  0.0  0.0  $(a_cn);
        "
    elseif name == "h2co"
        a1 = 1.21 * ratio
        a2 = 1.11 * ratio
        θ = deg2rad(118)
        ω1 = sin(θ / 2)
        ω2 = cos(θ / 2)
        geo = "
        C 0.0 0.0 0.0;
        O 0.0 0.0 $(a1);
        H $(a2*ω1) 0.0 $(-a2*ω2);
        H $(-a2*ω1) 0.0 $(-a2*ω2);
        "
    elseif name == "co"
        a = 1.128 * ratio
        geo = "
        C 0.0 0.0 0.0;
        O 0.0 0.0 $(a);
        "
    elseif name == "co2"
        a = 1.16 * ratio
        geo = "
        O 0.0 0.0 $(-a);
        C 0.0 0.0 0.0;
        O 0.0 0.0 $(a);
        "
    elseif name == "c2"
        a = 1.24 * ratio
        geo = "
        C 0.0 0.0 0.0;
        C 0.0 0.0 $(a);
        "
    elseif name == "o2"
        a = 1.21 * ratio
        geo = "
        O 0.0 0.0 0.0;
        O 0.0 0.0 $(a);
        "
    elseif name == "hf"
        a = 0.917 * ratio
        geo = "
        H 0.0 0.0 0.0;
        F 0.0 0.0 $(a);
        "
    elseif name == "hcl"
        a = 1.274 * ratio
        geo = "
        H 0.0 0.0 0.0;
        Cl 0.0 0.0 $(a);
        "
    elseif name == "ch4"
        a = 1.09 * ratio
        x = a / sqrt(3)
        geo = "
        C   0.000000000000   0.000000000000   0.000000000000;
        H   $( x)  $( x)  $( x);
        H   $( x)  $(-x)  $(-x);
        H   $(-x)  $( x)  $(-x);
        H   $(-x)  $(-x)  $( x);
        "   
    elseif name == "sih4"
        a = 1.48 * ratio
        x = a / sqrt(3)
        geo = "
        C   0.000000000000   0.000000000000   0.000000000000;
        H   $( x)  $( x)  $( x);
        H   $( x)  $(-x)  $(-x);
        H   $(-x)  $( x)  $(-x);
        H   $(-x)  $(-x)  $( x);
        "
    elseif name == "c2h4"
        a1 = 1.33 * ratio
        a2 = 1.08 * ratio
        θ = deg2rad(180 - 121.3)

        (c1x, c1y, c1z) = (-a1 / 2, 0.0, 0.0)
        (c2x, c2y, c2z) = (a1 / 2, 0.0, 0.0)

        h1x = c1x - a2 * cos(θ)
        h1y = c1y + a2 * sin(θ)
        h1z = 0.0

        h2y = c1y - a2 * sin(θ)
        h2x = c1x - a2 * cos(θ)
        h2z = 0.0

        h3x = c2x + a2 * cos(θ)
        h3y = c2y + a2 * sin(θ)
        h3z = 0.0

        h4x = c2x + a2 * cos(θ)
        h4y = c2y - a2 * sin(θ)
        h4z = 0.0

        geo = "
        C $(c1x)  $(c1y)  $(c1z);
        C $(c2x)  $(c2y)  $(c2z);
        H $(h1x)  $(h1y)  $(h1z);
        H $(h2x)  $(h2y)  $(h2z);
        H $(h3x)  $(h3y)  $(h3z);
        H $(h4x)  $(h4y)  $(h4z);
        "
    elseif name == "c2h6"
        a1 = 1.54 * ratio
        a2 = 1.09 * ratio
        θ = deg2rad(109.5)
        (c1x, c1y, c1z) = (0.0, 0.0, 0.0)
        (c2x, c2y, c2z) = (a1, 0.0, 0.0)
        # C1上的氢原子（指向-X方向）
        h1x = a2 * cos(θ)
        h1y = a2 * sin(θ)
        h1z = 0.0
        h2x = a2 * cos(θ)
        h2y = a2 * sin(θ) * cos(deg2rad(120))
        h2z = a2 * sin(θ) * sin(deg2rad(120))
        h3x = a2 * cos(θ)
        h3y = a2 * sin(θ) * cos(deg2rad(240))
        h3z = a2 * sin(θ) * sin(deg2rad(240))
        # C2上的氢原子（指向+X方向，与C1上的氢交错60°）
        # 修正：使用 -cos(θ) 或 cos(π-θ)，因为θ>90°，cos(θ)<0
        h4x = a1 - a2 * cos(θ)  # 等价于 a1 + a2 * cos(π-θ)
        h4y = a2 * sin(θ) * cos(deg2rad(60))
        h4z = a2 * sin(θ) * sin(deg2rad(60))
        h5x = a1 - a2 * cos(θ)
        h5y = a2 * sin(θ) * cos(deg2rad(180))
        h5z = a2 * sin(θ) * sin(deg2rad(180))
        h6x = a1 - a2 * cos(θ)
        h6y = a2 * sin(θ) * cos(deg2rad(300))
        h6z = a2 * sin(θ) * sin(deg2rad(300))
        geo = "
        C $(c1x) $(c1y) $(c1z)
        C $(c2x) $(c2y) $(c2z)
        H $(h1x) $(h1y) $(h1z)
        H $(h2x) $(h2y) $(h2z)
        H $(h3x) $(h3y) $(h3z)
        H $(h4x) $(h4y) $(h4z)
        H $(h5x) $(h5y) $(h5z)
        H $(h6x) $(h6y) $(h6z)
        "    
    elseif name == "c6h6"
        cc_bond = 1.39 * ratio
        ch_bond = 1.08 * ratio
        ring_radius = cc_bond / (2 * sin(pi / 6))

        c_coords = NTuple{3,Float64}[]
        for i in 0:5
            angle = deg2rad(60i)
            x = ring_radius * cos(angle)
            y = ring_radius * sin(angle)
            push!(c_coords, (x, y, 0.0))
        end

        h_coords = NTuple{3,Float64}[]
        for i in 0:5
            angle = deg2rad(60i + 30)
            x = c_coords[i+1][1] + ch_bond * cos(angle)
            y = c_coords[i+1][2] + ch_bond * sin(angle)
            push!(h_coords, (x, y, 0.0))
        end

        geo = ""
        for (i, (x, y, z)) in enumerate(c_coords)
            geo *= "C $(x) $(y) $(z) $(i);"
        end
        for (i, (x, y, z)) in enumerate(h_coords)
            geo *= "H $(x) $(y) $(z) $(i);"
        end
    elseif name == "cr2"
        # Cr₂的平衡键长约为1.68 Å（168 pm）
        # 参考：J. Chem. Phys. 96, 6796 (1992) 等文献
        a = 1.68 * ratio
        
        geo = "
        Cr 0.0 0.0 0.0;
        Cr 0.0 0.0 $(a);
        "
    elseif name == "ch3oh"
        # 甲醇 (Methanol)
        geo = "
        C  $(-0.046 * ratio)  $( 0.662 * ratio)  $( 0.000 * ratio);
        O  $(-0.046 * ratio)  $(-0.758 * ratio)  $( 0.000 * ratio);
        H  $(-1.085 * ratio)  $( 1.030 * ratio)  $( 0.000 * ratio);
        H  $( 0.468 * ratio)  $( 1.034 * ratio)  $( 0.887 * ratio);
        H  $( 0.468 * ratio)  $( 1.034 * ratio)  $(-0.887 * ratio);
        H  $( 0.865 * ratio)  $(-1.077 * ratio)  $( 0.000 * ratio);
        "
    elseif name == "c2h5oh"
        # 乙醇 (Ethanol)
        geo = "
        C  $(-1.196 * ratio)  $(-0.231 * ratio)  $( 0.000 * ratio);
        C  $( 0.117 * ratio)  $( 0.525 * ratio)  $( 0.000 * ratio);
        O  $( 1.213 * ratio)  $(-0.380 * ratio)  $( 0.000 * ratio);
        H  $(-1.258 * ratio)  $(-0.871 * ratio)  $( 0.886 * ratio);
        H  $(-1.258 * ratio)  $(-0.871 * ratio)  $(-0.886 * ratio);
        H  $(-2.053 * ratio)  $( 0.446 * ratio)  $( 0.000 * ratio);
        H  $( 0.158 * ratio)  $( 1.176 * ratio)  $( 0.888 * ratio);
        H  $( 0.158 * ratio)  $( 1.176 * ratio)  $(-0.888 * ratio);
        H  $( 2.000 * ratio)  $( 0.165 * ratio)  $( 0.000 * ratio);
        "
    elseif name == "hcooh"
        # 甲酸 (Formic Acid)
        geo = "
        C  $( 0.138 * ratio)  $( 0.370 * ratio)  $( 0.000 * ratio);
        O  $(-0.957 * ratio)  $(-0.347 * ratio)  $( 0.000 * ratio);
        O  $( 1.196 * ratio)  $(-0.187 * ratio)  $( 0.000 * ratio);
        H  $(-1.745 * ratio)  $( 0.218 * ratio)  $( 0.000 * ratio);
        H  $( 0.091 * ratio)  $( 1.464 * ratio)  $( 0.000 * ratio);
        "
    elseif name == "ch3cooh"
        # 乙酸 (Acetic Acid)
        geo = "
        C  $(-1.396 * ratio)  $( 0.103 * ratio)  $( 0.000 * ratio);
        C  $( 0.061 * ratio)  $( 0.125 * ratio)  $( 0.000 * ratio);
        O  $( 0.638 * ratio)  $(-1.073 * ratio)  $( 0.000 * ratio);
        O  $( 0.730 * ratio)  $( 1.139 * ratio)  $( 0.000 * ratio);
        H  $( 1.597 * ratio)  $(-0.958 * ratio)  $( 0.000 * ratio);
        H  $(-1.761 * ratio)  $( 0.627 * ratio)  $( 0.888 * ratio);
        H  $(-1.761 * ratio)  $( 0.627 * ratio)  $(-0.888 * ratio);
        H  $(-1.803 * ratio)  $(-0.906 * ratio)  $( 0.000 * ratio);
        "
    elseif name == "c4h10"
        # 丁烷 (Butane, Anti-conformation)
        geo = "
        C  $(-1.921 * ratio)  $( 0.313 * ratio)  $( 0.000 * ratio);
        C  $(-0.583 * ratio)  $(-0.428 * ratio)  $( 0.000 * ratio);
        C  $( 0.583 * ratio)  $( 0.428 * ratio)  $( 0.000 * ratio);
        C  $( 1.921 * ratio)  $(-0.313 * ratio)  $( 0.000 * ratio);
        H  $(-1.961 * ratio)  $( 1.399 * ratio)  $( 0.000 * ratio);
        H  $(-2.428 * ratio)  $(-0.066 * ratio)  $( 0.885 * ratio);
        H  $(-2.428 * ratio)  $(-0.066 * ratio)  $(-0.885 * ratio);
        H  $(-0.543 * ratio)  $(-1.072 * ratio)  $( 0.879 * ratio);
        H  $(-0.543 * ratio)  $(-1.072 * ratio)  $(-0.879 * ratio);
        H  $( 0.543 * ratio)  $( 1.072 * ratio)  $( 0.879 * ratio);
        H  $( 0.543 * ratio)  $( 1.072 * ratio)  $(-0.879 * ratio);
        H  $( 1.961 * ratio)  $(-1.399 * ratio)  $( 0.000 * ratio);
        H  $( 2.428 * ratio)  $( 0.066 * ratio)  $( 0.885 * ratio);
        H  $( 2.428 * ratio)  $( 0.066 * ratio)  $(-0.885 * ratio);
        "
    elseif name == "c3h8"
        geo = "
        C   $( 0.000 * ratio)  $( 0.000 * ratio)  $( 0.586 * ratio);
        C   $( 0.000 * ratio)  $( 1.276 * ratio)  $(-0.259 * ratio);
        C   $( 0.000 * ratio)  $(-1.276 * ratio)  $(-0.259 * ratio);
        H   $( 0.885 * ratio)  $( 0.000 * ratio)  $( 1.229 * ratio);
        H   $(-0.885 * ratio)  $( 0.000 * ratio)  $( 1.229 * ratio);
        H   $( 0.000 * ratio)  $( 2.164 * ratio)  $( 0.379 * ratio);
        H   $( 0.882 * ratio)  $( 1.328 * ratio)  $(-0.895 * ratio);
        H   $(-0.882 * ratio)  $( 1.328 * ratio)  $(-0.895 * ratio);
        H   $( 0.000 * ratio)  $(-2.164 * ratio)  $( 0.379 * ratio);
        H   $( 0.882 * ratio)  $(-1.328 * ratio)  $(-0.895 * ratio);
        H   $(-0.882 * ratio)  $(-1.328 * ratio)  $(-0.895 * ratio);
        "
    elseif name == "c3h6"
        # 丙烯 (Propene), Cs 对称性
        geo = "
        C  $( 1.274 * ratio)  $( 0.273 * ratio)  $( 0.000 * ratio);
        C  $( 0.000 * ratio)  $(-0.188 * ratio)  $( 0.000 * ratio);
        C  $(-1.196 * ratio)  $( 0.725 * ratio)  $( 0.000 * ratio);
        H  $( 1.442 * ratio)  $( 1.343 * ratio)  $( 0.000 * ratio);
        H  $( 2.133 * ratio)  $(-0.385 * ratio)  $( 0.000 * ratio);
        H  $(-0.134 * ratio)  $(-1.268 * ratio)  $( 0.000 * ratio);
        H  $(-2.146 * ratio)  $( 0.187 * ratio)  $( 0.000 * ratio);
        H  $(-1.157 * ratio)  $( 1.365 * ratio)  $( 0.880 * ratio);
        H  $(-1.157 * ratio)  $( 1.365 * ratio)  $(-0.880 * ratio);
        "        
    else
        throw(DomainError("No corresponding geometry name!"))
    end

    return geo
end


function molecule_geometry(name::String, ratio::Float64)
    if name in ["h$i" for i in 2:2:120]
        nH = parse(Int, match(r"\d+", name).match)
        return h_chain(nH, ratio)
    else
        return mole_geo(name, ratio)
    end
end


function build_ethane_geometry(ratio=1.0, phi_deg=60.0)
    a1 = 1.54 * ratio
    a2 = 1.09 * ratio
    θ = deg2rad(109.5)
    
    (c1x, c1y, c1z) = (0.0, 0.0, 0.0)
    (c2x, c2y, c2z) = (a1, 0.0, 0.0)
    
    # C1上的氢原子（基准角度：0°, 120°, 240°）
    h1x = a2 * cos(θ)
    h1y = a2 * sin(θ) * cos(0.0) # 明确写出cos(0)和sin(0)以保持对称性
    h1z = a2 * sin(θ) * sin(0.0)
    
    h2x = a2 * cos(θ)
    h2y = a2 * sin(θ) * cos(deg2rad(120))
    h2z = a2 * sin(θ) * sin(deg2rad(120))
    
    h3x = a2 * cos(θ)
    h3y = a2 * sin(θ) * cos(deg2rad(240))
    h3z = a2 * sin(θ) * sin(deg2rad(240))
    
    # C2上的氢原子（引入二面角旋转参数 phi_deg）
    # 当 phi_deg = 0 时为重叠式，phi_deg = 60 时为交叉式
    h4x = a1 - a2 * cos(θ)  
    h4y = a2 * sin(θ) * cos(deg2rad(phi_deg))
    h4z = a2 * sin(θ) * sin(deg2rad(phi_deg))
    
    h5x = a1 - a2 * cos(θ)
    h5y = a2 * sin(θ) * cos(deg2rad(phi_deg + 120))
    h5z = a2 * sin(θ) * sin(deg2rad(phi_deg + 120))
    
    h6x = a1 - a2 * cos(θ)
    h6y = a2 * sin(θ) * cos(deg2rad(phi_deg + 240))
    h6z = a2 * sin(θ) * sin(deg2rad(phi_deg + 240))
    
    geo = """
    C $(c1x) $(c1y) $(c1z)
    C $(c2x) $(c2y) $(c2z)
    H $(h1x) $(h1y) $(h1z)
    H $(h2x) $(h2y) $(h2z)
    H $(h3x) $(h3y) $(h3z)
    H $(h4x) $(h4y) $(h4z)
    H $(h5x) $(h5y) $(h5z)
    H $(h6x) $(h6y) $(h6z)
    """

    return geo
end


function build_ethylene_geometry(ratio=1.0, phi_deg=0.0)
    # 乙烯的标准几何参数
    a_cc = 1.339 * ratio  # C=C 双键键长
    a_ch = 1.087 * ratio  # C-H 键长
    θ = deg2rad(121.2)    # H-C-C 键角
    
    # C1和C2的位置 (沿X轴分布)
    (c1x, c1y, c1z) = (0.0, 0.0, 0.0)
    (c2x, c2y, c2z) = (a_cc, 0.0, 0.0)
    
    # C1上的氢原子 (固定在 XY 平面，z = 0)
    # 向量向 -X 方向发散，因此 x 分量为负 (因为 cos(121.2°) < 0)
    h1x = a_ch * cos(θ)
    h1y = a_ch * sin(θ)
    h1z = 0.0
    
    h2x = a_ch * cos(θ)
    h2y = -a_ch * sin(θ)
    h2z = 0.0
    
    # C2上的氢原子基准位置 (尚未旋转，与C1共面，反向对称)
    # 向量向 +X 方向发散，所以相对 C2 的 x 偏移量取 -cos(θ) 使其为正
    base_h3x = -a_ch * cos(θ)
    base_h3y = a_ch * sin(θ)
    
    base_h4x = -a_ch * cos(θ)
    base_h4y = -a_ch * sin(θ)
    
    # 将旋转角度转换为弧度
    phi = deg2rad(phi_deg)
    
    # 应用绕 X 轴的旋转矩阵:
    # y' = y*cos(phi) - z*sin(phi)
    # z' = y*sin(phi) + z*cos(phi)
    # 因为基准态的 z = 0，所以简化为下式：
    h3x = c2x + base_h3x
    h3y = base_h3y * cos(phi)
    h3z = base_h3y * sin(phi)
    
    h4x = c2x + base_h4x
    h4y = base_h4y * cos(phi)
    h4z = base_h4y * sin(phi)
    
    geo = """
    C $(c1x) $(c1y) $(c1z)
    C $(c2x) $(c2y) $(c2z)
    H $(h1x) $(h1y) $(h1z)
    H $(h2x) $(h2y) $(h2z)
    H $(h3x) $(h3y) $(h3z)
    H $(h4x) $(h4y) $(h4z)
    """
    return geo
end


function init_scf(
    name::String,
    ratio::Float64;
    basis::String="sto-3g",
    run_cisd::Bool=true,
    run_mp2::Bool=true,
    run_ccsd::Bool=true,
    run_fci::Bool=true,
    is_save::Bool=false,
)
    e_scf  = 0.0
    e_cisd = 0.0
    e_mp2  = 0.0
    e_ccsd = 0.0
    e_fci  = 0.0

    geo = molecule_geometry(name, ratio)

    mol = gto.M(atom=geo, basis=basis, spin=0.0, symmetry=true)
    println("Use symmetry. Molecule point group: $(mol.groupname)")

    norb = mol.nao_nr()
    nelec::Tuple{Int64,Int64} = mol.nelec
    energy_nuc::Float64 = mol.energy_nuc()
    println("Norb: $(norb)   Ne: $(nelec)")

    norb > 120 && return ;

    mf = scf.RHF(mol)
    println("Running RHF...")
    mf.kernel()
    orbsym = mf.orbsym
    e_scf = mf.e_tot
    e_scale = mf.e_tot

    orbsym = hasproperty(mf, :orbsym) ? mf.orbsym : ones(Int64, norb)

    if run_cisd
        mf_ci = ci.CISD(mf)
        println("Running CISD ...")
        mf_ci.kernel()
        e_cisd = mf_ci.e_tot
        e_scale = mf_ci.e_tot
    end

    if run_mp2
        mf_mp2 = mp.MP2(mf)
        println("Running MP2 ...")
        mf_mp2.kernel()
        e_mp2 = mf_mp2.e_tot
        e_scale = mf_mp2.e_tot
    end

    if run_ccsd
        mf_cc = cc.CCSD(mf)
        println("Running CCSD ...")
        mf_cc.kernel()
        e_ccsd = mf_cc.e_tot
        e_scale = mf_cc.e_tot
    end

    if run_fci
        mf_fci = fci.FCI(mf)
        println("Running FCI ...")
        mf_fci.kernel()
        e_fci = mf_fci.e_tot
        e_scale = mf_fci.e_tot
        println(@sprintf("E(FCI) = %.14f  E_corr = %.14f", e_scale, e_scale - mf.e_tot))
    end

    one_body_mo::Array{Float64,2}, two_body_mo::Array{Float64,4} = pyfun.mol_int(mf)

    filepath = joinpath(jld2path, "$(basis)/$(name)-$(ratio)-$(basis).jld2")

    if is_save
        jldopen(filepath, "w") do file
            file["norb"] = norb
            file["nelec"] = nelec
            file["orbsym"] = orbsym
            file["energy_nuc"] = energy_nuc
            file["one_body_mo"] = one_body_mo
            file["two_body_mo"] = two_body_mo
            file["e_scale"] = e_scale
        end
        println("Saved to $(filepath)")
    end

    return e_scale, e_scf, e_cisd, e_mp2, e_ccsd, e_fci
end


function init_scf(
    phi_deg::Float64;
    run_cisd::Bool=true,
    run_mp2::Bool=true,
    run_ccsd::Bool=true,
    run_fci::Bool=true,
)
    e_scf  = 0.0
    e_cisd = 0.0
    e_mp2  = 0.0
    e_ccsd = 0.0
    e_fci  = 0.0

    geo = build_ethylene_geometry(1.0, phi_deg)
    mol = gto.M(atom=geo, basis="sto-3g", spin=0.0, symmetry=true)
    println("Use symmetry. Molecule point group: $(mol.groupname)")

    norb = mol.nao_nr()
    nelec::Tuple{Int64,Int64} = mol.nelec
    energy_nuc::Float64 = mol.energy_nuc()
    println("Norb: $(norb)   Ne: $(nelec)")

    norb > 120 && return ;

    mf = scf.RHF(mol)
    println("Running RHF...")
    mf.kernel()
    orbsym = mf.orbsym
    e_scf = mf.e_tot
    e_scale = mf.e_tot

    orbsym = hasproperty(mf, :orbsym) ? mf.orbsym : ones(Int64, norb)

    if run_cisd
        mf_ci = ci.CISD(mf)
        println("Running CISD ...")
        mf_ci.kernel()
        e_cisd = mf_ci.e_tot
        e_scale = mf_ci.e_tot
    end

    if run_mp2
        mf_mp2 = mp.MP2(mf)
        println("Running MP2 ...")
        mf_mp2.kernel()
        e_mp2 = mf_mp2.e_tot
        e_scale = mf_mp2.e_tot
    end

    if run_ccsd
        mf_cc = cc.CCSD(mf)
        println("Running CCSD ...")
        mf_cc.kernel()
        e_ccsd = mf_cc.e_tot
        e_scale = mf_cc.e_tot
    end

    if run_fci
        mf_fci = fci.FCI(mf)
        println("Running FCI ...")
        mf_fci.kernel()
        e_fci = mf_fci.e_tot
        e_scale = mf_fci.e_tot
        println(@sprintf("E(FCI) = %.14f  E_corr = %.14f", e_scale, e_scale - mf.e_tot))
    end

    one_body_mo::Array{Float64,2}, two_body_mo::Array{Float64,4} = pyfun.mol_int(mf)

    filepath = joinpath(jld2path, "c2h4-1.0-$(phi_deg)-sto-3g.jld2")
    
    jldopen(filepath, "w") do file
        file["norb"] = norb
        file["nelec"] = nelec
        file["orbsym"] = orbsym
        file["energy_nuc"] = energy_nuc
        file["one_body_mo"] = one_body_mo
        file["two_body_mo"] = two_body_mo
        file["e_scale"] = e_scale
    end
    println("Saved to $(filepath)")

    return e_scale, e_scf, e_cisd, e_mp2, e_ccsd, e_fci
end


if abspath(PROGRAM_FILE) == @__FILE__
    # for nh in 50:2:120
    #     init_scf(
    #         "h$(nh)", 1.0, 
    #         run_mp2=false, 
    #         run_ccsd=false, 
    #         run_fci=false, 
    #         is_save=true)
    # end
    # init_scf(
    #     ARGS[1], 1.0, basis=ARGS[2],
    #     run_cisd=false,
    #     run_mp2=false, 
    #     run_ccsd=false, 
    #     run_fci=false, 
    #     is_save=parse(Bool, ARGS[3]))

    # for n in ["lih", "beh2", "h2o", "nh3", "ch4", "n2", "co", "h2co", "c2h4", "c2h6", "co2", "hcn", "hf", "hcl", "c2", "o2", "sih4", "ch3oh", "c2h5oh", "hcooh", "ch3cooh", "c4h10"]
    #     for b in ["sto-3g", "6-31g", "cc-pvdz", "cc-pvtz"]
    #             init_scf(
    #                 n, 1.0, 
    #                 basis=b,
    #                 run_cisd=false,
    #                 run_mp2 =false, 
    #                 run_ccsd=false, 
    #                 run_fci =false, 
    #                 is_save =true)
    #     end
    # end

    # e_scfs = []
    # for r::Float64 in 0.5:0.1:2.5
    #     e_scale, e_scf, e_cisd, e_mp2, e_ccsd, e_fci = init_scf(ARGS[1], r, basis=ARGS[2], run_mp2=false, run_ccsd=false, run_fci=false, is_save=false)
    #     push!(e_scfs, e_scf)
    # end
    # for e in e_scfs
    #     println(e)
    # end
    init_scf("c2h4", 1.0, basis="sto-3g", run_mp2=false, run_ccsd=false, run_fci=false, is_save=false)
end

 