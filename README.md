# BinSim

BinSim 是一个用于量子多体/量子化学模型模拟的 Julia 代码库，核心计算由 Julia 调用本仓库 `src/extern` 中的 C++/CUDA 动态库完成。当前代码支持：

- 分子体系、周期性体系（PBC）和自定义模型体系；
- FCI / Davidson 对角化 / SCI (Selected Configuration Interaction) ；
- VQE、ADAPT-VQE、SSVQE、QSE、qEOM、ENPT2；
- 实时演化（PVQD、TDVA）、虚时演化、量子相位估计；
- CPU OpenMP 后端、MPI 分布式后端，以及可选 CUDA 后端。
- 位串类型支持 UInt32 / UInt64 / UInt128（对称和非对称基组）

## 目录结构

```text
.
├── Project.toml                   # Julia 项目依赖声明
├── Manifest.toml                  # 精确版本锁定（Julia 1.11.7）
├── jl/                            # Julia 源码
│   ├── binsim.jl                  # 主入口：加载依赖、全局常量、include 各模块
│   ├── sysinfo.jl                 # SysInfo 抽象类型 + Mole / Pbc 结构体定义
│   ├── geo.jl                     # 分子几何结构 + scf_mole + build(mole) 构造管线
│   ├── geo_pyscf_dist.jl          # Gao et al. Fujitsu FCI 基准集几何与能量
│   ├── network.jl                 # CPU 网络计算 (BasisManager, OTF, OTF_Functions)
│   ├── cunetwork.jl               # CUDA 网络计算 (CuBasisManager, CuOTF_Functions)
│   ├── distribute.jl              # MPI 分布式基础设施 (GlobalMemMap, SubTopology)
│   ├── distnetwork.jl             # CPU 分布式统一接口 (DistributedFunctions)
│   ├── cudistribute.jl            # CUDA 分布式基础设施 (CuSubTopology)
│   ├── cudistnetwork.jl           # CUDA 分布式统一接口 (Serial/NVLink/Hybrid)
│   ├── hamiltonian.jl             # 哈密顿量构建 (JW / Bravyi-Kitaev 变换)
│   ├── sci.jl                     # SCI 对称版 (SciBasisManager + select + hvec + pipeline)
│   ├── sci_nosym.jl               # SCI 无对称版 (SciBasisManagerNosym)
│   ├── ansatz.jl / vqe.jl         # VQE 相关 (ADAPT-VQE、SSVQE、exact-VQE)
│   ├── method.jl                  # 后处理 (ENPT2、QSE、qEOM)
│   ├── vqite.jl                   # 虚时演化 (Euler / RK4 / Krylov ITE)
│   ├── vqrte.jl                   # 实时演化 (PVQD / TDVA)
│   ├── symm.jl                    # 对称性处理
│   ├── davidson.jl                # Davidson 对角化
│   ├── tools.jl / integer.jl      # 基础工具与整数位操作
│   └── ...
├── src/
│   ├── Makefile                   # 默认构建脚本（znver4）
│   ├── a10080gmake                # A100 80G 构建配置
│   ├── dcumake                    # DCU 构建配置
│   ├── extern/                    # C++/CUDA 源文件
│   │   ├── sci_otf_native.cpp     # 原生算符 extern C 包装（Ti=uint32/64, 无 ankerl）
│   │   ├── sci_otf_select.cpp     # SCI select extern C 包装（Ti=uint32/64/128, 含 ankerl）
│   │   ├── ham.cpp / ham_real.cpp # 哈密顿量构建
│   │   ├── basis.cpp              # BasisManager 构造
│   │   ├── dist.cpp               # 分布式算符
│   │   ├── diag.cpp               # Davidson 对角化
│   │   └── cuotf.cu / cudist.cu   # CUDA 算符
│   ├── include/                   # C++/CUDA 头文件（按功能域分目录）
│   │   ├── core/                  # types.hpp  math.hpp  bit.hpp
│   │   ├── basis/                 # basis.hpp  sci_basis.hpp  nosym_basis.hpp
│   │   ├── ham/                   # ham.hpp  ham_test.hpp  otf.hpp
│   │   ├── op/                    # hvec.hpp  expm.hpp  grad.hpp  ...  dist.hpp
│   │   ├── select/                # utils.hpp  forward.hpp  nosym.hpp  declare.hpp
│   │   ├── diag/                  # davidson.hpp
│   │   └── cuda/                  # common.cuh  hvec.cuh  ...  sci_hvec.cuh
│   ├── lib/                       # 编译生成的 .so 动态库
│   └── third_party/               # 第三方头文件
│       └── ankerl/                # unordered_dense (fast hashmap)
├── example/                       # Julia 示例
│   ├── test.jl                    # 综合功能（FCI/VQE/ADAPT-VQE/ITE/QSE/qEOM）
│   ├── ising.jl / from_pyscf.jl   # 模型体系 / PySCF 直接调用
│   ├── test_sci_bitstr.jl         # SCI 对称版示例
│   ├── test_sci_nosym.jl          # SCI 无对称版示例
│   ├── test_distnetwork.jl        # CPU 分布式
│   ├── test_cunetwork.jl          # CUDA SerialOOC 分布式
│   └── ...
├── py/                            # Python 辅助脚本
├── jld2file/                      # PySCF 生成的积分缓存
├── bashs/                         # SLURM 批量提交脚本
└── data/                          # 基准测试输出数据
```

## 依赖版本

### 系统与编译器

| 组件 | 当前代码中的要求/默认值 | 说明 |
| --- | --- | --- |
| Linux | 推荐 | 动态库路径和 Makefile 按 Linux `.so` 组织。 |
| C++ 编译器 | GCC/G++ 13+ 或其他支持 C++20 的编译器 | `src/Makefile` 使用 `-std=c++20`、OpenMP、`-fPIC -shared`。 |
| CPU 指令集 | 默认 `-march=znver4 -mtune=znver4` | 需要按本机 CPU 修改，通用机器可改为 `-march=native -mtune=native`。 |
| CUDA Toolkit | 默认路径 `/usr/local/cuda-12.9/bin/nvcc` | 只在构建 CUDA 动态库时需要；无 GPU 时可注释 CUDA 目标。 |
| NVIDIA GPU 架构 | 默认 `sm_120` | 需要按实际 GPU 修改，例如 A100 常用 `sm_80`。 |
| MPI | 可选 | `jl/distribute.jl` 和 `example/test_dist.jl` 需要。 |

### Julia

建议使用 Julia 1.11 或更高版本。本仓库已提供 `Project.toml` 和 `Manifest.toml`（Julia 1.11.7），**推荐直接使用**：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

这会自动安装 `Manifest.toml` 中锁定的所有精确版本，保证可复现。

如果你的 Julia 版本与 `Manifest.toml` 不一致，可以手动添加依赖作为备选：

```bash
julia --project=. -e 'using Pkg; Pkg.add(["BitIntegers","JLD2","Combinatorics","Optim","NLSolversBase","LineSearches","CPUTime","FFTW","DataFrames","Arpack","LinearMaps","DifferentialEquations","RecursiveArrayTools","MPI","PyCall"])'
```

Julia 标准库依赖包括 `LinearAlgebra`、`SparseArrays`、`Random`、`Printf`、`Dates`、`Base.Threads` 等，不需要单独安装。

> **关于 CUDA.jl**：如果你使用 CUDA 后端（include `cunetwork.jl`），代码会自动检测 CUDA.jl 是否已安装，若未安装则自动调用 `Pkg.add("CUDA")` 进行安装，无需手动处理。

### Python（可选）

Python 只用于生成/读取分子积分、测试脚本和绘图脚本。按需安装：

```bash
python3 -m pip install numpy opt-einsum pyscf matplotlib openfermion
```

### 配置 PyCall 使用 Conda 虚拟环境

```bash
conda create -n binsim python=3.11 -y && conda activate binsim
pip install numpy opt-einsum pyscf matplotlib openfermion

julia --project=. -e '
    ENV["PYTHON"] = readchomp(`which python3`);
    using Pkg;
    Pkg.build("PyCall")
'
```

## 分子积分缓存

首次运行分子示例时，`build(mole)` 会自动调用 PySCF 生成单/双电子积分并保存到 `jld2file/` 目录。缓存文件路径格式为：

```text
jld2file/分子名-键长比例-基组.jld2
```

例如：`jld2file/n2-1.0-sto-3g.jld2`

**后续运行同一参数会自动读取缓存，跳过耗时的 PySCF 调用。**

- 缓存目录由 `jl/binsim.jl` 中的 `jld2path` 定义
- 如果你修改了 `jl/geo.jl` 中的分子几何参数（如键长），**需要手动删除对应的 `.jld2` 缓存文件**，下次运行才会重新调用 PySCF 生成新积分。

## 安装

### 1. 克隆仓库

```bash
git clone <repo-url>
cd BinSim
```

### 2. 安装 Julia 依赖

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 3. 安装 MPI 启动器 (mpiexecjl)

```bash
julia -e 'using MPI; MPI.install_mpiexecjl(destdir=joinpath(homedir(), ".julia", "bin"))'
export PATH=$HOME/.julia/bin:$PATH
```

### 4. 设置线程环境变量

`jl/binsim.jl` 会读取并解析 `OMP_NUM_THREADS`，因此运行前必须设置为整数。

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

MPI 分布式运行时需在脚本内解除线程绑定（见下方 MPI 分布式示例）。

### 5. 编译 C++/CUDA 动态库

进入 `src` 目录，先按机器修改 `Makefile`：

```bash
cd src
```

需要重点检查：

```makefile
CXX := g++
CXXFLAGS := ... -march=znver4 -mtune=znver4 ...
CUXX := /usr/local/cuda-12.9/bin/nvcc
CUXXFLAGS := ... -arch=sm_120 ...
```

常见修改：

```makefile
# 通用 CPU
CXXFLAGS 中改为 -march=native -mtune=native
# A100 GPU
CUXXFLAGS 中改为 -arch=sm_80
# 使用系统 nvcc
CUXX := nvcc
```

然后编译：

```bash
make clean && make -j
cd ..
```

如果没有 CUDA/NVIDIA GPU，只想运行 CPU 代码，注释 Makefile 中 `all` 目标里的以下行再执行 `make`：

```makefile
lib/libcuotf.so lib/libcudist.so
```

CPU 主要动态库：

- `src/lib/libham.so` / `lib/libham_real.so` — 哈密顿量构建
- `src/lib/libbasis.so` — BasisManager
- `src/lib/libdiag.so` — Davidson 对角化
- `src/lib/libsci_otf_native.so` — 原生算符（hvec/expm/grad 等, Ti=uint32/64, 编译最快）
- `src/lib/libsci_otf_select.so` — SCI select 全量（sym + nosym, Ti=uint32/64/128）
- `src/lib/libdist.so` — 分布式算符

> **版本控制约定**：`src/lib/*.so` 是本地编译产物，不随仓库提交。

## 运行示例

所有示例需在 `example/` 目录下运行，因为它们通过 `include("../jl/binsim.jl")` 加载主模块。

### 综合功能示例

```bash
cd example
export OMP_NUM_THREADS=8
julia --project=.. test.jl n2 sto-3g
```

### SCI (Selected Configuration Interaction) 示例

对称版 SCI（按点群对称性分块）：

```bash
cd example
export OMP_NUM_THREADS=8
julia --project=.. test_sci_bitstr.jl n2 1.0 sto-3g 1e-5
```

无对称版 SCI（平坦基组，无需 symmetry block）：

```bash
julia --project=.. test_sci_nosym.jl n2 1.0 sto-3g 1e-5
```

### CPU 分布式示例

```bash
cd example
export OMP_NUM_THREADS=4
mpiexecjl -n 2 --bind-to none julia --project=.. test_distnetwork.jl 4 h12 sto-3g
```

### CUDA 分布式示例

```bash
cd example
julia --project=.. test_cunetwork.jl h12 sto-3g 1.0 4
```

## 常见错误

### 1. `ArgumentError: invalid base 10 digit 'N' in "Not Set"`

未设置 `OMP_NUM_THREADS`。

解决：

```bash
export OMP_NUM_THREADS=8
```

### 2. `could not load library "src/lib/lib*.so"`

动态库不存在或未编译。

解决：

```bash
cd src && make clean && make && cd ..
```

### 3. `Illegal instruction (core dumped)`

CPU 架构参数不匹配。

解决：将 `src/Makefile` 中改为 `-march=native -mtune=native` 后重新编译。

### 4. `nvcc: command not found` 或 CUDA 编译失败

CUDA Toolkit 未安装、路径不对，或 GPU 架构参数不匹配。

解决：修改 `CUXX` 路径和 `-arch` 参数，或不使用 GPU 时注释 CUDA 目标。

### 5. 分子示例找不到积分数据或重复运行 PySCF 很慢

缓存文件路径变更。旧路径 `jld2file/基组/xxx.jld2` 已废弃，新路径为 `jld2file/xxx.jld2`（无 basis 子目录）。

解决：删除旧的 `jld2file/sto-3g/`、`jld2file/6-31g/`、`jld2file/cc-pvdz/` 目录，重新运行让 PySCF 生成新缓存。
