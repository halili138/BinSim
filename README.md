# BinSim

BinSim 是一个用于量子多体/量子化学模型模拟的 Julia 代码库，核心计算由 Julia 调用本仓库 `src/extern` 中的 C++/CUDA 动态库完成。当前代码支持：

- 分子体系、周期性体系（PBC）和自定义模型体系；
- FCI / Davidson 对角化；
- VQE、ADAPT-VQE、SSVQE、QSE、qEOM、ENPT2；
- 实时演化（PVQD、TDVA）、虚时演化、量子相位估计；
- CPU OpenMP 后端、MPI 分布式后端，以及可选 CUDA 后端。

## 目录结构

```text
.
├── Project.toml                   # Julia 项目依赖声明
├── Manifest.toml                  # 精确版本锁定（Julia 1.11.7）
├── jl/                            # Julia 源码
│   ├── binsim.jl                  # 主入口：加载依赖、全局常量、include 各模块
│   ├── cunetwork.jl               # CUDA 网络计算（自动安装 CUDA.jl）
│   ├── cudistribute.jl            # CUDA + MPI 分布式
│   ├── distribute.jl              # MPI 分布式
│   ├── geo.jl                     # 分子几何结构定义
│   ├── load_data.jl               # 体系信息加载（PySCF 调用与缓存）
│   ├── save_int.jl                # PySCF 积分生成并保存
│   ├── hamiltonian.jl             # 哈密顿量构建
│   ├── binqubitaabb.jl            # 算符表示（AABB 形式）
│   ├── binqubitabab.jl            # 算符表示（ABAB 形式）
│   ├── davidson.jl                # Davidson 对角化
│   ├── ansatz.jl / vqe.jl         # VQE 相关（含 ADAPT-VQE、SSVQE）
│   ├── method.jl                  # 后处理（ENPT2、QSE、qEOM）
│   ├── vqite.jl                   # 虚时演化（ITE）
│   ├── vqrte.jl                   # 实时演化（RTE）
│   ├── network.jl                 # CPU 网络计算
│   ├── symm.jl                    # 对称性处理
│   ├── tools.jl / integer.jl      # 基础工具
│   └── ...
├── src/
│   ├── Makefile                   # 默认构建脚本（znver4）
│   ├── a10080gmake                # A100 80G 构建配置
│   ├── dcumake                    # DCU 构建配置
│   ├── extern/                    # C++/CUDA 源文件
│   ├── include/                   # C++/CUDA 头文件
│   ├── lib/                       # 编译生成的 .so 动态库
│   └── third_party/               # 第三方头文件依赖
├── example/                       # Julia 示例
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
| LIKWID | 可选/当前未链接 | Makefile 中保留了 `LIKWID_*` 变量，但默认编译命令未使用。 |

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

其中需要 Python 及 `PyCall` 的脚本包括：

- `example/from_pyscf.jl` — 需 `PyCall` + `pyscf`、`numpy`
- `py/jw_test.py` — 需 `numpy`、`opt-einsum`、`pyscf`、`openfermion`
- `py/mole_pbc_int.py` — 需 `numpy`、`opt-einsum`、`pyscf`
- `py/dmrg.py`、`py/fci.py`、`py/nesite.py` — 需 `pyscf`
- `py/pict.py` — 需 `matplotlib`、`numpy`

> **注意**：大部分分子示例的 PySCF 调用已通过缓存机制自动化（见下方），通常无需手动安装 openfermion，除非直接运行 `py/jw_test.py`。

## 分子积分缓存

首次运行分子示例（如 `example/test.jl`、`example/post.jl` 等）时，`build(mole)` 会自动调用 PySCF 生成单/双电子积分，并通过 `init_scf` 保存到 `jld2file/` 目录。缓存文件路径格式为：

```text
jld2file/基组/分子名-键长比例-基组.jld2
```

例如：`jld2file/sto-3g/h4-1.0-sto-3g.jld2`

**后续运行同一参数会自动读取缓存，跳过耗时的 PySCF 调用。**

- 缓存目录由 `jl/binsim.jl:41` 中的 `jld2path` 定义：`joinpath(@__DIR__, "../jld2file/")`
- 如果你修改了 `jl/geo.jl` 中的分子几何参数（如键长），**需要手动删除对应的 `.jld2` 缓存文件**，下次运行才会重新调用 PySCF 生成新积分。

## 安装

### 1. 克隆仓库

```bash
git clone <repo-url>
cd binsim_temp
```

### 2. 安装 Julia 依赖

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 3. 设置线程环境变量

`jl/binsim.jl` 会读取并解析 `OMP_NUM_THREADS`，因此运行前必须设置为整数：

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

如果在 SLURM 中运行，可以让 `OMP_NUM_THREADS` 与 `SLURM_CPUS_PER_TASK` 保持一致。

### 4. 编译 C++/CUDA 动态库

进入 `src` 目录，先按机器修改 `Makefile`：

```bash
cd src
```

需要重点检查：

```makefile
CXX := g++
CXXARCH_FLAGS := -march=znver4 -mtune=znver4
CUXX := /usr/local/cuda-12.9/bin/nvcc
CUXXARCH_FLAGS := -arch=sm_120
```

常见修改：

```makefile
CXXARCH_FLAGS := -march=native -mtune=native
CUXX := nvcc
CUXXARCH_FLAGS := -arch=sm_80   # A100 示例
```

除默认 `Makefile` 外，仓库还提供预配置的构建脚本：
- `src/a10080gmake` — 适配 A100 80G GPU
- `src/dcumake` — 适配 DCU 平台

然后编译：

```bash
make clean && make -j
cd ..
```

如果没有 CUDA/NVIDIA GPU，只想运行 CPU 代码，可以临时从 `TARGETS` 中移除或注释以下目标，再执行 `make`：

```makefile
lib/libcuotf.so
lib/libcudist.so
```

CPU 主要动态库包括：

- `src/lib/libham.so`
- `src/lib/libbasis.so`
- `src/lib/libdiag.so`
- `src/lib/libotf.so`
- `src/lib/libdist.so`

CUDA 相关动态库包括：

- `src/lib/libcuotf.so`
- `src/lib/libcudist.so`

## 运行示例

所有示例需在 `example/` 目录下运行，因为它们通过 `include("../jl/binsim.jl")` 加载主模块。

### 综合功能示例

`example/test.jl` 接收分子名称和基组，运行 FCI、VQE、ADAPT-VQE、虚时演化、ENPT2、QSE、qEOM 等流程：

```bash
cd example
export OMP_NUM_THREADS=8 JULIA_NUM_THREADS=8
julia --project=.. -t 8 test.jl n2 sto-3g
```

首次运行会自动调用 PySCF 生成积分并保存到 `jld2file/`，之后再次运行会直接读取缓存。

### 直接从 PySCF 生成积分

`example/from_pyscf.jl` 绕过缓存机制，直接调用 PySCF 生成积分并运行计算：

```bash
cd example
export OMP_NUM_THREADS=8 JULIA_NUM_THREADS=8
julia --project=.. -t 8 from_pyscf.jl
```

### MPI 分布式示例

安装并配置 MPI 与 Julia `MPI.jl` 后，可以运行：

```bash
cd example
export OMP_NUM_THREADS=4 JULIA_NUM_THREADS=4
mpiexec -n 2 julia --project=.. -t 4 test_dist.jl
```

不同 MPI 实现可能需要先执行 `MPI.jl` 的构建/配置步骤，并确保 `mpiexec` 与 Julia 绑定的 MPI 库一致。

### CUDA 示例

确认 `src/lib/libcuotf.so` / `src/lib/libcudist.so` 已按本机 GPU 架构编译后，运行：

```bash
cd example
export OMP_NUM_THREADS=8 JULIA_NUM_THREADS=8
julia --project=.. -t 8 mole_cuda_test.jl
```

首次 include `cunetwork.jl` 时会自动检测并安装 CUDA.jl。

如果出现 CUDA 架构或驱动错误，请先确认 `CUXXARCH_FLAGS`、CUDA Toolkit 版本和 NVIDIA 驱动。

## 常见错误

### 1. `ArgumentError: invalid base 10 digit 'N' in "Not Set"` 或 `parse(Int, omp_threads)` 失败

原因：未设置 `OMP_NUM_THREADS`。当前 `binsim.jl` 默认读取不到环境变量时会得到字符串 `"Not Set"`，随后 `parse(Int, omp_threads)` 会失败。

解决：

```bash
export OMP_NUM_THREADS=8
export JULIA_NUM_THREADS=8
```

### 2. `could not load library "src/lib/lib*.so"`

原因：动态库不存在、未编译、路径不对，或编译出的库与当前系统不兼容。

解决：

```bash
cd src
make clean
make -j
cd ..
```

如果没有 CUDA，请从 `TARGETS` 中移除 CUDA 动态库目标后重新编译。

### 3. `Illegal instruction (core dumped)`

原因：Makefile 中的 CPU 架构参数与当前机器不匹配，例如在非 Zen4 CPU 上使用 `-march=znver4`。

解决：将 `src/Makefile` 中的 CPU 架构参数改为：

```makefile
CXXARCH_FLAGS := -march=native -mtune=native
```

或者改为集群节点对应的架构参数后重新 `make clean && make`。

### 4. `nvcc: command not found` 或 CUDA 编译失败

原因：CUDA Toolkit 未安装、路径不正确，或 GPU 架构参数不匹配。

解决：

- 已安装 CUDA：将 `CUXX := /usr/local/cuda-12.9/bin/nvcc` 改为实际路径，或改为 `CUXX := nvcc`；
- 检查 GPU 架构，例如 A100 使用 `-arch=sm_80`；
- 不使用 GPU：从 `TARGETS` 中移除 `lib/libcuotf.so` 和 `lib/libcudist.so`。

### 5. `Package ... not found in current path`

原因：Julia 包未安装到当前环境。

解决：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 6. `MPI` 相关加载或运行错误

原因：`MPI.jl` 与系统 MPI 不匹配，或使用了错误的 `mpiexec`。

解决：

- 确认系统 MPI、`mpiexec` 和 Julia `MPI.jl` 使用同一套 MPI；
- 必要时重新配置/构建 `MPI.jl`；
- 单机 CPU 运行也需要安装 `MPI` 包，因为当前 `binsim.jl` 会加载 `distribute.jl`。

### 7. 分子示例找不到积分数据或重复运行 PySCF 很慢

原因：缓存目录与 `jld2path` 不一致，或 `jl/geo.jl` 修改后未清除旧缓存。

解决：

- 确保 `jld2file/` 位于仓库根目录（与 `jl/` 同级）；
- 修改过 `jl/geo.jl` 后，删除对应的 `jld2file/基组/分子名-键长-基组.jld2` 缓存文件；
- 或修改 `jl/binsim.jl:41` 中的 `jld2path` 指向你的数据目录。

### 8. `PyCall` 找不到 Python 包

原因：`PyCall` 绑定的 Python 环境没有安装对应 Python 包，例如 `pyscf`、`numpy`、`opt_einsum`。

解决：

```bash
python3 -m pip install numpy opt-einsum pyscf matplotlib
```

如果 `PyCall` 绑定了其他 Python，可在 Julia 中重新指定：

```julia
ENV["PYTHON"] = "/path/to/python3"
using Pkg
Pkg.build("PyCall")
```

## 开发建议

- 修改 C++/CUDA 代码后，执行 `cd src && make clean && make`。
- 修改 Julia 算法代码后，优先运行较小规模的 `example/ising.jl` 做快速验证。
- 在集群环境中运行前，确认 `OMP_NUM_THREADS`、`JULIA_NUM_THREADS`、`SLURM_CPUS_PER_TASK` 和 BLAS 线程数符合预期。
- 新增 Julia 依赖时，记得更新 `Project.toml`（通过 `Pkg.add`）并提交更新后的 `Manifest.toml`。
