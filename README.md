# BinSim

BinSim 是一个用于量子多体/量子化学模型模拟的 Julia 代码库，核心计算由 Julia 调用本仓库 `src/extern` 中的 C++/CUDA 动态库完成。当前代码支持：

- 分子体系、周期性体系和自定义模型体系；
- FCI / Davidson 对角化；
- VQE、ADAPT-VQE、SSVQE、QSE、qEOM、ENPT2；
- 实时演化、虚时演化、量子相位估计；
- CPU OpenMP 后端、MPI 分布式后端，以及可选 CUDA 后端。

> 注意：本仓库当前没有 `Project.toml` / `Manifest.toml`，因此 Julia 依赖需要手动安装。下面的版本以当前源码和 Makefile 为准，建议在固定环境中自行生成 `Project.toml`/`Manifest.toml` 以保证可复现。

## 目录结构

```text
.
├── binsim.jl              # 主入口，加载 Julia 依赖和各功能模块
├── cunetwork.jl           # CUDA 版本网络计算入口
├── cudistribute.jl        # CUDA + MPI 分布式入口
├── distribute.jl          # MPI 分布式入口
├── src/
│   ├── Makefile           # C++/CUDA 动态库构建脚本
│   ├── extern/            # C++/CUDA 源文件
│   ├── include/           # C++/CUDA 头文件
│   ├── lib/               # 生成或已存在的 .so 动态库
│   └── third_party/       # 第三方头文件依赖
├── example/               # Julia 示例
└── py/                    # 生成分子积分/绘图等 Python 辅助脚本
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
| MPI | 可选 | `distribute.jl` 和 `example/test_dist.jl` 需要。 |
| LIKWID | 可选/当前未链接 | Makefile 中保留了 `LIKWID_*` 变量，但默认编译命令未使用。 |

### Julia

建议使用 Julia 1.10 或更高版本。当前主入口 `binsim.jl` 直接使用以下包：

```julia
using Pkg
Pkg.add([
    "BitIntegers",
    "JLD2",
    "Combinatorics",
    "Optim",
    "NLSolversBase",
    "LineSearches",
    "CPUTime",
    "FFTW",
    "DataFrames",
    "Arpack",
    "LinearMaps",
    "DifferentialEquations",
    "RecursiveArrayTools",
    "MPI",                 # 分布式功能需要；binsim.jl 当前会 include distribute.jl，因此也会加载 MPI
    "CUDA",                # 仅使用 cunetwork.jl / cudistribute.jl 时需要
    "PyCall"               # 仅部分示例和 py/py_output.jl 需要
])
```

Julia 标准库依赖包括 `LinearAlgebra`、`SparseArrays`、`Random`、`Printf`、`Dates`、`Base.Threads` 等，不需要单独安装。

### Python（可选）

Python 只用于生成/读取部分分子积分、测试脚本和绘图脚本。按需安装：

```bash
python3 -m pip install numpy opt-einsum pyscf matplotlib
```

其中：

- `py/jw_test.py` 需要 `numpy`、`opt-einsum`、`pyscf`；
- `py/mole_pbc_int.py` 需要 `numpy`、`opt-einsum`；
- `py/pict.py` 需要 `matplotlib`、`numpy`；
- `example/supf.jl` 和 `py/py_output.jl` 需要 Julia `PyCall`。

## 安装

### 1. 克隆仓库

```bash
git clone <repo-url>
cd binsim
```

### 2. 安装 Julia 依赖

如果你还没有为本仓库创建 Julia 环境，可以在仓库根目录执行：

```bash
julia --project=. -e 'using Pkg; Pkg.add(["BitIntegers","JLD2","Combinatorics","Optim","NLSolversBase","LineSearches","CPUTime","FFTW","DataFrames","Arpack","LinearMaps","DifferentialEquations","RecursiveArrayTools","MPI","CUDA","PyCall"])'
```

如果只运行 CPU 示例但仍加载 `binsim.jl`，也需要 `MPI`，因为当前 `binsim.jl` 会加载 `distribute.jl`。

### 3. 设置线程环境变量

`binsim.jl` 会读取并解析 `OMP_NUM_THREADS`，因此运行前必须设置为整数：

```bash
export OMP_NUM_THREADS=8
export JULIA_NUM_THREADS=8
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

然后编译：

```bash
make
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

### CPU 模型示例：Heisenberg / Ising

`example/ising.jl` 从命令行读取量子比特数：

```bash
export OMP_NUM_THREADS=8
export JULIA_NUM_THREADS=8
julia --project=. -t 8 example/ising.jl 8
```

脚本中当前默认构造 Heisenberg 模型；如果要改为 Ising 模型，可以参考文件中的注释切换 `ising_module` 和初态。

### 综合功能示例

`example/test.jl` 会构造 `n2 / sto-3g` 分子并运行 FCI、VQE、ADAPT-VQE、虚时演化、ENPT2、QSE、qEOM 等流程：

```bash
export OMP_NUM_THREADS=8
export JULIA_NUM_THREADS=8
julia --project=. -t 8 example/test.jl
```

该示例依赖分子积分数据。为避免每次运行都重新调用 PySCF，可预先生成并保存积分；数据格式可参考 `py/py_output.jl`。

### MPI 分布式示例

安装并配置 MPI 与 Julia `MPI.jl` 后，可以运行：

```bash
export OMP_NUM_THREADS=4
export JULIA_NUM_THREADS=4
mpiexec -n 2 julia --project=. -t 4 example/test_dist.jl
```

不同 MPI 实现可能需要先执行 `MPI.jl` 的构建/配置步骤，并确保 `mpiexec` 与 Julia 绑定的 MPI 库一致。

### CUDA 示例

确认 `src/lib/libcuotf.so` / `src/lib/libcudist.so` 已按本机 GPU 架构编译后，运行：

```bash
export OMP_NUM_THREADS=8
export JULIA_NUM_THREADS=8
julia --project=. -t 8 example/mole_cuda_test.jl
```

如果出现 CUDA 架构或驱动错误，请先确认 `CUXXARCH_FLAGS`、CUDA Toolkit 版本、NVIDIA 驱动和 `CUDA.jl` 可用性。

## 数据文件说明

默认数据目录为仓库同级的 `../jld2file/`：

```julia
const jld2path::String = joinpath(@__DIR__, "../jld2file/")
```

因此从仓库根目录运行时，积分数据通常应放在：

```text
/workspace/jld2file/
```

如果你希望数据目录在仓库内，例如 `binsim/jld2file/`，需要修改 `binsim.jl` 中的 `jld2path`。

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
make
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

解决：重新执行依赖安装命令，或进入 Julia REPL 后执行：

```julia
using Pkg
Pkg.activate(".")
Pkg.add("缺失的包名")
```

### 6. `MPI` 相关加载或运行错误

原因：`MPI.jl` 与系统 MPI 不匹配，或使用了错误的 `mpiexec`。

解决：

- 确认系统 MPI、`mpiexec` 和 Julia `MPI.jl` 使用同一套 MPI；
- 必要时重新配置/构建 `MPI.jl`；
- 单机 CPU 运行也需要安装 `MPI` 包，因为当前 `binsim.jl` 会加载 `distribute.jl`。

### 7. 分子示例找不到积分数据或重复运行 PySCF 很慢

原因：分子积分数据没有预生成，或者数据目录与 `jld2path` 不一致。

解决：

- 参考 `py/py_output.jl` 生成并保存积分；
- 检查数据是否位于 `../jld2file/`；
- 或修改 `binsim.jl` 中的 `jld2path` 指向你的数据目录。

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
- 为了可复现，建议后续添加 `Project.toml` 和 `Manifest.toml`，并记录 Julia 包版本。
