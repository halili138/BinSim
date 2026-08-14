# BinSim

BinSim 是一个用于量子多体/量子化学模型模拟的 Julia 代码库，核心计算由 Julia 调用本仓库 `src/extern` 中的 C++/CUDA 动态库完成。当前代码支持：

- 分子体系、周期性体系（PBC）和自定义模型体系；
- FCI / Davidson 对角化；
- VQE、ADAPT-VQE；
- 实时演化（TDVA / RK4）；
- CPU OpenMP 后端以及可选 CUDA 后端。
- 位串类型支持 UInt32 / UInt64 / UInt128（对称和非对称基组）

## 依赖版本

### 系统与编译器

| 组件 | 当前代码中的要求/默认值 | 说明 |
| --- | --- | --- |
| Linux | 推荐 | 动态库路径和 Makefile 按 Linux `.so` 组织。 |
| C++ 编译器 | GCC/G++ 13+ 或其他支持 C++20 的编译器 | `src/Makefile` 使用 `-std=c++20`、OpenMP、`-fPIC -shared`。 |
| CPU 指令集 | 默认 `-march=znver4 -mtune=znver4` | 需要按本机 CPU 修改，通用机器可改为 `-march=native -mtune=native`。 |
| CUDA Toolkit | 默认路径 `/usr/local/cuda-12.9/bin/nvcc` | 只在构建 CUDA 动态库时需要；无 GPU 时可注释 CUDA 目标。 |
| NVIDIA GPU 架构 | 默认 `sm_120` | 需要按实际 GPU 修改，例如 A100 常用 `sm_80`。 |

### Julia

建议使用 Julia 1.11 或更高版本。本仓库已提供 `Project.toml` 和 `Manifest.toml`（Julia 1.11.7），**推荐直接使用**：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

这会自动安装 `Manifest.toml` 中锁定的所有精确版本，保证可复现。

如果你的 Julia 版本与 `Manifest.toml` 不一致，可以手动添加依赖作为备选：

```bash
julia --project=. -e 'using Pkg; Pkg.add(["BitIntegers","JLD2","Combinatorics","Optim","NLSolversBase","LineSearches","DataFrames","Arpack","LinearMaps","PyCall"])'
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

### 3. 设置线程环境变量

`jl/binsim.jl` 会读取并解析 `OMP_NUM_THREADS`，因此运行前必须设置为整数。

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

### 4. 编译 C++/CUDA 动态库

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
lib/libcuotf.so
```

CPU 主要动态库：

- `src/lib/libham.so` / `lib/libham_real.so` — 哈密顿量构建
- `src/lib/libbasis.so` — BasisManager
- `src/lib/libdiag.so` — Davidson 对角化
- `src/lib/libotf_native.so` — 原生算符（hvec/expm/grad 等, Ti=uint32/64, 编译最快）

> **版本控制约定**：`src/lib/*.so` 是本地编译产物，不随仓库提交。

## 运行示例

所有示例需在 `example/` 目录下运行，因为它们通过 `include("../jl/binsim.jl")` 加载主模块。

### CPU VQE 示例

```bash
cd example
export OMP_NUM_THREADS=8
julia --project=.. host_vqe.jl c2h4 1.0 sto-3g
```

### CUDA VQE 示例（串行单 GPU）

从 `callback/vqe_uccgsd_n2_1.0_6-31g.jld2` 读取已优化初值 `x0` 后重启优化：

```bash
cd example
julia --project=.. cuda_vqe.jl n2 1.0 6-31g
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
