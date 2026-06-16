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
│   ├── network.jl                 # CPU 网络计算 (BasisManager, OTF, OTF_Functions)
│   ├── cunetwork.jl               # CUDA 网络计算 (CuBasisManager, CuOTF_Functions; 自动安装 CUDA.jl)
│   ├── distribute.jl              # MPI 分布式基础设施 (GlobalMemMap, SubTopology)
│   ├── distnetwork.jl             # CPU 分布式统一接口 (DistributedFunctions)
│   ├── cudistribute.jl            # CUDA 分布式基础设施 (CuSubTopology)
│   ├── cudistnetwork.jl           # CUDA 分布式统一接口 (CuDistributedFunctions: Serial/NVLink/Hybrid)
│   ├── hamiltonian.jl             # 哈密顿量构建 (JW / Bravyi-Kitaev 变换)
│   ├── geo.jl                     # 分子几何结构定义
│   ├── load_data.jl               # 体系信息加载 (PySCF 调用、缓存读写)
│   ├── save_int.jl                # PySCF 积分生成并保存为 .jld2
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
│   ├── include/                   # C++/CUDA 头文件
│   ├── lib/                       # 编译生成的 .so 动态库
│   └── third_party/               # 第三方头文件依赖
├── example/                       # Julia 示例
│   ├── test.jl                    # 综合功能（FCI/VQE/ADAPT-VQE/ITE/QSE/qEOM）
│   ├── ising.jl / from_pyscf.jl   # 模型体系 / PySCF 直接调用
│   ├── test_dist.jl               # CPU 分布式（原始手动 setup）
│   ├── test_distnetwork.jl        # CPU 分布式（DistributedFunctions 统一接口）
│   ├── test_cunetwork.jl          # CUDA SerialOOC 分布式（单卡突破显存上限）
│   ├── test_cudist_serial.jl      # CUDA OOC 原始手动实现（参考用）
│   ├── test_cudist_nvlink.jl      # CUDA NVLink 多卡原始实现（参考用）
│   ├── test_cudist_hybrid.jl      # CUDA HybridOOC 原始实现（参考用）
│   ├── ite.jl / rte.jl / qpe.jl   # 虚时演化 / 实时演化 / 量子相位估计
│   ├── post.jl / module.jl        # 后处理 / TDVA 模块
│   ├── pbc.jl                     # 周期性体系
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

### 配置 PyCall 使用 Conda 虚拟环境

如果你使用 Conda 管理 Python 环境：

```bash
# 1. 创建并激活 Conda 环境
conda create -n binsim python=3.11 -y && conda activate binsim

# 2. 安装 Python 依赖
pip install numpy opt-einsum pyscf matplotlib openfermion

# 3. 让 Julia 的 PyCall 绑定到此环境的 Python
julia --project=. -e '
    ENV["PYTHON"] = readchomp(`which python3`);
    using Pkg;
    Pkg.build("PyCall")
'
```

重启 Julia 后，`using PyCall` 将使用 Conda 环境中的 Python。可通过以下命令验证：

```bash
julia --project=. -e 'using PyCall; @pyimport sys; println(sys.executable)'
```

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

### 3. 安装 MPI 启动器 (mpiexecjl)

Julia 的 `MPI.jl` 推荐使用 `mpiexecjl` 作为启动器，避免系统 MPI 与 Julia MPI.jl 版本冲突：

```bash
julia -e 'using MPI; MPI.install_mpiexecjl(destdir=joinpath(homedir(), ".julia", "bin"))'
```

安装后将 `~/.julia/bin` 加入 `PATH`：

```bash
export PATH=$HOME/.julia/bin:$PATH
```

之后用 `mpiexecjl` 替代 `mpiexec` 运行分布式代码。

### 4. 设置线程环境变量

`jl/binsim.jl` 会读取并解析 `OMP_NUM_THREADS`，因此运行前必须设置为整数。**单机运行时**：

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

**MPI 分布式运行时**（详见下方 MPI 分布式示例），需在脚本内解除线程绑定：

```julia
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")
```

如果在 SLURM 中运行，可以让 `OMP_NUM_THREADS` 与 `SLURM_CPUS_PER_TASK` 保持一致。

### 5. 编译 C++/CUDA 动态库

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

### CPU 分布式示例

分布式计算需要 **三条必须配置**，缺少任意一条均会导致性能严重下降或进程被锁核：

1. **MPI 侧**：`mpiexec --bind-to none` — 禁止 MPI 将进程绑定到固定核心
2. **OpenMP 侧**：在脚本内执行 `delete!(ENV, "OMP_PROC_BIND")` / `delete!(ENV, "OMP_PLACES")` — 放开线程绑定，让 OS 自由调度
3. **Julia 侧**：`mpiexecjl`（或用 `mpiexec` 并确保绑定的 MPI 库一致）

```bash
cd example
export OMP_NUM_THREADS=4

# 使用 mpiexecjl（推荐）
mpiexecjl -n 2 --bind-to none julia --project=.. -t 4 test_distnetwork.jl 4 h12 sto-3g

# 或使用系统 mpiexec
mpiexec -n 2 --bind-to none julia --project=.. -t 4 test_dist.jl 4 h12 sto-3g
```

**`test_distnetwork.jl` 是新封装接口**，将构建 GlobalMemMap、build_distributed_otfs、SubTopology、缓冲区分
配等操作封装为 `DistributedFunctions` 结构体（`jl/distnetwork.jl`），调用方式与 `OTF_Functions` 一致：

```julia
funcs = DistributedFunctions(basis, ham, comm)   # 一行完成所有 setup
v  = funcs.get_hf(mole.nelec)                    # 获取局部 HF 初态
Hv = funcs.zeros()
funcs.hvec(v, Hv)                                 # 分布式 H|v⟩
funcs.normalize(v)                                # 全局归一化 (Allreduce)
e = funcs.inner(v, Hv)                            # 全局 ⟨v|Hv⟩
```

**`test_dist.jl` 是原始手动实现**，展示底层 setup 过程，适合理解内部机制。

> **注意**：进程数超过对称性块数时，多出的空进程会进行 MPI 忙等待（占用 ~100% CPU）。这是正常行为——它们通过占满 CPU 迫使系统锁定睿频，实际反而加速了有数据进程的计算。

### CUDA 分布式示例

CUDA 分布式提供三种模式，封装在 `jl/cudistnetwork.jl` 中：

| 模式 | 类 | GPU数 | 数据驻留 | 适用场景 |
|------|-----|-------|---------|---------|
| **SerialOOC** | `CuDistributedFunctions{ModeSerial}` | 1 | CPU Host RAM → GPU 分片轮询 | 单卡突破显存上限 |
| **NVLink** | `CuDistributedFunctions{ModeNVLink}` | N (MPI) | GPU VRAM 常驻 | 多卡，GPU-direct MPI |
| **HybridOOC** | `CuDistributedFunctions{ModeHybrid}` | N (MPI) | CPU Host RAM + MPI | 多卡 + 超大体系 |

**SerialOOC（单卡）**：

```bash
cd example
julia --project=.. -t 1 test_cunetwork.jl h12 sto-3g 1.0 4
```

最后一个参数 `4` 为虚拟切片数（`num_chunks`），切片越多显存峰值越低。

**CUDA 非分布式（`CuOTF_Functions`）**：

```bash
julia --project=.. -t 8 mole_cuda_test.jl
```

首次 include `cunetwork.jl` 时会自动检测并安装 CUDA.jl。

NVLink 多卡需要 CUDA-aware MPI，详见下方 [CUDA-aware MPI 配置](#cuda-aware-mpi-配置)。

## CUDA-aware MPI 配置（NVLink 多卡必读）

系统默认的 OpenMPI（如 `apt install openmpi-bin`）**不带 CUDA 支持**，GPU 间通信会退化到 CPU 内存中转（PCIe 换乘）。要激活真正的 GPU-direct NVLink 直通，必须使用 CUDA-aware MPI。

### 验证当前 MPI 是否支持 CUDA

```bash
julia --project=. -e 'using MPI; println(MPI.has_cuda())'
```

`false` 表示无法激活 NVLink 直通。

### 路线 A：源码编译 CUDA-aware OpenMPI（推荐，无 sudo）

**1. 下载并编译：**

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.bz2
tar -xvf openmpi-4.1.6.tar.bz2 && cd openmpi-4.1.6

# 自动探测 CUDA 路径（需确认 nvcc 在 PATH 中，或手动指定 --with-cuda=/path/to/cuda）
./configure --with-cuda --prefix=$HOME/openmpi_cuda
make -j $(nproc)
make install  # 安装到 $HOME/openmpi_cuda，无需 sudo
```

**2. 配置环境变量**（追加到 `~/.bashrc`）：

```bash
export PATH=$HOME/openmpi_cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openmpi_cuda/lib:$LD_LIBRARY_PATH
source ~/.bashrc
```

**3. 通知 Julia MPI.jl**：

```julia
using MPIPreferences
MPIPreferences.use_system_binary(mpiexec=expanduser("~/openmpi_cuda/bin/mpiexec"))
```

重启 Julia 后 `MPI.has_cuda()` 返回 `true`。

**HPC 集群（module 环境）**：如果 CUDA 通过 `module load` 管理，configure 时可省略路径让脚本自动探测：

```bash
module load cuda/12.6
./configure --with-cuda --prefix=$HOME/openmpi_cuda
```

每次运行代码前需先执行 `module load cuda/12.6` 以确保 CUDA 动态库可用。

### 路线 B：CPU bounce buffer（无需重编译，可立即使用）

不追求极致速度时，在 Julia 代码中手动做显存→CPU→MPI→显存的 D2H/H2D 中转。`jl/cudistnetwork.jl` 的 `ModeNVLink` 已内置独立的 `d_recv` 缓冲区用于此模式。

> 关键细节：`MPI.VBuffer` 接收端必须是全新 `CUDA.zeros` 产生的纯基指针（Base Pointer），**绝不能**使用 `@view` 偏移指针或 Julia 显存池的子指针——`cuIpcGetMemHandle` 只接受 `cudaMalloc` 原始地址。

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
### 9. MPI 分布式性能显著慢于预期 / CPU 占用锁定在 200%

原因：MPI 默认将进程绑定到固定核心（`--bind-to core`），OpenMP 线程在该核心上自旋等待，导致缓存争抢。

解决：在 `mpiexec` / `mpiexecjl` 后加上 `--bind-to none`；

同时确认脚本内已设置：
```julia
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")
```

```bash
mpiexecjl -n 2 --bind-to none julia --project=.. -t 4 test_distnetwork.jl 4 h12 sto-3g
```

## 开发建议

- 修改 C++/CUDA 代码后，执行 `cd src && make clean && make`。
- 修改 Julia 算法代码后，优先运行较小规模的 `example/ising.jl` 做快速验证。
- 在集群环境中运行前，确认 `OMP_NUM_THREADS`、`JULIA_NUM_THREADS`、`SLURM_CPUS_PER_TASK` 和 BLAS 线程数符合预期。
- 新增 Julia 依赖时，记得更新 `Project.toml`（通过 `Pkg.add`）并提交更新后的 `Manifest.toml`。
