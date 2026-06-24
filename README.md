# BinSim

量子多体/量子化学模型模拟 Julia 代码库，核心计算由 Julia 调用 C++/CUDA 动态库完成。支持分子体系、PBC、FCI/Davidson 对角化、VQE/ADAPT-VQE/SSVQE、ENPT2/QSE/qEOM、实/虚时演化、CPU OpenMP/MPI/CUDA 后端及虚拟对称性分区。

## 1. 编译动态库

进入 `src` 目录，先按机器修改 `Makefile`：

```bash
cd src
```

需要重点检查：

```makefile
CXX := g++
CXXARCH_FLAGS := -march=znver4 -mtune=znver4      # 按本机 CPU 修改
CUXX := /usr/local/cuda-12.9/bin/nvcc             # 按实际 CUDA 路径修改
CUXXARCH_FLAGS := -arch=sm_120                    # 按实际 GPU 架构修改
```

常见修改：

```makefile
CXXARCH_FLAGS := -march=native -mtune=native
CUXX := nvcc
CUXXARCH_FLAGS := -arch=sm_80   # A100 示例
```

然后编译：

```bash
make clean && make -j
cd ..
```

**没有 NVIDIA GPU 时**：临时从 `TARGETS` 中移除或注释 `lib/libcuotf.so` 和 `lib/libcudist.so` 目标后再 `make`。

CPU 动态库一览：

| 库文件 | 功能 |
|--------|------|
| `libham.so` | 哈密顿量生成（JW 变换） |
| `libbasis.so` | 基矢管理（对称性块、行列式枚举） |
| `libdiag.so` | 对角元计算 & Davidson 辅助函数 |
| `libotf.so` | CPU on-the-fly H\|v⟩ 计算 |
| `libdist.so` | 分布式索引映射 |

> **注意**：`src/lib/*.so` 是本地编译产物，拉取代码后请重新编译，以避免 CPU/GPU 架构、CUDA 版本和系统 ABI 不一致导致运行失败。

## 2. 安装 Julia 依赖

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

这会安装 `Manifest.toml` 中锁定的精确版本（Julia 1.11.7）。

若 Julia 版本不一致，可手动添加依赖：

```bash
julia --project=. -e 'using Pkg; Pkg.add(["BitIntegers","JLD2","Combinatorics","Optim","NLSolversBase","LineSearches","CPUTime","FFTW","DataFrames","Arpack","LinearMaps","DifferentialEquations","RecursiveArrayTools","MPI","PyCall"])'
```

> **CUDA.jl**：使用 CUDA 后端时会自动检测并安装 CUDA.jl，无需手动处理。

## 3. 设置环境变量

`OMP_NUM_THREADS` 等环境变量必须提前设置，否则 `binsim.jl` 启动时会报错。

**单机运行时**：

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

**MPI 分布式运行时**，需在 Julia 脚本内解除线程绑定：

```julia
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")
```

SLURM 环境中建议让 `OMP_NUM_THREADS` 与 `SLURM_CPUS_PER_TASK` 保持一致。

## 4. 快速运行

所有示例/测试需在仓库根目录或对应子目录下运行（通过 `include("../jl/binsim.jl")` 加载模块）。

### CPU VQE

```bash
cd example
export OMP_NUM_THREADS=8 JULIA_NUM_THREADS=8
julia --project=.. host_vqe.jl n2 1.0 sto-3g
```

参数：分子名、键长比例、基组。首次运行自动调用 PySCF 生成积分缓存到 `jld2file/`。

### CPU ADAPT-VQE

```bash
cd example
julia --project=.. host_adapt_vqe.jl n2 1.0 sto-3g
```

### CPU 性能基准

```bash
cd example
julia --project=.. host_time_benchmark.jl h12 1.0 sto-3g 4
```

### CUDA VQE / ADAPT-VQE

```bash
cd example
export OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
julia --project=.. cuda_vqe.jl n2 1.0 sto-3g
julia --project=.. cuda_adapt_vqe.jl n2 1.0 sto-3g
julia --project=.. cuda_time_benchmark.jl h12 1.0 sto-3g 4
```

### CPU 分布式

**三条必须配置**：

1. `mpiexec --bind-to none` — 禁止 MPI 绑定固定核心
2. 脚本内 `delete!(ENV, "OMP_PROC_BIND")` / `delete!(ENV, "OMP_PLACES")`
3. 使用 `mpiexecjl` 或系统 `mpiexec` 并确保 MPI 库一致

```bash
cd test
export OMP_NUM_THREADS=4

# DistributedFunctions 统一接口（推荐）
mpiexecjl -n 2 --bind-to none julia --project=.. test_distnetwork.jl 4 h12 sto-3g

# 原始手动实现
mpiexec -n 2 --bind-to none julia --project=.. test_dist.jl 4 h12 sto-3g
```

`DistributedFunctions` 用法：

```julia
funcs = DistributedFunctions(basis, ham, comm)   # 一行 setup
v  = funcs.get_hf(mole.nelec)
Hv = funcs.zeros()
funcs.hvec(v, Hv)                                 # 分布式 H|v⟩
funcs.normalize(v)                                # 全局归一化
e = funcs.inner(v, Hv)                            # 全局 ⟨v|Hv⟩
```

### CUDA 分布式

三种模式（`jl/cudistnetwork*.jl`）：

| 模式 | GPU数 | 数据驻留 | 场景 |
|------|-------|---------|------|
| **SerialOOC** | 1 | CPU Host RAM → GPU 分片轮询 | 单卡突破显存上限 |
| **NVLink** | N (MPI) | GPU VRAM 常驻 | 多卡 GPUDirect MPI |
| **HybridOOC** | N (MPI) | CPU Host RAM + MPI | 多卡 + 超大体系 |

SerialOOC 示例：

```bash
cd test
julia --project=.. test_cudist_serial.jl h12 sto-3g 1.0 4
```

NVLink 多卡需 CUDA-aware MPI（见 [CUDA-aware MPI 配置](#10-cuda-aware-mpi-配置)）。

## 5. 常见错误

### 1. `ArgumentError: invalid base 10 digit 'N' in "Not Set"`

未设置 `OMP_NUM_THREADS`。解决：

```bash
export OMP_NUM_THREADS=8 JULIA_NUM_THREADS=8
```

### 2. `could not load library "src/lib/lib*.so"`

动态库未编译或路径不对：

```bash
cd src && make clean && make -j && cd ..
```

### 3. `Illegal instruction (core dumped)`

CPU 架构不匹配。将 `Makefile` 中改为 `-march=native -mtune=native` 后重新编译。

### 4. CUDA 编译失败

- 检查 `CUXX` 路径；无 GPU 则从 `TARGETS` 中移除 CUDA 目标；
- A100 使用 `-arch=sm_80`。

### 5. `Package ... not found in current path`

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 6. MPI 相关加载或运行错误

确认系统 MPI、`mpiexec` 和 Julia `MPI.jl` 使用同一套 MPI。单机运行也需安装 `MPI` 包（`binsim.jl` 加载 `distnetwork.jl`）。

### 7. 找不到积分数据 / PySCF 重复运行

- 确保 `jld2file/` 位于仓库根目录；
- 修改 `jl/geo.jl` 后删除旧缓存 `.jld2` 文件；
- 或修改 `jl/binsim.jl:64` 中的 `jld2path`。

### 8. `PyCall` 找不到 Python 包

```bash
python3 -m pip install numpy opt-einsum pyscf matplotlib
```

或重新指定 Python 路径：

```julia
ENV["PYTHON"] = "/path/to/python3"
using Pkg; Pkg.build("PyCall")
```

### 9. MPI 分布式性能异常慢 / CPU 占用锁定

原因：MPI 默认 `--bind-to core` 与 OpenMP 冲突。

解决：`mpiexec -n 2 --bind-to none ...` 并在脚本内 `delete!(ENV, "OMP_PROC_BIND")`。

## 6. MPI 启动器安装

```bash
julia -e 'using MPI; MPI.install_mpiexecjl(destdir=joinpath(homedir(), ".julia", "bin"))'
export PATH=$HOME/.julia/bin:$PATH
```

之后用 `mpiexecjl` 替代 `mpiexec`，避免系统 MPI 版本冲突。

## 7. Python / PyCall 配置

按需安装 Python 包：

```bash
python3 -m pip install numpy opt-einsum pyscf matplotlib openfermion
```

Python 用于：
- `jl/save_int.jl` / `jl/load_data.jl` — PySCF 积分生成与读取
- `py/mole_pbc_int.py` — 周期性体系积分
- `py/jw_test.py` — JW 变换验证
- `py/dmrg.py`, `py/fci.py`, `py/nesite.py` — 参考基准
- `py/pict.py` — matplotlib 绘图

Conda 环境配置：

```bash
conda create -n binsim python=3.11 -y && conda activate binsim
pip install numpy opt-einsum pyscf matplotlib openfermion

julia --project=. -e '
    ENV["PYTHON"] = readchomp(`which python3`);
    using Pkg;
    Pkg.build("PyCall")
'
```

验证：`julia --project=. -e 'using PyCall; @pyimport sys; println(sys.executable)'`

## 8. 目录结构与代码文件介绍

```text
.
├── Project.toml                           # Julia 项目依赖声明
├── Manifest.toml                          # 精确版本锁定（Julia 1.11.7）
├── jl/                                    # Julia 源码（25 个文件）
│   ├── binsim.jl                          # 主入口：加载依赖、全局常量、include 各模块
│   ├── geo.jl                             # 分子几何结构定义
│   ├── load_data.jl                       # 体系信息加载（PySCF 调用、缓存读写）
│   ├── save_int.jl                        # PySCF 积分生成并保存为 .jld2
│   ├── integer.jl                         # 整数位操作（half/double width、popcount 等）
│   ├── tools.jl                           # 基础工具函数
│   ├── binqubitaabb.jl                    # AABB 格式 Pauli 串（BinaryQubitAABB）
│   ├── binqubitabab.jl                    # ABAB 格式 Pauli 串（BinaryQubitABAB）
│   ├── hamiltonian.jl                     # 哈密顿量构建（JW/BK 变换，uint64/128/256 后端）
│   ├── symm.jl                            # 对称性处理 & 虚拟对称性分区
│   ├── network.jl                         # CPU 网络计算（BasisManager, OTF, OTF_Functions）
│   ├── davidson.jl                        # Davidson 对角化 & Arpack 封装
│   ├── ansatz.jl                          # 激发算符池（FEB 等）
│   ├── vqe.jl                             # VQE / ADAPT-VQE / SSVQE 优化
│   ├── method.jl                          # 后处理：FCI、ENPT2、QSE、qEOM
│   ├── vqite.jl                           # 虚时演化（Euler / RK4 / Krylov ITE）
│   ├── vqrte.jl                           # 实时演化（PVQD / TDVA）
│   ├── cunetwork.jl                       # CUDA 网络计算（CuBasisManager, CuOTF_Functions）
│   ├── distnetwork.jl                     # CPU 分布式（GlobalMemMap, SubTopology, DistributedFunctions）
│   ├── cudistnetwork.jl                   # CUDA 分布式入口（聚合以下子模块）
│   ├── cudistnetwork_common.jl            # CUDA 分布式公共定义
│   ├── cudistnetwork_serial.jl            # CUDA SerialOOC 模式
│   ├── cudistnetwork_nvlink.jl            # CUDA NVLink 多卡 GPUDirect 模式
│   ├── cudistnetwork_hybrid.jl            # CUDA HybridOOC 模式
│   └── cudistnetwork_davidson.jl          # CUDA 分布式 Davidson 对角化
├── src/
│   ├── Makefile                           # 构建脚本（默认 znver4 优化）
│   ├── extern/                            # C++/CUDA 源文件（7 个）
│   │   ├── ham.cpp                        # 哈密顿量生成
│   │   ├── basis.cpp                      # 基矢管理
│   │   ├── otf.cpp                        # CPU on-the-fly H|v⟩
│   │   ├── diag.cpp                       # 对角元计算 & Davidson 辅助
│   │   ├── dist.cpp                       # 分布式索引映射
│   │   ├── cuotf.cu                       # CUDA on-the-fly H|v⟩
│   │   └── cudist.cu                      # CUDA 分布式通信框架
│   ├── include/                           # C++/CUDA 头文件（16 个）
│   │   ├── host_*.hpp                     # CPU 端声明
│   │   ├── cuda_*.cuh                     # CUDA 端声明
│   │   ├── bitintegers.hpp               # 大整数位操作
│   │   └── diag.hpp                       # 对角化声明
│   ├── lib/                               # 编译生成的 .so 动态库
│   └── third_party/ankerl/                # unordered_dense 哈希表
├── example/                               # 使用示例（6 个）
│   ├── host_vqe.jl                        # CPU VQE
│   ├── host_adapt_vqe.jl                  # CPU ADAPT-VQE
│   ├── host_time_benchmark.jl             # CPU hvec / cost 函数性能计时
│   ├── cuda_vqe.jl                        # CUDA VQE
│   ├── cuda_adapt_vqe.jl                  # CUDA ADAPT-VQE
│   └── cuda_time_benchmark.jl             # CUDA hvec / cost 函数性能计时
├── test/                                  # 测试 & 高级引用（21 个）
│   ├── test.jl                            # FCI 对角化（Davidson + Arpack）
│   ├── test_jw.jl                         # JW 哈密顿量构建验证
│   ├── test_diag.jl                       # Davidson 一致性测试
│   ├── test_ite.jl                        # 虚时演化 vs FCI 参考
│   ├── test_rte.jl                        # 实肘演化（Ising 模型 pVQD）
│   ├── test_post_methods.jl               # 后处理：期望值、方差、态保真度
│   ├── test_otf_pure_excitation_diags.jl  # 纯激发算符对角元单测
│   ├── test_dist.jl                       # CPU 分布式原始实现
│   ├── test_distnetwork.jl                # CPU 分布式统一接口
│   ├── test_dist_vqe.jl                   # CPU 分布式 VQE
│   ├── test_cudist_serial.jl              # CUDA SerialOOC 原始实现
│   ├── test_cudist_nvlink.jl              # CUDA NVLink 原始实现
│   ├── test_cudist_hybrid.jl              # CUDA HybridOOC 原始实现
│   ├── test_cudist_vqe_serial.jl          # CUDA SerialOOC VQE
│   ├── test_cudist_vqe_nvlink.jl          # CUDA NVLink VQE
│   ├── test_cudist_vqe_hybrid.jl          # CUDA HybridOOC VQE
│   ├── test_vorbsym.jl                    # 轨道对称性标签统计
│   ├── test_virtual_cuserial.jl           # 虚拟对称性 + CUDA SerialOOC
│   ├── test_virtual_cudist.jl             # 虚拟对称性 + CUDA 分布式（可切换模式）
│   ├── test_virtual_cudist_nvlink.jl      # 虚拟对称性 + CUDA NVLink
│   └── test_virtual_distnetwork.jl        # 虚拟对称性 + CPU 分布式
├── py/                                    # Python 辅助脚本
├── jld2file/                              # PySCF 积分缓存
└── .gitignore
```

### `jl/` 模块依赖关系

```
binsim.jl （入口）
 ├─ integer.jl          # 整数位操作
 ├─ tools.jl            # 基础工具
 ├─ geo.jl              # 分子几何
 ├─ save_int.jl         # PySCF 积分生成
 ├─ load_data.jl        # 体系加载
 ├─ binqubitabab.jl     # Pauli 串 ABAB 格式
 ├─ binqubitaabb.jl     # Pauli 串 AABB 格式
 ├─ hamiltonian.jl      # 哈密顿量构建
 ├─ symm.jl             # 对称性 & 虚拟分区
 ├─ network.jl          # CPU 网络计算
 ├─ davidson.jl         # 对角化
 ├─ ansatz.jl           # 算符池
 ├─ vqe.jl              # VQE 优化
 ├─ method.jl           # 后处理
 ├─ vqite.jl            # 虚时演化
 ├─ vqrte.jl            # 实肘演化
 └─ distnetwork.jl      # CPU 分布式
```

CUDA 扩展模块独立加载（不通过 `binsim.jl`）：

```text
cunetwork.jl                        # CUDA 基础（自包含）
  └─ include binsim.jl

cudistnetwork.jl                    # CUDA 分布式入口
  ├─ include cunetwork.jl
  ├─ cudistnetwork_common.jl
  ├─ cudistnetwork_serial.jl        # SerialOOC
  ├─ cudistnetwork_nvlink.jl        # NVLink
  ├─ cudistnetwork_hybrid.jl        # HybridOOC
  └─ cudistnetwork_davidson.jl      # Davidson
```

## 9. 依赖版本详情

### 系统与编译器

| 组件 | 要求/默认值 | 说明 |
| --- | --- | --- |
| Linux | 推荐 | 按 `.so` 组织 |
| C++ 编译器 | GCC/G++ 13+（C++20） | `-std=c++20`、OpenMP、`-fPIC -shared` |
| CPU 指令集 | `-march=znver4 -mtune=znver4` | 通用机改 `-march=native` |
| CUDA Toolkit | `/usr/local/cuda-12.9/bin/nvcc` | 无 GPU 时可跳过 |
| GPU 架构 | `sm_120` | 按实际 GPU 修改 |
| MPI | 可选 | 分布式功能需要 |

### Julia

Julia 1.11+，`Project.toml` 依赖：
`BitIntegers`, `JLD2`, `Combinatorics`, `Optim`, `NLSolversBase`, `LineSearches`, `CPUTime`, `FFTW`, `DataFrames`, `Arpack`, `LinearMaps`, `DifferentialEquations`, `RecursiveArrayTools`, `MPI`, `PyCall`

## 10. 分子积分缓存

首次运行分子计算时，`build(mole)` 自动调用 PySCF 生成积分，通过 `init_scf`（`jl/save_int.jl`）保存到 `jld2file/基组/分子名-键长-基组.jld2`（例：`jld2file/sto-3g/h4-1.0-sto-3g.jld2`）。后续运行直接读缓存。

- 缓存路径定义：`jl/binsim.jl:64` 的 `jld2path`
- 修改 `jl/geo.jl` 后需手动删除旧缓存

## 11. CUDA-aware MPI 配置

系统默认 OpenMPI **不带 CUDA 支持**，GPU 间通信退化为 PCIe 中转。NVLink 直通需要 CUDA-aware MPI。

验证：`julia --project=. -e 'using MPI; println(MPI.has_cuda())'` —— `false` 表示不支持。

### 源码编译 OpenMPI（推荐）

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.bz2
tar -xvf openmpi-4.1.6.tar.bz2 && cd openmpi-4.1.6
./configure --with-cuda --prefix=$HOME/openmpi_cuda
make -j $(nproc) && make install
```

环境变量（追加 `~/.bashrc`）：

```bash
export PATH=$HOME/openmpi_cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openmpi_cuda/lib:$LD_LIBRARY_PATH
```

通知 Julia：

```julia
using MPIPreferences
MPIPreferences.use_system_binary(mpiexec=expanduser("~/openmpi_cuda/bin/mpiexec"))
```

### CPU bounce buffer（备选）

不追求极致速度时，`jl/cudistnetwork_nvlink.jl` 内置 D2H/H2D 中转缓冲区。

> `MPI.VBuffer` 接收端必须用 `CUDA.zeros` 纯基指针，不能用 `@view` 偏移或显存池子指针——`cuIpcGetMemHandle` 只接受原始 `cudaMalloc` 地址。

## 12. 虚拟对称性

当物理对称性块数太少，导致负载不均衡或 MPI 并行度不足时，可使用**虚拟对称性分区**将 Hilbert 空间进一步细分：给每个轨道分配随机标签（k 位），使量子数相同但虚拟标签不同的行列式落入不同 "虚拟块"。

相关测试：`test_vorbsym.jl`、`test_virtual_cuserial.jl`、`test_virtual_cudist.jl`、`test_virtual_cudist_nvlink.jl`、`test_virtual_distnetwork.jl`。

## 13. 开发建议

- 修改 C++/CUDA 代码后：`cd src && make clean && make`
- 修改 Julia 代码后优先用 `test/` 对应脚本验证
- 集群运行前确认 `OMP_NUM_THREADS`、`JULIA_NUM_THREADS`、`SLURM_CPUS_PER_TASK`、BLAS 线程数一致
- 新增 Julia 依赖时通过 `Pkg.add` 更新 `Project.toml` 并提交 `Manifest.toml`
