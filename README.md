# BinSim

example下面有例子, 目前包含: 

diag, vqe, adapt_vqe

实时演化, 虚时演化, 量子相位估值

ssvqe, adapt_ssvqe, enpt2, qse, qEOM

支持分子体系, 周期性体系, 任意模型体系


example/test.jl能够运行即可

为了防止每次运行时, 都需要重跑pyscf, 因此可以将电子积分等事先存储下来, 如果没有, 参看py/py_output.jl, 自己按照格式存储所需电子积分即可, 也可以使用Pycall即时运行.

运行前需前往src/Makefile, 需修改cpu, gpu架构(就是CXXARCH_FLAGS := -march=znver4 -mtune=znver4, 简便起见可以直接全部使用native), 以及gcc, cuda路径(服务器上module load后, 直接就是g++, nvcc).
