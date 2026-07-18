CXX       := g++
CXXFLAGS  := -std=c++20 -march=znver4 -mtune=znver4 -O3 \
             -fopenmp -fPIC -shared \
             -fomit-frame-pointer -mpopcnt -mbmi2 \
             -fopenmp-simd -funsafe-math-optimizations -fassociative-math

CUXX      := /usr/local/cuda-12.9/bin/nvcc
CUXXFLAGS := -std=c++20 -arch=sm_120 -O3 --extended-lambda --expt-relaxed-constexpr \
             -Xcompiler -fPIC -shared --use_fast_math

INC        := -I./include
ANKERL_INC := -I third_party

# 头文件集合 —— 修改任意头文件触发重编译
CORE_HDRS   := $(wildcard include/core/*.hpp)
BASIS_HDRS  := $(wildcard include/basis/*.hpp)
HAM_HDRS    := $(wildcard include/ham/*.hpp)
OP_HDRS     := $(wildcard include/op/*.hpp)
SELECT_HDRS := $(wildcard include/select/*.hpp)
DIAG_HDRS   := $(wildcard include/diag/*.hpp)

# 编译命令模板
CXX_LINK   = $(CXX) $(CXXFLAGS) $(INC) -o $@ $<
CXX_ANKERL = $(CXX) $(CXXFLAGS) $(INC) $(ANKERL_INC) -o $@ $<
CUDA_LINK  = $(CUXX) $(CUXXFLAGS) $(INC) $(ANKERL_INC) -o $@ $<

# ================================================================
.PHONY: all clean

all: lib/libham.so \
	 lib/libham_real.so \
	 lib/libbasis.so \
	 lib/libdiag.so \
     lib/libotf_native.so \
	 lib/libotf_select.so \
     lib/libdist.so \
	 lib/libcuotf.so \
	 lib/libcudist.so \

lib:
	mkdir -p lib

# ---- Hamiltonian（需要 ankerl + core/bit.hpp） ----
lib/libham.so:      extern/ham.cpp      $(CORE_HDRS) $(HAM_HDRS) | lib ; $(CXX_ANKERL)
lib/libham_real.so: extern/ham_real.cpp $(CORE_HDRS) $(HAM_HDRS) | lib ; $(CXX_ANKERL)

# ---- Basis（需要 ankerl） ----
lib/libbasis.so: extern/basis.cpp $(CORE_HDRS) $(BASIS_HDRS) $(HAM_HDRS) | lib ; $(CXX_ANKERL)

# ---- 原生算符（不依赖 ankerl，编译最快） ----
lib/libotf_native.so: extern/otf_native.cpp $(CORE_HDRS) $(OP_HDRS) $(HAM_HDRS) $(BASIS_HDRS) | lib ; $(CXX_LINK)

# ---- SCI Select（需要 ankerl + core/bit.hpp，编译最慢） ----
lib/libotf_select.so: extern/otf_select.cpp $(CORE_HDRS) $(SELECT_HDRS) $(BASIS_HDRS) $(HAM_HDRS) | lib ; $(CXX_ANKERL)

# ---- 分布式 ----
lib/libdist.so: extern/dist.cpp $(CORE_HDRS) $(BASIS_HDRS) $(OP_HDRS) $(HAM_HDRS) | lib ; $(CXX_ANKERL)

# ---- 对角化（独立，无 ankerl） ----
lib/libdiag.so: extern/diag.cpp $(DIAG_HDRS) | lib ; $(CXX_LINK)

# ---- CUDA ----
lib/libcuotf.so:  extern/cuotf.cu  | lib ; $(CUDA_LINK)
lib/libcudist.so: extern/cudist.cu | lib ; $(CUDA_LINK)

clean:
	rm -rf lib/*.so
