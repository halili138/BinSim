CXX := g++
CXXARCH_FLAGS := -march=znver4 -mtune=znver4
# CXXARCH_FLAGS := -march=native -mtune=native
# CXXARCH_FLAGS := -march=skylake-avx512 -mtune=skylake-avx512
# CXXARCH_FLAGS := -march=icelake-server -mtune=sapphirerapids # dcu节点
# CXXARCH_FLAGS := -march=icelake-server -mtune=icelake-server # a100_80g节点
CXXFLAGS := -std=c++20 $(CXXARCH_FLAGS) -O3\
            -fopenmp -fPIC -shared \
            -fomit-frame-pointer -mpopcnt -mbmi2 \
            -fopenmp-simd -funsafe-math-optimizations -fassociative-math \

CUXX := /usr/local/cuda-12.9/bin/nvcc
# CUXX := nvcc
CUXXARCH_FLAGS := -arch=sm_120
CUXXFLAGS := -std=c++20 $(CUXXARCH_FLAGS) -O3 --extended-lambda --expt-relaxed-constexpr \
			 -Xcompiler -fPIC -shared \
			 --use_fast_math \


INC_FLAGS := -I./include
THIRD_PARTY_INC := -I third_party/ankerl/unordered_dense/include

LIKWID_BASE  := /public/software/likwid/install
LIKWID_MACRO := -DLIKWID_PERFMON -I$(LIKWID_BASE)/include
LIKWID_LIBS  := -L$(LIKWID_BASE)/lib -Wl,-rpath=$(LIKWID_BASE)/lib -llikwid

TARGETS := lib/libham.so lib/libbasis.so lib/libdiag.so lib/libotf.so \
		   lib/libcuotf.so \
		   lib/libdist.so lib/libcudist.so

.PHONY: all clean

all: $(TARGETS)

lib:
	mkdir -p lib

lib/libham.so: extern/ham.cpp | lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) $(THIRD_PARTY_INC) -o $@ $<

lib/libbasis.so: extern/basis.cpp| lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -o $@ $<

lib/libdiag.so: extern/diag.cpp | lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -o $@ $<

lib/libotf.so: extern/otf.cpp | lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -o $@ $<

lib/libcuotf.so: extern/cuotf.cu | lib
	$(CUXX) $(CUXXFLAGS) $(INC_FLAGS) -o $@ $<

lib/libdist.so: extern/dist.cpp | lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -o $@ $<

lib/libcudist.so: extern/cudist.cu | lib
	$(CUXX) $(CUXXFLAGS) $(INC_FLAGS) -o $@ $<

clean:
	rm -rf lib/*.so
