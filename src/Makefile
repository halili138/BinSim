CXX := g++
CXXARCH_FLAGS := -march=znver4 -mtune=znver4
# CXXARCH_FLAGS := -march=icelake-server -mtune=icelake-server -mprefer-vector-width=512

CXXFLAGS := -std=c++20 $(CXXARCH_FLAGS) -O3 \
            -fopenmp -fPIC -shared \
            -fomit-frame-pointer -mpopcnt -mbmi2 \
            -fopenmp-simd -funsafe-math-optimizations -fassociative-math \
            -ffast-math

# CUXX := nvcc
# CUXXARCH_FLAGS := -arch=sm_120
# CUXXFLAGS := -std=c++20 $(CUXXARCH_FLAGS) -O3 --extended-lambda\
# 			 -Xcompiler -fPIC -shared \

INC_FLAGS := -I./include
THIRD_PARTY_INC := -I ../../third_party/ankerl/unordered_dense/include

LIKWID_BASE  := /public/software/likwid/install
LIKWID_MACRO := -DLIKWID_PERFMON -I$(LIKWID_BASE)/include
LIKWID_LIBS  := -L$(LIKWID_BASE)/lib -Wl,-rpath=$(LIKWID_BASE)/lib -llikwid

TARGETS := lib/libham.so lib/libbasis.so lib/libdiag.so \
		   lib/libnet.so lib/libagg.so

.PHONY: all clean

all: $(TARGETS)

lib:
	mkdir -p lib

lib/libham.so: extern/ham.cpp | lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) $(THIRD_PARTY_INC) -o $@ $<

lib/libbasis.so: extern/basis.cpp | lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -o $@ $<

lib/libdiag.so: extern/diag.cpp | lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -o $@ $<

lib/libnet.so: extern/net.cpp cpp/net_build.cpp cpp/net_hvec.cpp cpp/net_tvec.cpp cpp/net_grad.cpp | lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -o $@ $^

# lib/libagg.so: extern/agg.cpp cpp/agg_build.cpp cpp/agg_hvec.cpp cpp/net_build.cpp cpp/net_hvec.cpp | lib
# 	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -o $@ $^

lib/libagg.so: extern/agg.cpp cpp/agg_build.cpp cpp/agg_hvec.cpp cpp/net_build.cpp cpp/net_hvec.cpp | lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) $(LIKWID_MACRO) -o $@ $^ $(LIKWID_LIBS)

clean:
	rm -rf lib/*.so
