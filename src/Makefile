CXX := g++
CXXARCH_FLAGS := -march=znver4 -mtune=znver4 -O3
CXXFLAGS := -std=c++20 $(CXXARCH_FLAGS) \
            -fopenmp -fPIC -shared \
            -fomit-frame-pointer -mpopcnt -mbmi2 \
            -fopenmp-simd -funsafe-math-optimizations -fassociative-math \
            -ffast-math

INC_FLAGS := -I./include
THIRD_PARTY_INC := -I ../../third_party/ankerl/unordered_dense/include

LIKWID_BASE  := /public/software/likwid/install
LIKWID_MACRO := -DLIKWID_PERFMON -I$(LIKWID_BASE)/include
LIKWID_LIBS  := -L$(LIKWID_BASE)/lib -Wl,-rpath=$(LIKWID_BASE)/lib -llikwid

TARGETS := lib/libham.so lib/libbasis.so lib/libdiag.so \
		   lib/libotf.so

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

clean:
	rm -rf lib/*.so
