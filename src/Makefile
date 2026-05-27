CXX := g++
CXXARCH_FLAGS := -march=znver4 -mtune=znver4 -O3
CXXFLAGS := -std=c++20 $(CXXARCH_FLAGS) \
            -fopenmp -fPIC -shared \
            -fomit-frame-pointer -mpopcnt -mbmi2 \
            -fopenmp-simd -funsafe-math-optimizations -fassociative-math \
            -ffast-math

CUXX := /usr/local/cuda-12.9/bin/nvcc
CUXXARCH_FLAGS := -arch=sm_120
CUXXFLAGS := -std=c++20 $(CUXXARCH_FLAGS) -O3 --extended-lambda --expt-relaxed-constexpr \
			 -Xcompiler -fPIC -shared

INC_FLAGS := -I./include
THIRD_PARTY_INC := -I ../../third_party/ankerl/unordered_dense/include

LIKWID_BASE  := /public/software/likwid/install
LIKWID_MACRO := -DLIKWID_PERFMON -I$(LIKWID_BASE)/include
LIKWID_LIBS  := -L$(LIKWID_BASE)/lib -Wl,-rpath=$(LIKWID_BASE)/lib -llikwid

# MPI.jl (used for Julia integration)
MPI_JL_HOME := /home/halili/.julia/artifacts/117d9048128625bb3418b0b4cca48739c5d28740
MPI_JL_INC  := -I$(MPI_JL_HOME)/include
MPI_JL_LIBS := -L$(MPI_JL_HOME)/lib -Wl,-rpath=$(MPI_JL_HOME)/lib -lmpi

TARGETS := lib/libham.so lib/libbasis.so lib/libdiag.so lib/libotf.so \
		   lib/libcuotf.so lib/libdist.so

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

lib/libdist.so: extern/distribute.cpp| lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) $(MPI_JL_INC) -o $@ $< $(MPI_JL_LIBS)
	
clean:
	rm -rf lib/*.so
