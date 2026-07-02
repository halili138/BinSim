CXX := g++
CXXARCH_FLAGS := -march=native -mtune=native
CXXFLAGS := -std=c++20 $(CXXARCH_FLAGS) -O3\
            -fopenmp -fPIC -shared \
            -fomit-frame-pointer -mpopcnt -mbmi2 \
            -fopenmp-simd -funsafe-math-optimizations -fassociative-math \

CUXX := /usr/local/cuda-12.9/bin/nvcc
CUXXARCH_FLAGS := -arch=sm_120
CUXXFLAGS := -std=c++20 $(CUXXARCH_FLAGS) -O3 --extended-lambda --expt-relaxed-constexpr \
			 -Xcompiler -fPIC -shared \
			 --use_fast_math \

INC_FLAGS := -I./include
THIRD_PARTY_INC := -I third_party/ankerl/unordered_dense/include

TARGETS := lib/libham.so lib/libbasis.so lib/libdiag.so lib/libotf.so \
		   lib/libcuotf.so \
		   lib/libdist.so lib/libcudist.so \
		   lib/libsci_otf_bitstr.so

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

lib/libsci_otf_bitstr.so: extern/sci_otf_bitstr.cpp | lib
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -o $@ $<

clean:
	rm -rf lib/*.so
