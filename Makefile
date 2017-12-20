.PHONY: clean
MATLABDIR ?= //usr/local/MATLAB/R2014a
# It should be the location of where matlab is installed
CXX ?= g++
CFLAGS = -std=c++11 -w -fopenmp -lm -Wall -Wconversion -O3 -fPIC -I$(MATLABDIR)/extern/include -I..
LDFLAGS = -fopenmp
LIBS = blas/blas.a
CC ?= gcc
MEX = $(MATLABDIR)/bin/mex
MEX_OPTION = CC="$(CXX)" CXX="$(CXX)" CFLAGS="$(CFLAGS)" CXXFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
# comment the following line if you use MATLAB on a 32-bit computer
MEX_OPTION += -largeArrayDims
MEX_EXT = $(shell $(MATLABDIR)/bin/mexext)
MAXCUTdir ?= MixingSDPSolve
MAXCUTlocal ?= $(MAXCUTdir)/MixMaxCutSparseAAT.cpp
MAXCUTlocal2 ?= $(MAXCUTdir)/MixMaxCut.cpp
all: clean w_solver.$(MEX_EXT) MixMaxCutSparseAAT.$(MEX_EXT) all_statistic.$(MEX_EXT) MixMaxCut.$(MEX_EXT)

w_solver.$(MEX_EXT): w_solver.cpp 
	$(MEX) $(MEX_OPTION) w_solver.cpp -Ilocallib/include -Llocallib/lib locallib/lib/liblbfgs-1.10.so 
MixMaxCutSparseAAT.$(MEX_EXT): $(MAXCUTlocal)
	$(MEX) $(MEX_OPTION) $(MAXCUTlocal) -I$(MAXCUTdir) 
MixMaxCut.$(MEX_EXT): $(MAXCUTlocal2)
	$(MEX) $(MEX_OPTION) $(MAXCUTlocal2) -I$(MAXCUTdir) 
all_statistic.$(MEX_EXT):all_statistic.cpp
	$(MEX) $(MEX_OPTION) all_statistic.cpp
clean:
	rm -f *~ *.o *.$(MEX_EXT)