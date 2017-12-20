all: cpp matlab

cpp:
	g++ -std=c++11 -O3 solve.cpp -o solve
matlab:
	/usr/bin/g++ -c -DMX_COMPAT_32   -D_GNU_SOURCE -DMATLAB_MEX_FILE  -I"/v/filer4b/software/matlab-r2015b/extern/include" -I"/v/filer4b/software/matlab-r2015b/simulink/include" -ansi -fexceptions -fPIC -fno-omit-frame-pointer -pthread -std=c++11 -O3 -ffast-math -DNDEBUG /scratch/cluster/ianyen/ConvexNMF/ScalableConvexNMF/ColumnGen/MixMaxCut.cpp -o MixMaxCut.o
	/usr/bin/g++ -pthread -Wl,--no-undefined  -shared -O3 -ffast-math -Wl,--version-script,"/v/filer4b/software/matlab-r2015b/extern/lib/glnxa64/mexFunction.map" MixMaxCut.o   -Wl,-rpath-link,/v/filer4b/software/matlab-r2015b/bin/glnxa64 -L"/v/filer4b/software/matlab-r2015b/bin/glnxa64" -lmx -lmex -lmat -lm -lstdc++ -o MixMaxCut.mexa64
