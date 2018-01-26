
Unfortunately, the process varies on every single machine. Here I show an example:

swig -c++ -python -o iSDR_module.cpp PyiSDRcpp.i

g++ -fno-strict-aliasing -g -fwrapv -O4 -Wall -fPIC -std=c++11 -I -c PyiSDRcpp.cpp -o PyiSDRcpp.o

gcc -c -fpic -I ../src/MxNE.cpp ../src/iSDR.cpp iSDR_modulec.cpp

gcc -shared -Wl,--no-as-needed PyiSDRcpp.o MxNE.o iSDR.o iSDR_module.o -O4 -Wall \
-DWITH_MKLBLAS -DMKL_ILP64 -m64 -march=native -DMKL_ILP64 -m64 -march=native -L \
-lmatio -lhdf5 -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -ldl \
-lmkl_rt -liomp5 -lstdc++ -o _PyiSDRcpp.so

An example of how to use the Python wrapper can be found in example_pyiSDR.ipy
