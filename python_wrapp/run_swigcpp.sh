

MKL_FOLDER=/home/bbelaouc/Wokspace/Needed/intel
MKLROOT=${MKL_FOLDER}/compilers_and_libraries_2018.1.163/linux/mkl


swig -c++ -python -I/home/bbelaouc/Wokspace/Project/iSDR_p/inc -I/home/bbelaouc/Wokspace/Needed/FLENS -I/home/bbelaouc/Wokspace/Needed/FLENS/flens -I/usr/local/include/ -I/usr/include -o iSDR_modulecpp.cpp PyiSDRcpp.i

 g++ -fno-strict-aliasing -g -fwrapv -O4 -Wall -fPIC -std=c++11 -I/home/bbelaouc/anaconda2/include/python2.7/ -I/home/bbelaouc/Wokspace/Project/iSDR_p/inc -I/home/bbelaouc/Wokspace/Needed/FLENS -I/home/bbelaouc/anaconda2/lib/pkgconfig -I/home/bbelaouc/anaconda2/lib/python2.7/config -c PyiSDRcpp.cpp -o PyiSDRcpp.o

 gcc -c -fpic -I/home/bbelaouc/anaconda2/include/python2.7/ -I/home/bbelaouc/Wokspace/Project/iSDR_p/inc -I/home/bbelaouc/Wokspace/Needed/FLENS -I/home/bbelaouc/anaconda2/lib/pkgconfig -I/home/bbelaouc/anaconda2/lib/python2.7/config ../src/MxNE.cpp ../src/iSDR.cpp iSDR_modulecpp.cpp

gcc -shared -Wl,--no-as-needed PyiSDRcpp.o MxNE.o iSDR.o iSDR_modulecpp.o -O4 -Wall -DWITH_MKLBLAS -DMKL_ILP64 -m64 -march=native \
-I/usr/include -I${MKL_FOLDER}/mkl/include -I/home/bbelaouc/Wokspace/Needed/FLENS -I/usr/local/include/ \
-I/home/bbelaouc/Wokspace/Project/iSDR_p/inc -DMKL_ILP64 -m64 -march=native -L${MKL_FOLDER}/mkl/lib/intel64_lin -L/usr/lib64/ \
-L/home/bbelaouc/Wokspace/Needed/hdf5/hdf5/lib/ -L/home/bbelaouc/Wokspace/Needed/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin/ \
 -lmatio -lhdf5 -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -ldl -lmkl_rt -liomp5 -lstdc++ -o _PyiSDRcpp.so


or


gcc -shared -Wl,--no-as-needed PyiSDRcpp.o MxNE.o iSDR.o iSDR_modulecpp.o -O4 -Wall -DWITH_MKLBLAS -DMKL_ILP64 -m64 -march=native \
-I/usr/include -I${MKL_FOLDER}/mkl/include -I/home/bbelaouc/Wokspace/Needed/FLENS -I/usr/local/include/ \
-I/home/bbelaouc/Wokspace/Project/iSDR_p/inc -DMKL_ILP64 -m64 -march=native -L${MKL_FOLDER}/mkl/lib/intel64_lin -L/usr/lib64/ \
-L/home/bbelaouc/Wokspace/Needed/hdf5/hdf5/lib/ -L/home/bbelaouc/Wokspace/Needed/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin/ \
 -lmatio -lhdf5 -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -ldl -lmkl_rt -fopenmp -lstdc++ -o _PyiSDRcpp.so
