MKL_FOLDER= 
P_PATH=
MKLROOT=${MKL_FOLDER}/compilers_and_libraries_2017.4.196/linux/mkl

SRCS =  src/MxNE.cpp src/iSDR.cpp src/ReadWriteMat.cpp
MKL_CXXFLAGS= -DMKL_ILP64 -m64 -march=native -I${MKLROOT}/include 
MKL_LDFLAGS=  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -ldl
FlensDir =${P_PATH}/FLENS/
Matio_Dir=${P_PATH}/matio-1.5.2/build/
CC = g++
INCLUDES = -I/usr/include -I${MKL_FOLDER}/mkl/include -I${FlensDir} -I${Matio_Dir}include -I inc
CXXFLAGS = -std=gnu++11 -O4 -Wall -DWITH_MKLBLAS $(MKL_CXXFLAGS) -g
LDFLAGS = -L${MKL_FOLDER}/mkl/lib/intel64_lin -L${Matio_Dir}/lib -lmkl_rt -lblas -llapack -lmatio -lhdf5
SOURCE = main.cpp
MYPROGRAM = iSDR_p
SOURCE2 = test/test_MxNE_iSDR.cpp
MYPROGRAM2 = test_MxNE_iSDR
SOURCE3 = iSDR_cv.cpp
MYPROGRAM3 = iSDR_cv
TARGET=iSDR/
TARGET2=test/

all: $(MYPROGRAM) $(MYPROGRAM3) $(MYPROGRAM2)
$(MYPROGRAM):$(SOURCE)
	$(CC) $(INCLUDES) $(CXXFLAGS) $(SRCS) $(SOURCE) -o $(TARGET)$(MYPROGRAM) $(LDFLAGS) `pkg-config --cflags --libs matio`

$(MYPROGRAM3):$(SOURCE3)
	$(CC) $(INCLUDES) $(CXXFLAGS) $(SRCS) $(SOURCE3) -o $(TARGET)$(MYPROGRAM3) $(LDFLAGS) `pkg-config --cflags --libs matio`

$(MYPROGRAM2):$(SOURCE2)
	$(CC) $(INCLUDES) $(CXXFLAGS) $(SRCS) $(SOURCE2) -o $(TARGET2)$(MYPROGRAM2) $(LDFLAGS) `pkg-config --cflags --libs matio`

clean:
	rm -f $(TARGET)$(MYPROGRAM) $(TARGET2)$(MYPROGRAM2) $(TARGET)$(MYPROGRAM3)

