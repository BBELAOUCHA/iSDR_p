add_library(iSDR SHARED MxNE.cpp iSDR.cpp CV_iSDR.cpp ReadWriteMat.cpp)
target_link_libraries(iSDR MATIO::MATIO ${MKL_LIBRARIES})
