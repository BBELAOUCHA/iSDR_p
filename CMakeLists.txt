cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 11)  # use c++11
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(iSDR C CXX)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#set(BLA_VENDOR Intel10_64lp)
#find_package(BLAS REQUIRED)

option(MKL_ROOT_DIR "MKL root directory" "")
option(MKL_USE_parallel "Use MKL parallel" True)
option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" False)
option(matio_DIR "matio directory" "")
option(FLENS_INCLUDE_DIR "flens directory" "")

set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")

set(MKL_ROOT_DIR /misc/opt/intel2018/mkl)
find_package(MKL REQUIRED)

find_package(matio REQUIRED)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(OPENMP OpenMP::OpenMP_CXX)
endif()

include_directories(${CMAKE_SOURCE_DIR}/inc ${FLENS_INCLUDE_DIR} ${matio_INCLUDE_DIR})

find_path(FLENS flens.cxx HINTS ${FLENS_INCLUDE_DIR}/flens)
if (NOT FLENS)
    message("FLENS not found update the variable FLENS_DIR")
endif()
add_definitions(-DWITH_MKLBLAS)

link_directories(${MKL_LIBRARY_DIR}) 

add_subdirectory(src)

add_executable(iSDR_p main.cpp)
target_link_libraries(iSDR_p PUBLIC iSDR ${OPENMP})

add_executable(iSDR_cv main_cv.cpp)
target_link_libraries(iSDR_cv PUBLIC iSDR ${OPENMP})

add_subdirectory(test)
