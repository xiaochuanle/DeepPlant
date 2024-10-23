cmake_minimum_required(VERSION 3.22)
project(DeepPlant)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
    message(STATUS "CMAKE_BUILD_TYPE = Debug")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
    message(STATUS "CMAKE_BUILD_TYPE = Release")
endif ()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_COMPILER "/usr/bin/x86_64-linux-gnu-gcc-11")
set(CMAKE_CXX_COMPILER "/usr/bin/x86_64-linux-gnu-g++-11")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11)
set(CUDNN_INCLUDE_DIR /usr/local/cuda-11/include)
set(CUDNN_LIBRARY /usr/local/cuda-11/lib64)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
set(CAFFE2_USE_CUDNN TRUE)


find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)
find_package(spdlog REQUIRED)

link_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Pod5/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/htslib
)
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Pod5/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/htslib/include
)

add_executable(
        ${PROJECT_NAME}
        main.cpp

        utils/utils_func.cpp
        utils/utils_func.h
        utils/utils_thread.cpp
        utils/utils_thread.h

        DataLoader/SamRead.cpp
        DataLoader/SamRead.h
        DataLoader/Pod5Data.cpp
        DataLoader/Pod5Data.h
        DataLoader/Reference_Reader.cpp
        DataLoader/Reference_Reader.h

        Call_Modification/Call_Modification.cpp
        Call_Modification/Call_Modification.h

        Feature_Extractor/Feature_Extractor.cpp
        Feature_Extractor/Feature_Extractor.h

        3rdparty/cnpy/cnpy.cpp
        3rdparty/cnpy/cnpy.h
        3rdparty/threadpool/threadpool.h
        3rdparty/argparse/argparse.h
)

target_link_libraries(
        ${PROJECT_NAME}
        ${TORCH_LIBRARIES}
        ${ZLIB_LIBRARIES}
        "libpod5_format.so"
        "libhts.so"
        spdlog::spdlog_header_only
)
