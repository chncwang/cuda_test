project(N3LDG-classifier-gpu)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})


if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")   # Disable min/max macros in windef.h
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W1 /MP")   # -Wall produces 20k warnings. Enable parallel compilation
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -fno-finite-math-only -Wall -Wno-missing-braces -std=c++11 -Ofast -g -march=native")
endif()

# SET(CMAKE_BUILD_TYPE "Debug")  
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")  

if(CMAKE_BUILD_TYPE MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w -msse3 -funroll-loops  -std=c++11 -O1 -pg" )
else()
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -w -std=c++11 -Ofast -march=native" )
endif()

if(USE_FLOAT)
    add_definitions(-DUSE_FLOAT=1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_FLOAT")
endif()

if(USE_CUDA)
    add_definitions(-DUSE_GPU)
    if(MSVC)
        include_directories(D:/CUDA/sample/common/inc)
        set(CUDA_cublas_LIBRARY D:/CUDA/development/lib/x64)
        set(CUDA_cublas_device_LIBRARY D:/CUDA/development/lib/x64)
        set(LIBS ${LIBS} D:/CUDA/development/lib/x64/cublas.lib)
        set(LIBS ${LIBS} D:/CUDA/development/lib/x64/cublas_device.lib)
    else()
        include_directories(/usr/local/cuda-8.0/samples/common/inc)
    endif()
    find_package(CUDA REQUIRED)
    set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
    message("CUDA_INCLUDE: " ${CUDA_INCLUDE_DIRS})
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    #list(APPEND CUDA_LIBRARIES /usr/lib64/libpthread.so)
    MESSAGE("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    list(REMOVE_ITEM CUDA_LIBRARIES -lpthread)
    set(LIBS ${LIBS} ${CUDA_LIBRARIES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_HAS_CUDA_FP16 -DEIGEN_USE_GPU")
endif()


include_directories(matrix/include)
include_directories(matrix/cnmem/include)
include_directories(eigen)
include_directories(core)

add_subdirectory(matrix)
set(LIBS ${LIBS} matrix)
add_subdirectory(src)
