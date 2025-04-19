cmake_minimum_required(VERSION 3.10)

project(MatXtractProj CXX C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CPP_STANDARD 11)

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()


set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_80 -rdc=true)
option(ENABLE_CUDA_MEMCHECK "Enable cuda-memcheck support" OFF)
option(USE_FP64 "Enable fp64 support" ON)


if(ENABLE_CUDA_MEMCHECK)
    message(STATUS "cuda-memcheck is enabled")
    set(CMAKE_BUILD_TYPE Debug)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -O0")
else()
    message(STATUS "cuda-memcheck is disabled")
endif()

if(USE_FP64)
    message(STATUS "FP64 is enabled")
    add_definitions(-Dfp64)
else()
    message(STATUS "FP64 is disabled")
endif()

set(the_description "cuda")  
# set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src) 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB cuda_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cu")

include_directories( ${PROJECT_SOURCE_DIR}/include)

find_package(CUDA REQUIRED)
find_library(CUSPARSE_LIBRARY NAMES libcusparse.so PATHS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib")
find_package(OpenMP REQUIRED)


if(CUDA_FOUND AND CUSPARSE_LIBRARY)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    message(STATUS "cuSPARSE library found: ${CUSPARSE_LIBRARY}")
else()
    message(FATAL_ERROR "CUDA or cuSPARSE not found!")
endif()

message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")

cuda_add_library(MatXtractlib ${cuda_sources})


set(CUDA_TEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/test)
set(CUDA_UTILITIES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/utilities) 


file(GLOB cuda_utilities "${CUDA_UTILITIES_PATH}/*.cpp")
add_library(cuda_utilities_lib STATIC ${cuda_utilities})
target_link_libraries(cuda_utilities_lib ${CUSPARSE_LIBRARY})


file(GLOB CUDA_TEST_FILES "${CUDA_TEST_PATH}/test_cdspmv.cpp")
message(STATUS "CUDA_TEST_FILES: ${CUDA_TEST_FILES}")


file(GLOB CUDA_TCTEST_FILES "${CUDA_TEST_PATH}/test_autotc_spmv.cpp")
# file(GLOB CUDA_TCTEST_FILES "${CUDA_TEST_PATH}/test_tcspmv.cpp")

if(CUDA_TEST_FILES)
    #cd kernel
    add_executable(cuda_perftest ${CUDA_TEST_FILES})
    target_link_libraries(cuda_perftest MatXtractlib cuda_utilities_lib)
    target_include_directories(cuda_perftest PRIVATE ${PROJECT_SOURCE_DIR}/include)

    #auto kernel
    add_executable(matxtract_perftest ${CUDA_TCTEST_FILES})
    target_link_libraries(matxtract_perftest MatXtractlib cuda_utilities_lib OpenMP::OpenMP_CXX)
    target_include_directories(matxtract_perftest PRIVATE ${PROJECT_SOURCE_DIR}/include)
else()
    message(FATAL_ERROR "No test files found!")
endif()






# 添加 ML 文件夹路径
set(ML_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ML)

# 添加 extract_features.cpp 的构建规则
file(GLOB ML_SOURCES "${ML_PATH}/extract_features.cpp")

add_executable(extract_features ${ML_SOURCES})
target_include_directories(extract_features PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(extract_features cuda_utilities_lib MatXtractlib)