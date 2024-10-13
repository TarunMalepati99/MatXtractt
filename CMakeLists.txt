cmake_minimum_required(VERSION 3.10)

project(TCSpMVProj CXX C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CPP_STANDARD 11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_80) 

option(ENABLE_CUDA_MEMCHECK "Enable cuda-memcheck support" OFF)
option(USE_FP64 "Enable fp64 support" ON) # 添加 fp64 选项

if(ENABLE_CUDA_MEMCHECK)
    message(STATUS "cuda-memcheck is enabled")
    set(CMAKE_BUILD_TYPE Debug)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -O0")
else()
    message(STATUS "cuda-memcheck is disabled")
endif()

if(USE_FP64)
    message(STATUS "FP64 is enabled")
    add_definitions(-Dfp64)  # 向编译器传递 fp64 预处理宏
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

# 检查 cuSPARSE 是否被找到
if(CUDA_FOUND AND CUSPARSE_LIBRARY)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    message(STATUS "cuSPARSE library found: ${CUSPARSE_LIBRARY}")
else()
    message(FATAL_ERROR "CUDA or cuSPARSE not found!")
endif()

# 打印 CUDA 库的路径
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")

cuda_add_library(TCSpMVlib ${cuda_sources})


set(CUDA_TEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/test)
set(CUDA_UTILITIES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/utilities) 


# cusparse 包含在cuda_utilities_lib库中了
file(GLOB cuda_utilities "${CUDA_UTILITIES_PATH}/*.cpp")
add_library(cuda_utilities_lib STATIC ${cuda_utilities})
target_link_libraries(cuda_utilities_lib ${CUSPARSE_LIBRARY})


file(GLOB CUDA_TEST_FILES "${CUDA_TEST_PATH}/test_spmv.cpp")
message(STATUS "CUDA_TEST_FILES: ${CUDA_TEST_FILES}")

# file(GLOB CUDA_TCTEST_FILES "${CUDA_TEST_PATH}/tc_preprocess_fp64.cpp")
# file(GLOB CUDA_TCTEST_FILES "${CUDA_TEST_PATH}/tc_kernel_fp64.cpp")
file(GLOB CUDA_TCTEST_FILES "${CUDA_TEST_PATH}/test_tcspmv.cpp")

if(CUDA_TEST_FILES)
    #fp64
    add_executable(TCSpMVlib_perf ${CUDA_TEST_FILES})
    target_link_libraries(TCSpMVlib_perf TCSpMVlib cuda_utilities_lib)
    target_include_directories(TCSpMVlib_perf PRIVATE ${PROJECT_SOURCE_DIR}/include)

    #preprocess
    add_executable(TCSpMVlib_tcperftest ${CUDA_TCTEST_FILES})
    target_link_libraries(TCSpMVlib_tcperftest TCSpMVlib cuda_utilities_lib)
    target_include_directories(TCSpMVlib_tcperftest PRIVATE ${PROJECT_SOURCE_DIR}/include)
else()
    message(FATAL_ERROR "No test files found!")
endif()