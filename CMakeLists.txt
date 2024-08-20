cmake_minimum_required(VERSION 3.10)

project(TCSpMVlib CXX C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CPP_STANDARD 11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_80) 

set(the_description "cuda")  
# set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src) 


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

target_link_libraries(TCSpMVlib ${CUSPARSE_LIBRARY})

set(CUDA_TEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/test)
set(CUDA_UTILITIES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/utilities) 

file(GLOB cuda_utilities "${CUDA_UTILITIES_PATH}/*.cpp")
message(STATUS "CUDA_UTILITIES_FILES: ${cuda_utilities}")

file(GLOB CUDA_TEST_FILES "${CUDA_TEST_PATH}/test_spmv.cpp")
message(STATUS "CUDA_TEST_FILES: ${CUDA_TEST_FILES}")

file(GLOB CUDA_ANALYZE_FILES "${CUDA_TEST_PATH}/analyze.cpp")


if(CUDA_TEST_FILES)
    #fp64
    add_executable(TCSpMVlib_perf ${CUDA_TEST_FILES} ${cuda_utilities})
    target_link_libraries(TCSpMVlib_perf TCSpMVlib)
    target_include_directories(TCSpMVlib_perf PRIVATE ${PROJECT_SOURCE_DIR}/include)

    #analyze
    add_executable(TCSpMVlib_analyze ${CUDA_ANALYZE_FILES} ${cuda_utilities})
else()
    message(FATAL_ERROR "No test files found!")
endif()