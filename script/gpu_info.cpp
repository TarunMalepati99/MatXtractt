#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    cudaDeviceProp prop;
    int device;
    
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads dimensions: (" 
              << prop.maxThreadsDim[0] << ", " 
              << prop.maxThreadsDim[1] << ", " 
              << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max grid dimensions: (" 
              << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " 
              << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Memory clock rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "L2 cache size: " << prop.l2CacheSize / 1024.0 << " KB" << std::endl;

    return 0;
}