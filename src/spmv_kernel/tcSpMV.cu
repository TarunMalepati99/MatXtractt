#include "common.h"

#define WARP_SIZE 32

__device__ __forceinline__ void mma_m8n8k4(valT *acc, valT &frag_a, valT &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]) : "d"(frag_a), "d"(frag_b));
}

// Define cp_async_global_to_shared_async for copying a single element asynchronously
__device__ inline void cp_async_global_to_shared_async(void *shared_dest, const void *global_src, size_t bytes)
{
#if (__CUDA_ARCH__ >= 800)
    // Ensure the destination and source addresses are aligned to the required size
    if (bytes == 4)
    {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                "l"(shared_dest),
            "l"(global_src), "n"(4));
    }
    else if (bytes == 8)
    {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                "l"(shared_dest),
            "l"(global_src), "n"(8));
    }
    else if (bytes == 16)
    {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                "l"(shared_dest),
            "l"(global_src), "n"(16));
    }
    else
    {
        // For sizes not supported by cp.async, use synchronous copy
        memcpy(shared_dest, global_src, bytes);
    }
#else
    // For older architectures, fall back to synchronous copy
    memcpy(shared_dest, global_src, bytes);
#endif
}
// Define cp_async_bulk_global_to_shared_async for copying a bulk of data asynchronously
__device__ inline void cp_async_bulk_global_to_shared_async(void *shared_dest, const void *global_src, size_t bytes)
{
    char *shmem_ptr = reinterpret_cast<char *>(shared_dest);
    const char *gmem_ptr = reinterpret_cast<const char *>(global_src);
#if (__CUDA_ARCH__ >= 800)
    // Supported sizes for cp.async are 4, 8, or 16 bytes
    size_t offset = 0;
    // Handle initial unaligned bytes
    uintptr_t shmem_addr = reinterpret_cast<uintptr_t>(shmem_ptr + offset);
    size_t misalignment = shmem_addr % 16;
    if (misalignment != 0)
    {
        size_t align_correction = 16 - misalignment;
        memcpy(shmem_ptr + offset, gmem_ptr + offset, align_correction);
        offset += align_correction;
        bytes -= align_correction;
    }
    // Now both addresses are aligned to 16 bytes
    while (bytes >= 16)
    {
        // Ensure addresses are aligned to 16 bytes
        if (((reinterpret_cast<uintptr_t>(shmem_ptr + offset) % 16) == 0) &&
            ((reinterpret_cast<uintptr_t>(gmem_ptr + offset) % 16) == 0))
        {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                    "l"(shmem_ptr + offset),
                "l"(gmem_ptr + offset), "n"(16));
            offset += 16;
            bytes -= 16;
        }
        else
        {
            break; // Addresses not aligned for 16 bytes, handle smaller sizes
        }
    }
    while (bytes >= 8)
    {
        if (((reinterpret_cast<uintptr_t>(shmem_ptr + offset) % 8) == 0) &&
            ((reinterpret_cast<uintptr_t>(gmem_ptr + offset) % 8) == 0))
        {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                    "l"(shmem_ptr + offset),
                "l"(gmem_ptr + offset), "n"(8));
            offset += 8;
            bytes -= 8;
        }
        else
        {
            break; // Addresses not aligned for 8 bytes, handle smaller sizes
        }
    }
    while (bytes >= 4)
    {
        if (((reinterpret_cast<uintptr_t>(shmem_ptr + offset) % 4) == 0) &&
            ((reinterpret_cast<uintptr_t>(gmem_ptr + offset) % 4) == 0))
        {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                    "l"(shmem_ptr + offset),
                "l"(gmem_ptr + offset), "n"(4));
            offset += 4;
            bytes -= 4;
        }
        else
        {
            break; // Addresses not aligned, fallback to memcpy
        }
    }
    // Handle any remaining bytes with memcpy
    if (bytes > 0)
    {
        // memcpy(shmem_ptr + offset, gmem_ptr + offset, bytes);
        for (size_t i = 0; i < bytes; ++i)
        {
            shmem_ptr[offset + i] = gmem_ptr[offset + i];
        }
    }
#else
    // For older architectures, fall back to synchronous copy
    memcpy(shared_dest, global_src, bytes);
    // Copy remaining bytes manually
    // for (size_t i = 0; i < bytes; ++i)
    // {
    //     shmem_ptr[offset + i] = gmem_ptr[offset + i];
    // }

#endif
}

// Function to wait for all asynchronous copies to complete
__device__ inline void cp_async_wait_all()
{
#if (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
#else
    // No action needed for synchronous copies
#endif
}

// using namespace nvcuda::wmma;
// x_d: Input vector
// y_d: Output vector
// chunkPtr: Offsets of tcFrags for each rowChunk
// fragPtr: Fragment pointer array
// fragBit: Bitmaps for each tcFrag
// tcVal: Non-zero values in tcFrags
// sparse_AToX_index: Indices mapping tcFrag columns to x_d
__global__ void tcspmv_kernel(
    const double *__restrict__ x_d,
    double *__restrict__ y_d,
    const int *__restrict__ chunkPtr,
    const int *__restrict__ fragPtr,
    const uint32_t *__restrict__ fragBit,
    const double *__restrict__ tcVal,
    const int *__restrict__ sparse_AToX_index,
    int dRows)
{
    // printf("\n 11111111 \n");
    const int warpsPerBlock = 4;
    // const int threadsPerBlock = warpsPerBlock * 32;
    // Shared memory for double buffering tcVal and x_d
    extern __shared__ double shared_mem[];

    int per_buffer_size_tcVal = fragM * fragK;
    int per_buffer_size_x_d = fragK;
    int per_buffer_total = per_buffer_size_tcVal + per_buffer_size_x_d;

    int warpId = threadIdx.x / 32; // 0 to 3
    int laneId = threadIdx.x % 32; // 0 to 31

    double *shmem_warp = shared_mem + warpId * 2 * per_buffer_total;
    double *shmem_warp_buffer0 = shmem_warp;
    double *shmem_warp_buffer1 = shmem_warp + per_buffer_total;
    // Double buffering pointers
    double *shmem_tcVal_buffers[2] = {shmem_warp_buffer0, shmem_warp_buffer1};
    double *shmem_x_buffers[2] = {shmem_warp_buffer0 + per_buffer_size_tcVal, shmem_warp_buffer1 + per_buffer_size_tcVal};

    // Get the rowChunk this warp is responsible for
    int rowChunkIndex = blockIdx.x * warpsPerBlock + warpId;
    // int numRowChunks = (dRows + fragM - 1) / fragM;
    // if (rowChunkIndex >= numRowChunks)
    //     return;

    // 当前chunk的起始和终止frag
    int tcFragStart = chunkPtr[rowChunkIndex];
    int tcFragEnd = chunkPtr[rowChunkIndex + 1];
    int numTcFragsInChunk = tcFragEnd - tcFragStart;
    if (numTcFragsInChunk == 0)
        return; // Nothing to compute
    // Initialize the output vector segment in global memory
    int rowStart = rowChunkIndex * fragM;
    int rowEnd = min(rowStart + fragM, dRows);

    // Initiate prefetch for the first tcFrag before the loop
    int bufferIdx = 0; // Start with buffer 0
    {
        // Compute the number of non-zero elements in the first tcFrag
        int tcFragNnz = fragPtr[tcFragStart + 1] - fragPtr[tcFragStart];
        const double *tcValPtr = &tcVal[fragPtr[tcFragStart]];
        // Async copy tcVal for the first tcFrag
        size_t tcVal_bytes = tcFragNnz * sizeof(double);
        // Use cp.async to load tcVal into shared memory
        char *tcVal_shared_ptr = reinterpret_cast<char *>(shmem_tcVal_buffers[bufferIdx]);
        cp_async_bulk_global_to_shared_async(tcVal_shared_ptr, reinterpret_cast<const char *>(tcValPtr), tcVal_bytes);

        // Async copy x_d for the first tcFrag
        const int *sparse_AToX_idx = &sparse_AToX_index[tcFragStart * fragK];
        if (fragK < 32)
        {
            if (laneId < fragK)
            {
                int x_idx = sparse_AToX_idx[laneId];
                cp_async_global_to_shared_async(&shmem_x_buffers[bufferIdx][laneId], &x_d[x_idx], sizeof(double));
            }
        }
        else
        {
            for (int idx = laneId; idx < fragK; idx += WARP_SIZE)
            {
                int x_idx = sparse_AToX_idx[idx];
                cp_async_global_to_shared_async(&shmem_x_buffers[bufferIdx][idx], &x_d[x_idx], sizeof(double));
            }
        }
        // Wait for the async copies to complete
        cp_async_wait_all();
        __syncthreads();
    }

    // Loop over the tcFrags
    for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
    {
        // Compute indices for double buffering
        int currentBufferIdx = bufferIdx; // 初始时预取的第一个
        int nextBufferIdx = 1 - bufferIdx;

        // Initiate prefetch for the next tcFrag if not the last one
        if (tcFragIdx + 1 < tcFragEnd)
        {
            // Compute the number of non-zero elements in the next tcFrag
            int next_tcFragNnz = fragPtr[tcFragIdx + 2] - fragPtr[tcFragIdx + 1];
            const double *next_tcValPtr = &tcVal[fragPtr[tcFragIdx + 1]];

            size_t tcVal_bytes = next_tcFragNnz * sizeof(double);
            char *next_tcVal_shared_ptr = reinterpret_cast<char *>(shmem_tcVal_buffers[nextBufferIdx]);
            cp_async_bulk_global_to_shared_async(next_tcVal_shared_ptr, reinterpret_cast<const char *>(next_tcValPtr), tcVal_bytes);

            // Async copy x_d for the next tcFrag
            const int *next_sparse_AToX_idx = &sparse_AToX_index[(tcFragIdx + 1) * fragK];
            if (fragK < 32)
            {
                if (laneId < fragK)
                {
                    int x_idx = next_sparse_AToX_idx[laneId];
                    cp_async_global_to_shared_async(&shmem_x_buffers[nextBufferIdx][laneId], &x_d[x_idx], sizeof(double));
                }
            }
            else
            {
                for (int idx = laneId; idx < fragK; idx += WARP_SIZE)
                {
                    int x_idx = next_sparse_AToX_idx[idx];
                    cp_async_global_to_shared_async(&shmem_x_buffers[nextBufferIdx][idx], &x_d[x_idx], sizeof(double));
                }
            }
        }
        // Proceed to compute using data in currentBufferIdx
        // Wait for the async copies to complete (for the current fragment)
        cp_async_wait_all();
        __syncthreads();
        /////////////////////////////////////////////////////
        //// shared memory to register and compute
        /////////////////////////////////////////////////////
        uint32_t bitmap = fragBit[tcFragIdx];
        // int nzCount = fragPtr[tcFragIdx + 1] - fragPtr[tcFragIdx];
        const double *tcValShmPtr = shmem_tcVal_buffers[currentBufferIdx];
        // Load x_d segment into B fragment and broadcast
        double x_values[fragK];
        for (int k = 0; k < fragK; ++k)
        {
            x_values[k] = shmem_x_buffers[currentBufferIdx][k];
        }
        unsigned int lane_id;
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
        {
            // 计算矩阵 A、B、C 中的索引
            int m = lane_id >> 2;
            int k = lane_id & 3;
            // int n = 0;

            // 加载 A 矩阵元素到 a_value
            double a_value = 0.0;
            int bitPos = m * fragK + k;
            int bit = (bitmap >> bitPos) & 1;
            int valIdx = 0; // 非零值位置索引
            // 到当前bitPos处，累计了多少valIdx
            for (int i = 0; i < bitPos; ++i)
            {
                if ((bitmap >> i) & 1)
                {
                    valIdx++;
                }
            }
            if (bit)
            {
                a_value = tcValShmPtr[valIdx]; // TODO: ERROR
            }
            double b_value = x_values[k];
            double c_value[2] = {0.0, 0.0};
            asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 \n\t"
                "{%0, %1}, {%2}, {%3}, {%0, %1};\n\t"
                : "+d"(c_value[0]), "+d"(c_value[1])
                : "d"(a_value),
                  "d"(b_value));
            int y_idx = rowStart + m;
            if (y_idx < dRows)
            {
                // 原子地累加结果到 y_d
                atomicAdd(&y_d[y_idx], c_value[0]); // 累加第一个元素
                atomicAdd(&y_d[y_idx], c_value[1]); // 累加第二个元素
            }
        }

        // Swap buffers for next iteration
        bufferIdx = nextBufferIdx;
    } // End of tcFrag loop
}

void tcspmv(indT *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
            std::vector<double> tcVal, indT *sparse_AToX_index, valT *X_val,
            valT *Y_val, int rowA, int colA, int *row_order, double *necTime, double *necPre)
{
    struct timeval t1;
    struct timeval t2;
    // struct timeval tpre1;
    // struct timeval tpre2;
    int chunkNum = ceil((double)rowA / (double)fragM);
    int totalTcFrags = chunkPtr[chunkNum];
    valT *d_tcVal, *d_X_val, *d_Y_val;
    indT *d_sparse_AToX_index, *d_chunkPtr, *d_fragPtr;
    uint32_t *d_fragBit;

    cudaMalloc(&d_tcVal, tcVal.size() * sizeof(valT));
    cudaMalloc(&d_fragPtr, fragPtr.size() * sizeof(indT));
    cudaMalloc(&d_fragBit, fragBit.size() * sizeof(uint32_t));

    cudaMemcpy(d_tcVal, tcVal.data(), tcVal.size() * sizeof(valT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fragPtr, fragPtr.data(), fragPtr.size() * sizeof(indT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fragBit, fragBit.data(), fragBit.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_chunkPtr, sizeof(indT) * (chunkNum + 1));
    cudaMalloc(&d_sparse_AToX_index, sizeof(indT) * (totalTcFrags * fragK));
    cudaMalloc(&d_X_val, sizeof(valT) * colA);
    cudaMalloc(&d_Y_val, sizeof(valT) * rowA);

    cudaMemcpy(d_chunkPtr, chunkPtr, sizeof(indT) * (chunkNum + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sparse_AToX_index, sparse_AToX_index, sizeof(indT) * (totalTcFrags * fragK), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X_val, X_val, sizeof(valT) * colA, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_Y_val, Y_val, sizeof(valT) * rowA, cudaMemcpyHostToDevice);

    cudaMemset(d_Y_val, 0.0, sizeof(valT) * rowA);

    int warpsPerBlock = 4;
    int warpSize = 32;
    int threadsPerBlock = warpsPerBlock * warpSize;
    int numRowChunks = (rowA + fragM - 1) / fragM;
    int blocksPerGrid = (numRowChunks + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate shared memory size
    // TODO:  redundant
    int per_buffer_size_tcVal = fragM * fragK; // Max possible non-zero elements in a fragment
    int per_buffer_size_x_d = fragK;
    int per_buffer_total = per_buffer_size_tcVal + per_buffer_size_x_d;
    size_t sharedMemSize = warpsPerBlock * 2 * per_buffer_total * sizeof(double);

    int execute_time = 1;
    /*
    int warmup_time = 100;
    for (int i = 0; i < warmup_time; ++i)
    {
        tcspmv_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_X_val, d_Y_val, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA);
    }
    cudaDeviceSynchronize();
    */
    gettimeofday(&t1, NULL);

    for (int i = 0; i < execute_time; ++i)
    {
        tcspmv_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_X_val, d_Y_val, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(Y_val, d_Y_val, sizeof(valT) * rowA, cudaMemcpyDeviceToHost);

    gettimeofday(&t2, NULL);
    cudaFree(d_tcVal);
    cudaFree(d_fragPtr);
    cudaFree(d_fragBit);
    cudaFree(d_chunkPtr);
    cudaFree(d_sparse_AToX_index);
    cudaFree(d_X_val);
    cudaFree(d_Y_val);
}
