/*
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
        : "+d"(acc[0]), "+d"(acc[1]):
        "d"(frag_a), "d"(frag_b)
    );
}
__device__ __forceinline__ valT warpReduceSum(valT sum){
    sum += __shfl_down_sync(0xffffffff,sum,16);
    sum += __shfl_down_sync(0xffffffff,sum,8);
    sum += __shfl_down_sync(0xffffffff,sum,4);
    sum += __shfl_down_sync(0xffffffff,sum,2);
    sum += __shfl_down_sync(0xffffffff,sum,1);
    return sum;
}

// %0: 第一个输出操作数(在这里是r)
// [%1]: 第二个输入操作数的内存地址(在这里是a)
// "=d"(r): r是输出操作数,d表示它应该存储在双精度浮点寄存器中
// "l"(a): a是输入操作数,l表示它应该是一个64位内存地址,在64位CUDA架构中，所有指针都是64位的
__device__ __forceinline__ valT load_double_from_global_to_reg(const valT* a)
{
    valT r;
    asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(r) : "l"(a));
    return r;
}
__device__ __forceinline__ int load_int_from_global_to_reg(const int* a)
{
    int r;
    asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(r) : "l"(a));
    return r;
}

__device__ __forceinline__ void store_double_to_global(const valT* a, valT v)
{
    asm volatile("st.global.cs.f64 [%0], %1;" :: "l"(a), "d"(v));
}



//由于wmma不支持double类型，请将以下所有的wmma用ptx指令替换，要求不改变代码原始语义。并将修改的部分罗列出来
// Define cp_async_global_to_shared_async for copying a single element asynchronously
__device__ inline void cp_async_global_to_shared_async(void *shared_dest, const void *global_src, size_t bytes)
{
#if (__CUDA_ARCH__ >= 800)
    // Use cp.async intrinsic for architectures with compute capability 8.0 or higher
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                     "l"(shared_dest),
                 "l"(global_src), "n"(bytes));
#else
    // For older architectures, fall back to synchronous copy
    memcpy(shared_dest, global_src, bytes);
#endif
}

// Define cp_async_bulk_global_to_shared_async for copying a bulk of data asynchronously
__device__ inline void cp_async_bulk_global_to_shared_async(void *shared_dest, const void *global_src, size_t bytes)
{
#if (__CUDA_ARCH__ >= 800)
    // cp.async supports copying 4, 8, or 16 bytes per instruction
    const size_t max_copy_size = 16; // Maximum bytes per cp.async instruction
    char *shmem_ptr = reinterpret_cast<char *>(shared_dest);
    const char *gmem_ptr = reinterpret_cast<const char *>(global_src);

    size_t offset = 0;
    while (bytes > 0)
    {
        size_t chunk_size = (bytes >= max_copy_size) ? max_copy_size : ((bytes >= 8) ? 8 : 4);
        // Ensure chunk_size is 4, 8, or 16 bytes
        if (bytes < 4)
        {
            // For remaining bytes less than 4, use synchronous copy
            memcpy(shmem_ptr + offset, gmem_ptr + offset, bytes);
            break;
        }
        else
        {
            // Use cp.async for 4, 8, or 16 bytes
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                             "l"(shmem_ptr + offset),
                         "l"(gmem_ptr + offset), "n"(chunk_size));
            offset += chunk_size;
            bytes -= chunk_size;
        }
    }
#else
    // For older architectures, fall back to synchronous copy
    memcpy(shared_dest, global_src, bytes);
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
    int dRows,
    int fragM,
    int fragK)
{
    // Shared memory for double buffering tcVal and x_d
    extern __shared__ double shared_mem[];
    // Assuming we have enough shared memory, divide it into two buffers
    // Each buffer will hold the tcVal and x_d for one tcFrag
    int per_buffer_size_tcVal = fragM * fragK; // Max possible non-zero elements in a fragment
    int per_buffer_size_x_d = fragK;
    int per_buffer_total = per_buffer_size_tcVal + per_buffer_size_x_d;

    double *shmem_buffer0 = shared_mem;
    double *shmem_buffer1 = shared_mem + per_buffer_total;

    // Get the rowChunk this block is responsible for
    int rowChunkIndex = blockIdx.x;
    int numRowChunks = (dRows + fragM - 1) / fragM;

    if (rowChunkIndex >= numRowChunks)
        return;

    // Get the tcFrag range for this rowChunk
    int tcFragStart = chunkPtr[rowChunkIndex];
    int tcFragEnd = chunkPtr[rowChunkIndex + 1];
    int numTcFragsInChunk = tcFragEnd - tcFragStart;

    if (numTcFragsInChunk == 0)
        return; // Nothing to compute

    // Initialize the output vector segment in global memory
    int rowChunkStart = rowChunkIndex * fragM;
    int rowChunkEnd = min(rowChunkStart + fragM, dRows);

    // Optional: Initialize the output vector y_d for this chunk
    for (int idx = threadIdx.x; idx < fragM; idx += blockDim.x)
    {
        int y_idx = rowChunkStart + idx;
        if (y_idx < dRows)
            y_d[y_idx] = 0.0;
    }

    // Double buffering pointers
    double *shmem_tcVal_buffers[2] = {shmem_buffer0, shmem_buffer1};
    double *shmem_x_buffers[2] = {shmem_buffer0 + per_buffer_size_tcVal, shmem_buffer1 + per_buffer_size_tcVal};

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
        for (int idx = threadIdx.x; idx < fragK; idx += blockDim.x)
        {
            int x_idx = sparse_AToX_idx[idx];
            cp_async_global_to_shared_async(&shmem_x_buffers[bufferIdx][idx], &x_d[x_idx], sizeof(double));
        }

        // Wait for the async copies to complete
        cp_async_wait_all();
        __syncthreads();
    }

    // Loop over the tcFrags
    for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
    {
        // Compute indices for double buffering
        int currentBufferIdx = bufferIdx;
        int nextBufferIdx = 1 - bufferIdx;

        // Initiate prefetch for the next tcFrag if not the last one
        if (tcFragIdx + 1 < tcFragEnd)
        {
            // Compute the number of non-zero elements in the next tcFrag
            int next_tcFragNnz = fragPtr[tcFragIdx + 2] - fragPtr[tcFragIdx + 1];
            const double *next_tcValPtr = &tcVal[fragPtr[tcFragIdx + 1]];

            // Async copy tcVal for the next tcFrag
            size_t tcVal_bytes = next_tcFragNnz * sizeof(double);
            // Use cp.async to load tcVal into shared memory
            char *next_tcVal_shared_ptr = reinterpret_cast<char *>(shmem_tcVal_buffers[nextBufferIdx]);

            cp_async_bulk_global_to_shared_async(next_tcVal_shared_ptr, reinterpret_cast<const char *>(next_tcValPtr), tcVal_bytes);

            // Async copy x_d for the next tcFrag
            const int *next_sparse_AToX_idx = &sparse_AToX_index[(tcFragIdx + 1) * fragK];
            for (int idx = threadIdx.x; idx < fragK; idx += blockDim.x)
            {
                int x_idx = next_sparse_AToX_idx[idx];
                cp_async_global_to_shared_async(&shmem_x_buffers[nextBufferIdx][idx], &x_d[x_idx], sizeof(double));
            }
        }

        // Proceed to compute using data in currentBufferIdx
        // Wait for the async copies to complete (for the current fragment)
        cp_async_wait_all();
        __syncthreads();

        // Now compute for the current tcFrag
        uint32_t bitmap = fragBit[tcFragIdx];
        int nzCount = fragPtr[tcFragIdx + 1] - fragPtr[tcFragIdx];
        const double *tcValPtr = shmem_tcVal_buffers[currentBufferIdx];

        // Load x_d segment into B fragment and broadcast
        double x_values[fragK];
        for (int k = 0; k < fragK; ++k)
        {
            x_values[k] = shmem_x_buffers[currentBufferIdx][k];
        }
        
        // // Tensor Core fragments
        // wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::col_major> a_frag;
        // wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> b_frag;
        // wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

        // // Initialize fragments
        // wmma::fill_fragment(a_frag, 0.0);
        // wmma::fill_fragment(b_frag, 0.0);
        // wmma::fill_fragment(c_frag, 0.0);

        // // Reconstruct dense matrix A fragment from fragBit and tcVal
        // int valIdx = 0;
        // for (int m = 0; m < fragM; ++m)
        // {
        //     for (int k = 0; k < fragK; ++k)
        //     {
        //         int bitPos = m * fragK + k;
        //         int bit = (bitmap >> bitPos) & 1;

        //         if (bit)
        //         {
        //             double val = tcValPtr[valIdx++];
        //             a_frag.x[m * fragK + k] = val;
        //         }
        //     }
        // }

        // // Load x_values into B fragment and broadcast
        // for (int k = 0; k < fragK; ++k)
        // {
        //     double x_val = x_values[k];
        //     for (int n = 0; n < 8; ++n)
        //     {
        //         b_frag.x[k * 8 + n] = x_val;
        //     }
        // }

        // // Perform the Tensor Core operation
        // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // // Write the result back to y_d
        // for (int m = 0; m < fragM; ++m)
        // {
        //     double c_val = c_frag.x[m]; // Using first column as per requirement
        //     int y_idx = rowChunkStart + m;
        //     if (y_idx < dRows)
        //     {
        //         atomicAdd(&y_d[y_idx], c_val);
        //     }
        // }
        
        
        unsigned int lane_id;
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
        {
            // 计算矩阵 A、B、C 中的索引
            int m = lane_id % 8;       // 行索引 (0-7)
            int k = lane_id / 8;       // 列索引 (0-1)，fragK = 4，因此列索引 0 或 1
            int n = 0;                 // 列索引，从 0 开始

            // 加载 A 矩阵元素到 a_value
            double a_value = 0.0;
            int bitPos = m * fragK + k;
            int bit = (bitmap >> bitPos) & 1;
            int valIdx = 0; // 非零值位置索引

            // 需要计算 valIdx 的正确位置
            for (int i = 0; i < bitPos; ++i)
            {
                if ((bitmap >> i) & 1)
                {
                    valIdx++;
                }
            }

            if (bit)
            {
                a_value = tcValPtr[valIdx];
            }

            // 加载 B 矩阵元素到 b_value
            double b_value = x_values[k];

            // 初始化 C 矩阵累加器
            double c_value[2] = {0.0, 0.0};

            // 执行 MMA 操作
            asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 \n\t"
                "{%0, %1}, {%2}, {%3}, {%0, %1};\n\t"
                : "+d"(c_value[0]), "+d"(c_value[1]) // 输出：C 矩阵的两个元素
                : "d"(a_value),                      // 输入：A 矩阵元素
                  "d"(b_value)                       // 输入：B 矩阵元素
            );

            // 将结果写回到 y_d
            int y_idx = rowChunkStart + m;
            if (y_idx < dRows)
            {
                int col_idx0 = n;
                int col_idx1 = n + 1; // 第二个元素的列索引

                // 原子地累加结果到 y_d
                atomicAdd(&y_d[y_idx], c_value[0]); // 累加第一个元素
                atomicAdd(&y_d[y_idx], c_value[1]); // 累加第二个元素
            }
        }

        // Swap buffers for next iteration
        bufferIdx = nextBufferIdx;

        // If not the last fragment, prefetching for the next fragment has already been initiated
        // Wait for the prefetch to complete at the beginning of the next iteration
    } // End of tcFrag loop
}

void tcspmv(indT *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
            std::vector<double> tcVal, indT *sparse_AToX_index, valT *X_val,
            valT *Y_val, int rowA, int colA, int fragM, int fragK, int fragN, int *row_order, double *necTime, double *necPre)
{
    struct timeval t1;
    struct timeval t2;
    struct timeval tpre1;
    struct timeval tpre2;
    int chunkNum = ceil((double)rowA / (double)fragM);
    int totalTcFrags = chunkPtr[chunkNum];
    valT *d_tcVal, *d_sparse_AToX_index, *d_X_val, *d_Y_val;
    indT *d_chunkPtr, *d_fragPtr;
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
    cudaMemcpy(d_Y_val, Y_val, sizeof(valT) * rowA, cudaMemcpyHostToDevice);

    int warpsPerBlock = 4;
    int warpSize = 32;
    int threadsPerBlock = warpsPerBlock * warpSize;
    int numRowChunks = (rowA + fragM - 1) / fragM;
    int blocksPerGrid = (numRowChunks + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate shared memory size
    int per_buffer_size_tcVal = fragM * fragK; // Max possible non-zero elements in a fragment
    int per_buffer_size_x_d = fragK;
    int per_buffer_total = per_buffer_size_tcVal + per_buffer_size_x_d;
    size_t sharedMemSize = warpsPerBlock * 2 * per_buffer_total * sizeof(double);

    int warmup_time = 100;
    int execute_time = 1000;

    for (int i = 0; i < warmup_time; ++i)
    {
        // Launch the kernel
        tcspmv_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_X_val, d_Y_val, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, fragM, fragK);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    for (int i = 0; i < execute_time; ++i)
    {
        // Launch the kernel
        tcspmv_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_X_val, d_Y_val, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, fragM, fragK);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);

    // double pre_time = ((tpre2.tv_sec - tpre1.tv_sec) * 1000.0 + (tpre2.tv_usec - tpre1.tv_usec) / 1000.0) / 1;
    // *necPre = pre_time;
    // double nec_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / execute_time;
    // *necTime = nec_time;
    // double nec_gflops = (double)((long)nnzA * 2) / (nec_time * 1e6);

    // printf("SpMV_X:  %8.4lf ms, %8.4lf GFlop/s\n", nec_time, nec_gflops);

    // int iter = (int) pre_time / nec_time;
    // printf("iterate:  %d \n", iter);

    // printf("\nrowA = %d, row_long = %d, row_block = %d, row_short1 = %d, common13 = %d, row_short_3 = %d, row_short_4 = %d, row_short_2 = %d\n", rowA, row_long, row_block, short_row_1, common_13, short_row_3, short_row_4, short_row_2);

    // cudaMemcpy(Y_val, d_vecY_csr, sizeof(valT) * rowA, cudaMemcpyDeviceToHost);
    cudaFree(d_tcVal);
    cudaFree(d_fragPtr);
    cudaFree(d_fragBit);
    cudaFree(d_chunkPtr);
    cudaFree(d_sparse_AToX_index);
    cudaFree(d_X_val);
    cudaFree(d_Y_val);
}

*/