#include "common.h"

#define WARP_SIZE 32

template <int SizeInBytes>
__device__ __forceinline__ void cp_async(double *smem_ptr, const double *global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported for double");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

#define CUDA_CHECK_ERROR(call)                                            \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            fprintf(stderr, "CUDA Error: %s (error code: %d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);    \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

__device__ inline void cp_async_wait_all()
{
#if (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
#else
    // No action needed for synchronous copies
#endif
}

__global__ void tcspmv_kernel(
    const double *__restrict__ x_d,
    double *__restrict__ y_d,
    const int *__restrict__ chunkPtr,
    const int *__restrict__ fragPtr,
    const uint32_t *__restrict__ fragBit,
    const double *__restrict__ tcVal,
    const int *__restrict__ sparse_AToX_index,
    int dRows,
    int dCols)
{
    const int warpsPerBlock = 4;
    extern __shared__ double shared_mem[];

    int per_buffer_size_tcVal = fragM * fragK; // Maximum elements for fp64, but for A matrix only 32 elements (8x4)
    int per_buffer_size_x_d = fragK;           // 4 elements
    int per_buffer_total = per_buffer_size_tcVal + per_buffer_size_x_d;

    int warpId = threadIdx.x / 32; // Warp ID within the block
    int laneId = threadIdx.x % 32; // Lane ID within the warp

    double *shmem_warp = shared_mem + warpId * 2 * per_buffer_total;
    double *shmem_warp_buffer0 = shmem_warp;
    double *shmem_warp_buffer1 = shmem_warp + per_buffer_total;

    double *shmem_tcVal_buffers[2] = {shmem_warp_buffer0, shmem_warp_buffer1};                                             // Buffer for tcVal (matrix A)
    double *shmem_x_buffers[2] = {shmem_warp_buffer0 + per_buffer_size_tcVal, shmem_warp_buffer1 + per_buffer_size_tcVal}; // Buffer for x_d (matrix B)

    int rowChunkIndex = blockIdx.x * warpsPerBlock + warpId;

    int tcFragStart = chunkPtr[rowChunkIndex];
    int tcFragEnd = chunkPtr[rowChunkIndex + 1];
    int numTcFragsInChunk = tcFragEnd - tcFragStart;

    if (numTcFragsInChunk == 0)
    {
        printf("\n !!!now the tc frags number = 0 \n");
        return;
    }
    int rowStart = rowChunkIndex * fragM;
    int rowEnd = min(rowStart + fragM, dRows);

    int bufferIdx = 0; // Start with buffer 0

    // Preload the first tcFragment data
    int tcFragNnz = fragPtr[tcFragStart + 1] - fragPtr[tcFragStart];
    if (tcFragNnz > per_buffer_size_tcVal)
    {
        printf("Error: tcFragNnz (%d) exceeds per_buffer_size_tcVal (%d)\n", tcFragNnz, per_buffer_size_tcVal);
        return;
    }
    const double *tcValPtr = &tcVal[fragPtr[tcFragStart]];
    // size_t tcVal_bytes = tcFragNnz * sizeof(double);
    double *tcVal_shared_ptr = shmem_tcVal_buffers[bufferIdx];

    // cp_async_bulk_global_to_shared_async(tcVal_shared_ptr, reinterpret_cast<const char *>(tcValPtr), tcVal_bytes);

    if (laneId < tcFragNnz)
    {
        size_t shmem_addr = __cvta_generic_to_shared(tcVal_shared_ptr + laneId);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], %2;\n"
            :
            : "l"(shmem_addr),
              "l"(tcValPtr + laneId),
              "n"(8));
    }

    const int *sparse_AToX_idx = &sparse_AToX_index[tcFragStart * fragK];
    if (laneId < fragK)
    {
        int x_idx = sparse_AToX_idx[laneId];
        if (x_idx >= 0 && x_idx < dCols)
        {
            size_t shmem_addr = __cvta_generic_to_shared(&shmem_x_buffers[bufferIdx][laneId]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::
                             "l"(shmem_addr),
                         "l"(&x_d[x_idx]));
        }
    }

    cp_async_wait_all();
    __syncthreads();

    for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
    {
        int currentBufferIdx = bufferIdx; // Buffer with current tcFragment data
        int nextBufferIdx = 1 - bufferIdx;

        // Preload next tcFragment data while processing the current one
        if (tcFragIdx + 1 < tcFragEnd)
        {
            int next_tcFragNnz = fragPtr[tcFragIdx + 2] - fragPtr[tcFragIdx + 1];
            const double *next_tcValPtr = &tcVal[fragPtr[tcFragIdx + 1]];

            // size_t tcVal_bytes = next_tcFragNnz * sizeof(double);
            double *next_tcVal_shared_ptr = shmem_tcVal_buffers[nextBufferIdx];
            // cp_async_bulk_global_to_shared_async(next_tcVal_shared_ptr, reinterpret_cast<const char *>(next_tcValPtr), tcVal_bytes);
            if (laneId < next_tcFragNnz)
            {
                size_t shmem_addr = __cvta_generic_to_shared(next_tcVal_shared_ptr + laneId);
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], %2;\n"
                    :
                    : "l"(shmem_addr),
                      "l"(next_tcValPtr + laneId),
                      "n"(8));
            }

            // Async copy x_d for the next tcFrag
            const int *next_sparse_AToX_idx = &sparse_AToX_index[(tcFragIdx + 1) * fragK];
            if (laneId < fragK)
            {
                int x_idx = next_sparse_AToX_idx[laneId];
                if (x_idx >= 0 && x_idx < dCols)
                {
                    size_t shmem_addr = __cvta_generic_to_shared(&shmem_x_buffers[nextBufferIdx][laneId]);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::
                                     "l"(shmem_addr),
                                 "l"(&x_d[x_idx]));
                }
            }
        }

        cp_async_wait_all();
        __syncthreads();

        uint32_t bitmap = fragBit[tcFragIdx];
        const double *tcValShmPtr = shmem_tcVal_buffers[currentBufferIdx];
        const double *x_values = shmem_x_buffers[currentBufferIdx];

        // Define fragments for MMA
        double a_frag;
        double b_frag;
        double c_frag[2] = {0.0, 0.0}; // Each thread holds two FP64 elements for accumulator

        // Calculate positions according to the documentation
        int a_row = laneId >> 2; // laneId / 4
        int a_col = laneId & 3;  // laneId % 4

        // Load A fragment
        int a_bitPos = a_row * fragK + a_col;
        int a_valIdx = __popc(bitmap & ((1U << a_bitPos) - 1));
        if ((bitmap >> a_bitPos) & 1)
        {
            a_frag = tcValShmPtr[a_valIdx];
        }
        else
        {
            a_frag = 0.0;
        }

        // Load B fragment
        b_frag = x_values[a_col];


        // Perform MMA operation
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 \n\t"
            "{%0, %1}, {%2}, {%3}, {%0, %1};\n\t"
            : "+d"(c_frag[0]), "+d"(c_frag[1])
            : "d"(a_frag), "d"(b_frag));

        // Compute sum of accumulator elements
        if (a_col == 0)
        {
            // 计算写入的全局索引
            int y_idx = rowStart + a_row;

            // 写回全局内存
            if (y_idx < dRows)
            {
                atomicAdd(&y_d[y_idx], c_frag[0]);
            }
        }
        bufferIdx = nextBufferIdx;
    } // End of tcFrag loop
}

void tcspmv(indT *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
            std::vector<double> tcVal, indT *sparse_AToX_index, double *X_val,
            double *Y_val, int rowA, int colA, int *row_order, double *necTime, double *necPre)
{
    struct timeval t1;
    struct timeval t2;
    int fragM = 8;
    int fragK = 4;
    int chunkNum = ceil((double)rowA / (double)fragM);
    int totalTcFrags = chunkPtr[chunkNum];
    double *d_tcVal, *d_X_val, *d_Y_val;
    indT *d_sparse_AToX_index, *d_chunkPtr, *d_fragPtr;
    uint32_t *d_fragBit;

    //  tcVal
    CUDA_CHECK_ERROR(cudaMalloc(&d_tcVal, tcVal.size() * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_tcVal, tcVal.data(), tcVal.size() * sizeof(double), cudaMemcpyHostToDevice));

    //  fragPtr
    CUDA_CHECK_ERROR(cudaMalloc(&d_fragPtr, fragPtr.size() * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_fragPtr, fragPtr.data(), fragPtr.size() * sizeof(int), cudaMemcpyHostToDevice));

    //  fragBit
    CUDA_CHECK_ERROR(cudaMalloc(&d_fragBit, fragBit.size() * sizeof(uint32_t)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_fragBit, fragBit.data(), fragBit.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    //  chunkPtr
    CUDA_CHECK_ERROR(cudaMalloc(&d_chunkPtr, sizeof(indT) * (chunkNum + 1)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_chunkPtr, chunkPtr, sizeof(indT) * (chunkNum + 1), cudaMemcpyHostToDevice));

    //  sparse_AToX_index
    CUDA_CHECK_ERROR(cudaMalloc(&d_sparse_AToX_index, sizeof(indT) * (totalTcFrags * fragK)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_sparse_AToX_index, sparse_AToX_index, sizeof(indT) * (totalTcFrags * fragK), cudaMemcpyHostToDevice));

    //  X_val
    CUDA_CHECK_ERROR(cudaMalloc(&d_X_val, sizeof(double) * colA));
    CUDA_CHECK_ERROR(cudaMemcpy(d_X_val, X_val, sizeof(double) * colA, cudaMemcpyHostToDevice));

    //  Y_val
    CUDA_CHECK_ERROR(cudaMalloc(&d_Y_val, sizeof(double) * rowA));
    CUDA_CHECK_ERROR(cudaMemset(d_Y_val, 0, sizeof(double) * rowA));

    int warpsPerBlock = 4;
    int warpSize = 32;
    int threadsPerBlock = warpsPerBlock * warpSize;
    int numRowChunks = (rowA + fragM - 1) / fragM;
    int blocksPerGrid = (numRowChunks + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate shared memory size
    int per_buffer_size_tcVal_total = fragM * fragK; // Max possible non-zero elements in a fragment
    int per_buffer_size_x_d_total = fragK;
    int per_buffer_total = per_buffer_size_tcVal_total + per_buffer_size_x_d_total;
    size_t sharedMemSize = warpsPerBlock * 2 * per_buffer_total * sizeof(double);

    printf("Launching kernel with %d blocks, %d threads per block, sharedMemSize = %zu bytes\n",
           blocksPerGrid, threadsPerBlock, sharedMemSize);

    int execute_time = 1;

    gettimeofday(&t1, NULL);

    for (int i = 0; i < execute_time; ++i)
    {
        tcspmv_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_X_val, d_Y_val, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    printf("\n tcspmv_kernel end \n\n");
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    gettimeofday(&t2, NULL);

    CUDA_CHECK_ERROR(cudaMemcpy(Y_val, d_Y_val, sizeof(double) * rowA, cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_tcVal));
    CUDA_CHECK_ERROR(cudaFree(d_fragPtr));
    CUDA_CHECK_ERROR(cudaFree(d_fragBit));
    CUDA_CHECK_ERROR(cudaFree(d_chunkPtr));
    CUDA_CHECK_ERROR(cudaFree(d_sparse_AToX_index));
    CUDA_CHECK_ERROR(cudaFree(d_X_val));
    CUDA_CHECK_ERROR(cudaFree(d_Y_val));

    double elapsed = (t2.tv_sec - t1.tv_sec) * 1000.0;
    elapsed += (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Kernel execution time: %lf ms\n", elapsed);
}
