#include "common.h"

// 1 warp - 1 row chunk
__global__ void tcspmv_kernel_fp16_v2(
    const half *__restrict__ x_d,
    half *__restrict__ y_d,
    const int *__restrict__ chunkPtr,
    const int *__restrict__ fragPtr,
    const uint64_t *__restrict__ fragBit,
    const half *__restrict__ tcVal,
    const int *__restrict__ sparse_AToX_index,
    int dRows,
    int dCols)
{
    // __shared__ half x_shm[SHM_SIZE];
    // int num_elements = SHM_SIZE;
    // int num_threads = blockDim.x;
    // int elements_per_thread = (num_elements + num_threads - 1) / num_threads;

    // // Load data into shared memory
    // for (int i = 0; i < elements_per_thread; ++i)
    // {
    //     int idx = threadIdx.x + i * num_threads;
    //     if (idx < num_elements)
    //     {
    //         x_shm[idx] = x_d[idx];
    //     }
    // }
    // __syncthreads();
    const int warpsPerBlock = 4;
    int warpId = threadIdx.x >> 5; // Warp ID within the block
    int laneId = threadIdx.x & 31; // Lane ID within the warp
    int groupID = laneId >> 2;
    int threadID_in_group = laneId & 3;
    int double_threadID_in_group = threadID_in_group * 2;

    int rowChunkIndex = blockIdx.x * warpsPerBlock + warpId;

    int tcFragStart = chunkPtr[rowChunkIndex];
    int tcFragEnd = chunkPtr[rowChunkIndex + 1];
    int numTcFragsInChunk = tcFragEnd - tcFragStart;

    if (numTcFragsInChunk == 0)
    {
        return;
    }
    int rowStart = rowChunkIndex * fragM;

    // Initialize the accumulators
    half frag_a[4];
    half frag_b[4];
    half frag_d[4];
    half acc[4] = {__half(0)};
    half sum0 = __half(0);
    half sum1 = __half(0);
    int b_col = groupID;
    for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
    {
        uint64_t bitmap = fragBit[tcFragIdx];
        const half *tcValPtr = &tcVal[fragPtr[tcFragIdx]];
        // load BBB
        int b_row = double_threadID_in_group;
        int b_bitPos = b_col * fragK + b_row;
        int bit = (bitmap >> b_bitPos) & 1;
        frag_b[0] = bit ? tcValPtr[__popcll(bitmap & ((1ULL << b_bitPos) - 1))] : __float2half(0.0f);

        b_row = double_threadID_in_group + 1;
        b_bitPos = b_col * fragK + b_row;
        bit = (bitmap >> b_bitPos) & 1;
        frag_b[1] = bit ? tcValPtr[__popcll(bitmap & ((1ULL << b_bitPos) - 1))] : __float2half(0.0f);
        // load AAA
        const int *sparse_AToX_idx = &sparse_AToX_index[tcFragIdx * fragK];
        
        int x_idx = sparse_AToX_idx[double_threadID_in_group];
        frag_a[0] = __ldg(&x_d[x_idx]);

        x_idx = sparse_AToX_idx[double_threadID_in_group + 1];
        frag_a[1] = __ldg(&x_d[x_idx]);

        uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_a[0]);
        uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_b[0]);
        uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);
        uint32_t *D = reinterpret_cast<uint32_t *>(&frag_d[0]);

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
            : "=r"(D[0]), "=r"(D[1])
            : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(C[0]), "r"(C[1]));

        sum0 += frag_d[0];
        sum1 += frag_d[1];
    }
    // // Write the result to y_d
    if (groupID == 0) // t0 t1 t2 t3
    {
        int y_idx0 = rowStart + double_threadID_in_group;
        if (y_idx0 < dRows)
        {
            ushort *sum_u0 = reinterpret_cast<ushort *>(&sum0);
            asm volatile("st.global.cs.u16 [%0], %1;" ::"l"(y_d + y_idx0), "h"(*sum_u0));
        }

        int y_idx1 = rowStart + double_threadID_in_group + 1;
        if (y_idx1 < dRows)
        {
            ushort *sum_u1 = reinterpret_cast<ushort *>(&sum1);
            asm volatile("st.global.cs.u16 [%0], %1;" ::"l"(y_d + y_idx1), "h"(*sum_u1));
        }
    }
}

void du_tcspmv_fp16_v2(indT *chunkPtr, std::vector<int> fragPtr, std::vector<uint64_t> fragBit,
                       std::vector<half> tcVal, indT *sparse_AToX_index, half *X_val,
                       half *Y_val, int rowA, int colA, int *row_order, double *tcTime)
{
    int chunkNum = ceil((double)rowA / (double)fragM);
    int totalTcFrags = chunkPtr[chunkNum];
    half *d_tcVal, *d_X_val, *d_Y_val, *d_Y_val_perf;
    indT *d_sparse_AToX_index, *d_chunkPtr, *d_fragPtr;
    uint64_t *d_fragBit;

    // cudaMemcpyToSymbol(x_const, X_val, CONST_SIZE * sizeof(half));

    //  tcVal
    CUDA_CHECK_ERROR(cudaMalloc(&d_tcVal, tcVal.size() * sizeof(half)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_tcVal, tcVal.data(), tcVal.size() * sizeof(half), cudaMemcpyHostToDevice));

    //  fragPtr
    CUDA_CHECK_ERROR(cudaMalloc(&d_fragPtr, fragPtr.size() * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_fragPtr, fragPtr.data(), fragPtr.size() * sizeof(int), cudaMemcpyHostToDevice));

    //  fragBit
    CUDA_CHECK_ERROR(cudaMalloc(&d_fragBit, fragBit.size() * sizeof(uint64_t)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_fragBit, fragBit.data(), fragBit.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    //  chunkPtr
    CUDA_CHECK_ERROR(cudaMalloc(&d_chunkPtr, sizeof(indT) * (chunkNum + 1)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_chunkPtr, chunkPtr, sizeof(indT) * (chunkNum + 1), cudaMemcpyHostToDevice));

    //  sparse_AToX_index
    CUDA_CHECK_ERROR(cudaMalloc(&d_sparse_AToX_index, sizeof(indT) * (totalTcFrags * fragK)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_sparse_AToX_index, sparse_AToX_index, sizeof(indT) * (totalTcFrags * fragK), cudaMemcpyHostToDevice));

    //  X_val
    CUDA_CHECK_ERROR(cudaMalloc(&d_X_val, sizeof(half) * colA));
    CUDA_CHECK_ERROR(cudaMemcpy(d_X_val, X_val, sizeof(half) * colA, cudaMemcpyHostToDevice));

    //  Y_val
    CUDA_CHECK_ERROR(cudaMalloc(&d_Y_val, sizeof(half) * rowA));
    CUDA_CHECK_ERROR(cudaMemset(d_Y_val, 0, sizeof(half) * rowA));

    CUDA_CHECK_ERROR(cudaMalloc(&d_Y_val_perf, sizeof(half) * rowA));
    CUDA_CHECK_ERROR(cudaMemset(d_Y_val_perf, 0, sizeof(half) * rowA));

    int warpsPerBlock = 4;
    int warpSize = 32;
    int threadsPerBlock = warpsPerBlock * warpSize;
    int numRowChunks = (rowA + fragM - 1) / fragM;
    int blocksPerGrid = (numRowChunks + warpsPerBlock - 1) / warpsPerBlock;

    // printf("Launching tcspmv_kernel_fp16 with %d blocks, %d threads per block\n",
    //        blocksPerGrid, threadsPerBlock);

    int warm_iter = 200;
    for (int i = 0; i < warm_iter; ++i)
    {
        tcspmv_kernel_fp16_v2<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val_perf, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    int test_iter = 4000;
    cuda_time_test_start();
    for (int i = 0; i < test_iter; ++i)
    {
        tcspmv_kernel_fp16_v2<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val_perf, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    cuda_time_test_end();

    double runtime = (elapsedTime) / test_iter;
    printf("tcspmv_kernel_fp16: %g ms\n", runtime);
    *tcTime = runtime;

    tcspmv_kernel_fp16_v2<<<blocksPerGrid, threadsPerBlock>>>(
        d_X_val, d_Y_val, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
        d_sparse_AToX_index, rowA, colA);

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaMemcpy(Y_val, d_Y_val, sizeof(half) * rowA, cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_tcVal));
    CUDA_CHECK_ERROR(cudaFree(d_fragPtr));
    CUDA_CHECK_ERROR(cudaFree(d_fragBit));
    CUDA_CHECK_ERROR(cudaFree(d_chunkPtr));
    CUDA_CHECK_ERROR(cudaFree(d_sparse_AToX_index));
    CUDA_CHECK_ERROR(cudaFree(d_X_val));
    CUDA_CHECK_ERROR(cudaFree(d_Y_val));
    CUDA_CHECK_ERROR(cudaFree(d_Y_val_perf));
}
