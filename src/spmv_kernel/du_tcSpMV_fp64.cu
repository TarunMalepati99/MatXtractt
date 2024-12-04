#include "common.h"
#include "fuse_kernel.h"

#define SHM_SIZE 128  // Shared memory size in doubles (8 KB)
#define CONST_SIZE 4096  // 常量内存大小
 __constant__ double x_const[CONST_SIZE];

 __device__ __forceinline__ void store_double_to_global(const double* a, double v)
{
    asm volatile("st.global.cs.f64 [%0], %1;" :: "l"(a), "d"(v));
}

// 重新组织tcVal、fragPtr和sparse_AToX_index的数据布局，使得连续线程访问连续的内存地址
// 利用CUDA的Warp级别原语（如__shfl_down_sync）在warp内部高效地共享数据，减少同步开销

__global__ void tcspmv_kernel_fp64(
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
    int warpId = threadIdx.x / 32; // Warp ID within the block
    int laneId = threadIdx.x & 31; // Lane ID within the warp
    // double thr_accum = 0;

    int rowChunkIndex = blockIdx.x * warpsPerBlock + warpId;

    int tcFragStart = chunkPtr[rowChunkIndex];
    int tcFragEnd = chunkPtr[rowChunkIndex + 1];
    int numTcFragsInChunk = tcFragEnd - tcFragStart;

    if (numTcFragsInChunk == 0)
    {
        return;
    }
    int rowStart = rowChunkIndex * fragM;

    // Calculate positions according to the documentation
    int a_row = laneId >> 2; // laneId / 4
    int a_col = laneId & 3;  // laneId % 4
    int a_bitPos = a_row * fragK + a_col;
    // double sum = 0.0;
    for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
    {
        uint32_t bitmap = fragBit[tcFragIdx];
        const double *tcValPtr = &tcVal[fragPtr[tcFragIdx]];
        const int *sparse_AToX_idx = &sparse_AToX_index[tcFragIdx * fragK];

        double c_frag[2] = {0.0, 0.0};
        // Load A fragment
        int a_valIdx = __popc(bitmap & ((1U << a_bitPos) - 1));
        int bit = (bitmap >> a_bitPos) & 1;
        double a_frag = bit ? tcValPtr[a_valIdx] : 0.0;

        int x_idx = sparse_AToX_idx[a_col];
        // b_frag = x_d[x_idx];
        double b_frag = __ldg(&x_d[x_idx]);

        // Perform MMA operation
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 \n\t"
            "{%0, %1}, {%2}, {%3}, {%0, %1};\n\t"
            : "+d"(c_frag[0]), "+d"(c_frag[1])
            : "d"(a_frag), "d"(b_frag));
        // Compute sum of accumulator elements
        // thr_accum += c_frag[0];
        
        if (a_col == 0)
        {
            int y_idx = rowStart + a_row;
            if (y_idx < dRows)
            {
                // atomicAdd(&y_d[y_idx], thr_accum);
                atomicAdd(&y_d[y_idx], c_frag[0]);
            }
        }
        
        // sum += c_frag[0];

    } // End of tcFrag loop
    // if (a_col == 0)
    // {
    //     int y_idx = rowStart + a_row;
    //     if (y_idx < dRows)
    //     {
    //         store_double_to_global(y_d + y_idx, sum);
    //     }
    // }
}

__global__ void tcspmv_kernel_fp64__(
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
    // __shared__ double x_shm[SHM_SIZE];
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
    int warpId = threadIdx.x / 32; // Warp ID within the block
    int laneId = threadIdx.x & 31; // Lane ID within the warp
    // double thr_accum = 0;

    int rowChunkIndex = blockIdx.x * warpsPerBlock + warpId;

    int tcFragStart = chunkPtr[rowChunkIndex];
    int tcFragEnd = chunkPtr[rowChunkIndex + 1];
    int numTcFragsInChunk = tcFragEnd - tcFragStart;

    if (numTcFragsInChunk == 0)
    {
        return;
    }
    int rowStart = rowChunkIndex * fragM;

    // Calculate positions according to the documentation
    int a_row = laneId >> 2; // laneId / 4
    int a_col = laneId & 3;  // laneId % 4
    int a_bitPos = a_row * fragK + a_col;
    for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
    {
        uint32_t bitmap = fragBit[tcFragIdx];
        const double *tcValPtr = &tcVal[fragPtr[tcFragIdx]];
        const int *sparse_AToX_idx = &sparse_AToX_index[tcFragIdx * fragK];

        double c_frag[2] = {0.0, 0.0};
        // Load A fragment
        int a_valIdx = __popc(bitmap & ((1U << a_bitPos) - 1));
        int bit = (bitmap >> a_bitPos) & 1;
        double a_frag = bit ? tcValPtr[a_valIdx] : 0.0;

        int x_idx = sparse_AToX_idx[a_col];
        // double b_frag;
        // if (x_idx < SHM_SIZE)
        // {
        //     b_frag = x_shm[x_idx];
        // }
        // else
        // {
        //     b_frag = __ldg(&x_d[x_idx]);
        // }
        double b_frag = __ldg(&x_d[x_idx]);

        // Perform MMA operation
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 \n\t"
            "{%0, %1}, {%2}, {%3}, {%0, %1};\n\t"
            : "+d"(c_frag[0]), "+d"(c_frag[1])
            : "d"(a_frag), "d"(b_frag));
        // Compute sum of accumulator elements
        // thr_accum += c_frag[0];
        if (a_col == 0)
        {
            int y_idx = rowStart + a_row;
            if (y_idx < dRows)
            {
                // atomicAdd(&y_d[y_idx], thr_accum);
                atomicAdd(&y_d[y_idx], c_frag[0]);
            }
        }

    } // End of tcFrag loop
    
}



void tcspmv_fp64(indT *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
            std::vector<double> tcVal, indT *sparse_AToX_index, double *X_val,
            double *Y_val, int rowA, int colA, int *row_order, double *tcTime)
{
    int chunkNum = ceil((double)rowA / (double)fragM);
    int totalTcFrags = chunkPtr[chunkNum];
    double *d_tcVal, *d_X_val, *d_Y_val, *d_Y_val_perf;
    indT *d_sparse_AToX_index, *d_chunkPtr, *d_fragPtr;
    uint32_t *d_fragBit;
   
    // cudaMemcpyToSymbol(x_const, X_val, CONST_SIZE * sizeof(double));

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

    CUDA_CHECK_ERROR(cudaMalloc(&d_Y_val_perf, sizeof(double) * rowA));
    CUDA_CHECK_ERROR(cudaMemset(d_Y_val_perf, 0, sizeof(double) * rowA));

    int warpsPerBlock = 4;
    int warpSize = 32;
    int threadsPerBlock = warpsPerBlock * warpSize;
    int numRowChunks = (rowA + fragM - 1) / fragM;
    int blocksPerGrid = (numRowChunks + warpsPerBlock - 1) / warpsPerBlock;

    printf("Launching tcspmv_kernel_fp64 with %d blocks, %d threads per block\n",
           blocksPerGrid, threadsPerBlock);

    int warp_iter = 100;
    for (int i = 0; i < warp_iter; ++i)
    {
        tcspmv_kernel_fp64<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val_perf, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    int test_iter = 3000;
    cuda_time_test_start();
    for (int i = 0; i < test_iter; ++i)
    {
        tcspmv_kernel_fp64<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val_perf, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    cuda_time_test_end();


    double runtime = (elapsedTime) / test_iter;
    printf("tcspmv_kernel_fp64: %g ms\n", runtime);
    *tcTime = runtime;
    tcspmv_kernel_fp64<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaMemcpy(Y_val, d_Y_val, sizeof(double) * rowA, cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_tcVal));
    CUDA_CHECK_ERROR(cudaFree(d_fragPtr));
    CUDA_CHECK_ERROR(cudaFree(d_fragBit));
    CUDA_CHECK_ERROR(cudaFree(d_chunkPtr));
    CUDA_CHECK_ERROR(cudaFree(d_sparse_AToX_index));
    CUDA_CHECK_ERROR(cudaFree(d_X_val));
    CUDA_CHECK_ERROR(cudaFree(d_Y_val));
    CUDA_CHECK_ERROR(cudaFree(d_Y_val_perf));
}
