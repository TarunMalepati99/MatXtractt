
#include "common.h"
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
    int warpId = threadIdx.x / 32; // Warp ID within the block
    int laneId = threadIdx.x & 31; // Lane ID within the warp

    int rowChunkIndex = blockIdx.x * warpsPerBlock + warpId;

    int tcFragStart = chunkPtr[rowChunkIndex];
    int tcFragEnd = chunkPtr[rowChunkIndex + 1];
    int numTcFragsInChunk = tcFragEnd - tcFragStart;

    if (numTcFragsInChunk == 0)
    {
        return;
    }
    int rowStart = rowChunkIndex * fragM;
    // int rowEnd = (rowStart + fragM > dRows) ? dRows : rowStart + fragM;
    for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
    {
        uint32_t bitmap = fragBit[tcFragIdx];
        const double *tcValPtr = &tcVal[fragPtr[tcFragIdx]];
        const int *sparse_AToX_idx = &sparse_AToX_index[tcFragIdx * fragK];

        // Define fragments for MMA
        double a_frag = 0.0;
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
            a_frag = tcValPtr[a_valIdx];
        }
        int x_idx = sparse_AToX_idx[a_col];
        // if (x_idx >= 0 && x_idx < dCols)
        // {
        //     b_frag = x_d[x_idx];
        // }
        // else
        // {
        //     b_frag = 0.0;
        // }
        b_frag = x_d[x_idx];

        // Perform MMA operation
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 \n\t"
            "{%0, %1}, {%2}, {%3}, {%0, %1};\n\t"
            : "+d"(c_frag[0]), "+d"(c_frag[1])
            : "d"(a_frag), "d"(b_frag));

        // Compute sum of accumulator elements
        if (a_col == 0)
        {
            int y_idx = rowStart + a_row;
            if (y_idx < dRows)
            {
                atomicAdd(&y_d[y_idx], c_frag[0]);
            }
        }
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

    printf("Launching kernel with %d blocks, %d threads per block\n",
           blocksPerGrid, threadsPerBlock);

    int warp_iter = 100;
    for (int i = 0; i < warp_iter; ++i)
    {
        tcspmv_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    int test_iter = 1000;
    gettimeofday(&t1, NULL);
    cuda_time_test_start();
    for (int i = 0; i < test_iter; ++i)
    {
        tcspmv_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    cuda_time_test_end();

    double runtime = (elapsedTime) / test_iter;
    printf("\n SpMV CUDA kernel runtime = %g ms\n", runtime);
    gettimeofday(&t2, NULL);
    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaMemcpy(Y_val, d_Y_val, sizeof(double) * rowA, cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_tcVal));
    CUDA_CHECK_ERROR(cudaFree(d_fragPtr));
    CUDA_CHECK_ERROR(cudaFree(d_fragBit));
    CUDA_CHECK_ERROR(cudaFree(d_chunkPtr));
    CUDA_CHECK_ERROR(cudaFree(d_sparse_AToX_index));
    CUDA_CHECK_ERROR(cudaFree(d_X_val));
    CUDA_CHECK_ERROR(cudaFree(d_Y_val));
}