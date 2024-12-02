__device__ __forceinline__ void tcspmv_kernel_fp16_ptb(
    const half *__restrict__ x_d,
    half *__restrict__ y_d,
    const int *__restrict__ chunkPtr,
    const int *__restrict__ fragPtr,
    const uint32_t *__restrict__ fragBit,
    const half *__restrict__ tcVal,
    const int *__restrict__ sparse_AToX_index,
    int dRows,
    int dCols,
    int original_block_num,
    int issued_block_num,
    const int tid_in_block)
{
    for (int bid_in_grid = blockIdx.x; bid_in_grid < original_block_num; bid_in_grid += issued_block_num)
    {
        const int warpsPerBlock = 4;
        int warpId = threadIdx.x / 32; // Warp ID within the block
        int laneId = threadIdx.x & 31; // Lane ID within the warp
        int mmaIndex = (laneId < 16) ? laneId / 4 : (laneId - 16) / 4;

        int rowChunkIndex = bid_in_grid * warpsPerBlock + warpId;

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
        half sum = __half(0);
        int a_row = (laneId & 3) + ((laneId < 16) ? 0 : 4);

        for (int tcFragIdx_warp_start = tcFragStart; tcFragIdx_warp_start < tcFragEnd; tcFragIdx_warp_start += 4)
        {
            half acc[8] = {__half(0)};

            int dangfragIdx = tcFragIdx_warp_start + mmaIndex;
            int fragIdx = dangfragIdx >= tcFragEnd ? (tcFragEnd - 1) : dangfragIdx; // 确保其合法内存访问

            uint32_t bitmap = fragBit[fragIdx];
            const half *tcValPtr = &tcVal[fragPtr[fragIdx]];
            // load A
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int a_col = i; // Since i ranges from 0 to 3
                int a_bitPos = a_row * fragK + a_col;
                int a_valIdx = __popc(bitmap & ((1U << a_bitPos) - 1));
                int bit = (bitmap >> a_bitPos) & 1;
                frag_a[i] = bit ? tcValPtr[a_valIdx] : __float2half(0.0f);
            }
            // load B
            const int *sparse_AToX_idx = &sparse_AToX_index[fragIdx * fragK];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int b_row = i; // Since i ranges from 0 to 3
                int x_idx = sparse_AToX_idx[b_row];
                frag_b[i] = __ldg(&x_d[x_idx]);
            }
            uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_a[0]);
            uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_b[0]);
            uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);
            asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
                " { %0, %1, %2, %3 }, "
                " { %4, %5 }, "
                " { %6, %7 }, "
                " { %0, %1, %2, %3 };"
                : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]) : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));

            if (dangfragIdx >= tcFragEnd)
            {
                acc[0] = 0;
            }
            sum += acc[0];
        }

        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);

        // Write the result to y_d
        if (mmaIndex == 0)
        {
            int y_idx = rowStart + a_row;
            if (y_idx < dRows)
            {
                // atomicAdd(&y_d[y_idx], sum);
                ushort *sum_u = reinterpret_cast<ushort *>(&sum);
                asm volatile("st.global.cs.u16 [%0], %1;" ::"l"(y_d + y_idx), "h"(*sum_u));
            }
        }
    }
}

template <unsigned int threads_per_row>
__device__ __forceinline__ half warpReduceSum222(half sum)
{
    if (threads_per_row >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (threads_per_row >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, 2-10, etc.
    if (threads_per_row >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, 2-6, etc.
    if (threads_per_row >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, 4-6, 5-7, etc.
    if (threads_per_row >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1, 2-3, 4-5, etc.
    return sum;
}


template <indT THREADS_PER_VECTOR, typename I, typename T>
__device__ __forceinline__ void cdspmv_kernel_ptb_fp16(
    T *d_val,
    indT *d_ptr,
    indT *d_cols,
    indT rowA,
    T *d_vector,
    T *d_out,
    int original_block_num,
    int issued_block_num,
    const int tid_in_block)
{
    for (int bid_in_grid = blockIdx.x; bid_in_grid < original_block_num; bid_in_grid += issued_block_num)
    {
        const int thread_id = 256 * bid_in_grid + tid_in_block;
        const int thread_lane = tid_in_block & (THREADS_PER_VECTOR - 1);
        const int row_id = thread_id / THREADS_PER_VECTOR;

        if (row_id < rowA)
        {
            const int row_start = d_ptr[row_id]; // same as: row_start = Ap[row];
            const int row_end = d_ptr[row_id + 1];

            // initialize local sum
            T sum = 0;

            // accumulate local sums
            for (int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
                sum += d_val[jj] * d_vector[d_cols[jj]];

            sum = warpReduceSum222<THREADS_PER_VECTOR>(sum);
            if (thread_lane == 0)
            {
                d_out[row_id] = sum;
            }
        }
    }
}

template <indT thread_num_tc, indT thread_num_cd>
__global__ void fospmv_kernel_fp16_ptb(half *tcX, half *tcY, indT *chunkPtr, indT *fragPtr, uint32_t *fragBit, half *tcVal, indT *sparse_AToX_index, int tcRow, int tcCol,
                                       half *csrVal, indT *csrRowPtr, indT *csrColInd, int cdRow, half *cdX, half *cdY, indT mean_col_num,
                                       indT original_block_num_tc, indT original_block_num_cd, indT issued_block_num)
{
    if (threadIdx.x < thread_num_tc)
    {
        const int tid_in_block = threadIdx.x;
        tcspmv_kernel_fp16_ptb(tcX, tcY, chunkPtr, fragPtr, fragBit, tcVal,
                               sparse_AToX_index, tcRow, tcCol, original_block_num_tc, issued_block_num, tid_in_block);
    }
    else if (threadIdx.x < thread_num_tc * 1 + thread_num_cd)
    {
        const int tid_in_block = threadIdx.x - thread_num_tc * 1;
        if (mean_col_num <= 2)
        {
            const int THREADS_PER_VECTOR = 2;
            const unsigned int VECTORS_PER_BLOCK = 128;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((cdRow + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            cdspmv_kernel_ptb_fp16<THREADS_PER_VECTOR, indT, half>(csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, NUM_BLOCKS, issued_block_num, tid_in_block);
        }
        else if (mean_col_num > 2 && mean_col_num <= 4)
        {
            const int THREADS_PER_VECTOR = 4;
            const unsigned int VECTORS_PER_BLOCK = 64;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((cdRow + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            cdspmv_kernel_ptb_fp16<THREADS_PER_VECTOR, indT, half>(csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, NUM_BLOCKS, issued_block_num, tid_in_block);
        }
        else if (mean_col_num > 4 && mean_col_num <= 8)
        {
            const int THREADS_PER_VECTOR = 8;
            const unsigned int VECTORS_PER_BLOCK = 32;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((cdRow + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            cdspmv_kernel_ptb_fp16<THREADS_PER_VECTOR, indT, half>(csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, NUM_BLOCKS, issued_block_num, tid_in_block);
        }
        else if (mean_col_num > 8 && mean_col_num <= 16)
        {
            const int THREADS_PER_VECTOR = 16;
            const unsigned int VECTORS_PER_BLOCK = 16;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((cdRow + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            cdspmv_kernel_ptb_fp16<THREADS_PER_VECTOR, indT, half>(csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, NUM_BLOCKS, issued_block_num, tid_in_block);
        }
        else if (mean_col_num > 16)
        {
            const int THREADS_PER_VECTOR = 32;
            const unsigned int VECTORS_PER_BLOCK = 8;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((cdRow + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            cdspmv_kernel_ptb_fp16<THREADS_PER_VECTOR, indT, half>(csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, NUM_BLOCKS, issued_block_num, tid_in_block);
        }
    }
}