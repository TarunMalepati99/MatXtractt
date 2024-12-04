__device__ __forceinline__ void tcspmv_kernel_fp64_ptb(
    const double *__restrict__ x_d,
    double *__restrict__ y_d,
    const int *__restrict__ chunkPtr,
    const int *__restrict__ fragPtr,
    const uint32_t *__restrict__ fragBit,
    const double *__restrict__ tcVal,
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
        int warpId = tid_in_block / 32; // Warp ID within the block
        int laneId = tid_in_block & 31; // Lane ID within the warp
        // double thr_accum = 0;

        int rowChunkIndex = bid_in_grid * warpsPerBlock + warpId;

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
}

template <indT productNnzPerThread, indT THREADS_PER_BLOCK, typename I, typename T>
__device__ __forceinline__ void cdspmv_kernel_ptb_shared(
    T *d_val,
    indT *d_ptr,
    indT *d_cols,
    indT rowA,
    T *d_vector,
    T *d_out,
    I *__restrict__ startRowPerBlock,
    int original_block_num,
    int issued_block_num,
    const int tid_in_block)
{
    for (int bid_in_grid = blockIdx.x; bid_in_grid < original_block_num; bid_in_grid += issued_block_num)
    {
        const int NNZ_PER_BLOCK = THREADS_PER_BLOCK * productNnzPerThread;
        __shared__ T middle_s[NNZ_PER_BLOCK];
        const I lastElemId = d_ptr[rowA];

        int blockNnzStart = NNZ_PER_BLOCK * bid_in_grid;

// product and stream in Shared Memory
#pragma unroll
        for (int round = 0; round < productNnzPerThread; round++)
        {
            const I sIdx = tid_in_block + round * THREADS_PER_BLOCK;
            const I gIdx = min(blockNnzStart + sIdx, lastElemId - 1);
            middle_s[sIdx] = d_val[gIdx] * d_vector[d_cols[gIdx]];
        }
        __syncthreads();

        const I reduceStartRowId = min(startRowPerBlock[bid_in_grid], rowA);
        I reduceEndRowId = min(startRowPerBlock[bid_in_grid + 1], rowA);
        reduceEndRowId = (reduceEndRowId == 0) ? rowA : reduceEndRowId;
        if (d_ptr[reduceEndRowId] % NNZ_PER_BLOCK != 0 || reduceEndRowId == reduceStartRowId)
        {
            reduceEndRowId = min(reduceEndRowId + 1, rowA);
        }
        // online workload balance reduction
        const I n_reduce_rows_num = reduceEndRowId - reduceStartRowId;
/*
        // when threads = 128
        if (n_reduce_rows_num > 64)
        {
            lbNEC_reduce_oneRow_in_thread<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK>(tid_in_block, bid_in_grid,
                                                                                  reduceStartRowId, reduceEndRowId,
                                                                                  d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num == 1)
        {
            lbNEC_reduce_oneRow_in_block<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK>(tid_in_block, bid_in_grid,
                                                                                 reduceStartRowId, reduceEndRowId,
                                                                                 d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num == 2)
        {
            lbNEC_reduce_oneRow_in_vector_L<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 64>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                                        reduceStartRowId, reduceEndRowId,
                                                                                        d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num <= 4)
        {
            lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 32>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                                      reduceStartRowId, reduceEndRowId,
                                                                                      d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num <= 8)
        {
            lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 16>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                                      reduceStartRowId, reduceEndRowId,
                                                                                      d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num <= 16)
        {
            lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 8>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                                     reduceStartRowId, reduceEndRowId,
                                                                                     d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num <= 32)
        {
            lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 4>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                                     reduceStartRowId, reduceEndRowId,
                                                                                     d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num <= 64)
        {
            lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 2>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                                     reduceStartRowId, reduceEndRowId,
                                                                                     d_ptr, middle_s, d_out);
        }
*/
        // 线程256
        if (n_reduce_rows_num > 128)
        {
        lbNEC_reduce_oneRow_in_thread<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK>(tid_in_block, bid_in_grid,
                                                                    reduceStartRowId, reduceEndRowId,
                                                                    d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num == 1)
        {

        lbNEC_reduce_oneRow_in_block<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK>(tid_in_block, bid_in_grid,
                                                                    reduceStartRowId, reduceEndRowId,
                                                                    d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num == 2)
        {
        lbNEC_reduce_oneRow_in_vector_L<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 128>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                            reduceStartRowId, reduceEndRowId,
                                                                            d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num <= 4)
        {
        lbNEC_reduce_oneRow_in_vector_L<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 64>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                            reduceStartRowId, reduceEndRowId,
                                                                            d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num <= 8)
        {
        lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 32>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                            reduceStartRowId, reduceEndRowId,
                                                                            d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num <= 16)
        {
        lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 16>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                            reduceStartRowId, reduceEndRowId,
                                                                            d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num <= 32)
        {
        lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 8>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                        reduceStartRowId, reduceEndRowId,
                                                                        d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num <= 64)
        {
        lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 4>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                        reduceStartRowId, reduceEndRowId,
                                                                        d_ptr, middle_s, d_out);
        }
        else if (n_reduce_rows_num <= 128)
        {
        lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 2>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                        reduceStartRowId, reduceEndRowId,
                                                                        d_ptr, middle_s, d_out);
        }
    }
}

template <unsigned int threads_per_row>
__device__ __forceinline__ float warpReduceSum111(float sum)
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
__device__ __forceinline__ void cdspmv_kernel_ptb(
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

            sum = warpReduceSum111<THREADS_PER_VECTOR>(sum);
            if (thread_lane == 0)
            {
                d_out[row_id] = sum;
            }
        }
    }
}

template <indT thread_num_tc, indT thread_num_cd>
__global__ void fospmv_kernel_fp64_ptb(double *tcX, double *tcY, indT *chunkPtr, indT *fragPtr, uint32_t *fragBit, double *tcVal, indT *sparse_AToX_index, int tcRow, int tcCol,
                                       double *csrVal, indT *csrRowPtr, indT *csrColInd, int cdRow, double *cdX, double *cdY, indT *startRowPerBlock, indT mean_col_num,
                                       indT original_block_num_tc, indT original_block_num_cd, indT issued_block_num)
{
    if (threadIdx.x < thread_num_tc)
    {
        const int tid_in_block = threadIdx.x;
        tcspmv_kernel_fp64_ptb(tcX, tcY, chunkPtr, fragPtr, fragBit, tcVal,
                               sparse_AToX_index, tcRow, tcCol, original_block_num_tc, issued_block_num, tid_in_block);
    }
    else if (threadIdx.x < thread_num_tc * 1 + thread_num_cd)
    {
        const int tid_in_block = threadIdx.x - thread_num_tc * 1;
        // if (mean_col_num <= 2)
        // {
        //     const int THREADS_PER_VECTOR = 2;
        //     const unsigned int VECTORS_PER_BLOCK = 128;
        //     const unsigned int NUM_BLOCKS = static_cast<unsigned int>((cdRow + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
        //     cdspmv_kernel_ptb<THREADS_PER_VECTOR, indT, double>(csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, NUM_BLOCKS, issued_block_num, tid_in_block);
        // }
        // else if (mean_col_num > 2 && mean_col_num <= 4)
        // {
        //     const int THREADS_PER_VECTOR = 4;
        //     const unsigned int VECTORS_PER_BLOCK = 64;
        //     const unsigned int NUM_BLOCKS = static_cast<unsigned int>((cdRow + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
        //     cdspmv_kernel_ptb<THREADS_PER_VECTOR, indT, double>(csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, NUM_BLOCKS, issued_block_num, tid_in_block);
        // }
        // else if (mean_col_num > 4 && mean_col_num <= 8)
        // {
        //     const int THREADS_PER_VECTOR = 8;
        //     const unsigned int VECTORS_PER_BLOCK = 32;
        //     const unsigned int NUM_BLOCKS = static_cast<unsigned int>((cdRow + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
        //     cdspmv_kernel_ptb<THREADS_PER_VECTOR, indT, double>(csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, NUM_BLOCKS, issued_block_num, tid_in_block);
        // }
        // else if (mean_col_num > 8 && mean_col_num <= 16)
        // {
        //     const int THREADS_PER_VECTOR = 16;
        //     const unsigned int VECTORS_PER_BLOCK = 16;
        //     const unsigned int NUM_BLOCKS = static_cast<unsigned int>((cdRow + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
        //     cdspmv_kernel_ptb<THREADS_PER_VECTOR, indT, double>(csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, NUM_BLOCKS, issued_block_num, tid_in_block);
        // }
        // else if (mean_col_num > 16)
        // {
        //     const int THREADS_PER_VECTOR = 32;
        //     const unsigned int VECTORS_PER_BLOCK = 8;
        //     const unsigned int NUM_BLOCKS = static_cast<unsigned int>((cdRow + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
        //     cdspmv_kernel_ptb<THREADS_PER_VECTOR, indT, double>(csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, NUM_BLOCKS, issued_block_num, tid_in_block);
        // }
        cdspmv_kernel_ptb_shared<PRONNZT, thread_num_cd, indT, double>(csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, startRowPerBlock, original_block_num_cd, issued_block_num, tid_in_block);
    }
}