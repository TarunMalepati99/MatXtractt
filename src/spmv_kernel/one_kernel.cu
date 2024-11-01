#include "common.h"
#include "fuse_kernel.h"
const int PRONNZT = 4;
cudaStream_t stream_tc, stream_cd;

template <int BREAK_STRIDE, typename I>
__global__ void pre_startRowPerBlock(const I *__restrict__ row_ptr,
                                     const I m,
                                     I *__restrict__ startRowPerBlock)
{
  const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_id > m + 1)
    return;
  int a = row_ptr[global_thread_id];
  int b = row_ptr[min(global_thread_id + 1, (int)m + 1)];

  int blocka = divup<int>(a, BREAK_STRIDE);
  int blockb = (b - 1) / static_cast<int>(BREAK_STRIDE);

  if (a != b)
    for (; blocka <= blockb; ++blocka)
      startRowPerBlock[blocka] = global_thread_id;
}

template <typename I, typename T, indT NNZ_PER_BLOCK, indT THREADS_PER_BLOCK>
__device__ __forceinline__ void lbNEC_reduce_oneRow_in_thread(const int tid_in_block, const int block_id,
                                                              const I reduceStartRowId, const I reduceEndRowId,
                                                              const I *__restrict__ row_ptr,
                                                              const T *__restrict__ smem, T *__restrict__ y)
{
  I reduce_row_id = reduceStartRowId + tid_in_block;
  I nnz_id_before = block_id * NNZ_PER_BLOCK;
  for (; reduce_row_id < reduceEndRowId; reduce_row_id += THREADS_PER_BLOCK)
  {
    T sum = 0;
    // const I reduce_start_idx = max((indT)0, row_ptr[reduce_row_id] - nnz_id_before);
    // const I reduce_end_idx = min(NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - nnz_id_before);
    const I reduce_start_idx = (row_ptr[reduce_row_id] - nnz_id_before) < 0 ? 0 : (row_ptr[reduce_row_id] - nnz_id_before);
    const I reduce_end_idx = (row_ptr[reduce_row_id + 1] - nnz_id_before) > NNZ_PER_BLOCK ? NNZ_PER_BLOCK : (row_ptr[reduce_row_id + 1] - nnz_id_before);
    for (int i = reduce_start_idx; i < reduce_end_idx; i++)
    {
      sum += smem[i];
    }
    atomicAdd(y + reduce_row_id, sum);
  }
}

template <typename I, typename T, indT NNZ_PER_BLOCK, indT THREADS_PER_BLOCK>
__device__ __forceinline__ void lbNEC_reduce_oneRow_in_block(const int tid_in_block, const int block_id,
                                                             const I reduceStartRowId, const I reduceEndRowId,
                                                             const I *__restrict__ row_ptr,
                                                             const T *__restrict__ smem, T *__restrict__ y)
{
  __shared__ T LDS[THREADS_PER_BLOCK];
  // __shared__ volatile T LDS[THREADS_PER_BLOCK];

  T sum = 0;
  const I reduce_start_idx = max((indT)0, row_ptr[reduceStartRowId] - block_id * NNZ_PER_BLOCK);
  const I reduce_end_idx = min(NNZ_PER_BLOCK, row_ptr[reduceStartRowId + 1] - block_id * NNZ_PER_BLOCK);

  for (int j = reduce_start_idx + threadIdx.x; j < reduce_end_idx; j += blockDim.x)
  {
    sum += smem[j];
  }
  LDS[threadIdx.x] = sum;
  __syncthreads();

  // Reduce partial sums
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (threadIdx.x < stride)
      LDS[threadIdx.x] += LDS[threadIdx.x + stride];
  }
  // Write result
  if (threadIdx.x == 0)
    atomicAdd(y + reduceStartRowId, LDS[threadIdx.x]);
}

template <int VEC_SIZE>
__device__ __forceinline__ float warpReduceSum(float sum)
{
  if (VEC_SIZE >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
  if (VEC_SIZE >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, 2-10, etc.
  if (VEC_SIZE >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, 2-6, etc.
  if (VEC_SIZE >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, 4-6, 5-7, etc.
  if (VEC_SIZE >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1, 2-3, 4-5, etc.
  return sum;
}

template <typename I, typename T, int NNZ_PER_BLOCK, int THREADS_PER_BLOCK, int VECTOR_SIZE>
__device__ __forceinline__ void
lbNEC_reduce_oneRow_in_vector(const int n_reduce_rows_num, const int tid_in_block, const int block_id,
                              const I reduceStartRowId, const I reduceEndRowId,
                              const I *__restrict__ row_ptr, const T *__restrict__ smem, T *__restrict__ y)
{
  // use `vec_num` vectors, each vector can process reduction of one row by involving `vec_size` threads.
  const I vec_size = VECTOR_SIZE;
  const I vec_num = THREADS_PER_BLOCK / vec_size;
  const I vec_id = tid_in_block / vec_size;
  const I tid_in_vec = tid_in_block & (vec_size - 1);

  I reduce_row_id = reduceStartRowId + vec_id;
  for (; reduce_row_id < reduceEndRowId; reduce_row_id += vec_num)
  {
    const I reduce_start_idx = max((indT)0, row_ptr[reduce_row_id] - block_id * NNZ_PER_BLOCK);
    const I reduce_end_idx = min((indT)NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - block_id * NNZ_PER_BLOCK);
    // reduce LDS via vectors.
    T sum = 0;
    for (int i = reduce_start_idx + tid_in_vec; i < reduce_end_idx; i += vec_size)
    {
      sum += smem[i];
    }
    sum = warpReduceSum<vec_size>(sum);
    // store value
    if (tid_in_vec == 0)
    {
      atomicAdd(y + reduce_row_id, sum);
    }
  }
}

template <typename I, typename T, int NNZ_PER_BLOCK, int THREADS_PER_BLOCK, int VECTOR_SIZE>
__device__ __forceinline__ void
lbNEC_reduce_oneRow_in_vector_L(const int n_reduce_rows_num, const int tid_in_block, const int block_id,
                                const I reduceStartRowId, const I reduceEndRowId,
                                const I *__restrict__ row_ptr, const T *__restrict__ smem, T *__restrict__ y)
{
  // use `vec_num` vectors, each vector can process reduction of one row by involving `vec_size` threads.
  const I vec_size = VECTOR_SIZE;
  const I vec_num = THREADS_PER_BLOCK / vec_size;
  const I vec_id = tid_in_block / vec_size;
  const I tid_in_vec = tid_in_block & (vec_size - 1);
  const I warp_lane_id = tid_in_block & 31;

  I reduce_row_id = reduceStartRowId + vec_id;
  for (; reduce_row_id < reduceEndRowId; reduce_row_id += vec_num)
  {
    const I reduce_start_idx = max((indT)0, row_ptr[reduce_row_id] - block_id * NNZ_PER_BLOCK);
    const I reduce_end_idx = min((indT)NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - block_id * NNZ_PER_BLOCK);
    // reduce LDS via vectors.
    T sum = 0;
    for (int i = reduce_start_idx + tid_in_vec; i < reduce_end_idx; i += vec_size)
    {
      sum += smem[i];
    }
    // sum = warpReduceSum<vec_size>(sum);
    for (int offset = 16; offset > 0; offset >>= 1)
    {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // store value
    if (warp_lane_id == 0)
    {
      atomicAdd(y + reduce_row_id, sum);
    }
  }
}

__device__ __forceinline__ void tcspmv_kernel_fp64_impl(
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

template <indT productNnzPerThread, indT THREADS_PER_BLOCK, typename I, typename T>
__device__ __forceinline__ void cdspmv_kernel_impl(
    const int tid_in_block,
    const int bid_in_grid,
    T *d_val,
    indT *d_ptr,
    indT *d_cols,
    indT rowA,
    T *d_vector,
    T *d_out,
    I *__restrict__ startRowPerBlock,
    int block_num_tc)
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
}


template <indT productNnzPerThread, indT THREADS_PER_BLOCK, typename I, typename T>
__global__ void cdspmv_kernel111(T *d_val,
                               indT *d_ptr,
                               indT *d_cols,
                               indT rowA,
                               T *d_vector,
                               T *d_out,
                               I *__restrict__ startRowPerBlock)
{
  const int tid_in_block = threadIdx.x;
  const int NNZ_PER_BLOCK = THREADS_PER_BLOCK * productNnzPerThread;
  __shared__ T middle_s[NNZ_PER_BLOCK];
  const I lastElemId = d_ptr[rowA];

  int blockNnzStart = NNZ_PER_BLOCK * blockIdx.x;

  // product and stream in Shared Memory
#pragma unroll
  for (int round = 0; round < productNnzPerThread; round++)
  {
    const I sIdx = tid_in_block + round * THREADS_PER_BLOCK;
    const I gIdx = min(blockNnzStart + sIdx, lastElemId - 1);
    middle_s[sIdx] = d_val[gIdx] * d_vector[d_cols[gIdx]];
  }
  __syncthreads();

  const I reduceStartRowId = min(startRowPerBlock[blockIdx.x], rowA);
  I reduceEndRowId = min(startRowPerBlock[blockIdx.x + 1], rowA);
  reduceEndRowId = (reduceEndRowId == 0) ? rowA : reduceEndRowId;
  if (d_ptr[reduceEndRowId] % NNZ_PER_BLOCK != 0 || reduceEndRowId == reduceStartRowId)
  {
    reduceEndRowId = min(reduceEndRowId + 1, rowA);
  }
  // online workload balance reduction
  const I n_reduce_rows_num = reduceEndRowId - reduceStartRowId;

  // when threads = 128
  if (n_reduce_rows_num > 64)
  {
    lbNEC_reduce_oneRow_in_thread<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK>(tid_in_block, blockIdx.x,
                                                                          reduceStartRowId, reduceEndRowId,
                                                                          d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num == 1)
  {
    lbNEC_reduce_oneRow_in_block<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK>(tid_in_block, blockIdx.x,
                                                                         reduceStartRowId, reduceEndRowId,
                                                                         d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num == 2)
  {
    lbNEC_reduce_oneRow_in_vector_L<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 64>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                                reduceStartRowId, reduceEndRowId,
                                                                                d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 4)
  {
    lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 32>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                              reduceStartRowId, reduceEndRowId,
                                                                              d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 8)
  {
    lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 16>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                              reduceStartRowId, reduceEndRowId,
                                                                              d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 16)
  {
    lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 8>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                             reduceStartRowId, reduceEndRowId,
                                                                             d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 32)
  {
    lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 4>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                             reduceStartRowId, reduceEndRowId,
                                                                             d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 64)
  {
    lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 2>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                             reduceStartRowId, reduceEndRowId,
                                                                             d_ptr, middle_s, d_out);
  }

  /*
  // 线程256
  if (n_reduce_rows_num > 128)
  {
    lbNEC_reduce_oneRow_in_thread<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK>(tid_in_block, blockIdx.x,
                                                               reduceStartRowId, reduceEndRowId,
                                                               d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num == 1)
  {

    lbNEC_reduce_oneRow_in_block<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK>(tid_in_block, blockIdx.x,
                                                                reduceStartRowId, reduceEndRowId,
                                                                d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num == 2)
  {
    lbNEC_reduce_oneRow_in_vector_L<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 128>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                         reduceStartRowId, reduceEndRowId,
                                                                         d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 4)
  {
    lbNEC_reduce_oneRow_in_vector_L<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 64>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                        reduceStartRowId, reduceEndRowId,
                                                                        d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 8)
  {
    lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 32>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                      reduceStartRowId, reduceEndRowId,
                                                                      d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 16)
  {
    lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 16>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                      reduceStartRowId, reduceEndRowId,
                                                                      d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 32)
  {
    lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 8>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                     reduceStartRowId, reduceEndRowId,
                                                                     d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 64)
  {
    lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 4>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                     reduceStartRowId, reduceEndRowId,
                                                                     d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 128)
  {
    lbNEC_reduce_oneRow_in_vector<I, T, NNZ_PER_BLOCK, THREADS_PER_BLOCK, 2>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                     reduceStartRowId, reduceEndRowId,
                                                                     d_ptr, middle_s, d_out);
  }
  */
}



template <indT thread_num_tc, indT thread_num_cd>
__global__ void fospmv_kernel_fp64_mix(double *tcX, double *tcY, indT *chunkPtr, indT *fragPtr, uint32_t *fragBit, double *tcVal, indT *sparse_AToX_index, int tcRow, int tcCol,
                                       double *csrVal, indT *csrRowPtr, indT *csrColInd, int cdRow, double *cdX, double *cdY, indT *startRowPerBlock,
                                       indT block_num_tc, indT block_num_cd)
{
  if (threadIdx.x < thread_num_tc && blockIdx.x < block_num_tc)
  {
    // tcspmv_kernel_fp64<<<block_num_tc, thread_num_tc>>>(
    //     tcX, tcY, chunkPtr, fragPtr, fragBit, tcVal,
    //     sparse_AToX_index, tcRow, tcCol);
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
      // b_frag = tcX[x_idx];
      double b_frag = __ldg(&tcX[x_idx]);

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
        if (y_idx < tcRow)
        {
          // atomicAdd(&tcY[y_idx], thr_accum);
          atomicAdd(&tcY[y_idx], c_frag[0]);
        }
      }
    }
  }
  else if (threadIdx.x >= thread_num_tc && threadIdx.x < thread_num_tc + thread_num_cd && blockIdx.x < block_num_cd)
  {
    // Adjust thread index for cdspmv_kernel
    const int tid_in_block = threadIdx.x - thread_num_tc;
    const int bid_in_grid = blockIdx.x;
    const int productNnzPerThread = 4;
    const int NNZ_PER_BLOCK = 128 * productNnzPerThread;
    __shared__ valT middle_s[NNZ_PER_BLOCK];
    const indT lastElemId = csrRowPtr[cdRow];

    int blockNnzStart = NNZ_PER_BLOCK * bid_in_grid;

    // product and stream in Shared Memory
#pragma unroll
    for (int round = 0; round < productNnzPerThread; round++)
    {
      const indT sIdx = tid_in_block + round * 128;
      const indT gIdx = min(blockNnzStart + sIdx, lastElemId - 1);
      middle_s[sIdx] = csrVal[gIdx] * cdX[csrColInd[gIdx]];
    }
    __syncthreads();

    const indT reduceStartRowId = min(startRowPerBlock[bid_in_grid], cdRow);
    indT reduceEndRowId = min(startRowPerBlock[bid_in_grid + 1], cdRow);
    reduceEndRowId = (reduceEndRowId == 0) ? cdRow : reduceEndRowId;
    if (csrRowPtr[reduceEndRowId] % NNZ_PER_BLOCK != 0 || reduceEndRowId == reduceStartRowId)
    {
      reduceEndRowId = min(reduceEndRowId + 1, cdRow);
    }
    // online workload balance reduction

    const indT n_reduce_rows_num = reduceEndRowId - reduceStartRowId;

    // when threads = 128
    if (n_reduce_rows_num > 64)
    {
      lbNEC_reduce_oneRow_in_thread<indT, valT, NNZ_PER_BLOCK, 128>(tid_in_block, bid_in_grid,
                                                                    reduceStartRowId, reduceEndRowId,
                                                                    csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num == 1)
    {
      lbNEC_reduce_oneRow_in_block<indT, valT, NNZ_PER_BLOCK, 128>(tid_in_block, bid_in_grid,
                                                                   reduceStartRowId, reduceEndRowId,
                                                                   csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num == 2)
    {
      lbNEC_reduce_oneRow_in_vector_L<indT, valT, NNZ_PER_BLOCK, 128, 64>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                          reduceStartRowId, reduceEndRowId,
                                                                          csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num <= 4)
    {
      lbNEC_reduce_oneRow_in_vector<indT, valT, NNZ_PER_BLOCK, 128, 32>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                        reduceStartRowId, reduceEndRowId,
                                                                        csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num <= 8)
    {
      lbNEC_reduce_oneRow_in_vector<indT, valT, NNZ_PER_BLOCK, 128, 16>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                        reduceStartRowId, reduceEndRowId,
                                                                        csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num <= 16)
    {
      lbNEC_reduce_oneRow_in_vector<indT, valT, NNZ_PER_BLOCK, 128, 8>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                       reduceStartRowId, reduceEndRowId,
                                                                       csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num <= 32)
    {
      lbNEC_reduce_oneRow_in_vector<indT, valT, NNZ_PER_BLOCK, 128, 4>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                       reduceStartRowId, reduceEndRowId,
                                                                       csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num <= 64)
    {
      lbNEC_reduce_oneRow_in_vector<indT, valT, NNZ_PER_BLOCK, 128, 2>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                       reduceStartRowId, reduceEndRowId,
                                                                       csrRowPtr, middle_s, cdY);
    }
  }
}

template <indT thread_num_tc, indT thread_num_cd>
__global__ void fospmv_kernel_fp64_sep(double *tcX, double *tcY, indT *chunkPtr, indT *fragPtr, uint32_t *fragBit, double *tcVal, indT *sparse_AToX_index, int tcRow, int tcCol,
                                       double *csrVal, indT *csrRowPtr, indT *csrColInd, int cdRow, double *cdX, double *cdY, indT *startRowPerBlock,
                                       indT block_num_tc, indT block_num_cd)
{
  // if (threadIdx.x < thread_num_tc && blockIdx.x < block_num_tc)
  if (blockIdx.x < block_num_tc)
  {
    // tcspmv_kernel_fp64<<<block_num_tc, thread_num_tc>>>(
    //     tcX, tcY, chunkPtr, fragPtr, fragBit, tcVal,
    //     sparse_AToX_index, tcRow, tcCol);
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
      // b_frag = tcX[x_idx];
      double b_frag = __ldg(&tcX[x_idx]);

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
        if (y_idx < tcRow)
        {
          // atomicAdd(&tcY[y_idx], thr_accum);
          atomicAdd(&tcY[y_idx], c_frag[0]);
        }
      }
    }
  }
  else // else if (threadIdx.x >= thread_num_tc && threadIdx.x < thread_num_tc + thread_num_cd && blockIdx.x < block_num_cd)
  {
    // cdspmv_kernel<productNnzPerThread, thread_num_cd, indT, double>
    //     <<<block_num_cd, thread_num_cd>>>(
    //         csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, startRowPerBlock);

    // Adjust thread index for cdspmv_kernel
    // const int tid_in_block = threadIdx.x - thread_num_tc;
    const int bid_in_grid = blockIdx.x - block_num_tc;
    const int tid_in_block = threadIdx.x;
    const int productNnzPerThread = 4;
    const int NNZ_PER_BLOCK = 128 * productNnzPerThread;
    __shared__ valT middle_s[NNZ_PER_BLOCK];
    const indT lastElemId = csrRowPtr[cdRow];

    int blockNnzStart = NNZ_PER_BLOCK * bid_in_grid;

    // product and stream in Shared Memory
#pragma unroll
    for (int round = 0; round < productNnzPerThread; round++)
    {
      const indT sIdx = tid_in_block + round * 128;
      const indT gIdx = min(blockNnzStart + sIdx, lastElemId - 1);
      middle_s[sIdx] = csrVal[gIdx] * cdX[csrColInd[gIdx]];
    }
    __syncthreads();

    const indT reduceStartRowId = min(startRowPerBlock[bid_in_grid], cdRow);
    indT reduceEndRowId = min(startRowPerBlock[bid_in_grid + 1], cdRow);
    reduceEndRowId = (reduceEndRowId == 0) ? cdRow : reduceEndRowId;
    if (csrRowPtr[reduceEndRowId] % NNZ_PER_BLOCK != 0 || reduceEndRowId == reduceStartRowId)
    {
      reduceEndRowId = min(reduceEndRowId + 1, cdRow);
    }
    // online workload balance reduction

    const indT n_reduce_rows_num = reduceEndRowId - reduceStartRowId;

    // when threads = 128
    if (n_reduce_rows_num > 64)
    {
      lbNEC_reduce_oneRow_in_thread<indT, valT, NNZ_PER_BLOCK, 128>(tid_in_block, bid_in_grid,
                                                                    reduceStartRowId, reduceEndRowId,
                                                                    csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num == 1)
    {
      lbNEC_reduce_oneRow_in_block<indT, valT, NNZ_PER_BLOCK, 128>(tid_in_block, bid_in_grid,
                                                                   reduceStartRowId, reduceEndRowId,
                                                                   csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num == 2)
    {
      lbNEC_reduce_oneRow_in_vector_L<indT, valT, NNZ_PER_BLOCK, 128, 64>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                          reduceStartRowId, reduceEndRowId,
                                                                          csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num <= 4)
    {
      lbNEC_reduce_oneRow_in_vector<indT, valT, NNZ_PER_BLOCK, 128, 32>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                        reduceStartRowId, reduceEndRowId,
                                                                        csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num <= 8)
    {
      lbNEC_reduce_oneRow_in_vector<indT, valT, NNZ_PER_BLOCK, 128, 16>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                        reduceStartRowId, reduceEndRowId,
                                                                        csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num <= 16)
    {
      lbNEC_reduce_oneRow_in_vector<indT, valT, NNZ_PER_BLOCK, 128, 8>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                       reduceStartRowId, reduceEndRowId,
                                                                       csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num <= 32)
    {
      lbNEC_reduce_oneRow_in_vector<indT, valT, NNZ_PER_BLOCK, 128, 4>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                       reduceStartRowId, reduceEndRowId,
                                                                       csrRowPtr, middle_s, cdY);
    }
    else if (n_reduce_rows_num <= 64)
    {
      lbNEC_reduce_oneRow_in_vector<indT, valT, NNZ_PER_BLOCK, 128, 2>(n_reduce_rows_num, tid_in_block, bid_in_grid,
                                                                       reduceStartRowId, reduceEndRowId,
                                                                       csrRowPtr, middle_s, cdY);
    }
  }
}

template <indT thread_num_tc, indT thread_num_cd>
__global__ void fospmv_kernel_fp64_sep1(double *tcX, double *tcY, indT *chunkPtr, indT *fragPtr, uint32_t *fragBit, double *tcVal, indT *sparse_AToX_index, int tcRow, int tcCol,
                                        double *csrVal, indT *csrRowPtr, indT *csrColInd, int cdRow, double *cdX, double *cdY, indT *startRowPerBlock,
                                        indT block_num_tc, indT block_num_cd)
{
  if (blockIdx.x < block_num_tc)
  {
    tcspmv_kernel_fp64_impl(tcX, tcY, chunkPtr, fragPtr, fragBit, tcVal,
                            sparse_AToX_index, tcRow, tcCol);
  }
  else
  {
    const int bid_in_grid = blockIdx.x - block_num_tc;
    const int tid_in_block = threadIdx.x;
    cdspmv_kernel_impl<PRONNZT, thread_num_cd, indT, double>(tid_in_block, bid_in_grid, csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, startRowPerBlock, block_num_cd);
  }
}
template <indT thread_num_tc, indT thread_num_cd>
void fospmv_kernel_fp64_stream(double *tcX, double *tcY, indT *chunkPtr, indT *fragPtr, uint32_t *fragBit, double *tcVal, indT *sparse_AToX_index, int tcRow, int tcCol,
                               double *csrVal, indT *csrRowPtr, indT *csrColInd, int cdRow, double *cdX, double *cdY, indT *startRowPerBlock,
                               indT block_num_tc, indT block_num_cd,
                               cudaStream_t stream_tc, cudaStream_t stream_cd)
{
  
  tcspmv_kernel_fp64<<<block_num_tc, thread_num_tc, 0, stream_tc>>>(
      tcX, tcY, chunkPtr, fragPtr, fragBit, tcVal,
      sparse_AToX_index, tcRow, tcCol);

  cdspmv_kernel111<PRONNZT, thread_num_cd, indT, valT><<<block_num_cd, thread_num_cd, 0, stream_cd>>>(
      csrVal, csrRowPtr, csrColInd, cdRow, cdX, cdY, startRowPerBlock);

  // cudaStreamSynchronize(stream_tc); // 默认流已全局同步
  // cudaStreamSynchronize(stream_cd);

}

void fospmv_fp64(indT *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit, std::vector<double> tcVal, indT *sparse_AToX_index, double *tcX, double *tcY, int tcRow, int tcCol,
                 double *csrVal, indT *csrRowPtr, indT *csrColInd, double *cdX, double *cdY, int cdRow, int cdCol, indT cdnnzA)
{
  int chunkNum = ceil((double)tcRow / (double)fragM);
  int totalTcFrags = chunkPtr[chunkNum];
  double *d_tcVal, *d_tcX, *d_tcY;
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

  //  tcX
  CUDA_CHECK_ERROR(cudaMalloc(&d_tcX, sizeof(double) * tcCol));
  CUDA_CHECK_ERROR(cudaMemcpy(d_tcX, tcX, sizeof(double) * tcCol, cudaMemcpyHostToDevice));

  //  tcY
  CUDA_CHECK_ERROR(cudaMalloc(&d_tcY, sizeof(double) * tcRow));
  CUDA_CHECK_ERROR(cudaMemset(d_tcY, 0, sizeof(double) * tcRow));

  double *d_cdY, *d_cdY_perf, *d_cdX, *d_val;
  indT *d_indices, *d_ptr;

  CUDA_CHECK_ERROR(cudaMalloc(&d_cdY, sizeof(double) * cdRow));
  CUDA_CHECK_ERROR(cudaMalloc(&d_cdY_perf, sizeof(double) * cdRow));
  CUDA_CHECK_ERROR(cudaMalloc(&d_cdX, sizeof(double) * cdCol));
  CUDA_CHECK_ERROR(cudaMalloc(&d_val, sizeof(double) * cdnnzA));
  CUDA_CHECK_ERROR(cudaMalloc(&d_indices, sizeof(indT) * cdnnzA));
  CUDA_CHECK_ERROR(cudaMalloc(&d_ptr, sizeof(indT) * (cdRow + 1)));

  CUDA_CHECK_ERROR(cudaMemcpy(d_val, csrVal, sizeof(double) * cdnnzA, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_indices, csrColInd, sizeof(indT) * cdnnzA, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_ptr, csrRowPtr, sizeof(indT) * (cdRow + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_cdX, cdX, sizeof(double) * cdCol, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_cdY, cdY, sizeof(double) * cdRow, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_cdY_perf, cdY, sizeof(double) * cdRow, cudaMemcpyHostToDevice));
  cudaMemset(d_cdY, 0.0, sizeof(double) * cdRow);

  // tc thread grid
  const int warpsPerBlock = 4;
  const int warpSize = 32;
  const int thread_num_tc = warpsPerBlock * warpSize;
  const int numRowChunks = (tcRow + fragM - 1) / fragM;
  const int block_num_tc = (numRowChunks + warpsPerBlock - 1) / warpsPerBlock;
  printf("Launching kernel with %d blocks, %d threads per block\n",
         block_num_tc, thread_num_tc);

  // cd thread grid
  const int productNnzPerThread = PRONNZT;
  const int thread_num_cd = 128;
  const int block_num_cd = cdnnzA / (productNnzPerThread * thread_num_cd) + ((cdnnzA % (productNnzPerThread * thread_num_cd) == 0) ? 0 : 1);

  // cd preprocess
  const indT startRowPerBlock_len = block_num_cd + 1;
  indT *startRowPerBlock;
  CUDA_CHECK_ERROR(cudaMalloc((void **)&startRowPerBlock, sizeof(indT) * startRowPerBlock_len));
  CUDA_CHECK_ERROR(cudaMemset(startRowPerBlock, 0, sizeof(indT) * startRowPerBlock_len));
  pre_startRowPerBlock<productNnzPerThread * thread_num_cd, indT><<<divup<uint32_t>(cdRow + 1, 256), 256>>>(d_ptr, cdRow, startRowPerBlock);

  int sep_grid, sep_block;
  sep_grid = block_num_tc + block_num_cd;
  sep_block = thread_num_cd > thread_num_tc ? thread_num_cd : thread_num_tc;

  int mix_grid, mix_block;
  mix_grid = (block_num_tc > block_num_cd) ? block_num_tc : block_num_cd;
  mix_block = thread_num_tc + thread_num_cd;

  printf("Launching fospmv_kernel_fp64_sep with %d blocks, %d threads per block\n",
         sep_grid, sep_block);
  cudaStream_t stream_tc, stream_cd;
  cudaStreamCreate(&stream_tc);
  cudaStreamCreate(&stream_cd);
  int warp_iter = 100;
  for (int i = 0; i < warp_iter; ++i)
  {
    fospmv_kernel_fp64_stream<thread_num_tc, thread_num_cd>(
        d_tcX, d_tcY, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal, d_sparse_AToX_index, tcRow, tcCol,
        d_val, d_ptr, d_indices, cdRow, d_cdX, d_cdY, startRowPerBlock,
        block_num_tc, block_num_cd, stream_tc, stream_cd);

    // fospmv_kernel_fp64_sep1<thread_num_tc, thread_num_cd><<<sep_grid, sep_block>>>(
    //     d_tcX, d_tcY, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal, d_sparse_AToX_index, tcRow, tcCol,
    //     d_val, d_ptr, d_indices, cdRow, d_cdX, d_cdY, startRowPerBlock,
    //     block_num_tc, block_num_cd);

    // fospmv_kernel_fp64_mix<thread_num_tc, thread_num_cd><<<mix_grid, mix_block>>>(
    //     d_tcX, d_tcY, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal, d_sparse_AToX_index, tcRow, tcCol,
    //     d_val, d_ptr, d_indices, cdRow, d_cdX, d_cdY, startRowPerBlock,
    //     block_num_tc, block_num_cd);
  }
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  int test_iter = 1000;
  cuda_time_test_start();
  for (int i = 0; i < test_iter; ++i)
  {
    fospmv_kernel_fp64_stream<thread_num_tc, thread_num_cd>(
        d_tcX, d_tcY, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal, d_sparse_AToX_index, tcRow, tcCol,
        d_val, d_ptr, d_indices, cdRow, d_cdX, d_cdY, startRowPerBlock,
        block_num_tc, block_num_cd, stream_tc, stream_cd);

    // fospmv_kernel_fp64_sep1<thread_num_tc, thread_num_cd><<<sep_grid, sep_block>>>(
    //     d_tcX, d_tcY, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal, d_sparse_AToX_index, tcRow, tcCol,
    //     d_val, d_ptr, d_indices, cdRow, d_cdX, d_cdY, startRowPerBlock,
    //     block_num_tc, block_num_cd);

    // fospmv_kernel_fp64_mix<thread_num_tc, thread_num_cd><<<mix_grid, mix_block>>>(
    //     d_tcX, d_tcY, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal, d_sparse_AToX_index, tcRow, tcCol,
    //     d_val, d_ptr, d_indices, cdRow, d_cdX, d_cdY, startRowPerBlock,
    //     block_num_tc, block_num_cd);
  }
  cuda_time_test_end();

  cudaStreamDestroy(stream_tc);
  cudaStreamDestroy(stream_cd);
  double runtime = (elapsedTime) / test_iter;
  printf("\n Fused one SpMV CUDA kernel runtime = %g ms\n", runtime);

  CUDA_CHECK_ERROR(cudaGetLastError());

  CUDA_CHECK_ERROR(cudaMemcpy(tcY, d_tcY, sizeof(double) * tcRow, cudaMemcpyDeviceToHost));
  CUDA_CHECK_ERROR(cudaMemcpy(cdY, d_cdY, sizeof(double) * cdRow, cudaMemcpyDeviceToHost));

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  CUDA_CHECK_ERROR(cudaFree(d_tcVal));
  CUDA_CHECK_ERROR(cudaFree(d_fragPtr));
  CUDA_CHECK_ERROR(cudaFree(d_fragBit));
  CUDA_CHECK_ERROR(cudaFree(d_chunkPtr));
  CUDA_CHECK_ERROR(cudaFree(d_sparse_AToX_index));
  CUDA_CHECK_ERROR(cudaFree(d_tcX));
  CUDA_CHECK_ERROR(cudaFree(d_tcY));

  CUDA_CHECK_ERROR(cudaFree(d_cdY_perf));
  CUDA_CHECK_ERROR(cudaFree(d_cdY));
  CUDA_CHECK_ERROR(cudaFree(d_cdX));
  CUDA_CHECK_ERROR(cudaFree(d_val));
  CUDA_CHECK_ERROR(cudaFree(d_indices));
  CUDA_CHECK_ERROR(cudaFree(d_ptr));
  CUDA_CHECK_ERROR(cudaFree(startRowPerBlock));
}

