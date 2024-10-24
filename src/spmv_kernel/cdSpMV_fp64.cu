#include "common.h"

// typedef int indT;

template <typename T>
__host__ __device__ __forceinline__ T divup(T a, T b)
{
  return (a + b - 1) / b;
}

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

template <indT productNnzPerThread, indT THREADS_PER_BLOCK, typename I, typename T>
__global__ void cdspmv_kernel(T *d_val,
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

void cdspmv(char *filename, valT *csrVal, indT *csrRowPtr, indT *csrColInd,
             valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA,
             double *necTime, double *necPre)
{
  struct timeval t1;
  struct timeval t2;
  struct timeval tpre1;
  struct timeval tpre2;

  valT *d_vecY_csr, *d_vecX_csr, *d_val;
  indT *d_indices, *d_ptr;

  cudaMalloc(&d_vecY_csr, sizeof(valT) * rowA);
  cudaMalloc(&d_vecX_csr, sizeof(valT) * colA);
  cudaMalloc(&d_val, sizeof(valT) * nnzA);
  cudaMalloc(&d_indices, sizeof(indT) * nnzA);
  cudaMalloc(&d_ptr, sizeof(indT) * (rowA + 1));

  cudaMemcpy(d_val, csrVal, sizeof(valT) * nnzA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, csrColInd, sizeof(indT) * nnzA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptr, csrRowPtr, sizeof(indT) * (rowA + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vecX_csr, X_val, sizeof(valT) * colA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vecY_csr, Y_val, sizeof(valT) * rowA, cudaMemcpyHostToDevice);
  // cudaMemset(d_vecY_csr, 0.0, sizeof(valT) * rowA);
#ifdef fp64
  const int productNnzPerThread = 4;
#else
  const int productNnzPerThread = 16;
#endif
  const int THREADS_PER_BLOCK = 128;

  const int WORK_BLOCKS = nnzA / (productNnzPerThread * THREADS_PER_BLOCK) + ((nnzA % (productNnzPerThread * THREADS_PER_BLOCK) == 0) ? 0 : 1);

  const indT startRowPerBlock_len = WORK_BLOCKS + 1;

  indT *startRowPerBlock;
  cudaMalloc((void **)&startRowPerBlock, sizeof(indT) * startRowPerBlock_len);
  cudaMemset(startRowPerBlock, 0, sizeof(indT) * startRowPerBlock_len);
  gettimeofday(&tpre1, NULL);
  pre_startRowPerBlock<productNnzPerThread * THREADS_PER_BLOCK, indT><<<divup<uint32_t>(rowA + 1, 256), 256>>>(d_ptr, rowA, startRowPerBlock);

  gettimeofday(&tpre2, NULL);
  int warmup_time = 100;
  int execute_time = 1000;

  for (int i = 0; i < warmup_time; ++i)
  {
    cdspmv_kernel<productNnzPerThread, THREADS_PER_BLOCK, indT, valT><<<(WORK_BLOCKS), (THREADS_PER_BLOCK)>>>(d_val, d_ptr, d_indices, rowA, d_vecX_csr, d_vecY_csr, startRowPerBlock);

  }
  cudaDeviceSynchronize();
  gettimeofday(&t1, NULL);
  for (int i = 0; i < execute_time; ++i)
  {
    cdspmv_kernel<productNnzPerThread, THREADS_PER_BLOCK, indT, valT><<<(WORK_BLOCKS), (THREADS_PER_BLOCK)>>>(d_val, d_ptr, d_indices, rowA, d_vecX_csr, d_vecY_csr, startRowPerBlock);

  }
  cudaDeviceSynchronize();

  gettimeofday(&t2, NULL);

  double pre_time = ((tpre2.tv_sec - tpre1.tv_sec) * 1000.0 + (tpre2.tv_usec - tpre1.tv_usec) / 1000.0) / 1;
  *necPre = pre_time;
  double nec_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / execute_time;
  *necTime = nec_time;
  double nec_gflops = (double)((long)nnzA * 2) / (nec_time * 1e6);

  // printf("SpMV_X:  %8.4lf ms, %8.4lf GFlop/s\n", nec_time, nec_gflops);

  // int iter = (int) pre_time / nec_time;
  // printf("iterate:  %d \n", iter);

  // printf("\nrowA = %d, row_long = %d, row_block = %d, row_short1 = %d, common13 = %d, row_short_3 = %d, row_short_4 = %d, row_short_2 = %d\n", rowA, row_long, row_block, short_row_1, common_13, short_row_3, short_row_4, short_row_2);

  cudaMemcpy(Y_val, d_vecY_csr, sizeof(valT) * rowA, cudaMemcpyDeviceToHost);
  cudaFree(d_vecY_csr);
  cudaFree(d_vecX_csr);
  cudaFree(d_val);
  cudaFree(d_indices);
  cudaFree(d_ptr);
  cudaFree(startRowPerBlock);
}
