#include "common.h"


__global__ void pre_startRowPerBlock(const int *__restrict__ row_ptr,
                                     const int m,
                                     int *__restrict__ startRowPerBlock, 
                                     int BREAK_STRIDE)
{
  const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (global_thread_id >= m)
    return;

  if (global_thread_id == 0)
  {
    startRowPerBlock[0] = 0;
  }
  int a = row_ptr[global_thread_id];
  int b = row_ptr[min(global_thread_id + 1, (int)m)];

  int blocka = divup<int>(a, BREAK_STRIDE);
  int blockb = (b - 1) / static_cast<int>(BREAK_STRIDE);

  if (a != b)
    for (; blocka <= blockb; ++blocka)
      startRowPerBlock[blocka] = global_thread_id;
}


template <int THREADS_PER_BLOCK>
__device__ __forceinline__ void lbNEC_reduce_oneRow_in_thread(int NNZ_PER_BLOCK, const int tid_in_block, const int block_id,
                                                              const int reduceStartRowId, const int reduceEndRowId,
                                                              const int *__restrict__ row_ptr,
                                                              const valT *__restrict__ smem, valT *__restrict__ y)
{
  int reduce_row_id = reduceStartRowId + tid_in_block;
  int nnz_id_before = block_id * NNZ_PER_BLOCK;
  for (; reduce_row_id < reduceEndRowId; reduce_row_id += THREADS_PER_BLOCK)
  {
    valT sum = 0;
    // const int reduce_start_idx = max((int)0, row_ptr[reduce_row_id] - nnz_id_before);
    // const int reduce_end_idx = min(NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - nnz_id_before);
    const int reduce_start_idx = (row_ptr[reduce_row_id] - nnz_id_before) < 0 ? 0 : (row_ptr[reduce_row_id] - nnz_id_before);
    const int reduce_end_idx = (row_ptr[reduce_row_id + 1] - nnz_id_before) > NNZ_PER_BLOCK ? NNZ_PER_BLOCK : (row_ptr[reduce_row_id + 1] - nnz_id_before);
    for (int i = reduce_start_idx; i < reduce_end_idx; i++)
    {
      sum += smem[i];
    }
    atomicAdd(y + reduce_row_id, sum);
  }
}

template <int THREADS_PER_BLOCK>
__device__ __forceinline__ void lbNEC_reduce_oneRow_in_block(int NNZ_PER_BLOCK, const int tid_in_block, const int block_id,
                                                             const int reduceStartRowId, const int reduceEndRowId,
                                                             const int *__restrict__ row_ptr,
                                                             const valT *__restrict__ smem, valT *__restrict__ y)
{
  __shared__ valT LDS[THREADS_PER_BLOCK];
  // __shared__ volatile valT LDS[THREADS_PER_BLOCK];

  valT sum = 0;
  const int reduce_start_idx = max((int)0, row_ptr[reduceStartRowId] - block_id * NNZ_PER_BLOCK);
  const int reduce_end_idx = min(NNZ_PER_BLOCK, row_ptr[reduceStartRowId + 1] - block_id * NNZ_PER_BLOCK);

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
__device__ __forceinline__ valT warpReduceSum(valT sum)
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

template <int THREADS_PER_BLOCK, int VECTOR_SIZE>
__device__ __forceinline__ void
lbNEC_reduce_oneRow_in_vector(int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
                              const int reduceStartRowId, const int reduceEndRowId,
                              const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{
  // use `vec_num` vectors, each vector can process reduction of one row by involving `vec_size` threads.
  const int vec_size = VECTOR_SIZE;
  const int vec_num = THREADS_PER_BLOCK / vec_size;
  const int vec_id = tid_in_block / vec_size;
  const int tid_in_vec = tid_in_block & (vec_size - 1);

  int reduce_row_id = reduceStartRowId + vec_id;
  for (; reduce_row_id < reduceEndRowId; reduce_row_id += vec_num)
  {
    const int reduce_start_idx = max((int)0, row_ptr[reduce_row_id] - block_id * NNZ_PER_BLOCK);
    const int reduce_end_idx = min((int)NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - block_id * NNZ_PER_BLOCK);
    // reduce LDS via vectors.
    valT sum = 0;
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

template <int THREADS_PER_BLOCK, int VECTOR_SIZE>
__device__ __forceinline__ void
lbNEC_reduce_oneRow_in_vector_L(int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
                                const int reduceStartRowId, const int reduceEndRowId,
                                const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{
  // use `vec_num` vectors, each vector can process reduction of one row by involving `vec_size` threads.
  const int vec_size = VECTOR_SIZE;
  const int vec_num = THREADS_PER_BLOCK / vec_size;
  const int vec_id = tid_in_block / vec_size;
  const int tid_in_vec = tid_in_block & (vec_size - 1);
  const int warp_lane_id = tid_in_block & 31;

  int reduce_row_id = reduceStartRowId + vec_id;
  for (; reduce_row_id < reduceEndRowId; reduce_row_id += vec_num)
  {
    const int reduce_start_idx = max((int)0, row_ptr[reduce_row_id] - block_id * NNZ_PER_BLOCK);
    const int reduce_end_idx = min((int)NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - block_id * NNZ_PER_BLOCK);
    // reduce LDS via vectors.
    valT sum = 0;
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

template <int THREADS_PER_BLOCK>
__global__ void cdspmv_kernel(valT *__restrict__ d_val,
                              int *__restrict__ d_ptr,
                              int *__restrict__ d_cols,
                              int rowA,
                              valT *__restrict__ d_vector,
                              valT *__restrict__ d_out,
                              int *__restrict__ startRowPerBlock, 
                              int productNnzPerThread,
                              int productNnzPerBlock)
{
  const int tid_in_block = threadIdx.x;
  // int NNZ_PER_BLOCK = THREADS_PER_BLOCK * productNnzPerThread;
  extern __shared__ valT middle_s[];
  // __shared__ valT middle_s[NNZ_PER_BLOCK];
  const int lastElemId = d_ptr[rowA];

  int blockNnzStart = productNnzPerBlock * blockIdx.x;

  // product and stream in Shared Memory
#pragma unroll
  for (int round = 0; round < productNnzPerThread; round++)
  {
    const int sIdx = tid_in_block + round * THREADS_PER_BLOCK;
    const int gIdx = min(blockNnzStart + sIdx, lastElemId - 1);
    middle_s[sIdx] = d_val[gIdx] * d_vector[d_cols[gIdx]];
  }
  __syncthreads();

  const int reduceStartRowId = min(startRowPerBlock[blockIdx.x], rowA);
  int reduceEndRowId = min(startRowPerBlock[blockIdx.x + 1], rowA);
  reduceEndRowId = (reduceEndRowId == 0) ? rowA : reduceEndRowId;
  if (d_ptr[reduceEndRowId] % productNnzPerBlock != 0 || reduceEndRowId == reduceStartRowId)
  {
    reduceEndRowId = min(reduceEndRowId + 1, rowA);
  }
  // online workload balance reduction
  const int n_reduce_rows_num = reduceEndRowId - reduceStartRowId;

  // when threads = 128
  if (n_reduce_rows_num > 64)
  {
    lbNEC_reduce_oneRow_in_thread<THREADS_PER_BLOCK>(productNnzPerBlock, tid_in_block, blockIdx.x,
                                                                          reduceStartRowId, reduceEndRowId,
                                                                          d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num == 1)
  {
    lbNEC_reduce_oneRow_in_block<THREADS_PER_BLOCK>(productNnzPerBlock, tid_in_block, blockIdx.x,
                                                                         reduceStartRowId, reduceEndRowId,
                                                                         d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num == 2)
  {
    lbNEC_reduce_oneRow_in_vector_L<THREADS_PER_BLOCK, 64>(productNnzPerBlock, n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                                reduceStartRowId, reduceEndRowId,
                                                                                d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 4)
  {
    lbNEC_reduce_oneRow_in_vector<THREADS_PER_BLOCK, 32>(productNnzPerBlock, n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                              reduceStartRowId, reduceEndRowId,
                                                                              d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 8)
  {
    lbNEC_reduce_oneRow_in_vector<THREADS_PER_BLOCK, 16>(productNnzPerBlock, n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                              reduceStartRowId, reduceEndRowId,
                                                                              d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 16)
  {
    lbNEC_reduce_oneRow_in_vector<THREADS_PER_BLOCK, 8>(productNnzPerBlock, n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                             reduceStartRowId, reduceEndRowId,
                                                                             d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 32)
  {
    lbNEC_reduce_oneRow_in_vector<THREADS_PER_BLOCK, 4>(productNnzPerBlock, n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                             reduceStartRowId, reduceEndRowId,
                                                                             d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 64)
  {
    lbNEC_reduce_oneRow_in_vector<THREADS_PER_BLOCK, 2>(productNnzPerBlock, n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                             reduceStartRowId, reduceEndRowId,
                                                                             d_ptr, middle_s, d_out);
  }

  /*
  // 线程256
  if (n_reduce_rows_num > 128)
  {
    lbNEC_reduce_oneRow_in_thread<THREADS_PER_BLOCK>(tid_in_block, blockIdx.x,
                                                               reduceStartRowId, reduceEndRowId,
                                                               d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num == 1)
  {

    lbNEC_reduce_oneRow_in_block<THREADS_PER_BLOCK>(tid_in_block, blockIdx.x,
                                                                reduceStartRowId, reduceEndRowId,
                                                                d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num == 2)
  {
    lbNEC_reduce_oneRow_in_vector_L<THREADS_PER_BLOCK, 128>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                         reduceStartRowId, reduceEndRowId,
                                                                         d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 4)
  {
    lbNEC_reduce_oneRow_in_vector_L<THREADS_PER_BLOCK, 64>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                        reduceStartRowId, reduceEndRowId,
                                                                        d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 8)
  {
    lbNEC_reduce_oneRow_in_vector<THREADS_PER_BLOCK, 32>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                      reduceStartRowId, reduceEndRowId,
                                                                      d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 16)
  {
    lbNEC_reduce_oneRow_in_vector<THREADS_PER_BLOCK, 16>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                      reduceStartRowId, reduceEndRowId,
                                                                      d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 32)
  {
    lbNEC_reduce_oneRow_in_vector<THREADS_PER_BLOCK, 8>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                     reduceStartRowId, reduceEndRowId,
                                                                     d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 64)
  {
    lbNEC_reduce_oneRow_in_vector<THREADS_PER_BLOCK, 4>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                     reduceStartRowId, reduceEndRowId,
                                                                     d_ptr, middle_s, d_out);
  }
  else if (n_reduce_rows_num <= 128)
  {
    lbNEC_reduce_oneRow_in_vector<THREADS_PER_BLOCK, 2>(n_reduce_rows_num, tid_in_block, blockIdx.x,
                                                                     reduceStartRowId, reduceEndRowId,
                                                                     d_ptr, middle_s, d_out);
  }
  */
}


void cdspmv(char *filename, valT *csrVal, int *csrRowPtr, int *csrColInd,
            valT *X_val, valT *Y_val, int rowA, int colA, int nnzA,
            double *necTime, double *necPre)
{
  struct timeval t1;
  struct timeval t2;
  struct timeval tpre1;
  struct timeval tpre2;

  valT *d_vecY_csr, *d_vecY_csr_perf, *d_vecX_csr, *d_val;
  int *d_indices, *d_ptr;

  cudaMalloc(&d_vecY_csr, sizeof(valT) * rowA);
  cudaMalloc(&d_vecY_csr_perf, sizeof(valT) * rowA);
  cudaMalloc(&d_vecX_csr, sizeof(valT) * colA);
  cudaMalloc(&d_val, sizeof(valT) * nnzA);
  cudaMalloc(&d_indices, sizeof(int) * nnzA);
  cudaMalloc(&d_ptr, sizeof(int) * (rowA + 2));

  cudaMemcpy(d_val, csrVal, sizeof(valT) * nnzA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, csrColInd, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptr, csrRowPtr, sizeof(int) * (rowA + 2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vecX_csr, X_val, sizeof(valT) * colA, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_vecY_csr, Y_val, sizeof(valT) * rowA, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_vecY_csr_perf, Y_val, sizeof(valT) * rowA, cudaMemcpyHostToDevice);
  cudaMemset(d_vecY_csr, 0.0, sizeof(valT) * rowA);
  cudaMemset(d_vecY_csr_perf, 0.0, sizeof(valT) * rowA);
#ifdef fp64
  int productNnzPerThread = 4;
#else
  int productNnzPerThread = (nnzA > 300000) ? 16 : 4;
#endif
  const int THREADS_PER_BLOCK = 128;

  const int WORK_BLOCKS = nnzA / (productNnzPerThread * THREADS_PER_BLOCK) + ((nnzA % (productNnzPerThread * THREADS_PER_BLOCK) == 0) ? 0 : 1);

  const int startRowPerBlock_len = WORK_BLOCKS + 1;

  int *startRowPerBlock;
  cudaMalloc(&startRowPerBlock, sizeof(int) * startRowPerBlock_len);
  cudaMemset(startRowPerBlock, 0, sizeof(int) * startRowPerBlock_len);
  gettimeofday(&tpre1, NULL);
  pre_startRowPerBlock<<<divup<uint32_t>(rowA + 1, 128), 128>>>(d_ptr, rowA, startRowPerBlock, productNnzPerThread * THREADS_PER_BLOCK);
  CUDA_CHECK_ERROR(cudaGetLastError());
  gettimeofday(&tpre2, NULL);
  cudaDeviceSynchronize(); // 确保 pre_startRowPerBlock 执行完成
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("pre_startRowPerBlock kernel launch failed: %s\n", cudaGetErrorString(err));
  }

  double mean_col_num = (double)(nnzA + rowA) / (double)rowA;
  int productNnzPerBlock = THREADS_PER_BLOCK * productNnzPerThread;

  printf("Launching cdspmv_kernel with %d blocks, %d threads per block\n",
         WORK_BLOCKS, THREADS_PER_BLOCK);

  int warmup_time = 100;
  int execute_time = 3000;

  for (int i = 0; i < warmup_time; ++i)
  {
    cdspmv_kernel<THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), productNnzPerBlock * sizeof(valT)>>>(d_val, d_ptr, d_indices, rowA, d_vecX_csr, d_vecY_csr_perf, startRowPerBlock, productNnzPerThread, productNnzPerBlock);
    
  }
  cudaDeviceSynchronize();
  gettimeofday(&t1, NULL);
  for (int i = 0; i < execute_time; ++i)
  {
    cdspmv_kernel<THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), productNnzPerBlock * sizeof(valT)>>>(d_val, d_ptr, d_indices, rowA, d_vecX_csr, d_vecY_csr_perf, startRowPerBlock, productNnzPerThread, productNnzPerBlock);
    
  }
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);

  cdspmv_kernel<THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), productNnzPerBlock * sizeof(valT)>>>(d_val, d_ptr, d_indices, rowA, d_vecX_csr, d_vecY_csr, startRowPerBlock, productNnzPerThread, productNnzPerBlock);
  
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR(cudaGetLastError());

  double pre_time = ((tpre2.tv_sec - tpre1.tv_sec) * 1000.0 + (tpre2.tv_usec - tpre1.tv_usec) / 1000.0) / 1;
  *necPre = pre_time;
  double nec_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / execute_time;
  *necTime = nec_time;
  double nec_gflops = (double)((long)nnzA * 2) / (nec_time * 1e6);

  CUDA_CHECK_ERROR(cudaMemcpy(Y_val, d_vecY_csr, sizeof(valT) * rowA, cudaMemcpyDeviceToHost));
  cudaFree(d_vecY_csr_perf);
  cudaFree(d_vecY_csr);
  cudaFree(d_vecX_csr);
  cudaFree(d_val);
  cudaFree(d_indices);
  cudaFree(d_ptr);
  cudaFree(startRowPerBlock);
}
