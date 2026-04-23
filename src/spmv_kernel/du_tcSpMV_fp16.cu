#ifdef _WIN32
#include <windows.h>
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
struct timezone { int tz_minuteswest; int tz_dsttime; };
static inline int gettimeofday(struct timeval* tv, struct timezone* tz) {
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    unsigned long long t = ((unsigned long long)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    t -= 116444736000000000ULL;
    tv->tv_sec  = (long)(t / 10000000);
    tv->tv_usec = (long)((t % 10000000) / 10);
    return 0;
}
#else
#include <sys/time.h>
#endif
#include "common.h"
#ifdef _WIN32
#include <windows.h>
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
#endif
#define SHM_SIZE 128 // Shared memory size in halves (8 KB)

__device__ __forceinline__ void mma_m8n8k4_fp16_v2(half *acc, uint32_t *A, half *frag_b)
{
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_b[0]);
    uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6, %7 }, "
        " { %0, %1, %2, %3 };"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]) : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));
}
__device__ __forceinline__ void store_half_to_global(const half *a, half v)
{
    ushort *v_u = reinterpret_cast<ushort *>(&v);
    asm volatile("st.global.cs.u16 [%0], %1;" ::"l"(a), "h"(*v_u));
}

// 1 warp - 1 row chunk
__global__ void tcspmv_kernel_fp16_v1(
    const half *__restrict__ x_d,
    half *__restrict__ y_d,
    const int *__restrict__ chunkPtr,
    const int *__restrict__ fragPtr,
    const uint32_t *__restrict__ fragBit,
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
    int warpId = threadIdx.x / 32; // Warp ID within the block
    int laneId = threadIdx.x & 31; // Lane ID within the warp
    int mmaIndex = (laneId < 16) ? laneId / 4 : (laneId - 16) / 4;

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
    half sum = __half(0);
    int a_row = (laneId & 3) + ((laneId < 16) ? 0 : 4);

    for (int tcFragIdx_warp_start = tcFragStart; tcFragIdx_warp_start < tcFragEnd; tcFragIdx_warp_start += 4)
    {
        half acc[8] = {__half(0)};

        int dangfragIdx = tcFragIdx_warp_start + mmaIndex;
        int fragIdx = dangfragIdx >= tcFragEnd ? (tcFragEnd - 1) : dangfragIdx;//Ensure legal memory access

        uint32_t bitmap = fragBit[fragIdx];
        const half *tcValPtr = &tcVal[fragPtr[fragIdx]];
        // load A
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int a_col = i; // Since i ranges from 0 to 3
            int a_bitPos = a_row * fragK + a_col;
            // int a_valIdx = __popc(bitmap & ((1U << a_bitPos) - 1));
            int bit = (bitmap >> a_bitPos) & 1;
            frag_a[i] = bit ? tcValPtr[__popc(bitmap & ((1U << a_bitPos) - 1))] : __float2half(0.0f);
        }
        // load B
        const int *sparse_AToX_idx = &sparse_AToX_index[fragIdx * fragK];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int b_row = i; // Since i ranges from 0 to 3
            int x_idx = sparse_AToX_idx[b_row];
            frag_b[i] = __ldg(&x_d[x_idx]);
            // frag_b[i] = (x_idx < SHM_SIZE) ? x_shm[x_idx] : __ldg(&x_d[x_idx]);
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
    // 0 4 8 12 threads
    // 1 5 9 13 threads
    // 2 6 10 14 threads
    // 3 7 11 15 threads
    // 16 20 24 28 threads
    // 17 21 25 29 threads
    // 18 22 26 30 threads
    // 19 23 27 31 threads

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

__global__ void tcspmv_kernel_fp16_v0(
    const half *__restrict__ x_d,
    half *__restrict__ y_d,
    const int *__restrict__ chunkPtr,
    const int *__restrict__ fragPtr,
    const uint32_t *__restrict__ fragBit,
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

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x & 31;
    int mmaIndex = (laneId < 16) ? laneId / 4 : (laneId - 16) / 4;

    // int rowChunkIndex = blockIdx.x * 8 + warpId * 4 + mmaIndex;
    int rowChunkIndex = blockIdx.x * 16 + warpId * 4 + mmaIndex;

    int tcFragStart = chunkPtr[rowChunkIndex];
    int tcFragEnd = chunkPtr[rowChunkIndex + 1];
    int numTcFragsInChunk = tcFragEnd - tcFragStart;
    if (numTcFragsInChunk == 0)
    {
        return;
    }
    int mmaRowStart = rowChunkIndex * fragM;

    half frag_a[4];
    half frag_b[4];
    half sum = __half(0);
    int a_row = (laneId & 3) + ((laneId < 16) ? 0 : 4);

    for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
    {
        half acc[8] = {__half(0), __half(0), __half(0), __half(0),
                       __half(0), __half(0), __half(0), __half(0)};

        uint32_t bitmap = fragBit[tcFragIdx];
        const half *tcValPtr = &tcVal[fragPtr[tcFragIdx]];
// load A
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int a_col = i; // Since i ranges from 0 to 3
            int a_bitPos = a_row * fragK + a_col;
            int a_valIdx = __popc(bitmap & ((1U << a_bitPos) - 1));
            int bit = (bitmap >> a_bitPos) & 1;
            frag_a[i] = bit ? tcValPtr[a_valIdx] : __half(0);
            // frag_a[i] = bit ? load_half_from_global(tcValPtr + a_valIdx) : __half(0);
        }
        // load B
        const int *sparse_AToX_idx = &sparse_AToX_index[tcFragIdx * fragK];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int b_row = i; // Since i ranges from 0 to 3
            int x_idx = sparse_AToX_idx[b_row];
            // int x_idx = load_int_from_global(sparse_AToX_idx + b_row);
            // frag_b[i] = (x_idx < SHM_SIZE) ? x_shm[x_idx] : __ldg(&x_d[x_idx]);
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

        sum += acc[0];
    }
    // Write the result to y_d
    int y_idx = mmaRowStart + a_row;
    if (y_idx < dRows)
    {
        ushort *sum_u = reinterpret_cast<ushort *>(&sum);
        asm volatile("st.global.cs.u16 [%0], %1;" ::"l"(y_d + y_idx), "h"(*sum_u));
    }
}

void du_tcspmv_fp16_v0(indT *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
                 std::vector<half> tcVal, indT *sparse_AToX_index, half *X_val,
                 half *Y_val, int rowA, int colA, int *row_order, double *tcTime)
{
    int chunkNum = ceil((double)rowA / (double)fragM);
    int totalTcFrags = chunkPtr[chunkNum];
    half *d_tcVal, *d_X_val, *d_Y_val, *d_Y_val_perf;
    indT *d_sparse_AToX_index, *d_chunkPtr, *d_fragPtr;
    uint32_t *d_fragBit;

    //  tcVal
    CUDA_CHECK_ERROR(cudaMalloc(&d_tcVal, tcVal.size() * sizeof(half)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_tcVal, tcVal.data(), tcVal.size() * sizeof(half), cudaMemcpyHostToDevice));

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
    CUDA_CHECK_ERROR(cudaMalloc(&d_X_val, sizeof(half) * colA));
    CUDA_CHECK_ERROR(cudaMemcpy(d_X_val, X_val, sizeof(half) * colA, cudaMemcpyHostToDevice));

    //  Y_val
    CUDA_CHECK_ERROR(cudaMalloc(&d_Y_val, sizeof(half) * rowA));
    CUDA_CHECK_ERROR(cudaMemset(d_Y_val, 0, sizeof(half) * rowA));

    CUDA_CHECK_ERROR(cudaMalloc(&d_Y_val_perf, sizeof(half) * rowA));
    CUDA_CHECK_ERROR(cudaMemset(d_Y_val_perf, 0, sizeof(half) * rowA));

    int warpsPerBlock = 4;
    int chunksPerBlock = warpsPerBlock * 4;
    int warpSize = 32;
    int threadsPerBlock = warpsPerBlock * warpSize;
    int numRowChunks = (rowA + fragM - 1) / fragM;
    int blocksPerGrid = (numRowChunks + chunksPerBlock - 1) / chunksPerBlock;

    // printf("Launching kernel with %d blocks, %d threads per block\n",
    //        blocksPerGrid, threadsPerBlock);

    int warm_iter = 200;
    for (int i = 0; i < warm_iter; ++i)
    {
        tcspmv_kernel_fp16_v0<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val_perf, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    int test_iter = 4000;
    cuda_time_test_start();
    for (int i = 0; i < test_iter; ++i)
    {
        tcspmv_kernel_fp16_v0<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val_perf, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    cuda_time_test_end();

    double runtime = (elapsedTime) / test_iter;
    printf("\n tcspmv_kernel_fp16_v0 = %g ms\n", runtime);
    *tcTime = runtime;

    tcspmv_kernel_fp16_v0<<<blocksPerGrid, threadsPerBlock>>>(
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

void du_tcspmv_fp16_v1(indT *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
                  std::vector<half> tcVal, indT *sparse_AToX_index, half *X_val,
                  half *Y_val, int rowA, int colA, int *row_order, double *tcTime)
{
    int chunkNum = ceil((double)rowA / (double)fragM);
    int totalTcFrags = chunkPtr[chunkNum];
    half *d_tcVal, *d_X_val, *d_Y_val, *d_Y_val_perf;
    indT *d_sparse_AToX_index, *d_chunkPtr, *d_fragPtr;
    uint32_t *d_fragBit;

    // cudaMemcpyToSymbol(x_const, X_val, CONST_SIZE * sizeof(half));

    //  tcVal
    CUDA_CHECK_ERROR(cudaMalloc(&d_tcVal, tcVal.size() * sizeof(half)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_tcVal, tcVal.data(), tcVal.size() * sizeof(half), cudaMemcpyHostToDevice));

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
        tcspmv_kernel_fp16_v1<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val_perf, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    int test_iter = 4000;
    cuda_time_test_start();
    for (int i = 0; i < test_iter; ++i)
    {
        tcspmv_kernel_fp16_v1<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val_perf, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    cuda_time_test_end();

    double runtime = (elapsedTime) / test_iter;
    printf("tcspmv_kernel_fp16: %g ms\n", runtime);
    *tcTime = runtime;

    tcspmv_kernel_fp16_v1<<<blocksPerGrid, threadsPerBlock>>>(
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

