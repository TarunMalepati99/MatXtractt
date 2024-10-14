#include "common.h"
/*
using namespace nvcuda;


// __global__ void tcspmv_kernel_fp16(
//     const half *__restrict__ x_d,
//     half *__restrict__ y_d,
//     const int *__restrict__ chunkPtr,
//     const int *__restrict__ fragPtr,
//     const uint64_t *__restrict__ fragBit,
//     const half *__restrict__ tcVal,
//     const int *__restrict__ sparse_AToX_index,
//     int dRows,
//     int dCols)
// {
//     const int warpsPerBlock = 4;
//     int warpId = threadIdx.x / 32;
//     int laneId = threadIdx.x & 31;

//     int rowChunkIndex = blockIdx.x * warpsPerBlock + warpId;

//     if (rowChunkIndex >= (dRows + fragM - 1) / fragM)
//     {
//         return;
//     }

//     int tcFragStart = chunkPtr[rowChunkIndex];
//     int tcFragEnd = chunkPtr[rowChunkIndex + 1];

//     if (tcFragStart == tcFragEnd)
//     {
//         return;
//     }

//     int rowStart = rowChunkIndex * fragM;

//     int a_row = laneId / fragK; // 0~1（fragM / warpSize = 16 / 32）
//     int a_col = laneId % fragK; // 0~7
//     int a_bitPos = a_row * fragK + a_col;

//     for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
//     {
//         const uint64_t *bitmapArray = &fragBit[tcFragIdx * 2];
//         const half *tcValPtr = &tcVal[fragPtr[tcFragIdx]];
//         const int *sparse_AToX_idx = &sparse_AToX_index[tcFragIdx * fragK];

//         int bitsBefore = 0;
//         int bit = 0;
//         if (a_bitPos < 64)
//         {
//             uint64_t bits = bitmapArray[0];
//             bitsBefore = __popcll(bits & ((1ULL << a_bitPos) - 1ULL));
//             bit = (bits >> a_bitPos) & 1;
//         }
//         else
//         {
//             uint64_t bits = bitmapArray[1];
//             bitsBefore = __popcll(bitmapArray[0]);
//             bitsBefore += __popcll(bits & ((1ULL << (a_bitPos - 64)) - 1ULL));
//             bit = (bits >> (a_bitPos - 64)) & 1;
//         }

//         half a_frag = __float2half(0.0f);
//         if (bit)
//         {
//             a_frag = tcValPtr[bitsBefore];
//         }

//         int x_idx = sparse_AToX_idx[a_col];
//         half b_frag = __ldg(&x_d[x_idx]);

//         // 定义累加器
//         float c_frag = 0.0f;

//         // 执行mma同步指令
//         asm volatile(
//             "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0}, {%1}, {%2}, {%0};\n\t"
//             : "+f"(c_frag)
//             : "h"(a_frag), "h"(b_frag));

//         if (a_col == 0)
//         {
//             int y_idx = rowStart + a_row;
//             if (y_idx < dRows)
//             {
//                 atomicAdd(&y_d[y_idx], __float2half(c_frag));
//             }
//         }
//     }
// }


__global__ void tcspmv_kernel_fp16(
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
    const int fragM = 16;
    const int fragK = 8;
    const int warpsPerBlock = 4;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x & 31;

    int rowChunkIndex = blockIdx.x * warpsPerBlock + warpId;

    if (rowChunkIndex >= (dRows + fragM - 1) / fragM)
    {
        return;
    }

    int tcFragStart = chunkPtr[rowChunkIndex];
    int tcFragEnd = chunkPtr[rowChunkIndex + 1];

    if (tcFragStart == tcFragEnd)
    {
        return;
    }

    int rowStart = rowChunkIndex * fragM;

    // 计算 groupID 和 threadID_in_group
    int groupID = laneId >> 2;           // laneId / 4
    int threadID_in_group = laneId & 3;  // laneId % 4

    for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
    {
        const uint64_t *bitmapArray = &fragBit[tcFragIdx * 2];
        const half *tcValPtr = &tcVal[fragPtr[tcFragIdx]];
        const int *sparse_AToX_idx = &sparse_AToX_index[tcFragIdx * fragK];

        // 初始化累加器片段
        float c_frag[8] = {0.0f};

        // 加载 A 矩阵片段
        unsigned short a_frag_bits[4];
        for (int i = 0; i < 4; ++i)
        {
            int row = groupID + ((i >= 2) ? 8 : 0);
            int col = threadID_in_group * 2 + (i & 1);
            int a_bitPos = row * fragK + col;
            int bitsBefore = 0;
            int bit = 0;
            if (a_bitPos < 64)
            {
                uint64_t bits = bitmapArray[0];
                bitsBefore = __popcll(bits & ((1ULL << a_bitPos) - 1ULL));
                bit = (bits >> a_bitPos) & 1;
            }
            else
            {
                uint64_t bits = bitmapArray[1];
                bitsBefore = __popcll(bitmapArray[0]);
                bitsBefore += __popcll(bits & ((1ULL << (a_bitPos - 64)) - 1ULL));
                bit = (bits >> (a_bitPos - 64)) & 1;
            }
            __half a_value;
            if (bit)
            {
                a_value = tcValPtr[bitsBefore];
            }
            else
            {
                a_value = __float2half(0.0f);
            }
            // 将 __half 转换为 unsigned short
            a_frag_bits[i] = __half_as_ushort(a_value);
        }

        // 加载 B 矩阵片段
        unsigned short b_frag_bits[2];
        for (int i = 0; i < 2; ++i)
        {
            int b_row = threadID_in_group * 2 + i;
            int x_idx = sparse_AToX_idx[b_row];
            __half b_value = __ldg(&x_d[x_idx]);
            // 将 __half 转换为 unsigned short
            b_frag_bits[i] = __half_as_ushort(b_value);
        }

        // 执行 MMA 操作
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n\t"
            "{%0,%1,%2,%3,%4,%5,%6,%7}, \n\t"
            "{%8,%9,%10,%11}, \n\t"
            "{%12,%13}, \n\t"
            "{%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
            : "+f"(c_frag[0]), "+f"(c_frag[1]), "+f"(c_frag[2]), "+f"(c_frag[3]),
              "+f"(c_frag[4]), "+f"(c_frag[5]), "+f"(c_frag[6]), "+f"(c_frag[7])
            : "h"(a_frag_bits[0]), "h"(a_frag_bits[1]), "h"(a_frag_bits[2]), "h"(a_frag_bits[3]),
              "h"(b_frag_bits[0]), "h"(b_frag_bits[1]));

        // 将结果累加到 y_d
        for (int i = 0; i < 8; ++i)
        {
            int row = groupID + ((i >= 4) ? 8 : 0);
            int y_idx = rowStart + row;
            if (y_idx < dRows)
            {
                atomicAdd(&y_d[y_idx], __float2half(c_frag[i]));
            }
        }
    }
}
*/


// __global__ void tcspmv_kernel_fp16(
//     const half *__restrict__ x_d,
//     half *__restrict__ y_d,
//     const int *__restrict__ chunkPtr,
//     const int *__restrict__ fragPtr,
//     const uint64_t *__restrict__ fragBit,
//     const half *__restrict__ tcVal,
//     const int *__restrict__ sparse_AToX_index,
//     int dRows,
//     int dCols)
// {
//     const int warpsPerBlock = 4;
//     int warpId = threadIdx.x / 32; // Warp ID within the block
//     int laneId = threadIdx.x & 31; // Lane ID within the warp

//     int rowChunkIndex = blockIdx.x * warpsPerBlock + warpId;
//     int tcFragStart = chunkPtr[rowChunkIndex];
//     int tcFragEnd = chunkPtr[rowChunkIndex + 1];

//     if (tcFragStart == tcFragEnd)
//     {
//         return;
//     }

//     int rowStart = rowChunkIndex * fragM;
//     int groupID = laneId >> 2;            // laneId / 4
//     int threadID_in_group = laneId & 0x3; // laneId % 4

//     for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
//     {
//         const uint64_t *bitmapArray = &fragBit[tcFragIdx * 4]; 
//         const half *tcValPtr = &tcVal[fragPtr[tcFragIdx]];
//         const int *x_indices = &sparse_AToX_index[tcFragIdx * fragK];

//         __half2 c_frag[2];
//         c_frag[0] = __float2half2_rn(0.0f);
//         c_frag[1] = __float2half2_rn(0.0f);

//         __half2 a_frag[4]; // Use __half2 for Tensor Core compatibility
//         for (int i = 0; i < 4; ++i)
//         {
//             int row, col;
//             if (i < 2)
//             {
//                 row = groupID;
//                 col = (threadID_in_group * 2) + (i & 0x1);
//             }
//             else
//             {
//                 row = groupID + 8;
//                 col = (threadID_in_group * 2) + (i & 0x1) + 8;
//             }

//             int bitPos = row * fragK + col;
//             int bitArrayIndex = bitPos / 64;
//             int bitOffset = bitPos % 64;

//             uint64_t bitmapWord = bitmapArray[bitArrayIndex];
//             int bit = (bitmapWord >> bitOffset) & 1;

//             int a_valIdx = 0;
//             for (int j = 0; j < bitArrayIndex; ++j)
//             {
//                 a_valIdx += __popcll(bitmapArray[j]);
//             }
//             uint64_t mask = ((uint64_t)1 << bitOffset) - 1;
//             a_valIdx += __popcll(bitmapWord & mask);

//             a_frag[i] = bit ? __halves2half2(tcValPtr[a_valIdx], tcValPtr[a_valIdx]) : __float2half2_rn(0.0f);
//         }

//         __half2 b_frag[2];
//         for (int i = 0; i < 2; ++i)
//         {
//             int row_b = (i < 1) ? (threadID_in_group * 2) + (i & 0x1) : (threadID_in_group * 2) + (i & 0x1) + 8;
//             int x_idx = x_indices[row_b];
//             b_frag[i] = __halves2half2(__ldg(&x_d[x_idx]), __ldg(&x_d[x_idx]));
//         }

//         asm volatile(
//             "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \n\t"
//             "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n\t"
//             : "+r"(c_frag[0]), "+r"(c_frag[1])
//             : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
//               "r"(b_frag[0]), "r"(b_frag[1]));

//         if (threadID_in_group == 0)
//         {
//             int y_idx = rowStart + groupID;
//             if (y_idx < dRows)
//             {
//                 atomicAdd(&y_d[y_idx], __low2half(c_frag[0]));
//             }
//         }
//     }
// }

__global__ void tcspmv_kernel_fp16_(
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
    const int warpsPerBlock = 4;
    int warpId = threadIdx.x / 32; // Warp ID within the block
    int laneId = threadIdx.x & 31; // Lane ID within the warp

    int rowChunkIndex = blockIdx.x * warpsPerBlock + warpId;
    int tcFragStart = chunkPtr[rowChunkIndex];
    int tcFragEnd = chunkPtr[rowChunkIndex + 1];

    if (tcFragStart == tcFragEnd)
    {
        return;
    }

    int rowStart = rowChunkIndex * fragM;
    int groupID = laneId >> 2;            // laneId / 4  四个线程为一组
    int threadID_in_group = laneId & 0x3; // laneId % 4

    for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)//遍历当前行块的所有稀疏矩阵片段（tcFrag）
    {
        const uint64_t *bitmapArray = &fragBit[tcFragIdx * 4]; //// 每个 tcFrag 有 4 个 uint64_t
        const half *tcValPtr = &tcVal[fragPtr[tcFragIdx]];
        const int *x_indices = &sparse_AToX_index[tcFragIdx * fragK];

        __half2 c_frag[2];
        c_frag[0] = __float2half2_rn(0.0f);
        c_frag[1] = __float2half2_rn(0.0f);

        __half2 a_frag[4]; // Use __half2 for Tensor Core compatibility
        for (int i = 0; i < 4; ++i)
        {
            int row, col;
            if (i < 2)
            {
                row = groupID;
                col = (threadID_in_group * 2) + (i & 0x1);
            }
            else
            {
                row = groupID + 8;
                col = (threadID_in_group * 2) + (i & 0x1) + 8;
            }

            int bitPos = row * fragK + col;
            int bitArrayIndex = bitPos / 64;
            int bitOffset = bitPos % 64;

            uint64_t bitmapWord = bitmapArray[bitArrayIndex];
            int bit = (bitmapWord >> bitOffset) & 1;

            int a_valIdx = 0;
            for (int j = 0; j < bitArrayIndex; ++j)
            {
                a_valIdx += __popcll(bitmapArray[j]);
            }
            uint64_t mask = ((uint64_t)1 << bitOffset) - 1;
            a_valIdx += __popcll(bitmapWord & mask);

            a_frag[i] = bit ? __halves2half2(tcValPtr[a_valIdx], tcValPtr[a_valIdx]) : __float2half2_rn(0.0f);
        }

        __half2 b_frag[2];
        for (int i = 0; i < 2; ++i)
        {
            int row_b = (i < 1) ? (threadID_in_group * 2) + (i & 0x1) : (threadID_in_group * 2) + (i & 0x1) + 8;
            int x_idx = x_indices[row_b];
            b_frag[i] = __halves2half2(__ldg(&x_d[x_idx]), __ldg(&x_d[x_idx]));
        }

        // Convert __half2 to unsigned int for assembly compatibility
        unsigned int *a_frag_int = reinterpret_cast<unsigned int*>(&a_frag[0]);
        unsigned int *b_frag_int = reinterpret_cast<unsigned int*>(&b_frag[0]);
        unsigned int *c_frag_int = reinterpret_cast<unsigned int*>(&c_frag[0]);

        // Assembly code using scalar types (unsigned int)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \n\t"
            "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n\t"
            : "+r"(c_frag_int[0]), "+r"(c_frag_int[1])
            : "r"(a_frag_int[0]), "r"(a_frag_int[1]), "r"(a_frag_int[2]), "r"(a_frag_int[3]),
              "r"(b_frag_int[0]), "r"(b_frag_int[1]));

        if (threadID_in_group == 0)
        {
            int y_idx = rowStart + groupID;
            if (y_idx < dRows)
            {
                atomicAdd(&y_d[y_idx], __low2half(c_frag[0]));
            }
        }
    }
}

__global__ void tcspmv_kernel_fp16(
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
    const int warpsPerBlock = 4;
    int warpId = threadIdx.x / 32; // Warp ID within the block
    int laneId = threadIdx.x & 31; // Lane ID within the warp

    int rowChunkIndex = blockIdx.x * warpsPerBlock + warpId;
    int tcFragStart = chunkPtr[rowChunkIndex];
    int tcFragEnd = chunkPtr[rowChunkIndex + 1];

    if (tcFragStart == tcFragEnd)
    {
        return;
    }

    int rowStart = rowChunkIndex * fragM;
    int groupID = laneId >> 2;            // laneId / 4  四个线程为一组
    int threadID_in_group = laneId & 0x3; // laneId % 4

    for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)//遍历当前行块的所有稀疏矩阵片段（tcFrag）
    {
        const uint64_t *bitmapArray = &fragBit[tcFragIdx * 4]; // 每个 tcFrag 有 4 个 uint64_t
        const half *tcValPtr = &tcVal[fragPtr[tcFragIdx]];
        const int *x_indices = &sparse_AToX_index[tcFragIdx * fragK];

        // 每个线程负责8个寄存器的a_frag
        half a_frag[8]; 

        for (int i = 0; i < 8; ++i)
        {
            int row, col;
            if (i < 2 || (i >= 4 && i < 6))
            {
                row = groupID;
            }
            else
            {
                row = groupID + 8;
            }

            if (i < 4)
            {
                col = (threadID_in_group * 2) + (i & 0x1);
            }
            else
            {
                col = (threadID_in_group * 2) + (i & 0x1) + 8;
            }

            int bitPos = row * fragK + col;
            int bitArrayIndex = bitPos / 64;
            int bitOffset = bitPos % 64;

            uint64_t bitmapWord = bitmapArray[bitArrayIndex];
            int bit = (bitmapWord >> bitOffset) & 1;

            int a_valIdx = 0;
            for (int j = 0; j < bitArrayIndex; ++j)
            {
                a_valIdx += __popcll(bitmapArray[j]);
            }
            uint64_t mask = ((uint64_t)1 << bitOffset) - 1;
            a_valIdx += __popcll(bitmapWord & mask);

            a_frag[i] = bit ? tcValPtr[a_valIdx] : __float2half(0.0f);
        }

        // 每个线程负责4个寄存器的b_frag, 只算第一列, 前四个线程即可
        half b_frag[4];
        for (int i = 0; i < 4; ++i)
        {
            int row_b;
            if (i < 2)
            {
                row_b = (threadID_in_group * 2) + (i & 0x1);
            }
            else
            {
                row_b = (threadID_in_group * 2) + (i & 0x1) + 8;
            }
            int x_idx = x_indices[row_b];
            b_frag[i] = __ldg(&x_d[x_idx]);
        }

        // 将单独的half组合成__half2
        __half2 a_frag2[4];
        for (int i = 0; i < 4; ++i)
        {
            a_frag2[i] = __halves2half2(a_frag[2 * i], a_frag[2 * i + 1]);
        }

        __half2 b_frag2[2];
        for (int i = 0; i < 2; ++i)
        {
            b_frag2[i] = __halves2half2(b_frag[2 * i], b_frag[2 * i + 1]);
        }

        // Initialize c_frag
        __half2 c_frag[2];
        c_frag[0] = __float2half2_rn(0.0f);
        c_frag[1] = __float2half2_rn(0.0f);

        // 将__half2 转换为无符号整数类型，方便在汇编中操作
        unsigned int *a_frag_int = reinterpret_cast<unsigned int *>(&a_frag2[0]);
        unsigned int *b_frag_int = reinterpret_cast<unsigned int *>(&b_frag2[0]);
        unsigned int *c_frag_int = reinterpret_cast<unsigned int *>(&c_frag[0]);

        // Assembly code using scalar types (unsigned int)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \n\t"
            "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n\t"
            : "+r"(c_frag_int[0]), "+r"(c_frag_int[1])
            : "r"(a_frag_int[0]), "r"(a_frag_int[1]), "r"(a_frag_int[2]), "r"(a_frag_int[3]),
              "r"(b_frag_int[0]), "r"(b_frag_int[1]));

        if (threadID_in_group == 0)
        {
            int y_idx_0 = rowStart + groupID;
            int y_idx_1 = y_idx_0 + 8;
            if (y_idx_0 < dRows)
            {
                atomicAdd(&y_d[y_idx_0], __low2half(c_frag[0]));
            }
            if (y_idx_1 < dRows)
            {
                atomicAdd(&y_d[y_idx_1], __low2half(c_frag[1]));
            }
        }
    }
}






void tcspmv_fp16(int *chunkPtr, std::vector<int> fragPtr, std::vector<std::array<uint64_t, 4>> fragBit,
                 std::vector<half> tcVal, int *sparse_AToX_index, half *X_val,
                 half *Y_val, int rowA, int colA, int *row_order, double *necTime, double *necPre)
{
    int chunkNum = (rowA + fragM - 1) / fragM;
    int totalTcFrags = chunkPtr[chunkNum];
    half *d_tcVal, *d_X_val, *d_Y_val;
    int *d_sparse_AToX_index, *d_chunkPtr, *d_fragPtr;
    uint64_t *d_fragBit;

    // Allocate and copy tcVal to device
    CUDA_CHECK_ERROR(cudaMalloc(&d_tcVal, tcVal.size() * sizeof(half)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_tcVal, tcVal.data(), tcVal.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Allocate and copy fragPtr to device
    CUDA_CHECK_ERROR(cudaMalloc(&d_fragPtr, fragPtr.size() * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_fragPtr, fragPtr.data(), fragPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // Allocate and copy fragBit to device
    size_t fragBitSize = fragBit.size() * sizeof(std::array<uint64_t, 4>);
    CUDA_CHECK_ERROR(cudaMalloc(&d_fragBit, fragBitSize));
    CUDA_CHECK_ERROR(cudaMemcpy(d_fragBit, fragBit.data(), fragBitSize, cudaMemcpyHostToDevice));

    // Allocate and copy chunkPtr to device
    CUDA_CHECK_ERROR(cudaMalloc(&d_chunkPtr, sizeof(int) * (chunkNum + 1)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_chunkPtr, chunkPtr, sizeof(int) * (chunkNum + 1), cudaMemcpyHostToDevice));

    // Allocate and copy sparse_AToX_index to device
    CUDA_CHECK_ERROR(cudaMalloc(&d_sparse_AToX_index, sizeof(int) * (totalTcFrags * fragK)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_sparse_AToX_index, sparse_AToX_index, sizeof(int) * (totalTcFrags * fragK), cudaMemcpyHostToDevice));

    // Allocate and copy X_val to device
    CUDA_CHECK_ERROR(cudaMalloc(&d_X_val, sizeof(half) * colA));
    CUDA_CHECK_ERROR(cudaMemcpy(d_X_val, X_val, sizeof(half) * colA, cudaMemcpyHostToDevice));

    // Allocate and initialize Y_val to 0 on the device
    CUDA_CHECK_ERROR(cudaMalloc(&d_Y_val, sizeof(half) * rowA));
    CUDA_CHECK_ERROR(cudaMemset(d_Y_val, 0, sizeof(half) * rowA));


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
        tcspmv_kernel_fp16<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    int test_iter = 1000;

    cuda_time_test_start();
    for (int i = 0; i < test_iter; ++i)
    {
        tcspmv_kernel_fp16<<<blocksPerGrid, threadsPerBlock>>>(
            d_X_val, d_Y_val, d_chunkPtr, d_fragPtr, d_fragBit, d_tcVal,
            d_sparse_AToX_index, rowA, colA);
    }
    cuda_time_test_end();

    double runtime = (elapsedTime) / test_iter;
    printf("\n SpMV CUDA kernel runtime = %g ms\n", runtime);

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaMemcpy(Y_val, d_Y_val, sizeof(half) * rowA, cudaMemcpyDeviceToHost));

    // 释放设备内存
    CUDA_CHECK_ERROR(cudaFree(d_tcVal));
    CUDA_CHECK_ERROR(cudaFree(d_fragPtr));
    CUDA_CHECK_ERROR(cudaFree(d_fragBit));
    CUDA_CHECK_ERROR(cudaFree(d_chunkPtr));
    CUDA_CHECK_ERROR(cudaFree(d_sparse_AToX_index));
    CUDA_CHECK_ERROR(cudaFree(d_X_val));
    CUDA_CHECK_ERROR(cudaFree(d_Y_val));
}


