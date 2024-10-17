/*
 * Dual-Phase Compression
 * Author: Luhan Wang
 * Date: 2024.9.24
 */
#include "mmio.h"
#include "csr2csc.h"
#include "TCSpMV.h"

// const float colProp = 0.82;
const float rowProp = 0.62;
const float colProp = 0.8;

bool isDenseTC = 0;
typedef struct
{
    int count;
    int index;
} CountWithIndex;

typedef struct
{
    int index;
    bool isIn;
} HashTable;

int compare_desc_structure(const void *a, const void *b)
{
    return ((CountWithIndex *)b)->count - ((CountWithIndex *)a)->count;
}

// int eQcheck(valT *tmp1, valT *tmp2, int length)
// {
//     for (int i = 0; i < length; i++)
//     {
//         if (fabs(tmp1[i] - tmp2[i]) > 1e-5)
//         {
//             printf("error in (%d), cpu(%4.2f), our(%4.2f),please check your code!\n", i, tmp1[i], tmp2[i]);
//             return -1;
//         }
//     }
//     printf("Y(%d), compute succeed!\n", length);
//     return 0;
// }

// int eQcheck(valT *tmp1, valT *tmp2, int length)
// {
// #ifdef fp64
//     // When fp64 is defined, use double precision comparison
//     for (int i = 0; i < length; i++)
//     {
//         if (fabs(tmp1[i] - tmp2[i]) > 1e-8) // Set tolerance for double precision
//         {
//             printf("Error at index (%d), res(%4.6f), our(%4.6f), please check your code!\n", i, tmp1[i], tmp2[i]);
//             return -1;
//         }
//     }
// #else
//     // When fp64 is not defined, use half precision comparison
//     for (int i = 0; i < length; i++)
//     {
//         // Convert __half to float for computation
//         float val1 = static_cast<float>(tmp1[i]);
//         float val2 = static_cast<float>(tmp2[i]);
//         if (fabs(val1 - val2) > (1e-2)) // Set tolerance for half precision
//         {
//             printf("Error at index (%d), res(%4.2f), our(%4.2f), please check your code!\n", i, val1, val2);
//             return -1;
//         }
//     }
// #endif
//     printf("Success! All values match within the tolerance for %d elements.\n", length);
//     return 0;
// }

int eQcheck(valT *tmp1, valT *tmp2, int length)
{
#ifdef fp64
    // Use double precision (fp64), check for 15 significant digits
    const double tolerance = 1e-8;  // 15 significant digits for double precision
    for (int i = 0; i < length; i++)
    {
        double val1 = tmp1[i];
        double val2 = tmp2[i];
        if (fabs(val1 - val2) / fmax(fabs(val1), fabs(val2)) > tolerance)
        {
            printf("Error at index (%d), res(%4.15f), our(%4.15f), please check your code!\n", i, val1, val2);
            return -1;
        }
    }
#else
    // Use half precision (fp16), check for 3-4 significant digits
    const float tolerance = 1e-2;  // 3 significant digits for half precision
    for (int i = 0; i < length; i++)
    {
        // Convert __half to float for computation
        float val1 = static_cast<float>(tmp1[i]);
        float val2 = static_cast<float>(tmp2[i]);
        if (fabs(val1 - val2) / fmax(fabs(val1), fabs(val2)) > tolerance)
        {
            printf("Error at index (%d), res(%4.3f), our(%4.3f), please check your code!\n", i, val1, val2);
            return -1;
        }
    }
#endif
    printf("Success! All values match within the tolerance for %d elements.\n", length);
    return 0;
}


int compare_desc(const void *a, const void *b)
{
    return (*(int *)b - *(int *)a);
}

void spmv_serial(valT *csrVal, indT *csrRowPtr, indT *csrColInd,
                 valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA)
{
    valT t;
    for (indT i = 0; i < rowA; i++)
    {
        t = static_cast<valT>(0.0);
        indT ptr_start = csrRowPtr[i];
        indT n_one_line = csrRowPtr[i + 1] - ptr_start;
        // printf("%d\n",i);
        for (indT j = 0; j < n_one_line; j++)
        {
            indT v_idx = csrColInd[j + ptr_start];
            t += csrVal[j + ptr_start] * X_val[v_idx];
        }
        Y_val[i] += t;
    }
}

void spmv_serial_csc(valT *cscVal, indT *cscColPtr, indT *cscRowInd,
                     valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA)
{
    for (indT i = 0; i < colA; i++)
    {
        // printf(" i = %d \n", i);
        indT col_start = cscColPtr[i];
        indT col_end = cscColPtr[i + 1];

        // printf("%d\n",i);
        for (indT j = col_start; j < col_end; j++)
        {
            indT v_idx = cscRowInd[j];
            Y_val[v_idx] += cscVal[j] * X_val[i];
        }
        // printf(" i = %d ------ \n", i);
    }
}

void spmv_serial_(valT *csrVal, indT *csrRowPtr, indT *csrColInd,
                  valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA, indT *row_order)
{
    valT t;
    for (indT i = 0; i < rowA; i++)
    {
        t = static_cast<valT>(0.0);
        indT ptr_start = csrRowPtr[i];
        indT n_one_line = csrRowPtr[i + 1] - ptr_start;
        for (indT j = 0; j < n_one_line; j++)
        {
            indT v_idx = csrColInd[j + ptr_start];
            t += csrVal[j + ptr_start] * X_val[v_idx];
        }
        Y_val[row_order[i]] += t;
        // if(i == 0) printf("!!!!!!!  %d  %f\n", row_order[i], Y_val[row_order[i]]);
    }
}

void spmv_serial_ecr(valT *csrVal, indT *csrRowPtr, indT *csrColInd,
                     valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA, indT *row_order, int *ecrId, int **use_x_id)
{
    valT t;
    for (int i = 0; i < rowA; i++)
    {
        int windowId = i / fragM;
        t = static_cast<valT>(0.0);
        indT ptr_start = csrRowPtr[i];
        indT n_one_line = csrRowPtr[i + 1] - ptr_start;
        for (indT j = 0; j < n_one_line; j++)
        {
            indT v_idx = use_x_id[windowId][ecrId[j + ptr_start]];
            t += csrVal[j + ptr_start] * X_val[v_idx];
        }
        Y_val[row_order[i]] += t;
    }
}

// condense an sorted array with duplication: [1,2,2,3,4,5,5]
// after condense, it becomes: [1,2,3,4,5].
// Also, mapping the origin value to the corresponding new location in the new array.
// 1->[0], 2->[1], 3->[2], 4->[3], 5->[4].
std::map<int, int> inplace_deduplication(int *array, int length)
{
    if (length == 0)
        return {};

    int loc = 0;
    int cur = 1;
    std::map<int, int> nb2col;
    nb2col[array[0]] = 0;

    while (cur < length)
    {
        if (array[cur] != array[cur - 1])
        {
            loc++;
            array[loc] = array[cur];
            nb2col[array[cur]] = loc;
        }
        cur++;
    }
    return nb2col;
}

void ecrPreprocess(
    int *csrColInd,
    int *csrRowPtr,
    int rowA,
    int colA,
    int fragSize_h,
    int fragSize_w,
    int *blockPartition,
    int *ecrId, // output
    int **use_x_id,
    int *nec_num)
{
    int block_counter = 0;
    // #pragma omp parallel for reduction(+ : block_counter)
    for (int iter = 0; iter < rowA; iter += fragSize_h)
    {
        int windowId = iter / fragSize_h;
        int block_start = csrRowPtr[iter];
        int iter_plus = iter + fragSize_h;

        int block_end = csrRowPtr[(iter_plus > rowA) ? rowA : iter_plus];
        int num_window_nnzs = block_end - block_start;

        if (num_window_nnzs <= 0)
        {
            blockPartition[windowId] = 0;
            continue;
        }

        int *neighbor_window = (int *)malloc(num_window_nnzs * sizeof(int));
        if (neighbor_window == nullptr)
        {
            fprintf(stderr, "Memory allocation failed for neighbor_window\n");
            exit(EXIT_FAILURE);
        }
        std::memcpy(neighbor_window, &csrColInd[block_start], num_window_nnzs * sizeof(int));

        std::sort(neighbor_window, neighbor_window + num_window_nnzs);

        std::map<int, int> clean_edges2col = inplace_deduplication(neighbor_window, num_window_nnzs);

        use_x_id[windowId] = (int *)malloc(clean_edges2col.size() * sizeof(int));
        memset(use_x_id[windowId], 0, clean_edges2col.size() * sizeof(int));
        std::map<int, int>::iterator it;
        int index = 0;
        for (it = clean_edges2col.begin(); it != clean_edges2col.end(); ++it)
        {
            use_x_id[windowId][index++] = it->first;
        }
        nec_num[windowId] = clean_edges2col.size();

        blockPartition[windowId] = (clean_edges2col.size() + fragSize_w - 1) / fragSize_w;
        block_counter += blockPartition[windowId];
        // printf("[%d] %d \n",windowId, block_counter);

        for (int e_index = block_start; e_index < block_end; e_index++)
        {
            int eid = csrColInd[e_index];

            auto it = clean_edges2col.find(eid);
            if (it != clean_edges2col.end())
            {
                ecrId[e_index] = it->second;
            }
            else
            {
                ecrId[e_index] = -1;
                fprintf(stderr, "Element ID not found in clean_edges2col map\n");
                exit(EXIT_FAILURE);
            }
        }
        free(neighbor_window);
    }
    // printf("block_counter:%d \n", block_counter);
}

void generateFormat(
    const valT *vals,               // Input: Non-zero values in CSR format
    const int *rowPtr,              // Input: Row pointers in CSR format
    const int *ecrId,               // Input: Adjusted column indices after empty column removal
    int dRows,                      // Input: Number of rows
    int dCols,                      // Input: Number of columns
    int *chunkPtr,                  // Input: Offsets of tcFrags for each rowChunk
    std::vector<int> &fragPtr,      // Output: Fragment pointer array
    std::vector<uint32_t> &fragBit, // Output: fragBit array for each tcFrag
    std::vector<valT> &tcVal        // Output: Non-zero values in tcFrags
)
{
    int numRowChunks = ceil((double)dRows / (double)fragM);
    int totalTcFrags = chunkPtr[numRowChunks];

    // Initialize fragPtr and per-tcFrag non-zero counts
    fragPtr.resize(totalTcFrags + 1, 0);
    std::vector<int> perTcFragNnz(totalTcFrags, 0);

    // Initialize fragBit
    fragBit.resize(totalTcFrags, 0);

    // Temporary data structures for values in tcFrags
    std::vector<std::vector<std::vector<valT>>> valueGrids;    // [tcFrag][row][col]
    std::vector<std::vector<std::vector<bool>>> hasValueGrids; // [tcFrag][row][col]

    for (int rowChunkIndex = 0; rowChunkIndex < numRowChunks; ++rowChunkIndex)
    {
        int rowChunkStart = rowChunkIndex * fragM;
        int rowChunkEnd = std::min(rowChunkStart + fragM, dRows);
        int numTcFragsInChunk = chunkPtr[rowChunkIndex + 1] - chunkPtr[rowChunkIndex];

        if (numTcFragsInChunk == 0)
        {
            continue;
        }
        // Initialize valueGrids and hasValueGrids for the tcFrags in this rowChunk
        valueGrids.assign(numTcFragsInChunk, std::vector<std::vector<valT>>(fragM, std::vector<valT>(fragK, 0.0)));
        hasValueGrids.assign(numTcFragsInChunk, std::vector<std::vector<bool>>(fragM, std::vector<bool>(fragK, false)));

        // Process each row in the rowChunk
        for (int r = rowChunkStart; r < rowChunkEnd; ++r)
        {
            int rowLocalIndex = r - rowChunkStart;

            // Iterate over non-zero elements in the row
            for (int idx = rowPtr[r]; idx < rowPtr[r + 1]; ++idx)
            {
                int adjustedColId = ecrId[idx];
                if (adjustedColId < 0 || adjustedColId >= dCols)
                {
                    std::cerr << "Invalid adjustedColId: " << adjustedColId << std::endl;
                    continue;
                }
                int tcFragInChunkIndex = adjustedColId / fragK;
                int colLocalIndex = adjustedColId % fragK;

                // Validate indices
                if (tcFragInChunkIndex < 0 || tcFragInChunkIndex >= numTcFragsInChunk)
                {
                    std::cerr << "Invalid tcFragInChunkIndex: " << tcFragInChunkIndex << std::endl;
                    continue;
                }

                if (colLocalIndex < 0 || colLocalIndex >= fragK)
                {
                    std::cerr << "Invalid colLocalIndex: " << colLocalIndex << std::endl;
                    continue;
                }

                // Store the value in valueGrid
                valueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex] = vals[idx];
                hasValueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex] = true;

                // Update fragBit
                int tcFragIndex = chunkPtr[rowChunkIndex] + tcFragInChunkIndex;
                if (tcFragIndex < 0 || tcFragIndex >= totalTcFrags)
                {
                    std::cerr << "Invalid tcFragIndex: " << tcFragIndex << std::endl;
                    continue;
                }

                // Compute bit position
                int bitPosition = rowLocalIndex * fragK + colLocalIndex;

                // Ensure bitPosition is within 0 to 31
                if (bitPosition < 0 || bitPosition >= 32)
                {
                    std::cerr << "Invalid bitPosition: " << bitPosition << std::endl;
                    continue;
                }

                // Set the bit in the fragBit
                fragBit[tcFragIndex] |= (1U << bitPosition);
            }
        }
        // printf("\n 000 \n");
        // Traverse valueGrid and fill tcVal and perTcFragNnz
        for (int tcFragInChunkIndex = 0; tcFragInChunkIndex < numTcFragsInChunk; ++tcFragInChunkIndex)
        {
            int tcFragIndex = chunkPtr[rowChunkIndex] + tcFragInChunkIndex;
            if (tcFragIndex < 0 || tcFragIndex >= totalTcFrags)
            {
                std::cerr << "Invalid tcFragIndex: " << tcFragIndex << std::endl;
                continue;
            }
            int nzCount = 0;

            for (int rowLocalIndex = 0; rowLocalIndex < fragM; ++rowLocalIndex)
            {
                for (int colLocalIndex = 0; colLocalIndex < fragK; ++colLocalIndex)
                {
                    if (hasValueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex])
                    {
                        tcVal.push_back(valueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex]);
                        ++nzCount;
                    }
                }
            }

            perTcFragNnz[tcFragIndex] = nzCount;
        }
    }

    // Build fragPtr from perTcFragNnz
    fragPtr[0] = 0;
    for (int i = 0; i < totalTcFrags; ++i)
    {
        fragPtr[i + 1] = fragPtr[i] + perTcFragNnz[i];
    }
}

void tcspmv_serial(
    const valT *x_d,
    valT *y_d,
    const int *chunkPtr,
    const std::vector<int> &fragPtr,
    const std::vector<uint32_t> &fragBit,
    const std::vector<valT> &tcVal,
    const int *sparse_AToX_index,
    int dRows,
    int dCols,
    int fragM,
    int fragK)
{
    int chunkNum = (dRows + fragM - 1) / fragM; // 计算总的行块数

    // 初始化输出向量 y_d 为零
    for (int i = 0; i < dRows; ++i)
        y_d[i] = 0.0;

    // 遍历每个行块（chunk）
    for (int rowChunkIndex = 0; rowChunkIndex < chunkNum; ++rowChunkIndex)
    {
        int rowStart = rowChunkIndex * fragM;
        int rowEnd = std::min(rowStart + fragM, dRows);

        int tcFragStart = chunkPtr[rowChunkIndex];
        int tcFragEnd = chunkPtr[rowChunkIndex + 1];

        // 遍历该行块中的每个 Tc 碎片
        for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
        {
            // 获取位图
            uint32_t bitmap = fragBit[tcFragIdx];

            // 获取该 Tc 碎片在 tcVal 中的起始和结束索引
            int valStartIdx = fragPtr[tcFragIdx];
            int valEndIdx = fragPtr[tcFragIdx + 1];
            int tcValNnz = valEndIdx - valStartIdx;

            const valT *tcValPtr = &tcVal[valStartIdx];

            // 获取该 Tc 碎片对应的 x 的索引
            const int *x_indices = &sparse_AToX_index[tcFragIdx * fragK]; // 大小为 fragK

            // 遍历碎片中的每个位置 (m, k)
            int valIdx = 0; // tcValPtr 中的当前非零值索引

            for (int m = 0; m < fragM; ++m)
            {
                int rowIdx = rowStart + m;
                if (rowIdx >= dRows)
                    continue; // 超出矩阵的行数，跳过

                for (int k = 0; k < fragK; ++k)
                {
                    int bitPos = m * fragK + k;
                    if (bitPos >= 32)
                        continue; // 位图只有 32 位，超出则跳过

                    int bit = (bitmap >> bitPos) & 1;

                    if (bit)
                    {
                        // 碎片中位置 (m, k) 有非零元素
                        valT a_value = tcValPtr[valIdx];
                        valIdx++; // 移动到 tcValPtr 中的下一个非零值

                        int x_idx = x_indices[k];
                        if (x_idx >= dCols)
                        {
                            // 非法的 x 索引，输出错误信息
                            std::cerr << "Invalid x index: " << x_idx << std::endl;
                            continue;
                        }
                        valT x_value = x_d[x_idx];

                        // 进行乘积并累加到 y_d 中
                        y_d[rowIdx] += a_value * x_value;
                    }
                    // 否则该位置为零，跳过
                }
            }
        }
    }
}
/*
void generateFormat_half(
    const half *vals,                              // Input: Non-zero values in CSR format
    const int *rowPtr,                             // Input: Row pointers in CSR format
    const int *ecrId,                              // Input: Adjusted column indices after empty column removal
    int dRows,                                     // Input: Number of rows
    int dCols,                                     // Input: Number of columns
    int *chunkPtr,                                 // Input: Offsets of tcFrags for each rowChunk
    std::vector<int> &fragPtr,                     // Output: Fragment pointer array
    std::vector<std::array<uint64_t, 2>> &fragBit, // Output: fragBit array for each tcFrag
    std::vector<half> &tcVal                       // Output: Non-zero values in tcFrags
)
{
    int numRowChunks = (dRows + fragM - 1) / fragM;
    int totalTcFrags = chunkPtr[numRowChunks];

    // Initialize fragPtr and per-tcFrag non-zero counts
    fragPtr.resize(totalTcFrags + 1, 0);
    std::vector<int> perTcFragNnz(totalTcFrags, 0);

    // Initialize fragBit
    fragBit.resize(totalTcFrags);

    // Temporary data structures for values in tcFrags
    std::vector<std::vector<std::vector<half>>> valueGrids;    // [tcFrag][row][col]
    std::vector<std::vector<std::vector<bool>>> hasValueGrids; // [tcFrag][row][col]

    for (int rowChunkIndex = 0; rowChunkIndex < numRowChunks; ++rowChunkIndex)
    {
        int rowChunkStart = rowChunkIndex * fragM;
        int rowChunkEnd = std::min(rowChunkStart + fragM, dRows);
        int numTcFragsInChunk = chunkPtr[rowChunkIndex + 1] - chunkPtr[rowChunkIndex];

        if (numTcFragsInChunk == 0)
        {
            continue;
        }
        // Initialize valueGrids and hasValueGrids for the tcFrags in this rowChunk
        valueGrids.assign(numTcFragsInChunk, std::vector<std::vector<half>>(fragM, std::vector<half>(fragK, half(0.0f))));
        hasValueGrids.assign(numTcFragsInChunk, std::vector<std::vector<bool>>(fragM, std::vector<bool>(fragK, false)));

        // Process each row in the rowChunk
        for (int r = rowChunkStart; r < rowChunkEnd; ++r)
        {
            int rowLocalIndex = r - rowChunkStart;

            // Iterate over non-zero elements in the row
            for (int idx = rowPtr[r]; idx < rowPtr[r + 1]; ++idx)
            {
                int adjustedColId = ecrId[idx];
                if (adjustedColId < 0 || adjustedColId >= dCols)
                {
                    std::cerr << "Invalid adjustedColId: " << adjustedColId << std::endl;
                    continue;
                }
                int tcFragInChunkIndex = adjustedColId / fragK;
                int colLocalIndex = adjustedColId % fragK;

                // Validate indices
                if (tcFragInChunkIndex < 0 || tcFragInChunkIndex >= numTcFragsInChunk)
                {
                    std::cerr << "Invalid tcFragInChunkIndex: " << tcFragInChunkIndex << std::endl;
                    continue;
                }

                if (colLocalIndex < 0 || colLocalIndex >= fragK)
                {
                    std::cerr << "Invalid colLocalIndex: " << colLocalIndex << std::endl;
                    continue;
                }

                // Store the value in valueGrid
                valueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex] = vals[idx];
                hasValueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex] = true;

                // Update fragBit
                int tcFragIndex = chunkPtr[rowChunkIndex] + tcFragInChunkIndex;
                if (tcFragIndex < 0 || tcFragIndex >= totalTcFrags)
                {
                    std::cerr << "Invalid tcFragIndex: " << tcFragIndex << std::endl;
                    continue;
                }

                // Compute bit position
                int bitPosition = rowLocalIndex * fragK + colLocalIndex;

                // Ensure bitPosition is within 0 to 127
                if (bitPosition < 0 || bitPosition >= 128)
                {
                    std::cerr << "Invalid bitPosition: " << bitPosition << std::endl;
                    continue;
                }

                // Set the bit in fragBit[tcFragIndex]
                int bitArrayIndex = bitPosition / 64; // 0 or 1
                int bitOffset = bitPosition % 64;     // 0 to 63
                fragBit[tcFragIndex][bitArrayIndex] |= (uint64_t(1) << bitOffset);
            }
        }

        for (int tcFragInChunkIndex = 0; tcFragInChunkIndex < numTcFragsInChunk; ++tcFragInChunkIndex)
        {
            int tcFragIndex = chunkPtr[rowChunkIndex] + tcFragInChunkIndex;
            if (tcFragIndex < 0 || tcFragIndex >= totalTcFrags)
            {
                std::cerr << "Invalid tcFragIndex: " << tcFragIndex << std::endl;
                continue;
            }
            int nzCount = 0;

            for (int rowLocalIndex = 0; rowLocalIndex < fragM; ++rowLocalIndex)
            {
                for (int colLocalIndex = 0; colLocalIndex < fragK; ++colLocalIndex)
                {
                    if (hasValueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex])
                    {
                        tcVal.push_back(valueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex]);
                        ++nzCount;
                    }
                }
            }

            perTcFragNnz[tcFragIndex] = nzCount;
        }
    }
    fragPtr[0] = 0;
    for (int i = 0; i < totalTcFrags; ++i)
    {
        fragPtr[i + 1] = fragPtr[i] + perTcFragNnz[i];
    }
}

void tcspmv_serial_half(
    const half *x_d,                                     // Input vector x
    half *y_d,                                           // Output vector y
    const int *chunkPtr,                                 // Offsets of tcFrags for each rowChunk
    const std::vector<int> &fragPtr,                     // Fragment pointer array
    const std::vector<std::array<uint64_t, 2>> &fragBit, // fragBit array for each tcFrag
    const std::vector<half> &tcVal,                      // Non-zero values in tcFrags
    const int *sparse_AToX_index,                        // Mapping from tcFrag to x indices
    int dRows,                                           // Number of rows
    int dCols,                                           // Number of columns
    int fragM,                                           // Fragment row size (16)
    int fragK                                            // Fragment column size (8)
)
{
    int chunkNum = (dRows + fragM - 1) / fragM;

    for (int i = 0; i < dRows; ++i)
        y_d[i] = half(0.0f);

    for (int rowChunkIndex = 0; rowChunkIndex < chunkNum; ++rowChunkIndex)
    {
        int rowStart = rowChunkIndex * fragM;
        int rowEnd = std::min(rowStart + fragM, dRows);

        int tcFragStart = chunkPtr[rowChunkIndex];
        int tcFragEnd = chunkPtr[rowChunkIndex + 1];

        for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
        {
            const std::array<uint64_t, 2> &bitmapArray = fragBit[tcFragIdx];
            int valStartIdx = fragPtr[tcFragIdx];
            int valEndIdx = fragPtr[tcFragIdx + 1];
            int tcValNnz = valEndIdx - valStartIdx;

            const half *tcValPtr = &tcVal[valStartIdx];
            const int *x_indices = &sparse_AToX_index[tcFragIdx * fragK];

            int valIdx = 0;

            for (int m = 0; m < fragM; ++m)
            {
                int rowIdx = rowStart + m;
                if (rowIdx >= dRows)
                    continue;

                for (int k = 0; k < fragK; ++k)
                {
                    int bitPos = m * fragK + k;
                    if (bitPos >= 128)
                        continue;

                    int bitArrayIndex = bitPos / 64; // 0 or 1
                    int bitOffset = bitPos % 64;     // 0 to 63

                    int bit = (bitmapArray[bitArrayIndex] >> bitOffset) & 1;

                    if (bit)
                    {
                        half a_value = tcValPtr[valIdx];
                        valIdx++;

                        int x_idx = x_indices[k];
                        if (x_idx >= dCols)
                        {
                            std::cerr << "Invalid x index: " << x_idx << std::endl;
                            continue;
                        }
                        half x_value = x_d[x_idx];
                        y_d[rowIdx] += a_value * x_value;
                    }
                }
            }
        }
    }
}
*/

void generateFormat_half(
    const half *vals,                              // Input: Non-zero values in CSR format
    const int *rowPtr,                             // Input: Row pointers in CSR format
    const int *ecrId,                              // Input: Adjusted column indices after empty column removal
    int dRows,                                     // Input: Number of rows
    int dCols,                                     // Input: Number of columns
    int *chunkPtr,                                 // Input: Offsets of tcFrags for each rowChunk
    std::vector<int> &fragPtr,                     // Output: Fragment pointer array
    std::vector<std::array<uint64_t, 4>> &fragBit, // Output: fragBit array for each tcFrag
    std::vector<half> &tcVal                       // Output: Non-zero values in tcFrags
)
{
    int numRowChunks = (dRows + fragM - 1) / fragM;
    int totalTcFrags = chunkPtr[numRowChunks];

    // Initialize fragPtr and per-tcFrag non-zero counts
    fragPtr.resize(totalTcFrags + 1, 0);
    std::vector<int> perTcFragNnz(totalTcFrags, 0);

    // Initialize fragBit
    fragBit.resize(totalTcFrags);

    // Temporary data structures for values in tcFrags
    std::vector<std::vector<std::vector<half>>> valueGrids;    // [tcFrag][row][col]
    std::vector<std::vector<std::vector<bool>>> hasValueGrids; // [tcFrag][row][col]

    for (int rowChunkIndex = 0; rowChunkIndex < numRowChunks; ++rowChunkIndex)
    {
        int rowChunkStart = rowChunkIndex * fragM;
        int rowChunkEnd = std::min(rowChunkStart + fragM, dRows);
        int numTcFragsInChunk = chunkPtr[rowChunkIndex + 1] - chunkPtr[rowChunkIndex];

        if (numTcFragsInChunk == 0)
        {
            continue;
        }
        // Initialize valueGrids and hasValueGrids for the tcFrags in this rowChunk
        valueGrids.assign(numTcFragsInChunk, std::vector<std::vector<half>>(fragM, std::vector<half>(fragK, half(0.0f))));
        hasValueGrids.assign(numTcFragsInChunk, std::vector<std::vector<bool>>(fragM, std::vector<bool>(fragK, false)));

        // Process each row in the rowChunk
        for (int r = rowChunkStart; r < rowChunkEnd; ++r)
        {
            int rowLocalIndex = r - rowChunkStart;

            // Iterate over non-zero elements in the row
            for (int idx = rowPtr[r]; idx < rowPtr[r + 1]; ++idx)
            {
                int adjustedColId = ecrId[idx];
                if (adjustedColId < 0 || adjustedColId >= dCols)
                {
                    std::cerr << "Invalid adjustedColId: " << adjustedColId << std::endl;
                    continue;
                }
                int tcFragInChunkIndex = adjustedColId / fragK;
                int colLocalIndex = adjustedColId % fragK;

                // Validate indices
                if (tcFragInChunkIndex < 0 || tcFragInChunkIndex >= numTcFragsInChunk)
                {
                    std::cerr << "Invalid tcFragInChunkIndex: " << tcFragInChunkIndex << std::endl;
                    continue;
                }

                if (colLocalIndex < 0 || colLocalIndex >= fragK)
                {
                    std::cerr << "Invalid colLocalIndex: " << colLocalIndex << std::endl;
                    continue;
                }

                // Store the value in valueGrid
                valueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex] = vals[idx];
                hasValueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex] = true;

                // Update fragBit
                int tcFragIndex = chunkPtr[rowChunkIndex] + tcFragInChunkIndex;
                if (tcFragIndex < 0 || tcFragIndex >= totalTcFrags)
                {
                    std::cerr << "Invalid tcFragIndex: " << tcFragIndex << std::endl;
                    continue;
                }

                // Compute bit position
                int bitPosition = rowLocalIndex * fragK + colLocalIndex;

                // Ensure bitPosition is within 0 to 255
                if (bitPosition < 0 || bitPosition >= 256)
                {
                    std::cerr << "Invalid bitPosition: " << bitPosition << std::endl;
                    continue;
                }

                // Set the bit in fragBit[tcFragIndex]
                int bitArrayIndex = bitPosition / 64; // 0 to 3
                int bitOffset = bitPosition % 64;     // 0 to 63
                fragBit[tcFragIndex][bitArrayIndex] |= (uint64_t(1) << bitOffset);
            }
        }

        for (int tcFragInChunkIndex = 0; tcFragInChunkIndex < numTcFragsInChunk; ++tcFragInChunkIndex)
        {
            int tcFragIndex = chunkPtr[rowChunkIndex] + tcFragInChunkIndex;
            if (tcFragIndex < 0 || tcFragIndex >= totalTcFrags)
            {
                std::cerr << "Invalid tcFragIndex: " << tcFragIndex << std::endl;
                continue;
            }
            int nzCount = 0;

            for (int rowLocalIndex = 0; rowLocalIndex < fragM; ++rowLocalIndex)
            {
                for (int colLocalIndex = 0; colLocalIndex < fragK; ++colLocalIndex)
                {
                    if (hasValueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex])
                    {
                        tcVal.push_back(valueGrids[tcFragInChunkIndex][rowLocalIndex][colLocalIndex]);
                        ++nzCount;
                    }
                }
            }

            perTcFragNnz[tcFragIndex] = nzCount;
        }
    }
    fragPtr[0] = 0;
    for (int i = 0; i < totalTcFrags; ++i)
    {
        fragPtr[i + 1] = fragPtr[i] + perTcFragNnz[i];
    }
}


void tcspmv_serial_half(
    const half *x_d,                                     // Input vector x
    half *y_d,                                           // Output vector y
    const int *chunkPtr,                                 // Offsets of tcFrags for each rowChunk
    const std::vector<int> &fragPtr,                     // Fragment pointer array
    const std::vector<std::array<uint64_t, 4>> &fragBit, // fragBit array for each tcFrag
    const std::vector<half> &tcVal,                      // Non-zero values in tcFrags
    const int *sparse_AToX_index,                        // Mapping from tcFrag to x indices
    int dRows,                                           // Number of rows
    int dCols,                                           // Number of columns
    int fragM,                                           // Fragment row size (16)
    int fragK                                            // Fragment column size (16)
)
{
    int chunkNum = (dRows + fragM - 1) / fragM;

    for (int i = 0; i < dRows; ++i)
        y_d[i] = half(0.0f);

    for (int rowChunkIndex = 0; rowChunkIndex < chunkNum; ++rowChunkIndex)
    {
        int rowStart = rowChunkIndex * fragM;
        int rowEnd = std::min(rowStart + fragM, dRows);

        int tcFragStart = chunkPtr[rowChunkIndex];
        int tcFragEnd = chunkPtr[rowChunkIndex + 1];

        for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
        {
            const std::array<uint64_t, 4> &bitmapArray = fragBit[tcFragIdx];
            int valStartIdx = fragPtr[tcFragIdx];
            int valEndIdx = fragPtr[tcFragIdx + 1];
            int tcValNnz = valEndIdx - valStartIdx;

            const half *tcValPtr = &tcVal[valStartIdx];
            const int *x_indices = &sparse_AToX_index[tcFragIdx * fragK];

            int valIdx = 0;

            for (int m = 0; m < fragM; ++m)
            {
                int rowIdx = rowStart + m;
                if (rowIdx >= dRows)
                    continue;

                for (int k = 0; k < fragK; ++k)
                {
                    int bitPos = m * fragK + k;
                    if (bitPos >= 256)
                        continue;

                    int bitArrayIndex = bitPos / 64; // 0 to 3
                    int bitOffset = bitPos % 64;     // 0 to 63

                    int bit = (bitmapArray[bitArrayIndex] >> bitOffset) & 1;

                    if (bit)
                    {
                        half a_value = tcValPtr[valIdx];
                        valIdx++;

                        int x_idx = x_indices[k];
                        if (x_idx >= dCols)
                        {
                            std::cerr << "Invalid x index: " << x_idx << std::endl;
                            continue;
                        }
                        half x_value = x_d[x_idx];
                        y_d[rowIdx] += a_value * x_value;
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Run the code by './spmv_double matrix.mtx'. \n");
        return 0;
    }
    int rowA, colA;
    indT nnzA;
    int isSymmetricA;
    valT *csrVal;
    indT *csrColInd;
    indT *csrRowPtr;
    valT *cscVal;
    indT *cscColPtr;
    indT *cscRowInd;

    char *filename;
    filename = argv[1];
    printf("\n===%s===\n\n", filename);

    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtr, &csrColInd, &csrVal, filename);
    initVec(csrVal, nnzA);
    /***************************************************************
     *                 1.Sparsity-aware Compression                *
     ***************************************************************/

    /***************************************************************
     *                  1.1split to two csc format                 *
     ***************************************************************/

    csr2csc(csrVal, csrRowPtr, csrColInd, rowA, colA, nnzA,
            &cscVal, &cscColPtr, &cscRowInd);

    CountWithIndex *descColId = (CountWithIndex *)malloc(sizeof(CountWithIndex) * colA);
    memset(descColId, 0, sizeof(CountWithIndex) * colA);

    for (int i = 0; i < colA; i++)
    {
        descColId[i].count = cscColPtr[i + 1] - cscColPtr[i];
        descColId[i].index = i;
    }
    qsort(descColId, colA, sizeof(CountWithIndex), compare_desc_structure);
    int dColsNnz = nnzA * colProp;
    int nnzColD = 0, dCols = 0;
    for (int i = 0; i < rowA; i++)
    {
        nnzColD += descColId[i].count;
        if (nnzColD >= dColsNnz)
        {
            dCols = i + 1;
            break;
        }
    }
    printf("col_nnz_ratio = %f\n", (float)nnzColD / (float)nnzA);
    printf("cols_ratio = %f \n", (float)dCols / (float)colA);

    int nnzColS = nnzA - nnzColD;
    int sCols = colA - dCols;

    valT *dcscVal = (valT *)malloc(nnzColD * sizeof(valT));
    indT *dcscColPtr = (indT *)malloc((dCols + 1) * sizeof(indT));
    indT *dcscRowInd = (indT *)malloc(nnzColD * sizeof(indT));

    valT *scscVal = (valT *)malloc(nnzColS * sizeof(valT));
    indT *scscColPtr = (indT *)malloc((sCols + 1) * sizeof(indT));
    indT *scscRowInd = (indT *)malloc(nnzColS * sizeof(indT));

    // memset(dcscVal, 0, sizeof(valT) * nnzColD);
    std::fill(dcscVal, dcscVal + nnzColD, static_cast<valT>(0.0));
    memset(dcscRowInd, 0, sizeof(indT) * nnzColD);
    memset(dcscColPtr, 0, sizeof(indT) * (dCols + 1));

    // memset(scscVal, 0, sizeof(valT) * nnzColS);
    std::fill(scscVal, scscVal + nnzColS, static_cast<valT>(0.0));
    memset(scscRowInd, 0, sizeof(indT) * nnzColS);
    memset(scscColPtr, 0, sizeof(indT) * (sCols + 1));

    dcscColPtr[0] = 0;
    int accu_d_col_ptr = 0;
    int dc_ptr = 0;
    for (int i = 0; i < dCols; i++)
    {
        int col_idx = descColId[i].index;
        accu_d_col_ptr += cscColPtr[col_idx + 1] - cscColPtr[col_idx];
        dcscColPtr[i + 1] = accu_d_col_ptr;
        for (int j = cscColPtr[col_idx]; j < cscColPtr[col_idx + 1]; j++)
        {
            dcscRowInd[dc_ptr] = cscRowInd[j];
            dcscVal[dc_ptr] = cscVal[j];
            dc_ptr++;
        }
    }
    // Create a bitmap to mark present values
    char *bitmap = (char *)calloc((colA + 7) / 8, sizeof(char));
    if (!bitmap)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    // Mark values present in descColId
    for (int i = 0; i < dCols; i++)
    {
        int index = descColId[i].index;
        if (index < colA)
        {
            bitmap[index / 8] |= (1 << (index % 8));
        }
    }
    // Find missing values and add to newArray
    int *newArray = (int *)malloc(sizeof(int) * sCols);
    memset(newArray, 0, sizeof(int) * sCols);
    int newArraySize = 0;

    for (int i = 0; i < colA; i++)
    {
        if (!(bitmap[i / 8] & (1 << (i % 8))))
        {
            newArray[newArraySize] = i;
            newArraySize++;
        }
    }

    scscColPtr[0] = 0;
    int accu_s_col_ptr = 0;
    int sc_ptr = 0;
    for (int i = 0; i < sCols; i++)
    {
        int col_idx = newArray[i];
        accu_s_col_ptr += cscColPtr[col_idx + 1] - cscColPtr[col_idx];
        scscColPtr[i + 1] = accu_s_col_ptr;
        for (int j = cscColPtr[col_idx]; j < cscColPtr[col_idx + 1]; j++)
        {
            scscRowInd[sc_ptr] = cscRowInd[j];
            scscVal[sc_ptr] = cscVal[j];
            sc_ptr++;
        }
    }
    /***************************************************************
     *    1.2split  dense col-segment to two csr format            *
     *    dense  col-segment: dcsrVal, dcsrRowPtr, dcsrColInd      *
     *    dense  row-chunk: csrVal_dd, csrRowPtr_dd, csrColInd_dd  *
     *    sparse row-chunk: csrVal_ds, csrRowPtr_ds, csrColInd_ds  *
     ***************************************************************/
    valT *dcsrVal;
    indT *dcsrColInd;
    indT *dcsrRowPtr;
    valT *scsrVal;
    indT *scsrColInd;
    indT *scsrRowPtr;
    csc2csr(dcscVal, dcscColPtr, dcscRowInd, rowA, dCols, nnzColD,
            &dcsrVal, &dcsrRowPtr, &dcsrColInd);

    csc2csr(scscVal, scscColPtr, scscRowInd, rowA, sCols, nnzColS,
            &scsrVal, &scsrRowPtr, &scsrColInd);

    CountWithIndex *descRowId = (CountWithIndex *)malloc(sizeof(CountWithIndex) * rowA);
    memset(descRowId, 0, sizeof(CountWithIndex) * rowA);

    for (int i = 0; i < rowA; i++)
    {
        descRowId[i].count = dcsrRowPtr[i + 1] - dcsrRowPtr[i];
        descRowId[i].index = i;
    }
    qsort(descRowId, rowA, sizeof(CountWithIndex), compare_desc_structure);
    int dRowsNnz = nnzA * rowProp;
    int nnzRowD = 0, dRows = 0;
    for (int i = 0; i < rowA; i++)
    {
        nnzRowD += descRowId[i].count;
        if (nnzRowD >= dRowsNnz)
        {
            dRows = i + 1;
            break;
        }
    }
    printf("row_nnz_ratio = %f\n", (float)nnzRowD / (float)nnzA);
    printf("rows_ratio = %f \n", (float)dRows / (float)rowA);
    printf("square_ratio = %f \n", ((float)dRows / (float)rowA) * ((float)dCols / (float)colA));
    int *rId = (int *)malloc(sizeof(int) * dRows);
    for (int i = 0; i < dRows; i++)
    {
        rId[i] = descRowId[i].index;
    }
    int nnzRowS = nnzColD - nnzRowD;
    int sRows = rowA - dRows;

    valT *csrVal_dd = (valT *)malloc(nnzRowD * sizeof(valT));
    indT *csrRowPtr_dd = (indT *)malloc((dRows + 1) * sizeof(indT));
    indT *csrColInd_dd = (indT *)malloc(nnzRowD * sizeof(indT));

    valT *csrVal_ds = (valT *)malloc(nnzRowS * sizeof(valT));
    indT *csrRowPtr_ds = (indT *)malloc((sRows + 1) * sizeof(indT));
    indT *csrColInd_ds = (indT *)malloc(nnzRowS * sizeof(indT));

    // memset(csrVal_dd, 0, sizeof(valT) * nnzRowD);
    std::fill(csrVal_dd, csrVal_dd + nnzRowD, static_cast<valT>(0.0));

    memset(csrColInd_dd, 0, sizeof(indT) * nnzRowD);
    memset(csrRowPtr_dd, 0, sizeof(indT) * (dRows + 1));

    // memset(csrVal_ds, 0, sizeof(valT) * nnzRowS);
    std::fill(csrVal_ds, csrVal_ds + nnzRowS, static_cast<valT>(0.0));
    memset(csrColInd_ds, 0, sizeof(indT) * nnzRowS);
    memset(csrRowPtr_ds, 0, sizeof(indT) * (sRows + 1));

    csrRowPtr_dd[0] = 0;
    int accu_d_row_ptr = 0;
    int dr_ptr = 0;
    for (int i = 0; i < dRows; i++)
    {
        int row_idx = descRowId[i].index;
        accu_d_row_ptr += dcsrRowPtr[row_idx + 1] - dcsrRowPtr[row_idx];
        csrRowPtr_dd[i + 1] = accu_d_row_ptr;
        for (int j = dcsrRowPtr[row_idx]; j < dcsrRowPtr[row_idx + 1]; j++)
        {
            csrColInd_dd[dr_ptr] = dcsrColInd[j];
            csrVal_dd[dr_ptr] = dcsrVal[j];
            dr_ptr++;
        }
    }

    char *bitmap_ = (char *)calloc((rowA + 7) / 8, sizeof(char));
    if (!bitmap_)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }

    for (int i = 0; i < dRows; i++)
    {
        int index = descRowId[i].index;
        if (index < rowA)
        {
            bitmap_[index / 8] |= (1 << (index % 8));
        }
    }
    // Find missing values and add to newArray
    int *newArray_ = (int *)malloc(sizeof(int) * sRows);
    memset(newArray_, 0, sizeof(int) * sRows);
    int newArraySize_ = 0;

    for (int i = 0; i < rowA; i++)
    {
        if (!(bitmap_[i / 8] & (1 << (i % 8))))
        {
            newArray_[newArraySize_] = i;
            newArraySize_++;
        }
    }

    csrRowPtr_ds[0] = 0;
    int accu_s_row_ptr = 0;
    int sr_ptr = 0;
    for (int i = 0; i < sRows; i++)
    {
        int row_idx = newArray_[i];
        accu_s_row_ptr += dcsrRowPtr[row_idx + 1] - dcsrRowPtr[row_idx];
        csrRowPtr_ds[i + 1] = accu_s_row_ptr;
        for (int j = dcsrRowPtr[row_idx]; j < dcsrRowPtr[row_idx + 1]; j++)
        {
            csrColInd_ds[sr_ptr] = dcsrColInd[j];
            csrVal_ds[sr_ptr] = dcsrVal[j];
            sr_ptr++;
        }
    }

    /***************************************************************
     *                     2.TCU-aware Compression                 *
     *   dense  row-chunk: csrVal_dd, csrRowPtr_dd, csrColInd_dd   *
     ***************************************************************/

    int *ecrId = (int *)malloc(sizeof(int) * nnzRowD);
    memset(ecrId, 0, sizeof(int) * nnzRowD);
    int chunkNum = ceil((double)dRows / (double)fragM);
    int *blockPartition = (int *)malloc(sizeof(int) * (chunkNum + 1));
    memset(blockPartition, 0, sizeof(int) * (chunkNum + 1));
    int **use_x_id = (int **)malloc((chunkNum + 1) * sizeof(int *));
    int *nec_num = (int *)malloc((chunkNum + 1) * sizeof(int));

    ecrPreprocess(csrColInd_dd, csrRowPtr_dd, dRows, dCols, fragM, fragK, blockPartition, ecrId, use_x_id, nec_num);

    int *chunkPtr = (int *)malloc(sizeof(int) * (chunkNum + 1));
    memset(chunkPtr, 0, sizeof(int) * (chunkNum + 1));
    for (int i = 1; i <= chunkNum; i++)
    {
        chunkPtr[i] += chunkPtr[i - 1] + blockPartition[i - 1];
    }
    int totalTcFrags = chunkPtr[chunkNum];
    // printf("\n chunkPtr: %d ", chunkPtr[chunkNum]);
    // printf("!!!TC_sparsity_ratio = %lf\n", (1 - (double)nnzRowD / (double)(chunkPtr[chunkNum] * 4 * 8)));
    printf("---TC_nnz_ratio = %lf---\n", ((double)nnzRowD / ((double)chunkPtr[chunkNum] * fragM * fragK)));
    int *sparse_AToX_index = (int *)malloc(sizeof(int) * totalTcFrags * fragK);
    memset(sparse_AToX_index, 0, sizeof(int) * (totalTcFrags * fragK));

    for (int rowChunkIndex = 0; rowChunkIndex < chunkNum; ++rowChunkIndex)
    {
        int *use_x = use_x_id[rowChunkIndex];
        for (int j = 0; j < nec_num[rowChunkIndex]; j++)
        {
            if (chunkPtr[rowChunkIndex] * fragK > totalTcFrags * fragK)
            {
                printf("!!!!!!!!!!!!!!");
            }
            int *sparse_AToX = sparse_AToX_index + chunkPtr[rowChunkIndex] * fragK;
            sparse_AToX[j] = use_x[j];
        }
    }

    /// Outputs
    std::vector<int> fragPtr;
    std::vector<valT> tcVal;
// #ifdef fp64
    std::vector<uint32_t> fragBit;
    generateFormat(csrVal_dd, csrRowPtr_dd, ecrId, dRows, dCols, chunkPtr, fragPtr, fragBit, tcVal);
/*
#else
    // std::vector<std::array<uint64_t, 2>> fragBit;
    std::vector<std::array<uint64_t, 4>> fragBit;
    generateFormat_half(csrVal_dd, csrRowPtr_dd, ecrId, dRows, dCols, chunkPtr, fragPtr, fragBit, tcVal);
#endif
*/
    /***************************************************************
     *         check the Sparsity-TCU-aware compression            *
     ***************************************************************/

    valT *X_val = (valT *)malloc(sizeof(valT) * colA);
    initVec(X_val, colA);

    valT *ourY_val = (valT *)malloc(sizeof(valT) * rowA);
    valT *tryY_val = (valT *)malloc(sizeof(valT) * dRows);
    valT *tryY_val1 = (valT *)malloc(sizeof(valT) * dRows);

    valT *Y_val = (valT *)malloc(sizeof(valT) * rowA);

    // memset(tryY_val, 0.0, sizeof(valT) * dRows);
    // memset(ourY_val, 0, sizeof(valT) * rowA);
    // memset(Y_val, 0, sizeof(valT) * rowA);
    std::fill(tryY_val, tryY_val + dRows, static_cast<valT>(0.0));
    std::fill(tryY_val1, tryY_val1 + dRows, static_cast<valT>(0.0));

    std::fill(ourY_val, ourY_val + rowA, static_cast<valT>(0.0));
    std::fill(Y_val, Y_val + rowA, static_cast<valT>(0.0));

    valT *x_d = (valT *)malloc(sizeof(valT) * dCols);
    valT *x_s = (valT *)malloc(sizeof(valT) * sCols);
    for (int i = 0; i < dCols; i++)
    {
        x_d[i] = X_val[descColId[i].index];
    }
    for (int i = 0; i < sCols; i++)
    {
        x_s[i] = X_val[newArray[i]];
    }
    // Baseline
    spmv_serial(csrVal, csrRowPtr, csrColInd, X_val, Y_val, rowA, colA, nnzA);

    // Peripheral-Sparse Block
    spmv_serial(scsrVal, scsrRowPtr, scsrColInd, x_s, ourY_val, rowA, sCols, nnzColS);
    printf("Peripheral-Sparse nnz pre row = %f\n", (double)nnzColS / (double)rowA);
    // Edge-Sparse Block
    spmv_serial_(csrVal_ds, csrRowPtr_ds, csrColInd_ds, x_d, ourY_val, sRows, dCols, nnzRowS, newArray_);
    printf("Edge-Sparse nnz pre row = %f\n", (double)nnzRowS / (double)sRows);

    // Core-Dense Block
    printf("Core-Dense nnz pre row = %f\n", (double)nnzRowD / (double)dRows);
    // spmv_serial_ecr(csrVal_dd, csrRowPtr_dd, csrColInd_dd, x_d, ourY_val, dRows, dCols, nnzRowD, rId, ecrId, use_x_id);
    double necTime = 0, necPre = 0;
#ifdef fp64
    tcspmv_serial(x_d, tryY_val, chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, dRows, dCols, fragM, fragK);
    // tcspmv_fp64(chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, x_d, tryY_val, dRows, dCols, rId, &necTime, &necPre);
#else
    tcspmv_serial(x_d, tryY_val1, chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, dRows, dCols, fragM, fragK);

    tcspmv_fp16(chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, x_d, tryY_val, dRows, dCols, rId, &necTime, &necPre);
#endif

    int result = eQcheck(tryY_val1, tryY_val, dRows);



    for (int i = 0; i < dRows; i++)
    {
        ourY_val[rId[i]] += tryY_val[i];
    }

    // spmv_serial(dcsrVal, dcsrRowPtr, dcsrColInd, x_d, ourY_val, rowA, dCols, nnzColD);
    // spmv_serial_(csrVal_dd, csrRowPtr_dd, csrColInd_dd, x_d, ourY_val, dRows, dCols, nnzRowD, rId);

    // int result = eQcheck(Y_val, ourY_val, rowA);

    free(sparse_AToX_index);
    free(tryY_val);
    free(nec_num);
    for (int i = 0; i < chunkNum; i++)
    {
        if (use_x_id[i])
        {
            free(use_x_id[i]);
        }
    }
    free(use_x_id);
    free(ecrId);
    free(chunkPtr);
    free(bitmap);
    // free(colHash);
    free(descColId);
    free(csrColInd);
    free(csrRowPtr);
    free(csrVal);
    free(cscColPtr);
    free(cscRowInd);
    free(cscVal);
    free(newArray);
    free(scscRowInd);
    free(scscColPtr);
    free(scscVal);
    free(dcscRowInd);
    free(dcscColPtr);
    free(dcscVal);
    free(x_s);
    free(x_d);
    free(Y_val);
    free(ourY_val);
    free(X_val);

    free(csrVal_dd);
    free(csrRowPtr_dd);
    free(csrColInd_dd);
    free(csrVal_ds);
    free(csrRowPtr_ds);
    free(csrColInd_ds);

    free(dcsrColInd);
    free(dcsrRowPtr);
    free(dcsrVal);
    free(scsrColInd);
    free(scsrRowPtr);
    free(scsrVal);

    free(newArray_);
    free(bitmap_);
    free(descRowId);
    free(blockPartition);

    return 0;
}
