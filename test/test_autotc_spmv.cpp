#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
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
#include <unordered_map>
#include <algorithm>
#include <numeric>
/*
 * MatXtract SpMV Performance Test
 * Author: luuhwy
 * Date: 2025.9.24
 */
#include "mmio.h"
#include "csr2csc.h"
#include "MatXtract.h"

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

// Function to merge two CSR matrices S and D into A_CD
void merge2CSR(
    // Inputs for matrix S
    int rowA,
    int sCols,
    int nnzColS,
    const valT *csrVal_S,
    const int *csrRowPtr_S,
    const int *csrColInd_S,

    // Inputs for matrix D
    int sRows,
    int dCols,
    int nnzRowS,
    const valT *csrVal_D,
    const int *csrRowPtr_D,
    const int *csrColInd_D,
    // Mapping array
    const int *newArray,
    // Input vectors
    const valT *x_d,
    const valT *x_s,
    // Outputs for A_CD
    valT *&csrVal_CD,
    int *&csrRowPtr_CD,
    int *&csrColInd_CD,

    // Output vector x_CD
    valT *&x_CD)
{
    // Define A_CD's dimensions
    int rowCD = rowA;
    int colCD = sCols + dCols;
    int nnzCD = nnzColS + nnzRowS;

    // Step 1: Initialize x_CD as [x_d, x_s]
    x_CD = (valT *)malloc(colCD * sizeof(valT));
    for (int j = 0; j < dCols; ++j)
        x_CD[j] = x_d[j];
    for (int j = 0; j < sCols; ++j)
        x_CD[dCols + j] = x_s[j];

    // Step 2: Build csrRowPtr_CD
    csrRowPtr_CD = (int *)malloc((rowCD + 1) * sizeof(int));

    int *row_nnz = (int *)calloc(rowCD, sizeof(int)); // Temporary array for non-zero counts per row
    for (int i = 0; i < rowA; ++i)
    {
        row_nnz[i] += csrRowPtr_S[i + 1] - csrRowPtr_S[i];
    }
    for (int i = 0; i < sRows; ++i)
    {
        int row_in_A_CD = newArray[i];
        row_nnz[row_in_A_CD] += csrRowPtr_D[i + 1] - csrRowPtr_D[i];
    }

    // Step 2: Convert row_nnz to cumulative row pointer (csrRowPtr_CD)
    csrRowPtr_CD[0] = 0;
    for (int i = 0; i < rowCD; ++i)
    {
        csrRowPtr_CD[i + 1] = csrRowPtr_CD[i] + row_nnz[i];
    }

    // Optional check: Validate nnzCD
    if (csrRowPtr_CD[rowCD] != nnzCD)
    {
        std::cerr << "Warning: Calculated nnzCD (" << csrRowPtr_CD[rowCD]
                  << ") does not match expected nnzCD (" << nnzCD << ").\n";
    }

    // Step 3: Allocate csrVal_CD and csrColInd_CD
    csrVal_CD = (valT *)malloc(nnzCD * sizeof(valT));
    csrColInd_CD = (int *)malloc(nnzCD * sizeof(int));

    // Step 4: Initialize current position for each row
    int *current_pos = (int *)malloc(rowCD * sizeof(int));
    for (int i = 0; i < rowCD; ++i)
        current_pos[i] = csrRowPtr_CD[i];

    // Step 5: Insert S's non-zeros into A_CD
    for (int i = 0; i < rowA; ++i)
    {
        for (int k = csrRowPtr_S[i]; k < csrRowPtr_S[i + 1]; ++k)
        {
            int dest_pos = current_pos[i];
            csrVal_CD[dest_pos] = csrVal_S[k];
            csrColInd_CD[dest_pos] = dCols + csrColInd_S[k]; // Offset S's columns by dCols
            current_pos[i]++;
        }
    }

    // Step 6: Insert D's non-zeros into A_CD using unique mapping from newArray
    for (int i = 0; i < sRows; ++i)
    {
        int row_in_A_CD = newArray[i];
        for (int k = csrRowPtr_D[i]; k < csrRowPtr_D[i + 1]; ++k)
        {
            int dest_pos = current_pos[row_in_A_CD];
            csrVal_CD[dest_pos] = csrVal_D[k];
            csrColInd_CD[dest_pos] = csrColInd_D[k]; // D's columns are first in A_CD
            current_pos[row_in_A_CD]++;
        }
    }
    free(current_pos);
    free(row_nnz);
}

void merge2CSR_opt(
    // Inputs for matrix S
    int rowA,
    int sCols,
    int nnzColS,
    const valT *csrVal_S,
    const int *csrRowPtr_S,
    const int *csrColInd_S,

    // Inputs for matrix D
    int sRows,
    int dCols,
    int nnzRowS,
    const valT *csrVal_D,
    const int *csrRowPtr_D,
    const int *csrColInd_D,
    // Mapping array
    const int *newArray,
    // Input vectors
    const valT *x_d,
    const valT *x_s,
    // Outputs for A_CD
    valT *&csrVal_CD,
    int *&csrRowPtr_CD,
    int *&csrColInd_CD,

    // Output vector x_CD
    valT *&x_CD)
{
    int rowCD = rowA;
    int colCD = sCols + dCols;
    int nnzCD = nnzColS + nnzRowS;

    // Step 1: Initialize x_CD = [x_d, x_s]
    x_CD = (valT *)malloc(colCD * sizeof(valT));
    std::memcpy(x_CD, x_d, dCols * sizeof(valT));
    std::memcpy(x_CD + dCols, x_s, sCols * sizeof(valT));

    // Step 2: Build csrRowPtr_CD
    csrRowPtr_CD = (int *)malloc((rowCD + 1) * sizeof(int));

    int *row_nnz = (int *)calloc(rowCD, sizeof(int));
    {

#pragma omp parallel for
        for (int i = 0; i < rowA; ++i)
        {
            row_nnz[i] += csrRowPtr_S[i + 1] - csrRowPtr_S[i];
        }
    }
    {

#pragma omp parallel for
        for (int i = 0; i < sRows; ++i)
        {
            int row_in_A_CD = newArray[i];
            row_nnz[row_in_A_CD] += (csrRowPtr_D[i + 1] - csrRowPtr_D[i]);
        }
    }
    csrRowPtr_CD[0] = 0;
    std::partial_sum(row_nnz, row_nnz + rowCD, csrRowPtr_CD + 1);

    // Optional check
    if (csrRowPtr_CD[rowCD] != nnzCD)
    {
        std::cerr << "Warning: Calculated nnzCD (" << csrRowPtr_CD[rowCD]
                  << ") does not match expected nnzCD (" << nnzCD << ").\n";
    }

    // Step 3: Allocate memory for A_CD
    csrVal_CD = (valT *)malloc(nnzCD * sizeof(valT));
    csrColInd_CD = (int *)malloc(nnzCD * sizeof(int));

    // Step 4: Initialize current position
    int *current_pos = (int *)malloc(rowCD * sizeof(int));
    {

#pragma omp parallel for
        for (int i = 0; i < rowCD; ++i)
        {
            current_pos[i] = csrRowPtr_CD[i];
        }
    }

    // Step 5: Insert S's non-zeros into A_CD
    {

#pragma omp parallel for
        for (int i = 0; i < rowA; ++i)
        {
            int pos = current_pos[i];
            for (int k = csrRowPtr_S[i]; k < csrRowPtr_S[i + 1]; ++k)
            {
                csrVal_CD[pos] = csrVal_S[k];
                csrColInd_CD[pos] = dCols + csrColInd_S[k];
                pos++;
            }
            current_pos[i] = pos;
        }
    }

    // Step 6: Insert D's non-zeros into A_CD
    {

#pragma omp parallel for
        for (int i = 0; i < sRows; ++i)
        {
            int row_in_A_CD = newArray[i];
            int pos = current_pos[row_in_A_CD];
            for (int k = csrRowPtr_D[i]; k < csrRowPtr_D[i + 1]; ++k)
            {
                csrVal_CD[pos] = csrVal_D[k];
                csrColInd_CD[pos] = csrColInd_D[k];
                pos++;
            }
            current_pos[row_in_A_CD] = pos;
        }
    }

    free(current_pos);
    free(row_nnz);
}

int compare_desc_structure(const void *a, const void *b)
{
    return ((CountWithIndex *)b)->count - ((CountWithIndex *)a)->count;
}

int eQcheck(valT *tmp1, valT *tmp2, int length)
{
#ifdef fp64
    // Use double precision (fp64), check for 15 significant digits
    const double tolerance = 1e-8; // 15 significant digits for double precision
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
    const float tolerance = 1e-1; // 3 significant digits for half precision
    for (int i = 0; i < length; i++)
    {
        // Convert __half to float for computation
        float val1 = static_cast<float>(tmp1[i]);
        float val2 = static_cast<float>(tmp2[i]);
        // if (isinf(val1) || isinf(val2))
        // {
        //     printf("Inf detected at index (%d), val1(%4.3f), val2(%4.3f)\n", i, val1, val2);
        // }
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
        for (indT j = 0; j < n_one_line; j++)
        {
            indT v_idx = csrColInd[j + ptr_start];
            t = t + csrVal[j + ptr_start] * X_val[v_idx];
        }
        Y_val[i] = Y_val[i] + t;
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
}

void ecrPreprocess_opt(
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

    int max_window_size = 0;
    for (int i = 0; i < rowA; i += fragSize_h)
    {
        int end = (i + fragSize_h > rowA) ? rowA : (i + fragSize_h);
        max_window_size = std::max(max_window_size, csrRowPtr[end] - csrRowPtr[i]);
    }
    int *neighbor_window = (int *)malloc(max_window_size * sizeof(int));
    if (neighbor_window == nullptr)
    {
        fprintf(stderr, "Memory allocation failed for neighbor_window\n");
        exit(EXIT_FAILURE);
    }
#pragma omp parallel for reduction(+ : block_counter)
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

        std::memcpy(neighbor_window, &csrColInd[block_start], num_window_nnzs * sizeof(int));
        std::sort(neighbor_window, neighbor_window + num_window_nnzs);
        auto end_it = std::unique(neighbor_window, neighbor_window + num_window_nnzs);
        int unique_size = end_it - neighbor_window;

        std::unordered_map<int, int> clean_edges2col;
        for (int i = 0; i < unique_size; ++i)
        {
            clean_edges2col[neighbor_window[i]] = i;
        }

        use_x_id[windowId] = (int *)malloc(unique_size * sizeof(int));
        if (use_x_id[windowId] == nullptr)
        {
            fprintf(stderr, "Memory allocation failed for use_x_id[windowId]\n");
            exit(EXIT_FAILURE);
        }
        std::copy(neighbor_window, neighbor_window + unique_size, use_x_id[windowId]);
        nec_num[windowId] = unique_size;

        blockPartition[windowId] = (unique_size + fragSize_w - 1) / fragSize_w;
        block_counter += blockPartition[windowId];

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
    }

    free(neighbor_window);
}

void ecrPreprocess_opt1(
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
    int max_window_size = 0;
    for (int i = 0; i < rowA; i += fragSize_h)
    {
        int end = std::min(i + fragSize_h, rowA);
        max_window_size = std::max(max_window_size, csrRowPtr[end] - csrRowPtr[i]);
    }

#pragma omp parallel
    {
        int *neighbor_window = (int *)malloc(max_window_size * sizeof(int));
        if (neighbor_window == nullptr)
        {
            fprintf(stderr, "Memory allocation failed for neighbor_window\n");
            exit(EXIT_FAILURE);
        }

#pragma omp for reduction(+ : block_counter)
        for (int iter = 0; iter < rowA; iter += fragSize_h)
        {
            int windowId = iter / fragSize_h;
            int block_start = csrRowPtr[iter];
            int block_end = csrRowPtr[std::min(iter + fragSize_h, rowA)];
            int num_window_nnzs = block_end - block_start;

            if (num_window_nnzs <= 0)
            {
                blockPartition[windowId] = 0;
                continue;
            }
            std::memcpy(neighbor_window, &csrColInd[block_start], num_window_nnzs * sizeof(int));
            std::sort(neighbor_window, neighbor_window + num_window_nnzs);
            auto end_it = std::unique(neighbor_window, neighbor_window + num_window_nnzs);
            int unique_size = end_it - neighbor_window;

            use_x_id[windowId] = (int *)malloc(unique_size * sizeof(int));
            if (use_x_id[windowId] == nullptr)
            {
                fprintf(stderr, "Memory allocation failed for use_x_id[windowId]\n");
                exit(EXIT_FAILURE);
            }

            std::unordered_map<int, int> clean_edges2col;
            for (int i = 0; i < unique_size; ++i)
            {
                use_x_id[windowId][i] = neighbor_window[i];
                clean_edges2col[neighbor_window[i]] = i;
            }
            nec_num[windowId] = unique_size;
            blockPartition[windowId] = (unique_size + fragSize_w - 1) / fragSize_w;
            block_counter += blockPartition[windowId];

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
        }

        free(neighbor_window);
    }
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

void generateFormat_opt(
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
    int numRowChunks = static_cast<int>(std::ceil((double)dRows / (double)fragM));
    int totalTcFrags = chunkPtr[numRowChunks];

    // Initialize fragPtr and per-tcFrag non-zero counts
    fragPtr.resize(totalTcFrags + 1, 0);
    std::vector<int> perTcFragNnz(totalTcFrags, 0);

    // Initialize fragBit
    fragBit.resize(totalTcFrags, 0);

    for (int rowChunkIndex = 0; rowChunkIndex < numRowChunks; ++rowChunkIndex)
    {
        int rowChunkStart = rowChunkIndex * fragM;
        int rowChunkEnd = std::min(rowChunkStart + fragM, dRows);
        int numTcFragsInChunk = chunkPtr[rowChunkIndex + 1] - chunkPtr[rowChunkIndex];

        if (numTcFragsInChunk == 0)
        {
            continue;
        }

        size_t chunkSize = static_cast<size_t>(numTcFragsInChunk) * fragM * fragK;
        std::vector<valT> valueGrids(chunkSize, static_cast<valT>(0));
        std::vector<bool> hasValueGrids(chunkSize, false);

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

                int tcFragIndex = chunkPtr[rowChunkIndex] + tcFragInChunkIndex;
                if (tcFragIndex < 0 || tcFragIndex >= totalTcFrags)
                {
                    std::cerr << "Invalid tcFragIndex: " << tcFragIndex << std::endl;
                    continue;
                }

                // Compute bit position
                int bitPosition = rowLocalIndex * fragK + colLocalIndex;
                if (bitPosition < 0 || bitPosition >= 32)
                {
                    std::cerr << "Invalid bitPosition: " << bitPosition << std::endl;
                    continue;
                }

                // Flattened index for valueGrids/hasValueGrids
                size_t pos = static_cast<size_t>(tcFragInChunkIndex) * fragM * fragK + rowLocalIndex * fragK + colLocalIndex;

                // Store the value
                valueGrids[pos] = vals[idx];
                hasValueGrids[pos] = true;

                // Update fragBit
                fragBit[tcFragIndex] |= (1U << bitPosition);
            }
        }

        // Traverse valueGrids and fill tcVal and perTcFragNnz
        for (int tcFragInChunkIndex = 0; tcFragInChunkIndex < numTcFragsInChunk; ++tcFragInChunkIndex)
        {
            int tcFragIndex = chunkPtr[rowChunkIndex] + tcFragInChunkIndex;
            if (tcFragIndex < 0 || tcFragIndex >= totalTcFrags)
            {
                std::cerr << "Invalid tcFragIndex: " << tcFragIndex << std::endl;
                continue;
            }

            int nzCount = 0;
            size_t basePos = static_cast<size_t>(tcFragInChunkIndex) * fragM * fragK;
            for (int rowLocalIndex = 0; rowLocalIndex < fragM; ++rowLocalIndex)
            {
                size_t rowBase = basePos + rowLocalIndex * fragK;
                for (int colLocalIndex = 0; colLocalIndex < fragK; ++colLocalIndex)
                {
                    size_t pos = rowBase + colLocalIndex;
                    if (hasValueGrids[pos])
                    {
                        tcVal.push_back(valueGrids[pos]);
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

void generateFormat_opt_fp16(
    const valT *vals,               // Input: Non-zero values in CSR format
    const int *rowPtr,              // Input: Row pointers in CSR format
    const int *ecrId,               // Input: Adjusted column indices after empty column removal
    int dRows,                      // Input: Number of rows
    int dCols,                      // Input: Number of columns
    int *chunkPtr,                  // Input: Offsets of tcFrags for each rowChunk
    std::vector<int> &fragPtr,      // Output: Fragment pointer array
    std::vector<uint64_t> &fragBit, // Output: fragBit array for each tcFrag (changed to uint64_t)
    std::vector<valT> &tcVal        // Output: Non-zero values in tcFrags
)
{
    int numRowChunks = static_cast<int>(std::ceil((double)dRows / (double)fragM));
    int totalTcFrags = chunkPtr[numRowChunks];

    // Initialize fragPtr and per-tcFrag non-zero counts
    fragPtr.resize(totalTcFrags + 1, 0);
    std::vector<int> perTcFragNnz(totalTcFrags, 0);

    // Initialize fragBit
    fragBit.resize(totalTcFrags, 0); // Adjusted to use uint64_t

    for (int rowChunkIndex = 0; rowChunkIndex < numRowChunks; ++rowChunkIndex)
    {
        int rowChunkStart = rowChunkIndex * fragM;
        int rowChunkEnd = std::min(rowChunkStart + fragM, dRows);
        int numTcFragsInChunk = chunkPtr[rowChunkIndex + 1] - chunkPtr[rowChunkIndex];

        if (numTcFragsInChunk == 0)
        {
            continue;
        }

        size_t chunkSize = static_cast<size_t>(numTcFragsInChunk) * fragM * fragK;
        std::vector<valT> valueGrids(chunkSize, static_cast<valT>(0));
        std::vector<bool> hasValueGrids(chunkSize, false);

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

                int tcFragIndex = chunkPtr[rowChunkIndex] + tcFragInChunkIndex;
                if (tcFragIndex < 0 || tcFragIndex >= totalTcFrags)
                {
                    std::cerr << "Invalid tcFragIndex: " << tcFragIndex << std::endl;
                    continue;
                }

                // Compute bit position
                int bitPosition = rowLocalIndex * fragK + colLocalIndex;
                if (bitPosition < 0 || bitPosition >= 64) // Adjusted from 32 to 64
                {
                    std::cerr << "Invalid bitPosition: " << bitPosition << std::endl;
                    continue;
                }

                // Flattened index for valueGrids/hasValueGrids
                size_t pos = static_cast<size_t>(tcFragInChunkIndex) * fragM * fragK + rowLocalIndex * fragK + colLocalIndex;

                // Store the value
                valueGrids[pos] = vals[idx];
                hasValueGrids[pos] = true;

                // Update fragBit
                fragBit[tcFragIndex] |= (1ULL << bitPosition); // Adjusted to 1ULL for 64-bit
            }
        }

        // Traverse valueGrids and fill tcVal and perTcFragNnz
        for (int tcFragInChunkIndex = 0; tcFragInChunkIndex < numTcFragsInChunk; ++tcFragInChunkIndex)
        {
            int tcFragIndex = chunkPtr[rowChunkIndex] + tcFragInChunkIndex;
            if (tcFragIndex < 0 || tcFragIndex >= totalTcFrags)
            {
                std::cerr << "Invalid tcFragIndex: " << tcFragIndex << std::endl;
                continue;
            }

            int nzCount = 0;
            size_t basePos = static_cast<size_t>(tcFragInChunkIndex) * fragM * fragK;
            for (int rowLocalIndex = 0; rowLocalIndex < fragM; ++rowLocalIndex)
            {
                size_t rowBase = basePos + rowLocalIndex * fragK;
                for (int colLocalIndex = 0; colLocalIndex < fragK; ++colLocalIndex)
                {
                    size_t pos = rowBase + colLocalIndex;
                    if (hasValueGrids[pos])
                    {
                        tcVal.push_back(valueGrids[pos]);
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
    int chunkNum = (dRows + fragM - 1) / fragM;
    for (int i = 0; i < dRows; ++i)
        y_d[i] = 0.0;
    for (int rowChunkIndex = 0; rowChunkIndex < chunkNum; ++rowChunkIndex)
    {
        int rowStart = rowChunkIndex * fragM;
        int rowEnd = std::min(rowStart + fragM, dRows);

        int tcFragStart = chunkPtr[rowChunkIndex];
        int tcFragEnd = chunkPtr[rowChunkIndex + 1];

        for (int tcFragIdx = tcFragStart; tcFragIdx < tcFragEnd; ++tcFragIdx)
        {
            uint32_t bitmap = fragBit[tcFragIdx];
            int valStartIdx = fragPtr[tcFragIdx];
            int valEndIdx = fragPtr[tcFragIdx + 1];
            int tcValNnz = valEndIdx - valStartIdx;

            const valT *tcValPtr = &tcVal[valStartIdx];

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
                    if (bitPos >= 32)
                        continue;

                    int bit = (bitmap >> bitPos) & 1;

                    if (bit)
                    {
                        valT a_value = tcValPtr[valIdx];
                        valIdx++;

                        int x_idx = x_indices[k];
                        if (x_idx >= dCols)
                        {
                            std::cerr << "Invalid x index: " << x_idx << std::endl;
                            continue;
                        }
                        valT x_value = x_d[x_idx];
                        y_d[rowIdx] = y_d[rowIdx] + a_value * x_value;
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
        printf("Run the code by './matxtract_perftest (col_frac) (hot_frac) matrix.mtx'.\n");
        printf("col_frac and hot_frac are optional, defaulting to 0 if not provided.\n");
        return 1;
    }
    char *filename = argv[argc - 1];
    double col_frac = 0.0, hot_frac = 0.0;
    if (argc == 4)
    {
        col_frac = atof(argv[1]);
        hot_frac = atof(argv[2]);
        if (col_frac < hot_frac)
        {
            printf("Error: col_frac must be greater than hot_frac.\n");
            printf("Please run the program with valid parameters.\n");
            return 1;
        }
    }
    else if (argc > 4)
    {
        printf("Error: Too many arguments provided.\n");
        printf("Run the code by './spmv_double (col_frac) (hot_frac) matrix.mtx'.\n");
        return 1;
    }

    int rowA, colA;
    indT nnzA;
    int isSymmetricA;
    valT *csrVal;
    indT *csrColInd;
    indT *csrRowPtr;
#ifdef fp64
    printf("\n Processing the %s graph in FP64\n\n", filename);
#else
    printf("\n Processing the %s graph in FP16\n\n", filename);
#endif

    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtr, &csrColInd, &csrVal, filename);
    initVec(csrVal, nnzA);
    // printf("\n rows = %d, nnzs = %d \n", rowA, nnzA);
    /***************************************************************
     *                 1.Sparsity-aware Compression                *
     ***************************************************************/

    /***************************************************************
     *                  1.1split to two csc format                 *
     ***************************************************************/
    // printf("\n------Sparsity-aware Compression START------\n");
    // float rowProp = 0.62f;
    // for (float rowProp = 0.1f; rowProp <= 0.75f; rowProp += 0.05f)
    float colProp = col_frac;
    float rowProp = hot_frac;
    double matxtract_time = 0.0;
    if (!((rowProp == 0.0) && (colProp == 0.0)))
    {
        // float colProp = rowProp / 0.8f;
        // float colProp = 0.8f;
        valT *cscVal;
        indT *cscColPtr;
        indT *cscRowInd;
        std::cout << "---------------------------------------------------------" << std::endl;
        std::cout << "                     localRow: " << rowProp << std::endl;
        std::cout << "                     globalCol: " << colProp << std::endl;
        std::cout << "---------------------------------------------------------" << std::endl;

        std::cout << "---------------------Compression Info--------------------" << std::endl;
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
        printf("cols_ratio = %f \n", (float)dCols / (float)colA);

        int nnzColS = nnzA - nnzColD;
        int sCols = colA - dCols;

        valT *dcscVal = (valT *)malloc(nnzColD * sizeof(valT));
        indT *dcscColPtr = (indT *)malloc((dCols + 1) * sizeof(indT));
        indT *dcscRowInd = (indT *)malloc(nnzColD * sizeof(indT));

        valT *scscVal = (valT *)malloc(nnzColS * sizeof(valT));
        indT *scscColPtr = (indT *)malloc((sCols + 1) * sizeof(indT));
        indT *scscRowInd = (indT *)malloc(nnzColS * sizeof(indT));

        std::fill(dcscVal, dcscVal + nnzColD, static_cast<valT>(0.0));
        memset(dcscRowInd, 0, sizeof(indT) * nnzColD);
        memset(dcscColPtr, 0, sizeof(indT) * (dCols + 1));
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

        // printf("nnzD = %d\n", nnzRowD);
        // printf("rows_ratio = %f \n", (float)dRows / (float)rowA);
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

        std::fill(csrVal_dd, csrVal_dd + nnzRowD, static_cast<valT>(0.0));
        memset(csrColInd_dd, 0, sizeof(indT) * nnzRowD);
        memset(csrRowPtr_dd, 0, sizeof(indT) * (dRows + 1));
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

        // struct timeval t1, t2;
        // printf("\n------ecrPreprocess START------\n");
        // gettimeofday(&t1, NULL);
        ecrPreprocess_opt1(csrColInd_dd, csrRowPtr_dd, dRows, dCols, fragM, fragK, blockPartition, ecrId, use_x_id, nec_num);
        // gettimeofday(&t2, NULL);
        // double ecrPreprocessT = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        // std::cout << "ecrPreprocess execution time: " << ecrPreprocessT << " ms" << std::endl;
        // printf("\n------ecrPreprocess END------\n");

        int *chunkPtr = (int *)malloc(sizeof(int) * (chunkNum + 1));
        memset(chunkPtr, 0, sizeof(int) * (chunkNum + 1));
        for (int i = 1; i <= chunkNum; i++)
        {
            chunkPtr[i] += chunkPtr[i - 1] + blockPartition[i - 1];
        }
        int totalTcFrags = chunkPtr[chunkNum];
        double tc_nnz_ratio = ((double)nnzRowD / ((double)chunkPtr[chunkNum] * fragM * fragK));

        printf("TC_nnz_ratio = %lf\n", tc_nnz_ratio);

        int *sparse_AToX_index = (int *)malloc(sizeof(int) * totalTcFrags * fragK);
        memset(sparse_AToX_index, 0, sizeof(int) * (totalTcFrags * fragK));

// for (int rowChunkIndex = 0; rowChunkIndex < chunkNum; ++rowChunkIndex)
// {
//     int *use_x = use_x_id[rowChunkIndex];
//     for (int j = 0; j < nec_num[rowChunkIndex]; j++)
//     {
//         if (chunkPtr[rowChunkIndex] * fragK > totalTcFrags * fragK)
//         {
//             printf("!!!!!!!!!!!!!!");
//         }
//         int *sparse_AToX = sparse_AToX_index + chunkPtr[rowChunkIndex] * fragK;
//         sparse_AToX[j] = use_x[j];
//     }
// }
#pragma omp parallel for
        for (int rowChunkIndex = 0; rowChunkIndex < chunkNum; ++rowChunkIndex)
        {
            int *use_x = use_x_id[rowChunkIndex];
            int offset = chunkPtr[rowChunkIndex] * fragK;
            assert(offset <= totalTcFrags * fragK && "Error: Offset exceeds total fragment size.");
            int *sparse_AToX = sparse_AToX_index + offset;
            std::memcpy(sparse_AToX, use_x, nec_num[rowChunkIndex] * sizeof(int));
        }

        // printf("\n------sparse_AToX END------\n");

        /// Outputs
        std::vector<int> fragPtr;
        std::vector<valT> tcVal;
        // printf("\n------generateFormat START------\n");
        // struct timeval t3, t4;
        // gettimeofday(&t3, NULL);
#ifdef fp64
        std::vector<uint32_t> fragBit;
        generateFormat_opt(csrVal_dd, csrRowPtr_dd, ecrId, dRows, dCols, chunkPtr, fragPtr, fragBit, tcVal);
#else
        std::vector<uint64_t> fragBit;
        generateFormat_opt_fp16(csrVal_dd, csrRowPtr_dd, ecrId, dRows, dCols, chunkPtr, fragPtr, fragBit, tcVal);

#endif   
        // gettimeofday(&t4, NULL);
        // double generateFormatT = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
        // std::cout << "generateFormat execution time: " << generateFormatT << " ms" << std::endl;
        // printf("\n------generateFormat END------\n");

        std::cout << "---------------------Performance Info--------------------" << std::endl;

        /***************************************************************
         *                              TEST                           *
         ***************************************************************/

        valT *X_val = (valT *)malloc(sizeof(valT) * colA);
        initVec(X_val, colA);

        valT *coldY_val = (valT *)malloc(sizeof(valT) * rowA);
        valT *coldY_val_solo = (valT *)malloc(sizeof(valT) * rowA);
        valT *hotY_val = (valT *)malloc(sizeof(valT) * dRows);
        valT *hotY_val_solo_du = (valT *)malloc(sizeof(valT) * dRows);
        valT *hotY_val_solo_se = (valT *)malloc(sizeof(valT) * dRows);
        valT *Y_val = (valT *)malloc(sizeof(valT) * rowA);

        // memset(hotY_val, 0.0, sizeof(valT) * dRows);
        // memset(coldY_val, 0, sizeof(valT) * rowA);
        // memset(Y_val, 0, sizeof(valT) * rowA);
        std::fill(hotY_val, hotY_val + dRows, static_cast<valT>(0.0));
        std::fill(hotY_val_solo_du, hotY_val_solo_du + dRows, static_cast<valT>(0.0));
        std::fill(hotY_val_solo_se, hotY_val_solo_se + dRows, static_cast<valT>(0.0));

        std::fill(coldY_val, coldY_val + rowA, static_cast<valT>(0.0));
        std::fill(coldY_val_solo, coldY_val_solo + rowA, static_cast<valT>(0.0));
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
        ////////////////////////////////////////////////////////////////////////////////////////////
        /////////////Baseline
        ////////////////////////////////////////////////////////////////////////////////////////////
        spmv_serial(csrVal, csrRowPtr, csrColInd, X_val, Y_val, rowA, colA, nnzA);

        /*
         *    Peripheral-Sparse Block: scsrVal, scsrRowPtr, scsrColInd : rowA, sCols, nnzColS        *
         *    Edge-Sparse Block: csrVal_ds, csrRowPtr_ds, csrColInd_ds : sRows, dCols, nnzRowS,      *
         *
         *    Core-Dense Block: csrVal_dd, csrRowPtr_dd, csrColInd_dd  *
         *    dense  col-segment: dcsrVal, dcsrRowPtr, dcsrColInd      *
         */

        int nnzCD = nnzColS + nnzRowS;
        int rowCD = rowA;
        int colCD = colA;

        valT *csrVal_CD;
        int *csrRowPtr_CD;
        int *csrColInd_CD;
        valT *x_CD;

        merge2CSR_opt(
            rowA, sCols, nnzColS, scsrVal, scsrRowPtr, scsrColInd,
            sRows, dCols, nnzRowS, csrVal_ds, csrRowPtr_ds, csrColInd_ds,
            newArray_,
            x_d, x_s,
            csrVal_CD, csrRowPtr_CD, csrColInd_CD,
            x_CD);
        double cdTime = 0, cdPre = 0;
        double tcTime_du = 0;
        double tcTime_se = 0;
        
        ////////////////////////////////////////////////////////////////////////////////////////////
        /////////////tensor core partition
        ////////////////////////////////////////////////////////////////////////////////////////////

        int NUM = 4;
        int block_longest = 256;
        double threshold = 0.75;
        int *new_order = (int *)malloc(sizeof(int) * rowA);

#ifdef fp64
        bool denseUnfold = ((nnzA >= 22322336) && (tc_nnz_ratio >= 0.55)) ? true : false;
        if (denseUnfold)
        {
            du_tcspmv_fp64(chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, x_d, hotY_val_solo_du, dRows, dCols, rId, &tcTime_du);
            // tcspmv_serial(x_d, hotY_val_solo_du, chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, dRows, dCols, fragM, fragK);
        }
        else // sparseEncode
        {
            se_tcspmv_fp64(csrVal_dd, csrRowPtr_dd, csrColInd_dd, x_d, hotY_val_solo_se, new_order, dRows, dCols, nnzRowD, NUM, threshold, block_longest, &tcTime_se);
        }
#else
        bool denseUnfold = ((nnzA >= 298113762) && (tc_nnz_ratio >= 0.688047)) ? true : false; // TODO: further evaluation
        if (denseUnfold)
        {
            du_tcspmv_fp16_v2(chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, x_d, hotY_val_solo_du, dRows, dCols, rId, &tcTime_du);
            // tcspmv_serial(x_d, hotY_val_solo_du, chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, dRows, dCols, fragM, fragK);
        }
        else // sparseEncode
        {
            se_tcspmv_fp16(csrVal_dd, csrRowPtr_dd, csrColInd_dd, x_d, hotY_val_solo_se, new_order, dRows, dCols, nnzRowD, NUM, threshold, block_longest, &tcTime_se);
        }
#endif
        cudaDeviceSynchronize();

        if (denseUnfold)
        {
            for (int i = 0; i < dRows; i++)
            {
                coldY_val_solo[rId[i]] = hotY_val_solo_du[i];
            }
        }
        else
        {
            for (int i = 0; i < dRows; i++)
            {
                coldY_val_solo[rId[new_order[i]]] = hotY_val_solo_se[i];
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /////////////cuda core partition
        ////////////////////////////////////////////////////////////////////////////////////////////
        if(!((rowProp == 1.0) && (colProp == 1.0)))
        {
        // printf("nnzS = %d\n", nnzCD);
        cdspmv(csrVal_CD, csrRowPtr_CD, csrColInd_CD, x_CD, coldY_val_solo, rowCD, colCD, nnzCD, &cdTime, &cdPre);
        // printf("cdspmv:    %8.4lf ms, cdspmv pre:%8.4lf ms\n", cdTime, cdPre);
        /*printf("cdspmv:    %8.4lf ms\n", cdTime);*/
        }
        else //all tcu 
        {
            cdTime = 0;
        }

        if (denseUnfold)
        {
            /*printf("du_spmv:    %8.4lf ms\n", tcTime_du);*/
            printf("MatXtract time = %lf ms\n\n", tcTime_du + cdTime);
            matxtract_time = tcTime_du + cdTime;
        }
        else
        {
            /*printf("se_spmv:    %8.4lf ms\n", tcTime_se);*/
            printf("MatXtract time = %lf ms\n\n", tcTime_se + cdTime);
            matxtract_time = tcTime_se + cdTime;
        }
        int result_auto = eQcheck(Y_val, coldY_val_solo, rowA);

        // free memory
        free(sparse_AToX_index);
        free(nec_num);
        for (int i = 0; i < chunkNum; i++)
        {
            if (use_x_id[i])
            {
                free(use_x_id[i]);
            }
        }
        free(rId);
        free(new_order);
        free(use_x_id);
        free(ecrId);
        free(chunkPtr);
        free(blockPartition);

        free(csrVal_CD);
        free(csrRowPtr_CD);
        free(csrColInd_CD);
        free(x_CD);

        free(hotY_val);
        free(hotY_val_solo_du);
        free(hotY_val_solo_se);
        free(coldY_val);
        free(coldY_val_solo);
        free(bitmap);
        // free(colHash);
        free(descColId);
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
    }
    else
    {
        // printf("\n all cd spmv \n");
        valT *X_val = (valT *)malloc(sizeof(valT) * colA);
        initVec(X_val, colA);
        valT *Y_val = (valT *)malloc(sizeof(valT) * rowA);
        valT *ourY_val = (valT *)malloc(sizeof(valT) * rowA);
        std::fill(Y_val, Y_val + rowA, static_cast<valT>(0.0));
        std::fill(ourY_val, ourY_val + rowA, static_cast<valT>(0.0));

        spmv_serial(csrVal, csrRowPtr, csrColInd, X_val, Y_val, rowA, colA, nnzA);
        double cdTime = 0, cdPre = 0;
        cdspmv(csrVal, csrRowPtr, csrColInd, X_val, ourY_val, rowA, colA, nnzA, &cdTime, &cdPre);

        printf("MatXtract time = %lf ms\n", cdTime);
        matxtract_time = cdTime;
        int result_cd = eQcheck(Y_val, ourY_val, rowA);
        free(Y_val);
        free(ourY_val);
        free(X_val);
    }
    printf("MatXtract GFLOPS= %.2f\n",
        (2.0 * (double)nnzA)           // FLOPs
      / (matxtract_time * 1e-3)       // seconds
      / 1e9);                         // to GFLOPS
      
    free(csrColInd);
    free(csrRowPtr);
    free(csrVal);

    return 0;
}

