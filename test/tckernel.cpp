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
        // printf(" res = %f and our = %f \n",val1,val2);
        if (isinf(val1) || isinf(val2))
        {
            printf("Inf detected at index (%d), val1(%4.3f), val2(%4.3f)\n", i, val1, val2);
        }
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


void tcspmv_serial_fp64(
    const double *x_d,
    double *y_d,
    const int *chunkPtr,
    const std::vector<int> &fragPtr,
    const std::vector<uint32_t> &fragBit,
    const std::vector<double> &tcVal,
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

            const double *tcValPtr = &tcVal[valStartIdx];

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
                        double a_value = tcValPtr[valIdx];
                        valIdx++; // 移动到 tcValPtr 中的下一个非零值

                        int x_idx = x_indices[k];
                        if (x_idx >= dCols)
                        {
                            // 非法的 x 索引，输出错误信息
                            std::cerr << "Invalid x index: " << x_idx << std::endl;
                            continue;
                        }
                        double x_value = x_d[x_idx];

                        // 进行乘积并累加到 y_d 中
                        y_d[rowIdx] += a_value * x_value;
                    }
                    // 否则该位置为零，跳过
                }
            }
        }
    }
}


void tcspmv_serial_half_(
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
     *         check the Sparsity-TCU-aware compression            *
     ***************************************************************/
#ifdef fp64
    std::string infile_name = std::string(filename) + "_preprocessed.dat";
#else
    std::string infile_name = std::string(filename) + "_preprocessed_fp16.dat";
#endif
    std::ifstream infile(infile_name, std::ios::binary);

    if (!infile) {
        std::cerr << "Error: Cannot open file " << infile_name << std::endl;
        return 1;
    }
    // 读取 chunkPtr
    int chunkNum;
    infile.read(reinterpret_cast<char*>(&chunkNum), sizeof(int));
    int *chunkPtr = (int *)malloc(sizeof(int) * (chunkNum + 1));
    infile.read(reinterpret_cast<char*>(chunkPtr), sizeof(int) * (chunkNum + 1));

    // 读取 sparse_AToX_index
    int totalTcFrags, fragK;
    infile.read(reinterpret_cast<char*>(&totalTcFrags), sizeof(int));
    infile.read(reinterpret_cast<char*>(&fragK), sizeof(int));
    int *sparse_AToX_index = (int *)malloc(sizeof(int) * totalTcFrags * fragK);
    infile.read(reinterpret_cast<char*>(sparse_AToX_index), sizeof(int) * totalTcFrags * fragK);

    // 读取 fragPtr
    size_t fragPtr_size;
    infile.read(reinterpret_cast<char*>(&fragPtr_size), sizeof(size_t));
    std::vector<int> fragPtr(fragPtr_size);
    infile.read(reinterpret_cast<char*>(fragPtr.data()), sizeof(int) * fragPtr_size);
/*
#ifdef fp64
*/
    size_t fragBit_size;
    infile.read(reinterpret_cast<char*>(&fragBit_size), sizeof(size_t));
    std::vector<uint32_t> fragBit(fragBit_size);
    infile.read(reinterpret_cast<char*>(fragBit.data()), sizeof(uint32_t) * fragBit_size);
/*
#else
    // 读取 fragBit
    size_t fragBit_size;
    infile.read(reinterpret_cast<char*>(&fragBit_size), sizeof(size_t));
    std::vector<std::array<uint64_t, 4>> fragBit(fragBit_size);
    infile.read(reinterpret_cast<char*>(fragBit.data()), sizeof(uint64_t) * 4 * fragBit_size);
#endif
*/
    // 读取 tcVal
    size_t tcVal_size;
    infile.read(reinterpret_cast<char*>(&tcVal_size), sizeof(size_t));
    std::vector<valT> tcVal(tcVal_size);
    infile.read(reinterpret_cast<char*>(tcVal.data()), sizeof(valT) * tcVal_size);

    infile.close();

    // 验证读取是否成功
    if (infile.fail()) {
        std::cerr << "Error occurred while reading the file." << std::endl;
        return 1;
    }
    


    printf("---TC_nnz_ratio = %lf---\n",((double)nnzRowD / ((double)chunkPtr[chunkNum] * fragM * fragK)));

  
    // std::cout << "Number of chunks: " << chunkNum << std::endl;



    valT *X_val = (valT *)malloc(sizeof(valT) * colA);
    initVec(X_val, colA);

    valT *ourY_val = (valT *)malloc(sizeof(valT) * rowA);
    valT *tryY_val = (valT *)malloc(sizeof(valT) * dRows);

    valT *Y_val = (valT *)malloc(sizeof(valT) * rowA);

    // memset(tryY_val, 0.0, sizeof(valT) * dRows);
    // memset(ourY_val, 0, sizeof(valT) * rowA);
    // memset(Y_val, 0, sizeof(valT) * rowA);
    std::fill(tryY_val, tryY_val + dRows, static_cast<valT>(0.0));
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
    // tcspmv_serial_fp64(x_d, tryY_val, chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, dRows, dCols, fragM, fragK);
    du_tcspmv_fp64(chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, x_d, tryY_val, dRows, dCols, rId, &necTime, &necPre);
#else
    // tcspmv_serial_half_(x_d, tryY_val, chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, dRows, dCols, fragM, fragK);
    du_tcspmv_fp16_v1(chunkPtr, fragPtr, fragBit, tcVal, sparse_AToX_index, x_d, tryY_val, dRows, dCols, rId, &necTime, &necPre);
#endif
    
    for (int i = 0; i < dRows; i++)
    {
        ourY_val[rId[i]] += tryY_val[i];
    }

    // spmv_serial(dcsrVal, dcsrRowPtr, dcsrColInd, x_d, ourY_val, rowA, dCols, nnzColD);
    // spmv_serial_(csrVal_dd, csrRowPtr_dd, csrColInd_dd, x_d, ourY_val, dRows, dCols, nnzRowD, rId);

    int result = eQcheck(Y_val, ourY_val, rowA);


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

    return 0;
}
