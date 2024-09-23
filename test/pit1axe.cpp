#include "mmio.h"
#include "csr2csc.h"

// const float colProp = 0.82;
const float rowProp = 0.6;
const float colProp = 0.8;

const int fragM = 8;
const int fragK = 4;
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

int sum_array(const int *arr, int size)
{
    int sum = 0;
    for (int i = 0; i < size; ++i)
    {
        sum += arr[i]; // 直接通过指针访问数组元素
    }
    return sum;
}

int compare_desc_structure(const void *a, const void *b)
{
    return ((CountWithIndex *)b)->count - ((CountWithIndex *)a)->count;
}

int eQcheck(valT *tmp1, valT *tmp2, int length)
{
    for (int i = 0; i < length; i++)
    {
        if (fabs(tmp1[i] - tmp2[i]) > 1e-5)
        {
            printf("error in (%d), cpu(%4.2f), our(%4.2f),please check your code!\n", i, tmp1[i], tmp2[i]);
            return -1;
        }
    }
    printf("Y(%d), compute succeed!\n", length);
    return 0;
}
int eQcheck111(indT *tmp1, indT *tmp2, int length)
{
    for (int i = 0; i < length; i++)
    {
        if (tmp1[i] != tmp2[i])
        {
            printf("error in (%d), our(%d), cpu(%d),please check your code!\n", i, tmp1[i], tmp2[i]);
            return -1;
        }
    }
    printf("Y(%d), compute succeed!\n", length);
    return 0;
}

int compare_desc(const void *a, const void *b)
{
    return (*(int *)b - *(int *)a);
}

void spmv_fp64_serial(valT *csrVal, indT *csrRowPtr, indT *csrColInd,
                      valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA)
{
    valT t;
    for (indT i = 0; i < rowA; i++)
    {
        t = 0.0f;
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

void spmv_fp64_serial_csc(valT *cscVal, indT *cscColPtr, indT *cscRowInd,
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

void spmv_fp64_serial_(valT *csrVal, indT *csrRowPtr, indT *csrColInd,
                       valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA, indT *row_order)
{
    valT t;
    for (indT i = 0; i < rowA; i++)
    {
        t = 0.0f;
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
// TODO:
void spmv_fp64_serial_ecr(valT *csrVal, indT *csrRowPtr, indT *csrColInd,
                          valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA, indT *row_order, int *ecrId, int **use_x_id)
{
    valT t;
    for (int i = 0; i < rowA; i ++)
    {
        int windowId = i / fragM;
        t = 0.0f;
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
// 去重函数
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
            nb2col[array[cur]] = loc; // 从eid到TC_block列索引的映射
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
    int *chunkPtr,
    int *ecrId, // output
    int **use_x_id,
    int *nec_num)
{
    int block_counter = 0;
    chunkPtr[0] = block_counter;
#pragma omp parallel for reduction(+ : block_counter)
    for (int iter = 0; iter < rowA + 1; iter += fragSize_h)
    {
        int windowId = iter / fragSize_h;
        int block_start = csrRowPtr[iter];
        int iter_plus = iter + fragSize_h;
        // 防止越界，取较小值
        int block_end = csrRowPtr[(iter_plus > rowA) ? rowA : iter_plus];
        int num_window_nnzs = block_end - block_start;

        if (num_window_nnzs <= 0)
        {
            chunkPtr[windowId] = 0;
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

        // 生成chunkPtr：每个窗口中的TC_block数量
        int partition = (clean_edges2col.size() + fragSize_w - 1) / fragSize_w;
        // chunkPtr[windowId] = partition;
        // block_counter += partition;
        block_counter += partition;
        chunkPtr[windowId + 1] = block_counter;

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

        // 释放内存
        free(neighbor_window);
    }

    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 4 * 8);
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

    memset(dcscVal, 0, sizeof(valT) * nnzColD);
    memset(dcscRowInd, 0, sizeof(indT) * nnzColD);
    memset(dcscColPtr, 0, sizeof(indT) * (dCols + 1));

    memset(scscVal, 0, sizeof(valT) * nnzColS);
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

    memset(csrVal_dd, 0, sizeof(valT) * nnzRowD);
    memset(csrColInd_dd, 0, sizeof(indT) * nnzRowD);
    memset(csrRowPtr_dd, 0, sizeof(indT) * (dRows + 1));

    memset(csrVal_ds, 0, sizeof(valT) * nnzRowS);
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
    // TODO: chunkPtr and nec_num not check
    int *ecrId = (int *)malloc(sizeof(int) * nnzRowD);
    memset(ecrId, 0, sizeof(int) * nnzRowD);
    int chunkNum = ceil(dRows / fragM);
    int *chunkPtr = (int *)malloc(sizeof(int) * (chunkNum + 1));
    memset(chunkPtr, 0, sizeof(int) * (chunkNum + 1));

    int **use_x_id = (int **)malloc((chunkNum + 1) * sizeof(int *));
    int *nec_num = (int *)malloc((chunkNum + 1) * sizeof(int));
    ecrPreprocess(csrColInd_dd, csrRowPtr_dd, dRows, dCols, fragM, fragK, chunkPtr, ecrId, use_x_id, nec_num);
    printf("TC_ratio = %f\n", (float)nnzRowD / (float)(chunkPtr[chunkNum] * 4 * 8));

    /***************************************************************
     *         check the Sparsity-TCU-aware compression            *
     ***************************************************************/

    valT *X_val = (valT *)malloc(sizeof(valT) * colA);
    initVec(X_val, colA);

    valT *ourY_val = (valT *)malloc(sizeof(valT) * rowA);
    valT *Y_val = (valT *)malloc(sizeof(valT) * rowA);

    memset(ourY_val, 0, sizeof(valT) * rowA);
    memset(Y_val, 0, sizeof(valT) * rowA);

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
    spmv_fp64_serial(csrVal, csrRowPtr, csrColInd, X_val, Y_val, rowA, colA, nnzA);



    // Peripheral-Sparse Block
    spmv_fp64_serial(scsrVal, scsrRowPtr, scsrColInd, x_s, ourY_val, rowA, sCols, nnzColS);
    // Edge-Sparse Block
    spmv_fp64_serial_(csrVal_ds, csrRowPtr_ds, csrColInd_ds, x_d, ourY_val, sRows, dCols, nnzRowS, newArray_);
    // Core-Dense Block
    spmv_fp64_serial_ecr(csrVal_dd, csrRowPtr_dd, csrColInd_dd, x_d, ourY_val, dRows, dCols, nnzRowD, rId, ecrId, use_x_id);
    // spmv_fp64_serial(dcsrVal, dcsrRowPtr, dcsrColInd, x_d, ourY_val, rowA, dCols, nnzColD);
    // spmv_fp64_serial_(csrVal_dd, csrRowPtr_dd, csrColInd_dd, x_d, ourY_val, dRows, dCols, nnzRowD, rId);

    int result = eQcheck(Y_val, ourY_val, rowA);

    free(nec_num);
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

    return 0;
}
