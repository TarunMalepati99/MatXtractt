// TODO: CSR如何快速拆分为iCSR与rCSR。

#include "mmio.h"
#include "csr2csc.h"

// const float colProp = 0.82;
const float rowProp = 0.8;
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

    // valT *csrVal_;
    // indT *csrColInd_;
    // indT *csrRowPtr_;

    char *filename;
    filename = argv[1];
    printf("\n===%s===\n\n", filename);

    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtr, &csrColInd, &csrVal, filename);
    initVec(csrVal, nnzA);
    //-------------------------------------------------------------------//
    // generate dense rId

    csr2csc(csrVal, csrRowPtr, csrColInd, rowA, colA, nnzA,
            &cscVal, &cscColPtr, &cscRowInd);
    /*
    // csc2csr(cscVal, cscColPtr, cscRowInd, colA, rowA, nnzA,
    //         &csrVal_, &csrRowPtr_, &csrColInd_);

    // valT *X_val = (valT *)malloc(sizeof(valT) * colA);
    // initVec(X_val, colA);

    // valT *cscY_val = (valT *)malloc(sizeof(valT) * rowA);
    // valT *Y_val = (valT *)malloc(sizeof(valT) * rowA);
    // memset(cscY_val, 0, sizeof(int) * rowA);
    // memset(Y_val, 0, sizeof(int) * rowA);

    // spmv_fp64_serial(csrVal, csrRowPtr, csrColInd, X_val, Y_val, rowA, colA, nnzA);
    // spmv_fp64_serial_csc(cscVal, cscColPtr, cscRowInd, X_val, cscY_val, rowA, colA, nnzA);
    // int result = eQcheck(Y_val, cscY_val, rowA);
    */

    /*
    // check
    for(int i = 0; i < nnzA; i++)
    {
        if(csrVal_[i] != csrVal[i])
        {
            printf(" csrVal failure \n");
        }
        if(csrColInd_[i] != csrColInd[i])
        {
            printf(" csrColInd failure \n");
        }
    }
    for(int i = 0; i <= rowA; i++)
    {
        if(csrRowPtr_[i] != csrRowPtr[i])
        {
            printf(" csrRowPtr failure \n");
        }
    }
    */

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
    // 稠密列部分包含nnzColD个非零元，dCols个列
    //-------------------------------------------------------------------//
    // 将分为两个csc格式
    int nnzColS = nnzA - nnzColD;// TODO: 问题
    int sCols = colA - dCols;
    // printf("dCols = %d and sCols = %d and colA = %d", dCols, sCols, colA);

    valT *dcscVal = (valT *)malloc(nnzColD * sizeof(valT));
    indT *dcscColPtr = (indT *)malloc((dCols + 1) * sizeof(indT));
    indT *dcscRowInd = (indT *)malloc(nnzColD * sizeof(indT));

    valT *scscVal = (valT *)malloc(nnzColS * sizeof(valT));
    indT *scscColPtr = (indT *)malloc((sCols + 1) * sizeof(indT));
    indT *scscRowInd = (indT *)malloc(nnzColS * sizeof(indT));
    // printf(" %d %d | %d %d \n", nnzColD, dCols, nnzColS, sCols);

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
    /*
    // Create a bitmap to mark present values
    HashTable *colHash = (HashTable *)malloc(sizeof(HashTable) * colA);
    for (int i = 0; i < colA; i++)
    {
        colHash[i].index = i;
        colHash[i].isIn = false;
    }
    // Mark values present in descColId
    for (int i = 0; i < dCols; i++)
    {
        int id = descColId[i].index;
        colHash[id].isIn = true;
    }
    // Find missing values and add to newArray
    int *newArray = (int *)malloc(sizeof(int) * sCols);
    memset(newArray, 0, sizeof(int) * sCols);
    int newArraySize = 0;

    for (int i = 0; i < colA; i++)
    {
        if (!(colHash[i].isIn))
        {
            // newArray[newArraySize] = i;
            // newArraySize++;
            if (newArraySize < sCols)
            {
                newArray[newArraySize] = i;
                newArraySize++;
            }
            else
            {
                printf("Error: newArray size exceeded!\n");
                break;
            }
        }
    }
    */
    
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
        if (index < colA) {
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

    valT *X_val = (valT *)malloc(sizeof(valT) * colA);
    initVec(X_val, colA);

    valT *cscY_val = (valT *)malloc(sizeof(valT) * rowA);
    valT *Y_val = (valT *)malloc(sizeof(valT) * rowA);

    memset(cscY_val, 0, sizeof(valT) * rowA);
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

    spmv_fp64_serial(csrVal, csrRowPtr, csrColInd, X_val, Y_val, rowA, colA, nnzA);

    spmv_fp64_serial_csc(dcscVal, dcscColPtr, dcscRowInd, x_d, cscY_val, rowA, dCols, nnzColD);
    spmv_fp64_serial_csc(scscVal, scscColPtr, scscRowInd, x_s, cscY_val, rowA, sCols, nnzColS);

    int result = eQcheck(Y_val, cscY_val, rowA);
    //-------------------------------------------------------------------//
    /*
    // 行维度的压缩
    CountWithIndex *descRowId = (CountWithIndex *)malloc(sizeof(CountWithIndex) * rowA);
    memset(descRowId, 0, sizeof(CountWithIndex) * rowA);
    for (int i = 0; i < rowA; i++)
    {
        descRowId[i].count = csrRowPtr[i + 1] - csrRowPtr[i];
        descRowId[i].index = i;
    }
    qsort(descRowId, rowA, sizeof(CountWithIndex), compare_desc_structure);
    // int avgColNnz = (int)(nnzA / rowA);
    // float max_mean = (float)(descRowId[1].count + descRowId[2].count + descRowId[3].count) / (float)(avgColNnz * 3);
    // printf("max_mean = %f\n", max_mean);
    // int dRows = (int)rowA / 10;
    // int dRows = 0;
    int dRowsNnz = nnzA * rowProp;
    int acc2 = 0, rowThreshNum;
    for (int i = 0; i < rowA; i++)
    {
        acc2 += descRowId[i].count;
        if (acc2 >= dRowsNnz)
        {
            rowThreshNum = i;
            break;
        }
    }
    printf("row_nnz_ratio = %f\n", (float)acc2 / (float)nnzA);
    printf("rows_ratio = %f \n", (float)rowThreshNum / (float)rowA);
    */
    /*
    //----------------------------2nd  compress----------------------------//
    // int dRows =  (int)((rowA * 0.2) / (double)fragM) * fragM;
    //ECR操作
    int numStripe = rowA / fragM;
    int *strpBuffer = (int *)malloc(sizeof(int) * colA);
    int *strpNEC = (int *)malloc(sizeof(int) * numStripe);
    int *TCnnzs = (int *)malloc(sizeof(int) * numStripe);
    memset(strpNEC, 0, sizeof(int) * numStripe);
    memset(TCnnzs, 0, sizeof(int) * numStripe);

    for (int s = 0; s < numStripe; s++)
    {
        memset(strpBuffer, 0, sizeof(int) * colA);
        int firstRow = s * fragM;
        int lastRow = firstRow + fragM;
        for(int r = firstRow; r < lastRow; r++)
        {
            TCnnzs[s] += descRowId[r].count;
            // 对于stripe的每一行
            for(int j = csrRowPtr[descRowId[r].index]; j < csrRowPtr[descRowId[r].index + 1]; j++)
            {
                for(int k = 0; k < dCols; k++)
                {
                    if(descColId[k].index == csrColInd[j])
                    {
                        strpBuffer[k] = 1;
                        break;
                    }
                }
            }
        }
        strpNEC[s] = sum_array(strpBuffer, colA);
        // printf("\n [%d] row nnzs = %d \n", i, descRowId[i].count);
        // printf("\n [%d] mean nnz = %d \n",i, (descColId[i].count));
    }
    // float ratioAll = (float)TCnnzs / (float)(nnzA);
    // printf("\n ratioAll = %f", ratioAll);
    float ss_sum = 0;
    for(int s = 0; s < numStripe; s++)
    {
        ss_sum += (float)TCnnzs[s] / (float)(strpNEC[s] * fragM);
    }
    printf("\n ss       = %f", ss_sum / numStripe);
    // printf("\n nnzA = %d", nnzA);
    int dense_nnzs = 0;
    for (int i = 0; i < dCols; i++)
    {
        dense_nnzs += descColId[i].count;
    }
    double dense_ratio = (double)dense_nnzs / (double)nnzA;
    printf("\n ratioDenseCol = %lf \n", dense_ratio);
    // printf("\n r = %lf \n", ratioAll/dense_ratio);
    */
    // free(descRowId);
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
    free(cscY_val);
    free(X_val);

    return 0;
}