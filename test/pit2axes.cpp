// TODO: CSR如何快速拆分为iCSR与rCSR。

#include "mmio.h"
#include "csr2csc.h"

const float colProp = 0.82;
const float rowProp = 0.61;

const int fragM = 8;
const int fragK = 4;
bool isDenseTC = 0;
typedef struct
{
    int count;
    int index;
} CountWithIndex;

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
    //-------------------------------------------------------------------//
    //-------------------------------------------------------------------//
    //-------------------------------------------------------------------//
    // generate dense cId and rId
    csr2csc(csrVal, csrRowPtr, csrColInd, rowA, colA, nnzA,
            &cscVal, &cscColPtr, &cscRowInd);

    CountWithIndex *descColId = (CountWithIndex *)malloc(sizeof(CountWithIndex) * colA);
    memset(descColId, 0, sizeof(CountWithIndex) * colA);
    int *originColId = (int *)malloc(sizeof(int) * colA);
    memset(originColId, 0, sizeof(int) * colA);

    for (int i = 0; i < colA; i++)
    {
        descColId[i].count = cscColPtr[i + 1] - cscColPtr[i];
        originColId[i] = descColId[i].count;
        descColId[i].index = i;
    }
    qsort(descColId, colA, sizeof(CountWithIndex), compare_desc_structure);
    int avgColNnz = (int)(nnzA / colA);
    // float max_mean = (float)(descColId[1].count + descColId[2].count + descColId[3].count) / (float)(avgColNnz * 3);
    // printf("max_mean = %f\n", max_mean);
    // int dCols = (int)colA / 10;
    int dCols = 0;
    int dColsNnz = nnzA * colProp;
    int dRowsNnz = nnzA * rowProp;
    int acc = 0, colThreshNum, colThreshNnz, rowThreshNum, rowThreshNnz;
    for (int i = 0; i < colA; i++)
    {
        acc += descColId[i].count;
        if (acc >= dColsNnz)
        {
            colThreshNum = i;
            colThreshNnz = descColId[i].count;
            break;
        }
    }

    int *cId = (int *)malloc(sizeof(int) * colThreshNum);
    memset(cId, 0, sizeof(int) * colThreshNum);
    int ci = 0;
    int accc1 = 0;
    for (int i = 0; i < colA; i++)
    {
        if (originColId[i] > colThreshNnz)
        {
            cId[ci++] = i;
            accc1 += originColId[i];
        }
    }
    // for (int i = 0; i < 20; i++)
    // {
    //     printf(" cId[%d] = %d \n", i, cId[i]);
    // }
    printf("accc1 = %f\n", (float)accc1 / (float)nnzA);
    printf("ci = %f \n", (float)ci / (float)colA);
    CountWithIndex *descRowId = (CountWithIndex *)malloc(sizeof(CountWithIndex) * rowA);
    for (int i = 0; i < rowA; i++)
    {
        descRowId[i].count = 0;
        descRowId[i].index = i;
    }
    int *originRowId = (int *)malloc(sizeof(int) * rowA);
    memset(originRowId, 0, sizeof(int) * rowA);

    for (int c = 0; c < ci; c++)
    {
        for (int r = cscColPtr[cId[c]]; r < cscColPtr[cId[c] + 1]; r++)
        {

            descRowId[cscRowInd[r]].count += 1;
            // cscRowInd[row, descRowId[cscRowInd[r]].count]
        }
    }
    for (int i = 0; i < rowA; i++)
    {
        originRowId[i] = descRowId[i].count;
    }
    qsort(descRowId, rowA, sizeof(CountWithIndex), compare_desc_structure);
    acc = 0;
    for (int i = 0; i < rowA; i++)
    {
        acc += descRowId[i].count;
        if (acc >= dRowsNnz)
        {
            rowThreshNum = i;
            rowThreshNnz = descRowId[i].count;
            break;
        }
    }
    int *rId = (int *)malloc(sizeof(int) * rowThreshNum);
    memset(rId, 0, sizeof(int) * rowThreshNum);
    int ri = 0;
    int accc2 = 0;
    for (int i = 0; i < rowA; i++)
    {
        if (originRowId[i] > rowThreshNnz)
        {
            rId[ri++] = i;
            accc2 += originRowId[i];
        }
    }

    // for (int i = 0; i < 20; i++)
    // {
    //     printf(" rId[%d] = %d \n", i, rId[i]);
    // }
    printf("accc2 = %f\n", (float)accc2 / (float)nnzA);
    printf("ri = %f \n", (float)ri / (float)rowA);

    //-------------------------------------------------------------------//
    //-------------------------------------------------------------------//
    //-------------------------------------------------------------------//
    // Convert CSR to dCSR and sCSR
    // rId[0...ri], cId[0...ci]

    int nnzD = accc2;
    int nnzS = nnzA - nnzD;
    // int nnzS = nnzA;
    int rowD = ri + 1;
    int colD = ci + 1;
    int rowS = rowA, colS = colA;

    valT *dcsrVal_alias = (valT *)malloc(nnzD * sizeof(valT));
    indT *dcsrRowPtr_alias = (indT *)malloc((rowD + 1) * sizeof(indT));
    indT *dcsrColInd_alias = (indT *)malloc(nnzD * sizeof(indT));

    valT *scsrVal_alias = (valT *)malloc(nnzS * sizeof(valT));
    indT *scsrRowPtr_alias = (indT *)malloc((rowS + 1) * sizeof(indT));
    indT *scsrColInd_alias = (indT *)malloc(nnzS * sizeof(indT));
    scsrRowPtr_alias[0] = 0;
    dcsrRowPtr_alias[0] = 0;
    int dprow = 0, dpcol, dpnnz = 0, spnnz = 0;

    for (int i = 0; i < rowA; i++)
    {
        dpcol = 0;
        int p = 0;
        // int q = csrRowPtr[i];
        if (i == rId[dprow])
        {
            // printf("i = %d and rId = %d \n",i,rId[dprow]);
            dprow++;
            int j = csrRowPtr[i];
            while (p < colD && j < csrRowPtr[i + 1])
            {
                if (cId[p] < csrColInd[j])
                {
                    p++;
                }
                else if (cId[p] == csrColInd[j])
                {

                    dcsrVal_alias[dpnnz] = csrVal[j];
                    dcsrColInd_alias[dpnnz] = p;
                    dpnnz++;
                    p++;
                    j++;
                }
                else
                {
                    scsrVal_alias[spnnz] = csrVal[j];
                    scsrColInd_alias[spnnz] = csrColInd[j];
                    spnnz++;
                    j++;
                }
            }
            while (j < csrRowPtr[i + 1])
            {
                scsrVal_alias[spnnz] = csrVal[j];
                scsrColInd_alias[spnnz] = csrColInd[j];
                spnnz++;
                j++;
            }
            dcsrRowPtr_alias[dprow] = dpnnz;
            // scsrRowPtr_alias[i + 1] = spnnz;
        }
        else
        {
            for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++)
            {
                scsrVal_alias[spnnz] = csrVal[j];
                scsrColInd_alias[spnnz] = csrColInd[j];
                spnnz++;
            }
        }
        scsrRowPtr_alias[i + 1] = spnnz;
    }
    // printf("\n dprow = %d and rowD = %d \n", dprow, rowD);
    // printf("\n dpnnz = %d and nnzD = %d \n", dpnnz, nnzD);
    printf("\n ----------------------------------------- \n");
    valT *X_val = (valT *)malloc(sizeof(valT) * colA);
    valT *X_val_d = (valT *)malloc(sizeof(valT) * colD);
    initVec(X_val, colA);
    for (int i = 0; i < colD; i++)
    {
        X_val_d[i] = X_val[cId[i]];
    }
    valT *pinY_val = (valT *)malloc(sizeof(valT) * rowA);
    valT *Y_val = (valT *)malloc(sizeof(valT) * rowA);
    memset(pinY_val, 0, sizeof(int) * rowA);
    memset(Y_val, 0, sizeof(int) * rowA);

    spmv_fp64_serial(csrVal, csrRowPtr, csrColInd, X_val, Y_val, rowA, colA, nnzA);
    spmv_fp64_serial_(dcsrVal_alias, dcsrRowPtr_alias, dcsrColInd_alias, X_val_d, pinY_val, rowD, colD, nnzD, rId);
    spmv_fp64_serial(scsrVal_alias, scsrRowPtr_alias, scsrColInd_alias, X_val, pinY_val, rowS, colS, nnzS);

    int result = eQcheck(Y_val, pinY_val, rowA);

    /*
    int dRows =  (int)((rowA * 0.2) / (double)fragM) * fragM;
    //ECR操作
    int numStripe = dRows / fragM;
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

    free(descRowId);
    free(descColId);
    free(csrColInd);
    free(csrRowPtr);
    free(csrVal);
    free(originColId);
    free(originRowId);
    free(pinY_val);
    free(Y_val);
    free(X_val_d);
    free(X_val);
    free(scsrColInd_alias);
    free(scsrRowPtr_alias);
    free(scsrVal_alias);
    free(dcsrColInd_alias);
    free(dcsrRowPtr_alias);
    free(dcsrVal_alias);
    free(rId);
    free(cId);
    /*
    free(strpBuffer);
    free(strpNEC);
    free(TCnnzs);
    */

    return 0;
}