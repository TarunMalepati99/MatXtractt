#include "mmio.h"
#include "csr2csc.h"

const int fragM = 8;
const int fragK = 4;
bool isDenseTC = 0;
typedef struct
{
    int count;
    int index;
} CountWithIndex;

int sum_array(const int *arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += arr[i]; // 直接通过指针访问数组元素
    }
    return sum;
}

int compare_desc_structure(const void *a, const void *b)
{
    return ((CountWithIndex *)b)->count - ((CountWithIndex *)a)->count;
}

int eQcheck(int *tmp1, int *tmp2, int length)
{
    for (int i = 0; i < length; i++)
    {
        if (fabs(tmp1[i] - tmp2[i]) > 1e-5)
        {
            printf("error in (%d), cusp(%d), cuda(%d),please check your code!\n", i, tmp1[i], tmp2[i]);
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
/*
void SF_all(char *filename, indT *csrRowPtr, indT *csrColInd,
            int rowA, int colA, indT nnzA, int *descColId)
{
    int *descRowIdId = (int *)malloc(sizeof(int) * rowA);
    memset(descRowIdId, 0, sizeof(int) * rowA);
    for (int i = 0; i < rowA; i++)
    {
        descRowIdId[i] = csrRowPtr[i + 1] - csrRowPtr[i];
    }

    for (int i = 0; i < nnzA; i++)
    {
        descColId[csrColInd[i]]++;
    }
    qsort(descColId, colA, sizeof(int), compare_desc);
    qsort(descRowIdId, rowA, sizeof(int), compare_desc);

    int denseCol_part = (int)colA / 10;
    int dRows_part = (int)rowA / 10;
    int denseCol_nnzs = 0;
    int dRows_nnzs = 0;
    for (int i = 0; i < denseCol_part; i++)
    {
        denseCol_nnzs += descColId[i];
    }
    for (int i = 0; i < dRows_part; i++)
    {
        dRows_nnzs += descRowIdId[i];
    }
    double denseCol_ratio = (double)denseCol_nnzs / (double)nnzA;
    double dRows_ratio = (double)dRows_nnzs / (double)nnzA;

    printf("\n dense col ratio = %lf \n", denseCol_ratio);
    printf("\n dense row ratio = %lf \n", dRows_ratio);
}
*/

/*
    cols = 10%  定下colid
    8x4 选4个最大colid，其最多的8行  若nnzs/32 >   一半

    16x8 加4colid，更新行的count并排出最多的8行

    。。。


*/
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
    
    int dCols = (int)colA / 10;
    //descColId[0-dCols].index的每一列, ,取出来放到一个CSR的buffer中
    CountWithIndex *descRowId = (CountWithIndex *)malloc(sizeof(CountWithIndex) * rowA);
    // memset(descRowId, 0, sizeof(CountWithIndex) * rowA);
    for (int i = 0; i < rowA; i++)
    {
        descRowId[i].count = 0;
        descRowId[i].index = i;
    }
    /*
    //-----------------------------------
    int dilateM = fragM;
    int dilateK = fragK;
    for (int c = 0; c < dCols; c++)
    {
        for (int j = cscColPtr[descColId[c].index]; j < cscColPtr[descColId[c].index + 1]; j++)
        {
            descRowId[cscRowInd[j]].count += 1;
        }
        // 检查i是否为2的幂次减1
        if ((c & (c + 1)) == 0 && c >= 3)
        { // 这是检查i是否为2的幂次的一个常用技巧
            
            // descRowId is a structure array, including count and index.
            // write a high-performance function with C code, to select N max descRowId.count, and return corresponding N descRowId.index
            int rows = (c + 1) * 2;

            qsort(descRowId, rowA, sizeof(CountWithIndex), compare_desc_structure);
            printf("========== %d x %d =============", (c + 1) * 2, (c + 1));
            int TCnnzs = 0;
            for(int ii = 0; ii < rows; ii++)
            {
                // printf("\n  row_count = %d", descRowId[ii].count);
                TCnnzs += descRowId[ii].count;
            }
            float ratioTC = (float) TCnnzs / (float) ((c + 1) * rows);
            printf("\n ratioTC = %f", ratioTC);
            // 这里可以放置你的条件判断
            if (ratioTC < 0.5)
            {
                printf(" TC frag is not dense, breaking out of the loop.\n");
                break; // 如果条件为真，跳出循环
            }
        }
        // 可以在这里添加其他循环逻辑
    }
    //-----------------------------------
    */
    // malloc dCSR
    //dcolCSR, 第一列连续读，stride写到csr中。问题：count没确定，不知道stride多少。ELL格式
    //然后连续读csr的最多行到dCSR中。
    for (int c = 0; c < dCols; c++)
    {
        for (int r = cscColPtr[descColId[c].index]; r < cscColPtr[descColId[c].index + 1]; r++)
        {
            
            descRowId[cscRowInd[r]].count += 1;
            // cscRowInd[row, descRowId[cscRowInd[r]].count]处
        }
    }
    qsort(descRowId, rowA, sizeof(CountWithIndex), compare_desc_structure);
    // [dRows行处与所有dCols相交的，去掉] rowPtr好弄，colId，vals
    
    int dRows =  (int)((rowA * 0.2) / (double)fragM) * fragM;
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

    free(descRowId);
    free(descColId);
    free(csrColInd);
    free(csrRowPtr);
    free(csrVal);
    free(strpBuffer);
    free(strpNEC);
    free(TCnnzs);
    

    return 0;
}

/*
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Run the code by './spmv_double matrix.mtx'. \n");
        return 0;
    }

    // struct timeval t1, t2;
    int rowA, colA;
    indT nnzA;
    int isSymmetricA;
    valT *csrVal;
    indT *csrColInd;
    indT *csrRowPtr;

    char *filename;
    filename = argv[1];

    printf("\n===%s===\n\n", filename);

    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtr, &csrColInd, &csrVal, filename);

    valT *cscVal;
    indT *cscColPtr;
    indT *cscRowInd;

    initVec(csrVal, nnzA);

    csr2csc(csrVal, csrRowPtr, csrColInd, rowA, colA, nnzA,
            &cscVal, &cscColPtr, &cscRowInd);

    CountWithIndex *descColId = (CountWithIndex *)malloc(sizeof(CountWithIndex) * colA);
    int *descColId_alias = (int *)malloc(sizeof(int) * colA);

    memset(descColId, 0, sizeof(CountWithIndex) * colA);
    memset(descColId_alias, 0, sizeof(int) * colA);
    for (int i = 0; i < colA; i++)
    {
        descColId[i].count = cscColPtr[i + 1] - cscColPtr[i];
        descColId[i].index = i;
    }

    // qsort(descColId, colA, sizeof(int), compare_desc);
    qsort(descColId, colA, sizeof(CountWithIndex), compare_desc_structure);

    int *row_empty_flag = (int *)malloc(sizeof(int) * rowA);
    memset(row_empty_flag, 0, sizeof(int) * rowA);

    int dCols = (int)colA / 10;

    int *descRowId = (int *)malloc(sizeof(int) * rowA);
    memset(descRowId, 0, sizeof(int) * rowA);

    for (int c = 0; c < dCols; c++)
    {
        for(int r = cscColPtr[descColId[c].index]; r < cscColPtr[descColId[c].index + 1]; r++)
        {
            row_empty_flag[cscRowInd[r]] = 1;
            descRowId[cscRowInd[r]] += 1;
        }
    }
    int non_empty_row = 0;
    for(int i = 0; i < rowA; i ++)
    {
        non_empty_row += row_empty_flag[i];
        // printf("\n %d", row_empty_flag[i]);
    }

    // printf("empty row ratio = %f",(double)(rowA - non_empty_row)/rowA);
    qsort(descRowId, rowA, sizeof(int), compare_desc);
    for (int i = 0; i < 32; i++)
    {
        printf("\n [%d] row nnzs = %d \n",i, descRowId[i]);
        // printf("\n [%d] mean nnz = %d \n",i, (descColId[i].count));
    }

    int dense_nnzs = 0;
    for (int i = 0; i < dCols; i++)
    {
        dense_nnzs += descColId[i].count;
        // printf("\n [%d] mean nnz ratio = %f \n",i, (float)(descColId[i].count)/(float)rowA);
        // printf("\n [%d] mean nnz = %d \n",i, (descColId[i].count));
    }
    double dense_ratio = (double)dense_nnzs / (double)nnzA;
    double ratio =( (double)dense_nnzs / ((double)(dCols))) *100 / (double)(non_empty_row);

    // printf("\n dense ratio CSC test = %lf \n", dense_ratio);
    // printf("\n !!!!!ratio CSC test = %lf%% ", (100 - ratio));

    // printf("INIT DONE\n");

    SF_all(filename, csrRowPtr, csrColInd, rowA, colA, nnzA, descColId_alias);

    // eQcheck(descColId, descColId_alias, colA);


    free(descColId);
    free(descColId_alias);
    free(csrColInd);
    free(csrRowPtr);
    free(csrVal);

    return 0;
}
*/