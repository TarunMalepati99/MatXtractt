#include "mmio.h"
#include "csr2csc.h"

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

void SF_all(char *filename, indT *csrRowPtr, int *csrColInd,
            int rowA, int colA, indT nnzA, int *count_colId)
{
    // int *count_colId = (int *)malloc(sizeof(int) * colA);
    // memset(count_colId, 0, sizeof(int) * colA);
    for (int i = 0; i < nnzA; i++)
    {
        count_colId[csrColInd[i]]++;
    }
    qsort(count_colId, colA, sizeof(int), compare_desc);

    int dense_part = (int)colA / 10;
    int dense_nnzs = 0;
    for (int i = 0; i < dense_part; i++)
    {
        dense_nnzs += count_colId[i];
    }
    double dense_ratio = (double)dense_nnzs / (double)nnzA;

    printf("\n dense ratio = %lf \n", dense_ratio);
}

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
    int *csrColInd;
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

    int *count_colId = (int *)malloc(sizeof(int) * colA);
    int *count_colId_alias = (int *)malloc(sizeof(int) * colA);

    memset(count_colId, 0, sizeof(int) * colA);
    memset(count_colId_alias, 0, sizeof(int) * colA);
    for (int i = 0; i < colA; i++)
    {
        count_colId[i] = cscColPtr[i + 1] - cscColPtr[i];
    }
    qsort(count_colId, colA, sizeof(int), compare_desc);

    int dense_part = (int)colA / 10;
    int dense_nnzs = 0;
    for (int i = 0; i < dense_part; i++)
    {
        dense_nnzs += count_colId[i];
    }
    double dense_ratio = (double)dense_nnzs / (double)nnzA;

    printf("\n dense ratio CSC test = %lf \n", dense_ratio);

    printf("INIT DONE\n");

    SF_all(filename, csrRowPtr, csrColInd, rowA, colA, nnzA, count_colId_alias);

    eQcheck(count_colId, count_colId_alias, colA);
    
    
    free(count_colId);
    free(count_colId_alias);
    free(csrColInd);
    free(csrRowPtr);
    free(csrVal);

    return 0;
}