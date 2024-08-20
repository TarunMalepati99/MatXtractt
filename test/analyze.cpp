#include "mmio.h"

int compare_desc(const void *a, const void *b)
{
    return (*(int *)b - *(int *)a);
}

void SF_all(char *filename, MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA,
            int rowA, int colA, MAT_PTR_TYPE nnzA)
{
    int *count_colId = (int *)malloc(sizeof(int) * colA);
    memset(count_colId, 0, sizeof(int) * colA);
    for (int i = 0; i < nnzA; i++)
    {
        count_colId[csrColIdxA[i]]++;
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

    free(count_colId);
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
    MAT_PTR_TYPE nnzA;
    int isSymmetricA;
    MAT_VAL_TYPE *csrValA;
    int *csrColIdxA;
    MAT_PTR_TYPE *csrRowPtrA;

    char *filename;
    filename = argv[1];

    printf("\n===%s===\n\n", filename);

    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtrA, &csrColIdxA, &csrValA, filename);
    MAT_VAL_TYPE *X_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * colA);
    initVec(X_val, colA);
    initVec(csrValA, nnzA);

    printf("INIT DONE\n");

    SF_all(filename, csrRowPtrA, csrColIdxA, rowA, colA, nnzA);

    free(csrColIdxA);
    free(csrRowPtrA);
    free(csrValA);

    return 0;
}