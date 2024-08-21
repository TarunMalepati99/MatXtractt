
// cusparseCsr2cscEx2
#include "csr2csc.h"
//TODO: csc格式需要在这里边分配内存，二级指针，参考mmio_all
void csr2csc(valT *csrVal, indT *csrRowPtr, int *csrColInd, int rowA, int colA, indT nnzA,
             valT **cscVal, indT **cscColPtr, indT **cscRowInd)
{
    struct timeval t1, t2;

    valT *d_csrVal;
    int *d_csrColInd;
    indT *d_csrRowPtr;

    valT *d_cscVal;
    indT *d_cscColPtr;
    indT *d_cscRowInd;

    valT *cscVal_alias = (valT *)malloc(nnzA * sizeof(valT));
    indT *cscColPtr_alias = (indT *)malloc((colA + 1) * sizeof(indT));
    indT *cscRowInd_alias = (indT *)malloc(nnzA * sizeof(indT));

    valT alpha = 1.0, beta = 0.0;

    cudaMalloc((void **)&d_csrVal, sizeof(valT) * nnzA);
    cudaMalloc((void **)&d_csrColInd, sizeof(int) * nnzA);
    cudaMalloc((void **)&d_csrRowPtr, sizeof(indT) * (rowA + 1));


    cudaMalloc((void **)&d_cscVal, sizeof(valT) * nnzA);
    cudaMalloc((void **)&d_cscColPtr, sizeof(indT) * (colA + 1));
    cudaMalloc((void **)&d_cscRowInd, sizeof(indT) * nnzA);


    cudaMemcpy(d_csrVal, csrVal, sizeof(valT) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, csrColInd, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtr, csrRowPtr, sizeof(indT) * (rowA + 1), cudaMemcpyHostToDevice);

    cudaMemset(d_cscVal, 0, sizeof(valT) * nnzA);
    cudaMemset(d_cscColPtr, 0, sizeof(indT) * (colA + 1));
    cudaMemset(d_cscRowInd, 0, sizeof(indT) * nnzA);

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    gettimeofday(&t1, NULL);

    cusparseCreate(&handle);
    // cusparseCreateCsr(&matA, rowA, colA, nnzA, d_csrRowPtr, d_csrColInd, d_csrVal,
    //                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                         &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
    //                         CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    cusparseCsr2cscEx2_bufferSize(handle,
                                  rowA,
                                  colA,
                                  nnzA,
                                  d_csrVal,
                                  d_csrRowPtr,
                                  d_csrColInd,
                                  d_cscVal,
                                  d_cscColPtr,
                                  d_cscRowInd,
                                  CUDA_R_64F,
                                  CUSPARSE_ACTION_SYMBOLIC,
                                  CUSPARSE_INDEX_BASE_ZERO,
                                  CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                  &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cusparseCsr2cscEx2(handle,
                       rowA,
                       colA,
                       nnzA,
                       d_csrVal,
                       d_csrRowPtr,
                       d_csrColInd,
                       d_cscVal,
                       d_cscColPtr,
                       d_cscRowInd,
                       CUDA_R_64F,
                       CUSPARSE_ACTION_SYMBOLIC,
                       CUSPARSE_INDEX_BASE_ZERO,
                       CUSPARSE_CSR2CSC_ALG_DEFAULT,
                       dBuffer);

        // cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        //              &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
        //              CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

    cudaDeviceSynchronize();
    

    // cusparseDestroySpMat(matA);
    // cusparseDestroyDnVec(vecX);
    // cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    cudaMemcpy(cscVal_alias, d_cscVal, sizeof(valT) * nnzA, cudaMemcpyDeviceToHost);
    cudaMemcpy(cscColPtr_alias, d_cscColPtr, sizeof(indT) * (colA + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(cscRowInd_alias, d_cscRowInd, sizeof(indT) * nnzA, cudaMemcpyDeviceToHost);

    *cscVal = cscVal_alias;
    *cscColPtr = cscColPtr_alias;
    *cscRowInd = cscRowInd_alias;

    cudaFree(d_csrVal);
    cudaFree(d_csrColInd);
    cudaFree(d_csrRowPtr);
    cudaFree(d_cscVal);
    cudaFree(d_cscColPtr);
    cudaFree(d_cscRowInd);
}
