
// cusparseCsr2cscEx2
#include "csr2csc.h"
void csr2csc(valT *csrVal, indT *csrRowPtr, indT *csrColInd, int rowA, int colA, indT nnzA,
             valT **cscVal, indT **cscColPtr, indT **cscRowInd)
{
    valT *d_csrVal;
    indT *d_csrColInd;
    indT *d_csrRowPtr;

    valT *d_cscVal;
    indT *d_cscColPtr;
    indT *d_cscRowInd;

    valT *cscVal_alias = (valT *)malloc(nnzA * sizeof(valT));
    indT *cscColPtr_alias = (indT *)malloc((colA + 1) * sizeof(indT));
    indT *cscRowInd_alias = (indT *)malloc(nnzA * sizeof(indT));

    valT alpha, beta;
#ifdef fp64
    alpha = 1.0;
    beta = 0.0;
#else
    alpha = __float2half(1.0f);
    beta = __float2half(0.0f);
#endif

    cudaMalloc((void **)&d_csrVal, sizeof(valT) * nnzA);
    cudaMalloc((void **)&d_csrColInd, sizeof(indT) * nnzA);
    cudaMalloc((void **)&d_csrRowPtr, sizeof(indT) * (rowA + 1));

    cudaMalloc((void **)&d_cscVal, sizeof(valT) * nnzA);
    cudaMalloc((void **)&d_cscColPtr, sizeof(indT) * (colA + 1));
    cudaMalloc((void **)&d_cscRowInd, sizeof(indT) * nnzA);

    cudaMemcpy(d_csrVal, csrVal, sizeof(valT) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, csrColInd, sizeof(indT) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtr, csrRowPtr, sizeof(indT) * (rowA + 1), cudaMemcpyHostToDevice);

    cudaMemset(d_cscVal, 0, sizeof(valT) * nnzA);
    cudaMemset(d_cscColPtr, 0, sizeof(indT) * (colA + 1));
    cudaMemset(d_cscRowInd, 0, sizeof(indT) * nnzA);

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    cusparseCreate(&handle);

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
                                  VAL_CUDA_R_TYPE,
                                  CUSPARSE_ACTION_NUMERIC,
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
                       VAL_CUDA_R_TYPE,
                       CUSPARSE_ACTION_NUMERIC,
                       CUSPARSE_INDEX_BASE_ZERO,
                       CUSPARSE_CSR2CSC_ALG_DEFAULT,
                       dBuffer);

    cudaDeviceSynchronize();

    cusparseDestroy(handle);

    cudaMemcpy(cscVal_alias, d_cscVal, sizeof(valT) * nnzA, cudaMemcpyDeviceToHost);
    cudaMemcpy(cscColPtr_alias, d_cscColPtr, sizeof(indT) * (colA + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(cscRowInd_alias, d_cscRowInd, sizeof(indT) * nnzA, cudaMemcpyDeviceToHost);

    *cscVal = cscVal_alias;
    *cscColPtr = cscColPtr_alias;
    *cscRowInd = cscRowInd_alias;

    cudaFree(dBuffer);
    cudaFree(d_csrVal);
    cudaFree(d_csrColInd);
    cudaFree(d_csrRowPtr);
    cudaFree(d_cscVal);
    cudaFree(d_cscColPtr);
    cudaFree(d_cscRowInd);
}

void csc2csr(valT *cscVal, indT *cscColPtr, indT *cscRowInd, int rowA, int colA, indT nnzA,
             valT **csrVal, indT **csrRowPtr, indT **csrColInd)
{
    valT *d_csrVal;
    indT *d_csrColInd;
    indT *d_csrRowPtr;
    valT *d_cscVal;
    indT *d_cscColPtr;
    indT *d_cscRowInd;

    valT *csrVal_alias = (valT *)malloc(nnzA * sizeof(valT));
    indT *csrRowPtr_alias = (indT *)malloc((rowA + 1) * sizeof(indT));
    indT *csrColInd_alias = (indT *)malloc(nnzA * sizeof(indT));

    valT alpha, beta;
#ifdef fp64
    alpha = 1.0;
    beta = 0.0;
#else
    alpha = __float2half(1.0f);
    beta = __float2half(0.0f);
#endif

    cudaMalloc((void **)&d_cscVal, sizeof(valT) * nnzA);
    cudaMalloc((void **)&d_cscColPtr, sizeof(indT) * (colA + 1));
    cudaMalloc((void **)&d_cscRowInd, sizeof(indT) * nnzA);

    cudaMalloc((void **)&d_csrVal, sizeof(valT) * nnzA);
    cudaMalloc((void **)&d_csrColInd, sizeof(indT) * nnzA);
    cudaMalloc((void **)&d_csrRowPtr, sizeof(indT) * (rowA + 1));

    cudaMemcpy(d_cscVal, cscVal, sizeof(valT) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowInd, cscRowInd, sizeof(indT) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscColPtr, cscColPtr, sizeof(indT) * (colA + 1), cudaMemcpyHostToDevice);

    cudaMemset(d_csrVal, 0, sizeof(valT) * nnzA);
    cudaMemset(d_csrRowPtr, 0, sizeof(indT) * (rowA + 1));
    cudaMemset(d_csrColInd, 0, sizeof(indT) * nnzA);

    cusparseHandle_t handle = NULL;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    cusparseCreate(&handle);

    cusparseCsr2cscEx2_bufferSize(handle,
                                  colA,
                                  rowA,
                                  nnzA,
                                  d_cscVal,
                                  d_cscColPtr,
                                  d_cscRowInd,
                                  d_csrVal,
                                  d_csrRowPtr,
                                  d_csrColInd,
                                  VAL_CUDA_R_TYPE,
                                  CUSPARSE_ACTION_NUMERIC,
                                  CUSPARSE_INDEX_BASE_ZERO,
                                  CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                  &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cusparseCsr2cscEx2(handle,
                       colA,
                       rowA,
                       nnzA,
                       d_cscVal,
                       d_cscColPtr,
                       d_cscRowInd,
                       d_csrVal,
                       d_csrRowPtr,
                       d_csrColInd,
                       VAL_CUDA_R_TYPE,
                       CUSPARSE_ACTION_NUMERIC,
                       CUSPARSE_INDEX_BASE_ZERO,
                       CUSPARSE_CSR2CSC_ALG_DEFAULT,
                       dBuffer);

    cudaDeviceSynchronize();

    cusparseDestroy(handle);

    cudaMemcpy(csrVal_alias, d_csrVal, sizeof(valT) * nnzA, cudaMemcpyDeviceToHost);
    cudaMemcpy(csrRowPtr_alias, d_csrRowPtr, sizeof(indT) * (rowA + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(csrColInd_alias, d_csrColInd, sizeof(indT) * nnzA, cudaMemcpyDeviceToHost);

    *csrVal = csrVal_alias;
    *csrRowPtr = csrRowPtr_alias;
    *csrColInd = csrColInd_alias;

    cudaFree(dBuffer);
    cudaFree(d_csrVal);
    cudaFree(d_csrColInd);
    cudaFree(d_csrRowPtr);
    cudaFree(d_cscVal);
    cudaFree(d_cscColPtr);
    cudaFree(d_cscRowInd);
}
