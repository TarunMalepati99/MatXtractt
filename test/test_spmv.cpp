#include "TCSpMV.h"
#include "mmio.h"
//-------------------------------------------------------------------------------
int resCompare(valT *our_val, valT *cuda_val, int length)
{
    for (int i = 0; i < length; i++)
    {
        if (fabs(our_val[i] - cuda_val[i]) > 1e-5)
        {
            printf("error in (%d), cusp(%4.2f), cuda(%4.2f),please check your code!\n", i, our_val[i], cuda_val[i]);
            return -1;
        }
    }
    printf("Y(%d), compute succeed!\n", length);
    return 0;
}

void cusparse_spmv_all(valT *cu_ValA, indT *cu_RowPtrA, int *cu_ColIdxA,
                       valT *cu_ValX, valT *cu_ValY, int rowA, int colA, indT nnzA,
                       long long int data_origin1, long long int data_origin2, double *cu_time, double *cu_gflops, double *cu_bandwidth1, double *cu_bandwidth2, double *cu_pre)
{
    struct timeval t1, t2;

    valT *dA_val, *dX, *dY;
    int *dA_cid;
    indT *dA_rpt;
    valT alpha = 1.0, beta = 0.0;

    cudaMalloc((void **)&dA_val, sizeof(valT) * nnzA);
    cudaMalloc((void **)&dA_cid, sizeof(int) * nnzA);
    cudaMalloc((void **)&dA_rpt, sizeof(indT) * (rowA + 1));
    cudaMalloc((void **)&dX, sizeof(valT) * colA);
    cudaMalloc((void **)&dY, sizeof(valT) * rowA);

    cudaMemcpy(dA_val, cu_ValA, sizeof(valT) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(indT) * (rowA + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, cu_ValX, sizeof(valT) * colA, cudaMemcpyHostToDevice);
    cudaMemset(dY, 0.0, sizeof(valT) * rowA);

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    gettimeofday(&t1, NULL);
    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, rowA, colA, nnzA, dA_rpt, dA_cid, dA_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, colA, dX, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, rowA, dY, CUDA_R_64F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    // cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double cusparse_pre = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    // printf("cusparse preprocessing time: %8.4lf ms\n", cusparse_pre);
    *cu_pre = cusparse_pre;
    int warp_iter = 100;
    int test_iter = 1000;

    for (int i = 0; i < warp_iter; ++i)
    {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                     CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t1, NULL);
    cuda_time_test_start();
    for (int i = 0; i < test_iter; ++i)
    {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                     CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    cuda_time_test_end();
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double runtime = (elapsedTime) / test_iter;
    // double gflops = (2.0 * matA_csr->nnz) / ((runtime / 1000) * 1e9);
    printf("\n CUSPARSE CUDA kernel runtime = %g ms\n", runtime);
    *cu_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / test_iter;
    *cu_gflops = (double)((long)nnzA * 2) / (*cu_time * 1e6);
    *cu_bandwidth1 = (double)data_origin1 / (*cu_time * 1e6);
    *cu_bandwidth2 = (double)data_origin2 / (*cu_time * 1e6);
    // printf("cusparse:%8.4lf ms, %8.4lf Gflop/s, %9.4lf GB/s, %9.4lf GB/s\n", *cu_time, *cu_gflops, *cu_bandwidth1, *cu_bandwidth2);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    cudaMemcpy(cu_ValY, dY, sizeof(valT) * rowA, cudaMemcpyDeviceToHost);

    cudaFree(dA_val);
    cudaFree(dA_cid);
    cudaFree(dA_rpt);
    cudaFree(dX);
    cudaFree(dY);
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
    indT *csrColInd;
    indT *csrRowPtr;

    char *filename;
    filename = argv[1];

    printf("\n===%s===\n\n", filename);

    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtr, &csrColInd, &csrVal, filename);
    valT *X_val = (valT *)malloc(sizeof(valT) * colA);
    initVec(X_val, colA);
    initVec(csrVal, nnzA);

    printf("INIT DONE\n");

    valT *cuY_val = (valT *)malloc(sizeof(valT) * rowA);
    valT *Y_val = (valT *)malloc(sizeof(valT) * rowA);

    double cu_time = 0, cu_gflops = 0, cu_bandwidth1 = 0, cu_bandwidth2 = 0, cu_pre = 0;
    long long int data_origin1 = (nnzA + colA + rowA) * sizeof(valT) + nnzA * sizeof(int) + (rowA + 1) * sizeof(indT);
    long long int data_origin2 = (nnzA + nnzA + rowA) * sizeof(valT) + nnzA * sizeof(int) + (rowA + 1) * sizeof(indT);

    cusparse_spmv_all(csrVal, csrRowPtr, csrColInd, X_val, cuY_val, rowA, colA, nnzA, data_origin1, data_origin2, &cu_time, &cu_gflops, &cu_bandwidth1, &cu_bandwidth2, &cu_pre);
    printf("cusparse end\n");
    double necTime = 0, necPre = 0;
    necspmv(filename, csrVal, csrRowPtr, csrColInd, X_val, Y_val, rowA, colA, nnzA, &necTime, &necPre);
    // spmv_fp64_serial(csrVal, csrRowPtr, csrColInd, X_val, Y_val, rowA, colA, nnzA);

    int iter = (int)((necPre - cu_pre) / (cu_time - necTime));

    printf("our_perf:    %8.4lf ms, our_pre:%8.4lf ms\n", necTime, necPre);
    printf("cusparse_perf:%8.4lf ms, cusparse_pre:%8.4lf ms\n", cu_time, cu_pre);
    // printf("\n iterate= %d\n", iter);

    // FILE *fout;
    // fout = fopen("data/spmv_f64_record.csv", "a");
    // fprintf(fout, "%lld,%lf,%lf,%lf,%lf\n", data_origin1, cu_time, cu_gflops, cu_bandwidth1, cu_bandwidth2);
    // fclose(fout);

    /* verify the result with cusparse */
    int result = resCompare(cuY_val, Y_val, rowA);

    free(X_val);
    free(Y_val);
    free(cuY_val);
    free(csrColInd);
    free(csrRowPtr);
    free(csrVal);

    return 0;
}
