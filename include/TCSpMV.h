#include "common.h"

void cdspmv(char *filename, valT *csrVal, indT *csrRowPtr, indT *csrColInd,
                          valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA,
                          double *necTime, double *necPre);

void tcspmv_fp64(indT *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
            std::vector<double> tcVal, indT *sparse_AToX_index, valT *X_val,
            valT *Y_val, int rowA, int colA, int *row_order, double *necTime, double *necPre);

void tcspmv_fp16_v0(int *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
                 std::vector<half> tcVal, int *sparse_AToX_index, half *X_val,
                 half *Y_val, int rowA, int colA, int *row_order, double *necTime, double *necPre);

void tcspmv_fp16_v1(int *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
                 std::vector<half> tcVal, int *sparse_AToX_index, half *X_val,
                 half *Y_val, int rowA, int colA, int *row_order, double *necTime, double *necPre);

void se_tcspmv_fp16(valT *csrValA, indT *csrRowPtrA, int *csrColIdxA, 
                      valT *X_val, valT *Y_val, int *order_rid, int rowA, int colA, indT nnzA, int NUM, double threshold, int block_longest);