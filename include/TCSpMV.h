#include "common.h"

void necspmv(char *filename, valT *csrVal, indT *csrRowPtr, indT *csrColInd,
                          valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA,
                          double *necTime, double *necPre);

void tcspmv_fp64(indT *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
            std::vector<double> tcVal, indT *sparse_AToX_index, valT *X_val,
            valT *Y_val, int rowA, int colA, int *row_order, double *necTime, double *necPre);

void tcspmv_fp16(int *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
                 std::vector<half> tcVal, int *sparse_AToX_index, half *X_val,
                 half *Y_val, int rowA, int colA, int *row_order, double *necTime, double *necPre);

void tcspmv_fp16_(int *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
                 std::vector<half> tcVal, int *sparse_AToX_index, half *X_val,
                 half *Y_val, int rowA, int colA, int *row_order, double *necTime, double *necPre);