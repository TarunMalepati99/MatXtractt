#include "common.h"

void necspmv(char *filename, valT *csrVal, indT *csrRowPtr, indT *csrColInd,
                          valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA,
                          double *necTime, double *necPre);

void tcspmv(indT *chunkPtr, std::vector<int> fragPtr, std::vector<uint32_t> fragBit,
            std::vector<double> tcVal, indT *sparse_AToX_index, valT *X_val,
            valT *Y_val, int rowA, int colA, int *row_order, double *necTime, double *necPre);