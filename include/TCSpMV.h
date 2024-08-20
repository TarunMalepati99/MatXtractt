#include "common.h"

void necspmv(char *filename, MAT_VAL_TYPE *csrVal, MAT_PTR_TYPE *csrRowPtr, int *csrColInd,
                          MAT_VAL_TYPE *X_val, MAT_VAL_TYPE *Y_val, int rowA, int colA, MAT_PTR_TYPE nnzA,
                          double *necTime, double *necPre);