#include "common.h"

void necspmv(char *filename, valT *csrVal, indT *csrRowPtr, int *csrColInd,
                          valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA,
                          double *necTime, double *necPre);