/* 
*   cusparse
*
*   See https://docs.nvidia.com/cuda/cusparse/#cusparsecsr2cscex2 for details.
*
*
*/
#include "common.h"

void csr2csc(valT *csrVal, indT *csrRowPtr, indT *csrColInd, int rowA, int colA, indT nnzA,
             valT **cscVal, indT **cscColPtr, indT **cscRowInd);