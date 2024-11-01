template <int BREAK_STRIDE, typename I>
__global__ void pre_startRowPerBlock(const I *__restrict__ row_ptr,
                                     const I m,
                                     I *__restrict__ startRowPerBlock);

__global__ void tcspmv_kernel_fp64(
    const double *__restrict__ x_d,
    double *__restrict__ y_d,
    const int *__restrict__ chunkPtr,
    const int *__restrict__ fragPtr,
    const uint32_t *__restrict__ fragBit,
    const double *__restrict__ tcVal,
    const int *__restrict__ sparse_AToX_index,
    int dRows,
    int dCols);


template <indT productNnzPerThread, indT THREADS_PER_BLOCK, typename I, typename T>
__global__ void cdspmv_kernel(T *d_val,
                              indT *d_ptr,
                              indT *d_cols,
                              indT rowA,
                              T *d_vector,
                              T *d_out,
                              I *__restrict__ startRowPerBlock);