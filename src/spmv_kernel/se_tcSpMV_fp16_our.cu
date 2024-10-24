/*
// TODO:
#include "common.h"

// 前缀和计算函数，对输入数组进行排除最后一个元素的前缀和计算
void exclusive_scan(indT *input, int length)
{
    if (length == 0 || length == 1)
        return;

    indT old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

#define BlockSize 8
#define groupNum 1
#define warpNum_short 4
#define loopNum_short 4
#define warpNum_long 4
#define loopNum_long 2

#define MMA_M 8
#define MMA_N 8
#define MMA_K 4



__device__ __forceinline__ valT warpReduceSum(valT sum){
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

__device__ __forceinline__ void mma_m8n8k4_fp16(half *acc, half *frag_a, half *frag_b)
{
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_a[0]);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_b[0]);
    uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6, %7 }, "
        " { %0, %1, %2, %3 };"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]):
        "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
    ); 
}

__device__ __forceinline__ void mma_m8n8k4_fp16_v2(half *acc, uint32_t *A, half *frag_b)
{
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_b[0]);
    uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6, %7 }, "
        " { %0, %1, %2, %3 };"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]):
        "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
    ); 
}

__device__ __forceinline__ void mma_m8n8k4_fp16_v3(uint32_t *C, uint32_t *A, uint32_t *B)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6, %7 }, "
        " { %0, %1, %2, %3 };"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]):
        "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
    ); 
}


__device__ __forceinline__ int load_int_from_global(const int* a)
{
    int r;
    asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(r) : "l"(a));
    return r;
}

__device__ __forceinline__ uint32_t load_uint_from_global(const uint32_t* a)
{
    uint32_t r;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(r) : "l"(a));
    return r;
}

__device__ __forceinline__ half load_half_from_global(const half* a)
{
    ushort r;
    asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(r) : "l"(a));
    half *r_half = reinterpret_cast<half *>(&r);
    return *r_half;
}

__device__ __forceinline__ void store_half_to_global(const half* a, half v)
{
    ushort *v_u = reinterpret_cast<ushort *>(&v);
    asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(a), "h"(*v_u));
}

__global__ void longPart_sum(int *dlongA_rpt, valT *dwarp_val, uint32_t *dY_val, int row_long)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int laneid = 31 & tid;
    int global_warpid = bid * warpNum_long + (tid >> 5);

    if (global_warpid >= row_long) return;

    valT *valY_half = reinterpret_cast<valT *>(&dY_val[0]);

    int offset_longA = load_int_from_global(dlongA_rpt + global_warpid);
    valT *cur_temp_val = dwarp_val + offset_longA;
    int len = load_int_from_global(dlongA_rpt + global_warpid + 1) - offset_longA;

    valT thread_val = 0;
    for (int i = laneid; i < len; i += WARP_SIZE)
    {
        thread_val += load_half_from_global(cur_temp_val + i);
    }
    thread_val = warpReduceSum(thread_val);

    if (laneid == 0)
        store_half_to_global(valY_half + global_warpid, thread_val);
}


template <int rowloop>  // this parameter must be 1 or 2 or 4
__global__ void dasp_spmv2(uint32_t *dX_val, uint32_t *dY_val,
                          uint32_t *dlongA_val, int *dlongA_cid, valT *dwarp_val, indT *dlongA_rpt, int row_long,
                          uint32_t *dregA_val, int *dregA_cid, indT *dblockA_ptr, int row_block, int blocknum, 
                          uint32_t *dirregA_val, int *dirregA_cid, indT *dirregA_rpt,
                          uint32_t *dshort_val, int *dshort_cid, int short_row_1, int common_13, int total_short_rows_34, int short_row_2,
                          int offset_reg, int offset_short1, int offset_short13, int offset_short34, int offset_short22,
                          indT fill0_nnz_short13, indT fill0_nnz_short34, indT fill0_nnz_short22)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int laneid = 31 & tid;
    
    int row = laneid < 16 ? (laneid >> 2) * 8 + (3 & laneid) : ((laneid - 16) >> 2) * 8 + (3 & laneid) + 4;
    int idx = row * MMA_K;
    int idx_val = row * 2;

    int target_idx = laneid < 16 ? (3 & laneid) : (3 & laneid) + 4;
    int new_id = (7 & laneid) < 4 ? (laneid >> 3) * 4 + (3 & laneid) : (laneid >> 3) * 4 + (3 & laneid) + 16;

    valT const *valX_half = reinterpret_cast<valT const *>(&dX_val[0]);
    valT *valY_half = reinterpret_cast<valT *>(&dY_val[0]);

    if (bid < offset_reg)
    {
        // long part
        int global_warpid = bid * warpNum_long + (tid >> 5);

        uint32_t fragA[2];
        valT fragB[4], fragC[8], res;
        
        fragC[target_idx] = 0.0;
        
        #pragma unroll
        for (int i = 0; i < loopNum_long; i++)
        {
            int offset_cid = (global_warpid * loopNum_long + i) * MMA_M * MMA_K * 4;
            int offset_val = offset_cid >> 1;
            
            uint32_t *curA_val = dlongA_val + offset_val;
            int *curA_cid = dlongA_cid + offset_cid;

            fragA[0] = load_uint_from_global(curA_val + idx_val);
            fragA[1] = load_uint_from_global(curA_val + idx_val + 1);
            fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
            fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
            fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
            fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];  
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
        }
        res = fragC[target_idx];
        res = warpReduceSum(res);

        if (laneid == 0)  
            store_half_to_global(dwarp_val + global_warpid, res);

        // if (global_warpid >= row_long) return;

        // int offset_long = load_int_from_global(dlongA_rpt + global_warpid);
        // valT *cur_temp_val = dwarp_val + offset_long;
        // int len = load_int_from_global(dlongA_rpt + global_warpid + 1) - offset_long;

        // valT thread_val = 0;
        // for (int i = laneid; i < len; i += WARP_SIZE)
        // {
        //     thread_val += load_half_from_global(cur_temp_val + i);
        // }
        // thread_val = warpReduceSum(thread_val);

        // if (laneid == 0)
        //     store_half_to_global(valY_half + global_warpid, thread_val); 
    }
    else if (bid >= offset_reg && bid < offset_short1)
    {
        // row-block part
        int bid_reg = bid - offset_reg;
        int warp_local = tid >> 5;

        uint32_t fragA[2];
        valT fragB[4], fragC[8], res;

        valT *valA_irreg = reinterpret_cast<valT *>(&dirregA_val[0]);

        if (rowloop == 1)
        {
            fragC[target_idx] = 0.0;

            int block_idx = bid_reg * 4 + warp_local;
            int offset_A = load_int_from_global(dblockA_ptr + block_idx);
            int blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;

            for (int i = 0; i < blocklen; i += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + i) >> 1);
                int *curA_cid = dregA_cid + offset_A + i;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];  
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 

            int offset_y = block_idx * BlockSize + laneid;
            if (laneid < 8 && offset_y < row_block)
            {
                int offset_irreg = load_int_from_global(dirregA_rpt + offset_y);
                int offset_irreg1 = load_int_from_global(dirregA_rpt + offset_y + 1);
                for (int i = offset_irreg; i < offset_irreg1; i ++)
                {   
                    res += load_half_from_global(valA_irreg + i) * valX_half[load_int_from_global(dirregA_cid + i)];
                }
                store_half_to_global(valY_half + row_long + offset_y, res);
                // valY_half[row_long + offset_y] = res;
            }
        }

        if (rowloop == 2)
        {
            valT result;
            fragC[target_idx] = 0.0;

            int block_idx = bid_reg * 8 + warp_local * 2;
            int offset_A = load_int_from_global(dblockA_ptr + block_idx);
            int blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];   
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 
            if (laneid < 8) result =  res;

            // i = 1
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = load_int_from_global(dblockA_ptr + block_idx);
            blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];   
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_up_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 1) result = res;

            int cur_row = bid_reg * 8 * BlockSize + warp_local * 2 * BlockSize + laneid;
            if (laneid < 16 && cur_row < row_block)
            {
                int offset_irreg = load_int_from_global(dirregA_rpt + cur_row);
                int offset_irreg1 = load_int_from_global(dirregA_rpt + cur_row + 1);
                for (int i = offset_irreg; i < offset_irreg1; i ++)
                {
                    result += load_half_from_global(valA_irreg + i) * valX_half[load_int_from_global(dirregA_cid + i)];
                }
                store_half_to_global(valY_half + row_long + cur_row, result);
            }
        }

        if (rowloop == 4)
        {
            valT result;

            // i = 0
            fragC[target_idx] = 0.0;

            int block_idx = bid_reg * 16 + warp_local * 4;
            int offset_A = load_int_from_global(dblockA_ptr + block_idx);
            int blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)]; 
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 
            if (laneid < 8) result =  res;

            // i = 1
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = load_int_from_global(dblockA_ptr + block_idx);
            blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_up_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 1) result = res;

            // i = 2
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = load_int_from_global(dblockA_ptr + block_idx);
            blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_up_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 2) result = res;

            // i = 3
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = load_int_from_global(dblockA_ptr + block_idx);
            blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)]; 
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_up_sync(0xffffffff, res, 16);
            res += __shfl_up_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 3) result = res;

            int cur_row = bid_reg * 16 * BlockSize + warp_local * 4 * BlockSize + laneid;
            if (cur_row < row_block)
            {
                int offset_irreg = load_int_from_global(dirregA_rpt + cur_row);
                int offset_irreg1 = load_int_from_global(dirregA_rpt + cur_row + 1);
                for (int i = offset_irreg; i < offset_irreg1; i ++)
                {
                    result += load_half_from_global(valA_irreg + i) * valX_half[load_int_from_global(dirregA_cid + i)];
                }
                store_half_to_global(valY_half + row_long + cur_row, result);
            }
        }
    }
    else if (bid >= offset_short1 && bid < offset_short13)
    // if (bid >= offset_short1 && bid < offset_short13)
    {
        // short part - 1 nnz/row
        int bid1 = bid - offset_short1;
        int tid1 = bid1 * blockDim.x + tid;
        uint32_t *cur_val = dshort_val + ((fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22) >> 1);
        int *cur_cid = dshort_cid + fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22;
        if (tid1 >= short_row_1)
        {
            return;
        }
        valT *valA = reinterpret_cast<valT *>(&cur_val[tid1 >> 1]);
        int x_idx = load_int_from_global(cur_cid + tid1);
        valT temp_y = load_half_from_global(valA + (1 & tid1)) * valX_half[x_idx];
        store_half_to_global(valY_half + row_long + row_block + common_13 * 2 + total_short_rows_34 + short_row_2 + tid1, temp_y);
        // valY_half[row_long + row_block + common_13 * 2 + total_short_rows_34 + short_row_2 + tid1] = valA[tid1 % 2] * valX_half[cur_cid[tid1]];
    }
    else if (bid >= offset_short13 && bid < offset_short34)
    // if (bid >= offset_short13 && bid < offset_short34)
    {
        // short part - block 1&3
        int warpid_local = tid >> 5;
        int bid13 = bid - offset_short13;

        uint32_t fragA[2];
        valT fragB[4], fragC[8], res;

        #pragma unroll
        for (int i = 0; i < groupNum; i ++)
        {
            // compute for 1
            fragC[target_idx] = 0.0;

            int offset = ((bid13 * groupNum + i) * warpNum_short + warpid_local) * MMA_M * MMA_K * 4;
            uint32_t *curA_val = dshort_val + (offset >> 1);
            int *curA_cid = dshort_cid + offset;

            fragA[0] = load_uint_from_global(curA_val + idx_val); 
            fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
            fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
            fragB[1] = 0, fragB[2] = 0, fragB[3] = 0;  
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);

            int offset_y = ((bid13 * groupNum + i) * warpNum_short  + warpid_local) * WARP_SIZE * 2 + laneid;
            if (offset_y < common_13 * 2) 
                valY_half[row_long + row_block + offset_y] = res;
            
            // compute for 3
            fragC[target_idx] = 0.0;

            fragB[0] = 0;
            fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
            fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
            fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];
            // fragB[1] = valX_half[curA_cid[idx + 1]], fragB[2] = valX_half[curA_cid[idx + 2]], fragB[3] = valX_half[curA_cid[idx + 3]];  
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            
            offset_y += WARP_SIZE;
            if (offset_y < common_13 * 2) 
                store_half_to_global(valY_half + row_long + row_block + offset_y, res);
                // valY_half[row_long + row_block + offset_y] = res;
            
        }
    }
    else if (bid >= offset_short34 && bid < offset_short22)
    // if (bid >= offset_short34 && bid < offset_short22)
    {
        // short part - block3 & block4
        int warpid_local = tid >> 5;
        int bid34 = bid - offset_short34;

        uint32_t fragA[2];
        valT fragB[4], fragC[8], res;

        #pragma unroll
        for (int j = 0; j < groupNum; j ++)
        {
            fragC[target_idx] = 0.0;

            int offset = fill0_nnz_short13 + ((bid34 * groupNum + j) * warpNum_short + warpid_local) * MMA_M * MMA_K * loopNum_short;
            uint32_t *curA_val = dshort_val + (offset >> 1);
            int *curA_cid = dshort_cid + offset;
            
            fragA[0] = load_uint_from_global(curA_val + idx_val); 
            fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
            fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
            fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
            fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
            fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            
            int offset_y = ((bid34 * groupNum + j) * warpNum_short + warpid_local) * WARP_SIZE + laneid;
            if (offset_y < total_short_rows_34) 
                store_half_to_global(valY_half + row_long + row_block + common_13 * 2 + offset_y, res);
        }
    }
    else
    {
        // short part - blocl 2&2
        int warpid_local = tid >> 5;
        int bid22 = bid - offset_short22;

        uint32_t fragA[2], fragB[2], fragC[4];
        valT res;
        
        valT *fragB_half = reinterpret_cast<valT *>(&fragB[0]);
        valT *fragC_half = reinterpret_cast<valT *>(&fragC[0]);
        
        #pragma unroll
        for (int i = 0; i < groupNum; i ++)
        {
            // compute for 2 (1)
            fragC_half[target_idx] = 0.0;

            int offset_block = ((bid22 * groupNum + i) * warpNum_short + warpid_local) * WARP_SIZE;
            int offset = fill0_nnz_short13 + fill0_nnz_short34 + offset_block * 4;
            int offset_y = offset_block * 2 + laneid;
            uint32_t *curA_val = dshort_val + (offset >> 1);
            int *curA_cid = dshort_cid + offset;

            // fragA[0] = curA_val[idx_val], fragA[1] = curA_val[idx_val + 1];
            fragA[0] = load_uint_from_global(curA_val + idx_val); 
            fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
            fragB_half[0] = valX_half[load_int_from_global(curA_cid + idx)]; 
            fragB_half[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
            fragB[1] = 0;  
            mma_m8n8k4_fp16_v3(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC_half[target_idx], new_id);

            if (offset_y < short_row_2) 
                store_half_to_global(valY_half + row_long + row_block + common_13 * 2 + total_short_rows_34 + offset_y, res);

                // valY_half[row_long + row_block + common_13 * 2 + total_short_rows_34 + offset_y] = res;
            
            // compute for 2 (2)        
            fragC_half[target_idx] = 0.0;

            fragB[0] = 0;
            fragB_half[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
            fragB_half[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];  
            mma_m8n8k4_fp16_v3(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC_half[target_idx], new_id);

            offset_y += WARP_SIZE;
            if (offset_y < short_row_2) 
                store_half_to_global(valY_half + row_long + row_block + common_13 * 2 + total_short_rows_34 + offset_y, res);
                // valY_half[row_long + row_block + common_13 * 2 + total_short_rows_34 + offset_y] = res;
        }
    }
}





__host__ void spmv_all(char *filename, valT *csrValA, indT *csrRowPtrA, int *csrColIdxA,
                       valT *X_val, valT *Y_val, int rowA, int colA, indT nnzA, int NUM, double threshold, int block_longest)
{
    struct timeval t1, t2, t3, pre_t1, pre_t2;

    // 三个部分：短行（长度为1、2、3、4的行），长行，行块（规则部分和非规则部分）
    gettimeofday(&pre_t1, NULL);
    indT nnz_short, nnz_long, origin_nnz_reg, fill0_nnz_reg, nnz_irreg;
    int num_long_rows = 0, num_block_rows = 0, num_zero_rows = 0;

    // 获取短行部分的数据
    int num_short_rows_1 = 0, num_short_rows_3 = 0, num_short_rows_2 = 0, num_short_rows_4 = 0;

    // 记录每一行的非零元数量
    int *row_lengths = (int *)malloc(sizeof(int) * rowA);
    for (int i = 0; i < rowA; i++)
    {
        row_lengths[i] = csrRowPtrA[i + 1] - csrRowPtrA[i];
    }

    // 定义各个类别的行范围边界
    int start_long = -1, end_long = -1;
    int start_block = -1, end_block = -1;
    int start_short_4 = -1, end_short_4 = -1;
    int start_short_3 = -1, end_short_3 = -1;
    int start_short_2 = -1, end_short_2 = -1;
    int start_short_1 = -1, end_short_1 = -1;
    int start_zero = -1, end_zero = -1;

    // 遍历行，确定每个类别的行范围
    int i = 0;
    while (i < rowA)
    {
        int row_len = row_lengths[i];

        if (row_len >= block_longest)
        {
            if (start_long == -1)
                start_long = i;
            while (i < rowA && row_lengths[i] >= block_longest)
            {
                num_long_rows++;
                i++;
            }
            end_long = i - 1;
        }
        else if (row_len > 4)
        {
            if (start_block == -1)
                start_block = i;
            while (i < rowA && row_lengths[i] > 4 && row_lengths[i] < block_longest)
            {
                num_block_rows++;
                i++;
            }
            end_block = i - 1;
        }
        else if (row_len == 4)
        {
            if (start_short_4 == -1)
                start_short_4 = i;
            while (i < rowA && row_lengths[i] == 4)
            {
                num_short_rows_4++;
                i++;
            }
            end_short_4 = i - 1;
        }
        else if (row_len == 3)
        {
            if (start_short_3 == -1)
                start_short_3 = i;
            while (i < rowA && row_lengths[i] == 3)
            {
                num_short_rows_3++;
                i++;
            }
            end_short_3 = i - 1;
        }
        else if (row_len == 2)
        {
            if (start_short_2 == -1)
                start_short_2 = i;
            while (i < rowA && row_lengths[i] == 2)
            {
                num_short_rows_2++;
                i++;
            }
            end_short_2 = i - 1;
        }
        else if (row_len == 1)
        {
            if (start_short_1 == -1)
                start_short_1 = i;
            while (i < rowA && row_lengths[i] == 1)
            {
                num_short_rows_1++;
                i++;
            }
            end_short_1 = i - 1;
        }
        else if (row_len == 0)
        {
            if (start_zero == -1)
                start_zero = i;
            while (i < rowA && row_lengths[i] == 0)
            {
                num_zero_rows++;
                i++;
            }
            end_zero = i - 1;
        }
        else
        {
            // 不应出现此情况
            i++;
        }
    }

    // 根据行块数量确定循环次数
    int rowloop;
    if (num_block_rows < 59990)
        rowloop = 1;
    else if (num_block_rows >= 59990 && num_block_rows < 400000)
        rowloop = 2;
    else
        rowloop = 4;

    // 计算短行部分的非零元总数
    nnz_short = num_short_rows_1 * 1 + num_short_rows_2 * 2 + num_short_rows_3 * 3 + num_short_rows_4 * 4;

    // 处理短行类型1和类型3的公共部分，计算可以成对处理的行数
    int common_13 = num_short_rows_1 < num_short_rows_3 ? num_short_rows_1 : num_short_rows_3;
    if (common_13 / BlockSize >= 16)
    {
        common_13 = BlockSize * 4 * (common_13 / (BlockSize * 4));
        num_short_rows_1 = num_short_rows_1 - common_13;
        num_short_rows_3 = num_short_rows_3 - common_13;
    }
    else
    {
        common_13 = 0;
    }

    // 计算短行部分的块数
    int short_block13 = (common_13 + BlockSize - 1) / BlockSize;
    int half_short_row_2 = (num_short_rows_2 + 1) / 2;
    int short_block22 = (half_short_row_2 + BlockSize - 1) / BlockSize;
    int total_short_rows_34 = num_short_rows_3 + num_short_rows_4;
    int short_block34 = (total_short_rows_34 + BlockSize - 1) / BlockSize;

    // 计算每个线程块处理的块数
    int block13_per_threadblock = warpNum_short * groupNum * 4;
    int block22_per_threadblock = warpNum_short * groupNum * 4;
    int block34_per_threadblock = warpNum_short * groupNum * loopNum_short;

    // 计算需要的线程块数
    int threadblock13 = (short_block13 + block13_per_threadblock - 1) / block13_per_threadblock;
    int threadblock22 = (short_block22 + block22_per_threadblock - 1) / block22_per_threadblock;
    int threadblock34 = (short_block34 + block34_per_threadblock - 1) / block34_per_threadblock;

    // 计算短行部分需要的填充非零元数量
    indT fill0_nnz_short13 = threadblock13 * block13_per_threadblock * MMA_M * MMA_K;
    indT fill0_nnz_short34 = threadblock34 * block34_per_threadblock * MMA_M * MMA_K;
    indT fill0_nnz_short22 = threadblock22 * block22_per_threadblock * MMA_M * MMA_K;
    indT fill0_nnz_short = ((num_short_rows_1 + 1) / 2) * 2 + fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22;

    // 分配短行部分的值和列索引数组，并初始化为0
    valT *short_val = (valT *)malloc(sizeof(valT) * fill0_nnz_short);
    int *short_cid = (int *)malloc(sizeof(int) * fill0_nnz_short);
    memset(short_val, 0.0, sizeof(valT) * fill0_nnz_short);
    memset(short_cid, 0, sizeof(int) * fill0_nnz_short);

    // 处理短行部分的代码，根据新的行范围进行遍历和填充

    // 由于我们不再使用行号数组，而是使用行范围，因此需要根据行范围进行遍历
    // 分别处理短行类型1和类型3的公共部分，以及剩余的短行

    // 定义变量，表示短行的起始索引
    int index_short_1 = start_short_1;
    int index_short_3 = start_short_3;

    // 处理短行类型1和类型3的公共部分（common_13）
    int group13 = common_13 / (4 * BlockSize);
#pragma omp parallel for
    for (int i = 0; i < group13; i++)
    {
        valT *cur_short_val = short_val + i * BlockSize * 4 * MMA_M * MMA_K;
        int *cur_short_cid = short_cid + i * BlockSize * 4 * MMA_M * MMA_K;

        for (int j = 0; j < BlockSize * 4; j++)
        {
            int row_1 = index_short_1 + i * BlockSize * 4 + j;
            int row_3 = index_short_3 + i * BlockSize * 4 + j;

            // 处理类型1的行
            cur_short_val[j * MMA_K] = csrValA[csrRowPtrA[row_1]];
            cur_short_cid[j * MMA_K] = csrColIdxA[csrRowPtrA[row_1]];

            // 处理类型3的行
            cur_short_val[(BlockSize * 4 + j) * MMA_K] = csrValA[csrRowPtrA[row_3]];
            cur_short_val[(BlockSize * 4 + j) * MMA_K + 1] = csrValA[csrRowPtrA[row_3] + 1];
            cur_short_val[(BlockSize * 4 + j) * MMA_K + 2] = csrValA[csrRowPtrA[row_3] + 2];

            cur_short_cid[(BlockSize * 4 + j) * MMA_K] = csrColIdxA[csrRowPtrA[row_3]];
            cur_short_cid[(BlockSize * 4 + j) * MMA_K + 1] = csrColIdxA[csrRowPtrA[row_3] + 1];
            cur_short_cid[(BlockSize * 4 + j) * MMA_K + 2] = csrColIdxA[csrRowPtrA[row_3] + 2];
        }
    }

    // 更新索引，跳过已处理的行
    index_short_1 += common_13;
    index_short_3 += common_13;

    // 处理剩余的短行类型3和类型4
    int remaining_short_rows_34 = num_short_rows_3 + num_short_rows_4;
#pragma omp parallel for
    for (int i = 0; i < remaining_short_rows_34; i++)
    {
        valT *cur_short_val = short_val + fill0_nnz_short13 + i * MMA_M * MMA_K;
        int *cur_short_cid = short_cid + fill0_nnz_short13 + i * MMA_M * MMA_K;

        int row;
        if (i < num_short_rows_3 - common_13)
        {
            // 处理剩余的短行类型3
            row = index_short_3 + i;
            cur_short_val[0] = csrValA[csrRowPtrA[row]];
            cur_short_val[1] = csrValA[csrRowPtrA[row] + 1];
            cur_short_val[2] = csrValA[csrRowPtrA[row] + 2];

            cur_short_cid[0] = csrColIdxA[csrRowPtrA[row]];
            cur_short_cid[1] = csrColIdxA[csrRowPtrA[row] + 1];
            cur_short_cid[2] = csrColIdxA[csrRowPtrA[row] + 2];
        }
        else
        {
            // 处理短行类型4
            row = start_short_4 + (i - (num_short_rows_3 - common_13));
            cur_short_val[0] = csrValA[csrRowPtrA[row]];
            cur_short_val[1] = csrValA[csrRowPtrA[row] + 1];
            cur_short_val[2] = csrValA[csrRowPtrA[row] + 2];
            cur_short_val[3] = csrValA[csrRowPtrA[row] + 3];

            cur_short_cid[0] = csrColIdxA[csrRowPtrA[row]];
            cur_short_cid[1] = csrColIdxA[csrRowPtrA[row] + 1];
            cur_short_cid[2] = csrColIdxA[csrRowPtrA[row] + 2];
            cur_short_cid[3] = csrColIdxA[csrRowPtrA[row] + 3];
        }
    }

    // 处理短行类型2，采用两行合并的方式
    int group22 = (short_block22 + 3) / 4;
#pragma omp parallel for
    for (int i = 0; i < group22; i++)
    {
        valT *cur_short_val = short_val + fill0_nnz_short13 + fill0_nnz_short34 + i * 4 * MMA_M * MMA_K;
        int *cur_short_cid = short_cid + fill0_nnz_short13 + fill0_nnz_short34 + i * 4 * MMA_M * MMA_K;

        for (int j = 0; j < (BlockSize * 4 * 2); j++)
        {
            int idx = i * BlockSize * 4 * 2 + j;
            if (idx < num_short_rows_2)
            {
                int row = start_short_2 + idx;
                int val_index = (j % (BlockSize * 4)) * MMA_K + (j / (BlockSize * 4)) * 2;

                cur_short_val[val_index] = csrValA[csrRowPtrA[row]];
                cur_short_val[val_index + 1] = csrValA[csrRowPtrA[row] + 1];

                cur_short_cid[val_index] = csrColIdxA[csrRowPtrA[row]];
                cur_short_cid[val_index + 1] = csrColIdxA[csrRowPtrA[row] + 1];
            }
        }
    }

    // 处理剩余的短行类型1
    int offset_short_row1 = fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22;
#pragma omp parallel for
    for (int i = 0; i < num_short_rows_1; i++)
    {
        int row = index_short_1 + i;
        short_val[offset_short_row1 + i] = csrValA[csrRowPtrA[row]];
        short_cid[offset_short_row1 + i] = csrColIdxA[csrRowPtrA[row]];
    }

    // 获取行块部分的行长度数组
    indT *rowBlockRowLens = (indT *)malloc(sizeof(indT) * (num_block_rows + 1));
    memset(rowBlockRowLens, 0, sizeof(indT) * (num_block_rows + 1));
    int block_row_index = 0;

    // 获取长行部分的行长度数组
    indT *longRowLens = (indT *)malloc(sizeof(indT) * (num_long_rows + 1));
    memset(longRowLens, 0, sizeof(indT) * (num_long_rows + 1));
    int long_row_index = 0;

    // 遍历行，根据行范围填充行长度数组
    for (int i = start_long; i <= end_long; i++)
    {
        longRowLens[long_row_index] = row_lengths[i];
        long_row_index++;
    }
    for (int i = start_block; i <= end_block; i++)
    {
        rowBlockRowLens[block_row_index] = row_lengths[i];
        block_row_index++;
    }

    // 计算前缀和
    exclusive_scan(rowBlockRowLens, num_block_rows + 1);
    exclusive_scan(longRowLens, num_long_rows + 1);
    nnz_long = longRowLens[num_long_rows];

    // 获取长行部分的数据
    indT *longRowWarpPtr = (indT *)malloc(sizeof(indT) * (num_long_rows + 1));
    memset(longRowWarpPtr, 0, sizeof(indT) * (num_long_rows + 1));
    int warp_number = 0;
#pragma omp parallel for
    for (int i = 0; i < num_long_rows; i++)
    {
        int nnz_num = longRowLens[i + 1] - longRowLens[i];
        int cur_warp_num = (nnz_num + MMA_M * MMA_K * loopNum_long * 4 - 1) / (MMA_M * MMA_K * loopNum_long * 4);
        longRowWarpPtr[i] = cur_warp_num;
    }
    exclusive_scan(longRowWarpPtr, num_long_rows + 1);
    warp_number = longRowWarpPtr[num_long_rows];

    // 计算长行部分的线程块数量和填充非零元数量
    int BlockNum_long = (warp_number + warpNum_long - 1) / warpNum_long;
    int fill0_nnz_long = BlockNum_long * warpNum_long * loopNum_long * 4 * MMA_M * MMA_K;
    warp_number = BlockNum_long * warpNum_long;

    // 分配长行部分的值和列索引数组，并初始化
    valT *val_by_warp = (valT *)malloc(sizeof(valT) * warp_number);
    int *row_indices_by_warp = (int *)malloc(sizeof(int) * warp_number);
    valT *long_val = (valT *)malloc(sizeof(valT) * fill0_nnz_long);
    memset(long_val, 0.0, sizeof(valT) * fill0_nnz_long);
    int *long_cid = (int *)malloc(sizeof(int) * fill0_nnz_long);
    memset(long_cid, 0, sizeof(int) * fill0_nnz_long);

// 填充长行部分的数据
#pragma omp parallel for
    for (int i = 0; i < num_long_rows; i++)
    {
        valT *cur_val = long_val + longRowWarpPtr[i] * loopNum_long * 4 * MMA_M * MMA_K;
        int *cur_cid = long_cid + longRowWarpPtr[i] * loopNum_long * 4 * MMA_M * MMA_K;
        int real_rid = start_long + i; // 行号

        for (int j = 0; j < longRowLens[i + 1] - longRowLens[i]; j++)
        {
            cur_val[j] = csrValA[csrRowPtrA[real_rid] + j];
            cur_cid[j] = csrColIdxA[csrRowPtrA[real_rid] + j];
        }

        for (int j = longRowWarpPtr[i]; j < longRowWarpPtr[i + 1]; j++)
        {
            row_indices_by_warp[j] = i;
        }
    }

    // 预处理行块部分：划分为规则部分和非规则部分
    int blocknum = (num_block_rows + BlockSize - 1) / BlockSize;
    blocknum = ((blocknum + rowloop * 4 - 1) / (rowloop * 4)) * rowloop * 4;
    indT *blockPtr = (indT *)malloc(sizeof(indT) * (blocknum + 1));
    memset(blockPtr, 0, sizeof(indT) * (blocknum + 1));

    indT *irregRowPtr = (indT *)malloc(sizeof(indT) * (num_block_rows + 1));
    memset(irregRowPtr, 0, sizeof(indT) * (num_block_rows + 1));

// 对每个块进行处理，确定规则部分和非规则部分
#pragma omp parallel for
    for (int i = 0; i < blocknum; i++)
    {
        int row_start = i * BlockSize;
        int row_end = (i + 1) * BlockSize >= num_block_rows ? num_block_rows : (i + 1) * BlockSize;
        int k = 1;
        while (1)
        {
            int block_nnz = 0;
            for (int cur_row = row_start; cur_row < row_end; cur_row++)
            {
                int row_len = rowBlockRowLens[cur_row + 1] - rowBlockRowLens[cur_row];
                if (row_len / MMA_K >= k)
                    block_nnz += MMA_K;
                else if (row_len / MMA_K == k - 1)
                    block_nnz += row_len % MMA_K;
            }

            if (block_nnz >= threshold * MMA_K * MMA_M)
            {
                blockPtr[i] += MMA_K * MMA_M;
            }
            else
            {
                for (int cur_row = row_start; cur_row < row_end; cur_row++)
                {
                    int row_len = rowBlockRowLens[cur_row + 1] - rowBlockRowLens[cur_row];
                    irregRowPtr[cur_row] = row_len - (k - 1) * MMA_K > 0 ? row_len - (k - 1) * MMA_K : 0;
                }
                break;
            }
            k++;
        }
        blockPtr[i] = ((blockPtr[i] + MMA_M * MMA_K * 4 - 1) / (MMA_M * MMA_K * 4)) * (MMA_M * MMA_K * 4);
    }

    // 计算规则部分和非规则部分的前缀和
    exclusive_scan(blockPtr, blocknum + 1);
    exclusive_scan(irregRowPtr, num_block_rows + 1);

    // 计算填充的非零元数量
    fill0_nnz_reg = blockPtr[blocknum];
    nnz_irreg = irregRowPtr[num_block_rows];
    origin_nnz_reg = nnzA - nnz_irreg - nnz_long - nnz_short;

    // 获取行块部分的非规则部分数据
    indT fill0_nnz_irreg = ((nnz_irreg + 1) / 2) * 2;
    valT *irreg_val = (valT *)malloc(sizeof(valT) * fill0_nnz_irreg);
    int *irreg_cid = (int *)malloc(sizeof(int) * nnz_irreg);
#pragma omp parallel for
    for (int i = 0; i < num_block_rows; i++)
    {
        int cur_rid = start_block + i;
        int irreg_offset = irregRowPtr[i];
        int irreg_len = irregRowPtr[i + 1] - irreg_offset;
        for (int j = 0; j < irreg_len; j++)
        {
            irreg_val[irreg_offset + j] = csrValA[csrRowPtrA[cur_rid + 1] - irreg_len + j];
            irreg_cid[irreg_offset + j] = csrColIdxA[csrRowPtrA[cur_rid + 1] - irreg_len + j];
        }
    }

    // 获取行块部分的规则部分数据
    valT *reg_val = (valT *)malloc(sizeof(valT) * fill0_nnz_reg);
    int *reg_cid = (int *)malloc(sizeof(int) * fill0_nnz_reg);

#pragma omp parallel for
    for (int bid = 0; bid < blocknum; bid++)
    {
        int nnz_block = (blockPtr[bid + 1] - blockPtr[bid]);
        int blocklen = nnz_block / BlockSize;

        for (int rowid = bid * BlockSize; rowid < (bid + 1) * BlockSize; rowid++)
        {
            int regA_start = blockPtr[bid] + blocklen * (rowid - bid * BlockSize);
            if (rowid < num_block_rows)
            {
                int real_id = start_block + rowid;
                int A_start = csrRowPtrA[real_id];
                int row_len = csrRowPtrA[real_id + 1] - A_start - (irregRowPtr[rowid + 1] - irregRowPtr[rowid]);
                for (int i = 0; i < blocklen; i++)
                {
                    if (i < row_len)
                    {
                        reg_val[regA_start + i] = csrValA[A_start + i];
                        reg_cid[regA_start + i] = csrColIdxA[A_start + i];
                    }
                    else
                    {
                        reg_val[regA_start + i] = 0;
                        reg_cid[regA_start + i] = 0;
                    }
                }
            }
            else
            {
                for (int i = 0; i < blocklen; i++)
                {
                    reg_val[regA_start + i] = 0.0;
                    reg_cid[regA_start + i] = 0;
                }
            }
        }

        // 对规则部分的数据进行重新排列
        valT *temp_val = (valT *)malloc(sizeof(valT) * nnz_block);
        int *temp_cid = (int *)malloc(sizeof(int) * nnz_block);
        valT *cur_val = reg_val + blockPtr[bid];
        int *cur_cid = reg_cid + blockPtr[bid];

        for (int i = 0; i < nnz_block; i++)
        {
            int new_id = ((i % blocklen) / MMA_K) * BlockSize * MMA_K + (i / blocklen) * MMA_K + i % MMA_K;
            temp_val[new_id] = cur_val[i];
            temp_cid[new_id] = cur_cid[i];
        }
        memcpy(cur_val, temp_val, sizeof(valT) * nnz_block);
        memcpy(cur_cid, temp_cid, sizeof(int) * nnz_block);
        free(temp_val);
        free(temp_cid);
    }
    gettimeofday(&pre_t2, NULL);
    double dasp_pre = (pre_t2.tv_sec - pre_t1.tv_sec) * 1000.0 + (pre_t2.tv_usec - pre_t1.tv_usec) / 1000.0;

    // 计算填充率
    long fill0_nnz = fill0_nnz_short + fill0_nnz_long + nnz_irreg + fill0_nnz_reg;
    double rate_fill0 = (double)(fill0_nnz - nnzA) / nnzA;

    // 计算数据传输量
    long long int data_X = (rowA + colA) * sizeof(valT) +
                           fill0_nnz_long * (sizeof(valT) + sizeof(int)) + warp_number * sizeof(valT) + (num_long_rows + 1) * sizeof(int) +
                           fill0_nnz_short * (sizeof(valT) + sizeof(int)) +
                           fill0_nnz_reg * (sizeof(valT) + sizeof(int)) + (blocknum + 1) * sizeof(indT) +
                           fill0_nnz_irreg * (sizeof(valT) + sizeof(int)) + (num_block_rows + 1) * sizeof(indT);

    long long int data_X2 = (rowA + nnzA) * sizeof(valT) +
                            fill0_nnz_long * (sizeof(valT) + sizeof(int)) + warp_number * sizeof(valT) + (num_long_rows + 1) * sizeof(int) +
                            fill0_nnz_short * (sizeof(valT) + sizeof(int)) +
                            fill0_nnz_reg * (sizeof(valT) + sizeof(int)) + (blocknum + 1) * sizeof(indT) +
                            fill0_nnz_irreg * (sizeof(valT) + sizeof(int)) + (num_block_rows + 1) * sizeof(indT);

    int BlockNum = (blocknum + rowloop * 4 - 1) / (rowloop * 4);

    int ThreadNum_short = warpNum_short * WARP_SIZE;
    int BlockNum_short_1 = (num_short_rows_1 + ThreadNum_short - 1) / ThreadNum_short;
    int BlockNum_short = BlockNum_short_1 + threadblock13 + threadblock34 + threadblock22;

    int offset_reg = BlockNum_long;
    int offset_short1 = offset_reg + BlockNum;
    int offset_short13 = offset_short1 + BlockNum_short_1;
    int offset_short34 = offset_short13 + threadblock13;
    int offset_short22 = offset_short34 + threadblock34;

    int BlockNum_all = BlockNum_long + BlockNum + BlockNum_short;
    int ThreadNum_all = 4 * WARP_SIZE;

    int sumBlockNum = (num_long_rows + 3) / 4;
    

    uint32_t *dX_val, *dY_val;

    // init cuda data of long part
    uint32_t *dlong_val;
    valT *dval_by_warp;
    indT *dlong_ptr_warp;
    int *dlong_cid;
    int *drid_by_warp;

    // init cuda data of short part
    uint32_t *dshort_val;
    int *dshort_cid;

    // init cuda data of reg & irreg part
    uint32_t *dreg_val;
    uint32_t *dirreg_val;
    indT *dblock_ptr, *dirreg_rpt;
    int *dreg_cid, *dirreg_cid;

    cudaMalloc((void **)&dX_val, sizeof(valT) * (((colA + 1) / 2) * 2));
    cudaMalloc((void **)&dY_val, sizeof(valT) * (((rowA + 1) / 2) * 2));
    cudaMemcpy(dX_val, X_val, sizeof(valT) * (((colA + 1) / 2) * 2), cudaMemcpyHostToDevice);
    cudaMemset(dY_val, 0.0, sizeof(valT) * (((rowA + 1) / 2) * 2));

    // cudaMalloc((void **)&dlong_val, sizeof(valT) * fill0_nnz_long);
    cudaMalloc((void **)&dlong_val, sizeof(valT) * fill0_nnz_long);
    cudaMalloc((void **)&dlong_cid, sizeof(int) * fill0_nnz_long);
    cudaMalloc((void **)&drid_by_warp, sizeof(int) * warp_number);
    cudaMalloc((void **)&dval_by_warp, sizeof(valT) * warp_number);
    cudaMalloc((void **)&dlong_ptr_warp, sizeof(indT) * (num_long_rows + 1));
    cudaMemcpy(dlong_val, long_val, sizeof(valT) * fill0_nnz_long, cudaMemcpyHostToDevice);
    cudaMemcpy(dlong_cid, long_cid, sizeof(int) * fill0_nnz_long, cudaMemcpyHostToDevice);
    cudaMemcpy(drid_by_warp, row_indices_by_warp, sizeof(int) * warp_number, cudaMemcpyHostToDevice);
    cudaMemcpy(dlong_ptr_warp, longRowWarpPtr, sizeof(indT) * (num_long_rows + 1), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dshort_val, sizeof(valT) * fill0_nnz_short);
    cudaMalloc((void **)&dshort_cid, sizeof(int) * fill0_nnz_short);
    cudaMemcpy(dshort_val, short_val, sizeof(valT) * fill0_nnz_short, cudaMemcpyHostToDevice);
    cudaMemcpy(dshort_cid, short_cid, sizeof(int) * fill0_nnz_short, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dreg_val, sizeof(valT) * fill0_nnz_reg);
    cudaMalloc((void **)&dreg_cid, sizeof(int) * fill0_nnz_reg);
    cudaMalloc((void **)&dblock_ptr, sizeof(indT) * (blocknum + 1));
    cudaMemcpy(dreg_val, reg_val, sizeof(valT) * fill0_nnz_reg, cudaMemcpyHostToDevice);
    cudaMemcpy(dreg_cid, reg_cid, sizeof(int) * fill0_nnz_reg, cudaMemcpyHostToDevice);
    cudaMemcpy(dblock_ptr, blockPtr, sizeof(indT) * (blocknum + 1), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dirreg_val, sizeof(valT) * fill0_nnz_irreg);
    cudaMalloc((void **)&dirreg_rpt, sizeof(indT) * (num_block_rows + 1));
    cudaMalloc((void **)&dirreg_cid, sizeof(int) * nnz_irreg);
    cudaMemcpy(dirreg_val, irreg_val, sizeof(valT) * fill0_nnz_irreg, cudaMemcpyHostToDevice);
    cudaMemcpy(dirreg_rpt, irregRowPtr, sizeof(indT) * (num_block_rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dirreg_cid, irreg_cid, sizeof(int) * nnz_irreg, cudaMemcpyHostToDevice);

    int carveout = 0;
    // cudaFuncSetAttribute(dasp_spmv<1>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    // cudaFuncSetAttribute(dasp_spmv<2>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    // cudaFuncSetAttribute(dasp_spmv<4>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(dasp_spmv2<1>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(dasp_spmv2<2>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(dasp_spmv2<4>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    
    int warmup_time = 100;
    int execute_time = 1000;
    {
        for (int i = 0; i < warmup_time; ++i)
        {
            dasp_spmv2<4><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, num_long_rows,
                                                    dreg_val, dreg_cid, dblock_ptr, num_block_rows, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, num_short_rows_1, common_13, total_short_rows_34, num_short_rows_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for (int i = 0; i < execute_time; ++i)
        {    
            dasp_spmv2<4><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, num_long_rows,
                                                    dreg_val, dreg_cid, dblock_ptr, num_block_rows, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, num_short_rows_1, common_13, total_short_rows_34, num_short_rows_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        if (num_long_rows)
        {
            for (int i = 0; i < execute_time; ++i)
            {
                longPart_sum<<<sumBlockNum, ThreadNum_all>>>(dlong_ptr_warp, dval_by_warp, dY_val, num_long_rows);
            }
            cudaDeviceSynchronize();
        }
        gettimeofday(&t2, NULL);
        for (int i = 0; i < execute_time; ++i)
        {    
            dasp_spmv2<4><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, num_long_rows,
                                                    dreg_val, dreg_cid, dblock_ptr, num_block_rows, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, num_short_rows_1, common_13, total_short_rows_34, num_short_rows_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        if (num_long_rows)
        {
            for (int i = 0; i < execute_time; ++i)
            {
                longPart_sum<<<sumBlockNum, ThreadNum_all>>>(dlong_ptr_warp, dval_by_warp, dY_val, num_long_rows);
            }
            cudaDeviceSynchronize();
        }
        gettimeofday(&t3, NULL);
    }

    double dasp_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / execute_time; 
    double dasp_gflops = (double)((long)nnzA * 2) / (dasp_time * 1e6);
    double dasp_time_bypass = ((t3.tv_sec - t2.tv_sec) * 1000.0 + (t3.tv_usec - t2.tv_usec) / 1000.0) / execute_time; 
    double dasp_gflops_bypass = (double)((long)nnzA * 2) / (dasp_time_bypass * 1e6);
    double dasp_bandwidth1 = (double)data_X / (dasp_time_bypass * 1e6);
    double dasp_bandwidth2 = (double)data_X2 / (dasp_time_bypass * 1e6);
    printf("SpMV_X:  %8.4lf ms, %8.4lf GFlop/s, %9.4lf GB/s, %9.4lf GB/s\n", dasp_time, dasp_gflops, dasp_bandwidth1, dasp_bandwidth2);
    printf("SpMV_X2: %8.4lf ms, %8.4lf GFlop/s, %9.4lf GB/s, %9.4lf GB/s\n", dasp_time_bypass, dasp_gflops_bypass, dasp_bandwidth1, dasp_bandwidth2);

    // printf("\nrowA = %d, num_long_rows = %d, num_block_rows = %d, row_short1 = %d, common13 = %d, row_short_3 = %d, row_short_4 = %d, row_short_2 = %d\n", rowA, num_long_rows, num_block_rows, num_short_rows_1, common_13, short_row_3, short_row_4, num_short_rows_2);

    cudaMemcpy(Y_val, dY_val, sizeof(valT) * rowA, cudaMemcpyDeviceToHost);

    
    cudaFree(dX_val);
    cudaFree(dY_val);

    cudaFree(dlong_val);
    cudaFree(dlong_cid);
    cudaFree(dval_by_warp);
    cudaFree(drid_by_warp);
    cudaFree(dlong_ptr_warp);

    cudaFree(dshort_cid);
    cudaFree(dshort_val);

    cudaFree(dreg_val);
    cudaFree(dreg_cid);
    cudaFree(dblock_ptr);
    cudaFree(dirreg_cid);
    cudaFree(dirreg_rpt);
    cudaFree(dirreg_val);

    // FILE *fout;
    // fout = fopen("data/spmv_f16_record.csv", "a");
    // fprintf(fout, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", filename, rowA, colA, nnzA, num_short_rows_1, common_13, short_row_3, short_row_4, num_short_rows_2, num_long_rows, num_block_rows, nnz_short, fill0_nnz_short, nnz_long, fill0_nnz_long, origin_nnz_reg, fill0_nnz_reg, nnz_irreg);
    // fprintf(fout, "%lf,%d,%lld,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", rate_fill0, block_longest, data_X, dasp_pre, dasp_time, dasp_gflops, dasp_time_bypass, dasp_gflops_bypass, dasp_bandwidth1, dasp_bandwidth2);
    // fclose(fout);

    printf("\n");
    // 释放内存
    
    free(row_lengths);
    free(rowBlockRowLens);
    free(longRowLens);
    free(longRowWarpPtr);
    free(val_by_warp);
    free(row_indices_by_warp);

    free(short_val);
    free(short_cid);

    free(long_cid);
    free(long_val);

    free(reg_val);
    free(reg_cid);
    free(blockPtr);

    free(irregRowPtr);
    free(irreg_cid);
    free(irreg_val);
}
*/