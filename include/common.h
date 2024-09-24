#pragma once

#include <map>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <omp.h>
#include <cstdlib> // 为了使用malloc和free

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// #include <helper_cuda.h>
// #include <helper_functions.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
// #include <thrust/sort.h>
// #include <thrust/unique.h>
// #include <torch/extension.h>
// #include <vector>

#include "omp.h"


// #ifdef f64
#define valT double
// #else
// #define valT half
// #endif


#define WARP_SIZE 32
#define BlockSize 8

#define MMA_M 8
#define MMA_N 8
#define MMA_K 4

#define indT int

#define NEW_CID_TYPE int


#define GET_BIT_REST(x)  ((unsigned int)(x << 2) >> 2)

#define SET_16_BIT(dst, src, index)  \
    dst &= ~(0xffff << (index << 4)); \
    dst |= (src << (index << 4))

#define SET_8_BIT(dst, src, index)  \
    dst &= ~(0xff << (index << 3)); \
    dst |= (src << (index << 3))

#define SET_4_BIT(dst, src, index) \
    dst &= ~(0xf << (index << 2)); \
    dst |= (src << (index << 2))

#define SET_2_BIT(dst, src) dst |= src << 30

#define GET_16_BIT(src, index) ((src >> (index << 4)) & 0xffff)
#define GET_8_BIT(src, index) ((src >> (index << 3)) & 0xff)
#define GET_4_BIT(src, index) ((src >> (index << 2)) & 0xf)
#define GET_2_BIT(src) ((src >> 30) & 0b11)
#define omp_valve 1e4



inline int BinarySearch(int *arr, int len, int target) {
	int low = 0;
	int high = len;
	int mid = 0;
	while (low <= high) {
		mid = (low + high) / 2;
		if (target < arr[mid]) high = mid - 1;
		else if (target > arr[mid]) low = mid + 1;
		else return mid;
	}
	return -1;
}

inline void swap_key(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

// quick sort key (child function)
inline int partition_key(int *key, int length, int pivot_index)
{
    int i = 0;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);

    for (; i < length; i++)
    {
        if (key[pivot_index + i] < pivot)
        {
            swap_key(&key[pivot_index + i], &key[small_length]);
            small_length++;
        }
    }

    swap_key(&key[pivot_index + length - 1], &key[small_length]);

    return small_length;
}

// quick sort key (child function)
inline int partition_key_idx(int *key, int *len, int length, int pivot_index)
{
    int i = 0;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);
    swap_key(&len[pivot_index], &len[pivot_index + (length - 1)]);

    for (; i < length; i++)
    {
        if (key[pivot_index + i] < pivot)
        {
            swap_key(&key[pivot_index + i], &key[small_length]);
            swap_key(&len[pivot_index + i], &len[small_length]);
            small_length++;
        }
    }

    swap_key(&key[pivot_index + length - 1], &key[small_length]);
    swap_key(&len[pivot_index + length - 1], &len[small_length]);

    return small_length;
}

// quick sort key (main function)
inline void quick_sort_key(int *key, int length)
{
    if (length == 0 || length == 1)
        return;

    int small_length = partition_key(key, length, 0);
    quick_sort_key(key, small_length);
    quick_sort_key(&key[small_length + 1], length - small_length - 1);
}

inline void quick_sort_key_idx(int *key, int *len, int length)
{
    if (length == 0 || length == 1)
        return;

    int small_length = partition_key_idx(key, len, length, 0);
    quick_sort_key_idx(key, len, small_length);
    quick_sort_key_idx(&key[small_length + 1], &len[small_length + 1], length - small_length - 1);
}

inline void initVec(valT *vec, int length)
{
    for (int i = 0; i < length; ++ i)
    {
        vec[i] = (i % 29);
    }
}

#ifdef f64
__device__ __forceinline__ void mma_m8n8k4(valT *acc, valT &frag_a, valT &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]):
        "d"(frag_a), "d"(frag_b)
    );
}
#endif


inline int get_max(int *arr, int len)
{
    int max = arr[0];
    for (int i = 1; i < len; i ++)
    {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}

inline void count_sort(int *arr, int *idx, int len, int exp)
{
    int *temp_arr = (int *)malloc(sizeof(int) * len);
    int *temp_idx = (int *)malloc(sizeof(int) * len);
    int buckets[10] = {0};

    for (int i = 0; i < len; i ++)
    {
        buckets[(arr[i] / exp) % 10] ++;
    }

    for (int i = 1; i < 10; i ++)
    {
        buckets[i] += buckets[i - 1];
    }

    for (int i = 0; i < len; i ++)
    {
        int offset = len - (buckets[(arr[i] / exp) % 10] - 1) - 1;
        temp_arr[offset] = arr[i];
        temp_idx[offset] = idx[i];
        buckets[(arr[i] / exp) % 10] --;
    }

    for (int i = 0; i < len; i ++)
    {
        arr[i] = temp_arr[i];
        idx[i] = temp_idx[i];
    }

    free(temp_arr);
    free(temp_idx);
}

inline void count_sort_asce(int *arr, int *idx, int len, int exp)
{
    int *temp_arr = (int *)malloc(sizeof(int) * len);
    int *temp_idx = (int *)malloc(sizeof(int) * len);
    int buckets[10] = {0};

    for (int i = 0; i < len; i ++)
    {
        buckets[(arr[i] / exp) % 10] ++;
    }

    for (int i = 1; i < 10; i ++)
    {
        buckets[i] += buckets[i - 1];
    }

    for (int i = len - 1; i >= 0; i ++)
    {
        int offset = buckets[(arr[i] / exp) % 10] - 1;
        temp_arr[offset] = arr[i];
        temp_idx[offset] = idx[i];
        buckets[(arr[i] / exp) % 10] --;
    }

    for (int i = 0; i < len; i ++)
    {
        arr[i] = temp_arr[i];
        idx[i] = temp_idx[i];
    }

    free(temp_arr);
    free(temp_idx);
}

inline void radix_sort(int *arr, int *idx, int len)
{
    int max = get_max(arr, len);
    for (int exp = 1; max / exp > 0; exp *= 10)
    {
        count_sort(arr, idx, len, exp);
    }
}

inline void radix_sort_asce(int *arr, int *idx, int len)
{
    int max = get_max(arr, len);
    for (int exp = 1; max / exp > 0; exp *= 10)
    {
        count_sort_asce(arr, idx, len, exp);
    }
}
