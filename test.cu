#include <cuda_runtime.h>
#include <stdio.h>

__global__ void shuffle_kernel(int *data) {
    int tid = threadIdx.x;
    int sum = data[tid];
    // 使用 __shfl_down_sync 进行值交换
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    data[tid] = sum;
}

int main() {
    const int size = 32;
    int h_data[size];
    for(int i = 0; i < size; ++i) h_data[i] = i;

    int *d_data;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    shuffle_kernel<<<1, size>>>(d_data);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < size; ++i) printf("%d ", h_data[i]);
    printf("\n");

    cudaFree(d_data);
    return 0;
}