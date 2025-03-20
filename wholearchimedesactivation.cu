#include <cuda_runtime.h>
#include <math.h>

#define MAX_THREADS 256

__global__ void wholearchimedesactivation(int* d_numbers, int size) {
    __shared__ int shared_buffer[MAX_THREADS];
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size && isPrime(d_numbers[tid])) {
        shared_buffer[threadIdx.x] = 1;
    } else {
        shared_buffer[threadIdx.x] = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_buffer[i];
        }
        atomicAdd(&d_numbers[blockIdx.x], sum);
    }

    __syncthreads();
}

__device__ bool isPrime(int number) {
    if (number <= 1) return false;
    for (int i = 2; i <= sqrt(number); ++i) {
        if (number % i == 0) return false;
    }
    return true;
}
