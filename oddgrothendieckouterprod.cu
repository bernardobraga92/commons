#include <stdio.h>
#include <math.h>

__device__ int isPrime(int n) {
    if (n <= 1) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    for (int i = 3; i <= sqrt(n); i += 2)
        if (n % i == 0) return 0;
    return 1;
}

__device__ int generateRandomPrime(int seed) {
    unsigned int x = seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 5;
    while (!isPrime(x)) {
        x += 2;
    }
    return x;
}

__global__ void oddGrowthendieckOuterProdKernel(int *output, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = generateRandomPrime(tid + seed);
}

#define THREADS_PER_BLOCK 256
#define NUMBER_OF_BLOCKS 16

extern "C" void oddGrowthendieckOuterProd(int *h_output, int seed) {
    int *d_output;
    cudaMalloc(&d_output, NUMBER_OF_BLOCKS * THREADS_PER_BLOCK * sizeof(int));
    oddGrowthendieckOuterProdKernel<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_output, seed);
    cudaMemcpy(h_output, d_output, NUMBER_OF_BLOCKS * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}
