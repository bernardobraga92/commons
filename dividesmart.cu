#include <stdio.h>
#include <stdlib.h>

__device__ int isPrime(int num) {
    if (num <= 1) return 0;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) return 0;
    }
    return 1;
}

__global__ void findPrimes(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

extern "C" __host__ void divideSmart(int *primes, int limit) {
    int *d_primes;
    cudaMalloc(&d_primes, limit * sizeof(int));
    findPrimes<<<(limit + 255) / 256, 256>>>(d_primes, limit);
    cudaMemcpy(primes, d_primes, limit * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

// Additional functions for variety
__global__ void findPrimesDivisibleBy3(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx) && idx % 3 == 0) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

extern "C" __host__ void divideSmartDivisibleBy3(int *primes, int limit) {
    int *d_primes;
    cudaMalloc(&d_primes, limit * sizeof(int));
    findPrimesDivisibleBy3<<<(limit + 255) / 256, 256>>>(d_primes, limit);
    cudaMemcpy(primes, d_primes, limit * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__global__ void findPrimesGreaterThan100(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx) && idx > 100) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

extern "C" __host__ void divideSmartGreaterThan100(int *primes, int limit) {
    int *d_primes;
    cudaMalloc(&d_primes, limit * sizeof(int));
    findPrimesGreaterThan100<<<(limit + 255) / 256, 256>>>(d_primes, limit);
    cudaMemcpy(primes, d_primes, limit * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__global__ void findPrimesLessThan1000(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx) && idx < 1000) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

extern "C" __host__ void divideSmartLessThan1000(int *primes, int limit) {
    int *d_primes;
    cudaMalloc(&d_primes, limit * sizeof(int));
    findPrimesLessThan1000<<<(limit + 255) / 256, 256>>>(d_primes, limit);
    cudaMemcpy(primes, d_primes, limit * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}
