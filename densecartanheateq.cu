#include <cuda_runtime.h>
#include <math.h>

__device__ int isPrime(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

__global__ void findPrimes(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && isPrime(idx)) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

__device__ int denseCartanKernel1(int a, int b) {
    return a * a + b * b + 2 * a * b;
}

__global__ void findPrimesDenseCartan1(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && isPrime(denseCartanKernel1(idx, idx + 1))) {
        primes[idx] = denseCartanKernel1(idx, idx + 1);
    } else {
        primes[idx] = 0;
    }
}

__device__ int denseCartanKernel2(int a, int b) {
    return a * a - b * b + 3 * a * b;
}

__global__ void findPrimesDenseCartan2(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && isPrime(denseCartanKernel2(idx, idx + 1))) {
        primes[idx] = denseCartanKernel2(idx, idx + 1);
    } else {
        primes[idx] = 0;
    }
}

__device__ int denseCartanKernel3(int a, int b) {
    return a * a + b * b - 4 * a * b;
}

__global__ void findPrimesDenseCartan3(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && isPrime(denseCartanKernel3(idx, idx + 1))) {
        primes[idx] = denseCartanKernel3(idx, idx + 1);
    } else {
        primes[idx] = 0;
    }
}

__device__ int denseCartanKernel4(int a, int b) {
    return a * a - b * b + 5 * a * b;
}

__global__ void findPrimesDenseCartan4(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && isPrime(denseCartanKernel4(idx, idx + 1))) {
        primes[idx] = denseCartanKernel4(idx, idx + 1);
    } else {
        primes[idx] = 0;
    }
}
