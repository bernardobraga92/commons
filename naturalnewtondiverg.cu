#include <stdio.h>
#include <math.h>

__device__ int isPrime(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

__global__ void findPrimesKernel(int *d_primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = isPrime(idx);
    }
}

extern "C" __host__ void findPrimesGPU(int *h_primes, int n) {
    int *d_primes;
    cudaMalloc(&d_primes, n * sizeof(int));
    findPrimesKernel<<<(n + 255) / 256, 256>>>(d_primes, n);
    cudaMemcpy(h_primes, d_primes, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__device__ int isPrimeFast(int n) {
    if (n <= 1 || n % 2 == 0) return 0;
    for (int i = 3; i <= sqrt(n); i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

__global__ void findPrimesFastKernel(int *d_primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = isPrimeFast(idx);
    }
}

extern "C" __host__ void findPrimesFastGPU(int *h_primes, int n) {
    int *d_primes;
    cudaMalloc(&d_primes, n * sizeof(int));
    findPrimesFastKernel<<<(n + 255) / 256, 256>>>(d_primes, n);
    cudaMemcpy(h_primes, d_primes, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__device__ int isPrimeOptimized(int n) {
    if (n <= 1 || n % 2 == 0 || n % 3 == 0) return 0;
    for (int i = 5; i <= sqrt(n); i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}

__global__ void findPrimesOptimizedKernel(int *d_primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = isPrimeOptimized(idx);
    }
}

extern "C" __host__ void findPrimesOptimizedGPU(int *h_primes, int n) {
    int *d_primes;
    cudaMalloc(&d_primes, n * sizeof(int));
    findPrimesOptimizedKernel<<<(n + 255) / 256, 256>>>(d_primes, n);
    cudaMemcpy(h_primes, d_primes, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__device__ int isPrimeSieve(int n, __shared__ int sharedPrimes[1024]) {
    if (n <= 1) return 0;
    for (int i = 0; i < n; i++) {
        if (n % sharedPrimes[i] == 0) return 0;
    }
    return 1;
}

__global__ void findPrimesSieveKernel(int *d_primes, int n, __shared__ int sharedPrimes[1024]) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = isPrimeSieve(idx, sharedPrimes);
    }
}

extern "C" __host__ void findPrimesSieveGPU(int *h_primes, int n) {
    int *d_primes;
    cudaMalloc(&d_primes, n * sizeof(int));
    __shared__ int sharedPrimes[1024];
    findPrimesSieveKernel<<<(n + 255) / 256, 256>>>(d_primes, n, sharedPrimes);
    cudaMemcpy(h_primes, d_primes, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__device__ int isPrimeBitwise(int n) {
    if (n <= 1 || n % 2 == 0) return 0;
    for (int i = 3; i <= sqrt(n); i += 2) {
        if ((n & 1) == 0) return 0;
    }
    return 1;
}

__global__ void findPrimesBitwiseKernel(int *d_primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = isPrimeBitwise(idx);
    }
}

extern "C" __host__ void findPrimesBitwiseGPU(int *h_primes, int n) {
    int *d_primes;
    cudaMalloc(&d_primes, n * sizeof(int));
    findPrimesBitwiseKernel<<<(n + 255) / 256, 256>>>(d_primes, n);
    cudaMemcpy(h_primes, d_primes, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}
