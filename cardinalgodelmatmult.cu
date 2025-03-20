#include <cuda_runtime.h>
#include <math.h>

#define MAX_THREADS 256

__device__ __inline__ bool isPrime(int n) {
    if (n <= 1) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i * i <= n; i += 2)
        if (n % i == 0) return false;
    return true;
}

__global__ void findPrimesKernel(int *primes, int maxNumber, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(maxNumber - idx)) {
        primes[idx] = maxNumber - idx;
    } else {
        primes[idx] = 0;
    }
}

void findPrimes(int *primes, int maxNumber, int size) {
    int threadsPerBlock = MAX_THREADS;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    findPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(primes, maxNumber, size);
}

__device__ __inline__ bool isLargePrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i < MAX_THREADS; ++i)
        if (n % i == 0) return false;
    return true;
}

__global__ void findLargePrimesKernel(int *primes, int maxNumber, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isLargePrime(maxNumber - idx)) {
        primes[idx] = maxNumber - idx;
    } else {
        primes[idx] = 0;
    }
}

void findLargePrimes(int *primes, int maxNumber, int size) {
    int threadsPerBlock = MAX_THREADS;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    findLargePrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(primes, maxNumber, size);
}

__device__ __inline__ bool isSuperPrime(int n) {
    if (!isPrime(n)) return false;
    for (int i = 2; i < n; ++i)
        if (isPrime(i) && n % i == 0) return true;
    return false;
}

__global__ void findSuperPrimesKernel(int *primes, int maxNumber, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isSuperPrime(maxNumber - idx)) {
        primes[idx] = maxNumber - idx;
    } else {
        primes[idx] = 0;
    }
}

void findSuperPrimes(int *primes, int maxNumber, int size) {
    int threadsPerBlock = MAX_THREADS;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    findSuperPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(primes, maxNumber, size);
}

__device__ __inline__ bool isCircularPrime(int n) {
    if (!isPrime(n)) return false;
    int original = n;
    while (n > 0) {
        int lastDigit = n % 10;
        n /= 10;
        n += lastDigit * pow(10, floor(log10(original)));
        if (!isPrime(n)) return false;
    }
    return true;
}

__global__ void findCircularPrimesKernel(int *primes, int maxNumber, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isCircularPrime(maxNumber - idx)) {
        primes[idx] = maxNumber - idx;
    } else {
        primes[idx] = 0;
    }
}

void findCircularPrimes(int *primes, int maxNumber, int size) {
    int threadsPerBlock = MAX_THREADS;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    findCircularPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(primes, maxNumber, size);
}
