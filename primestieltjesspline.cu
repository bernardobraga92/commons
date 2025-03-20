#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesKernel(int* primes, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = start + idx;
    if (isPrime(num)) {
        atomicAdd(primes, num);
    }
}

extern "C" void findPrimes(int* d_primes, int start, int limit, int gridSize, int blockSize) {
    findPrimesKernel<<<gridSize, blockSize>>>(d_primes, start, limit);
}

__device__ bool isComposite(int num) {
    if (num <= 1) return true;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return true;
    }
    return false;
}

__global__ void findCompositesKernel(int* composites, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = start + idx;
    if (isComposite(num)) {
        atomicAdd(composites, num);
    }
}

extern "C" void findComposites(int* d_composites, int start, int limit, int gridSize, int blockSize) {
    findCompositesKernel<<<gridSize, blockSize>>>(d_composites, start, limit);
}

__device__ bool isEven(int num) {
    return num % 2 == 0;
}

__global__ void findEvensKernel(int* evens, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = start + idx;
    if (isEven(num)) {
        atomicAdd(evens, num);
    }
}

extern "C" void findEvens(int* d_evens, int start, int limit, int gridSize, int blockSize) {
    findEvensKernel<<<gridSize, blockSize>>>(d_evens, start, limit);
}

__device__ bool isOdd(int num) {
    return num % 2 != 0;
}

__global__ void findOddsKernel(int* odds, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = start + idx;
    if (isOdd(num)) {
        atomicAdd(odds, num);
    }
}

extern "C" void findOdds(int* d_odds, int start, int limit, int gridSize, int blockSize) {
    findOddsKernel<<<gridSize, blockSize>>>(d_odds, start, limit);
}

__device__ bool isPrimeSemiPerfectSquare(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0 && isPrime(i)) {
            return true;
        }
    }
    return false;
}

__global__ void findPrimeSemiPerfectSquaresKernel(int* primeSemiPerfectSquares, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = start + idx;
    if (isPrimeSemiPerfectSquare(num)) {
        atomicAdd(primeSemiPerfectSquares, num);
    }
}

extern "C" void findPrimeSemiPerfectSquares(int* d_primeSemiPerfectSquares, int start, int limit, int gridSize, int blockSize) {
    findPrimeSemiPerfectSquaresKernel<<<gridSize, blockSize>>>(d_primeSemiPerfectSquares, start, limit);
}

__device__ bool isPrimeSquareFree(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % (i * i) == 0) {
            return false;
        }
    }
    return true;
}

__global__ void findPrimeSquareFreeNumbersKernel(int* primeSquareFreeNumbers, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = start + idx;
    if (isPrimeSquareFree(num)) {
        atomicAdd(primeSquareFreeNumbers, num);
    }
}

extern "C" void findPrimeSquareFreeNumbers(int* d_primeSquareFreeNumbers, int start, int limit, int gridSize, int blockSize) {
    findPrimeSquareFreeNumbersKernel<<<gridSize, blockSize>>>(d_primeSquareFreeNumbers, start, limit);
}

__device__ bool isPrimeTriplet(int num1, int num2, int num3) {
    return isPrime(num1) && isPrime(num2) && isPrime(num3) && (num3 - num2 == 2) && (num2 - num1 == 2);
}

__global__ void findPrimeTripletsKernel(int* primeTriplets, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num1 = start + idx;
    if (isPrimeTriplet(num1, num1 + 2, num1 + 4)) {
        atomicAdd(primeTriplets, num1);
    }
}

extern "C" void findPrimeTriplets(int* d_primeTriplets, int start, int limit, int gridSize, int blockSize) {
    findPrimeTripletsKernel<<<gridSize, blockSize>>>(d_primeTriplets, start, limit);
}

__device__ bool isPrimeQuadruplet(int num1, int num2, int num3, int num4) {
    return isPrime(num1) && isPrime(num2) && isPrime(num3) && isPrime(num4) && (num4 - num3 == 2) && (num3 - num2 == 2) && (num2 - num1 == 2);
}

__global__ void findPrimeQuadrupletsKernel(int* primeQuadruplets, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num1 = start + idx;
    if (isPrimeQuadruplet(num1, num1 + 2, num1 + 4, num1 + 6)) {
        atomicAdd(primeQuadruplets, num1);
    }
}

extern "C" void findPrimeQuadruplets(int* d_primeQuadruplets, int start, int limit, int gridSize, int blockSize) {
    findPrimeQuadrupletsKernel<<<gridSize, blockSize>>>(d_primeQuadruplets, start, limit);
}

__device__ bool isPrimeTwin(int num1, int num2) {
    return isPrime(num1) && isPrime(num2) && (num2 - num1 == 2);
}

__global__ void findPrimeTwinsKernel(int* primeTwins, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num1 = start + idx;
    if (isPrimeTwin(num1, num1 + 2)) {
        atomicAdd(primeTwins, num1);
    }
}

extern "C" void findPrimeTwins(int* d_primeTwins, int start, int limit, int gridSize, int blockSize) {
    findPrimeTwinsKernel<<<gridSize, blockSize>>>(d_primeTwins, start, limit);
}

__device__ bool isPrimePentuplet(int num1, int num2, int num3, int num4, int num5) {
    return isPrime(num1) && isPrime(num2) && isPrime(num3) && isPrime(num4) && isPrime(num5) &&
           (num2 - num1 == 4) && (num3 - num2 == 2) && (num4 - num3 == 2) && (num5 - num4 == 4);
}

__global__ void findPrimePentupletsKernel(int* primePentuplets, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num1 = start + idx;
    if (isPrimePentuplet(num1, num1 + 4, num1 + 6, num1 + 8, num1 + 10)) {
        atomicAdd(primePentuplets, num1);
    }
}

extern "C" void findPrimePentuplets(int* d_primePentuplets, int start, int limit, int gridSize, int blockSize) {
    findPrimePentupletsKernel<<<gridSize, blockSize>>>(d_primePentuplets, start, limit);
}

__device__ bool isPrimeHexuplet(int num1, int num2, int num3, int num4, int num5, int num6) {
    return isPrime(num1) && isPrime(num2) && isPrime(num3) && isPrime(num4) && isPrime(num5) && isPrime(num6) &&
           (num2 - num1 == 2) && (num3 - num2 == 4) && (num4 - num3 == 2) && (num5 - num4 == 4) && (num6 - num5 == 2);
}

__global__ void findPrimeHexupletsKernel(int* primeHexuplets, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num1 = start + idx;
    if (isPrimeHexuplet(num1, num1 + 2, num1 + 6, num1 + 8, num1 + 12, num1 + 14)) {
        atomicAdd(primeHexuplets, num1);
    }
}

extern "C" void findPrimeHexuplets(int* d_primeHexuplets, int start, int limit, int gridSize, int blockSize) {
    findPrimeHexupletsKernel<<<gridSize, blockSize>>>(d_primeHexuplets, start, limit);
}

__device__ bool isPrimeSeptuplet(int num1, int num2, int num3, int num4, int num5, int num6, int num7) {
    return isPrime(num1) && isPrime(num2) && isPrime(num3) && isPrime(num4) && isPrime(num5) && isPrime(num6) && isPrime(num7) &&
           (num2 - num1 == 4) && (num3 - num2 == 2) && (num4 - num3 == 4) && (num5 - num4 == 2) && (num6 - num5 == 4) && (num7 - num6 == 2);
}

__global__ void findPrimeSeptupletsKernel(int* primeSeptuplets, int start, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num1 = start + idx;
    if (isPrimeSeptuplet(num1, num1 + 4, num1 + 6, num1 + 10, num1 + 12, num1 + 16, num1 + 18)) {
        atomicAdd(primeSeptuplets, num1);
    }
}

extern "C" void findPrimeSeptuplets(int* d_primeSeptuplets, int start, int limit, int gridSize, int blockSize) {
    findPrimeSeptupletsKernel<<<gridSize, blockSize>>>(d_primeSeptuplets, start, limit);
}
