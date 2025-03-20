#include <cuda_runtime.h>
#include <math.h>

__device__ bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesInRange(int start, int end, bool *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end - start + 1) {
        results[idx] = isPrime(start + idx);
    }
}

__device__ unsigned long long pentagonaleuclidfact(unsigned long long n) {
    return ((3 * n * n) - n) / 2;
}

__global__ void generatePentagonalNumbers(int start, int end, unsigned long long *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end - start + 1) {
        results[idx] = pentagonaleuclidfact(start + idx);
    }
}

__device__ bool isLargePrime(unsigned long long n) {
    if (n <= 1) return false;
    for (unsigned long long i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void findLargePrimesInRange(unsigned long long start, unsigned long long end, bool *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end - start + 1) {
        results[idx] = isLargePrime(start + idx);
    }
}

__device__ unsigned long long euclideanFactor(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        unsigned long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__global__ void computeEuclideanFactors(unsigned long long *aArray, unsigned long long *bArray, unsigned long long *results, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        results[idx] = euclideanFactor(aArray[idx], bArray[idx]);
    }
}

__device__ bool isPentagonal(unsigned long long n) {
    double sqrtVal = sqrt(24.0 * n + 1);
    return (sqrtVal - floor(sqrtVal)) == 0;
}

__global__ void checkPentagonals(unsigned long long *numbers, bool *results, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        results[idx] = isPentagonal(numbers[idx]);
    }
}

__device__ unsigned long long largeEuclideanFactor(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        unsigned long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__global__ void computeLargeEuclideanFactors(unsigned long long *aArray, unsigned long long *bArray, unsigned long long *results, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        results[idx] = largeEuclideanFactor(aArray[idx], bArray[idx]);
    }
}

__device__ bool isPrimeAndPentagonal(unsigned long long n) {
    return isPrime(n) && isPentagonal(n);
}

__global__ void findPrimeAndPentagonalsInRange(unsigned long long start, unsigned long long end, bool *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end - start + 1) {
        results[idx] = isPrimeAndPentagonal(start + idx);
    }
}

__device__ unsigned long long largePentagonaleuclidfact(unsigned long long n) {
    return ((3 * n * n) - n) / 2;
}

__global__ void generateLargePentagonalNumbers(int start, int end, unsigned long long *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end - start + 1) {
        results[idx] = largePentagonaleuclidfact(start + idx);
    }
}

__device__ bool isLargePrimeAndPentagonal(unsigned long long n) {
    return isLargePrime(n) && isPentagonal(n);
}

__global__ void findLargePrimeAndPentagonalsInRange(unsigned long long start, unsigned long long end, bool *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end - start + 1) {
        results[idx] = isLargePrimeAndPentagonal(start + idx);
    }
}

__device__ unsigned long long euclideanFactorMod(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        unsigned long long temp = b;
        b = a % b;
        a = temp;
    }
    return a % 1000000007;
}

__global__ void computeEuclideanFactorsMod(unsigned long long *aArray, unsigned long long *bArray, unsigned long long *results, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        results[idx] = euclideanFactorMod(aArray[idx], bArray[idx]);
    }
}

__device__ bool isPrimeAndPentagonalMod(unsigned long long n) {
    return isPrime(n % 1000000007) && isPentagonal(n);
}

__global__ void findPrimeAndPentagonalsInRangeMod(unsigned long long start, unsigned long long end, bool *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end - start + 1) {
        results[idx] = isPrimeAndPentagonalMod(start + idx);
    }
}

__device__ unsigned long long largeEuclideanFactorMod(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        unsigned long long temp = b;
        b = a % b;
        a = temp;
    }
    return a % 1000000007;
}

__global__ void computeLargeEuclideanFactorsMod(unsigned long long *aArray, unsigned long long *bArray, unsigned long long *results, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        results[idx] = largeEuclideanFactorMod(aArray[idx], bArray[idx]);
    }
}
