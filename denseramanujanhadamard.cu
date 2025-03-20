#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#define MAX_THREADS_PER_BLOCK 256

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i)
        if (num % i == 0) return false;
    return true;
}

__global__ void findPrimes(int* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = isPrime(numbers[idx]);
}

__device__ unsigned long long modularExponentiation(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    unsigned long long result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

__global__ void verifyFermatPrimes(unsigned long long* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = modularExponentiation(2ULL, numbers[idx], 2ULL * numbers[idx] + 1) == 2;
}

__device__ unsigned long long legendreSymbol(unsigned long long a, unsigned long long p) {
    return modularExponentiation(a, (p - 1) / 2, p);
}

__global__ void checkLegendreSymbols(unsigned long long* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = legendreSymbol(numbers[idx], 2ULL * numbers[idx] + 1) == -1;
}

__device__ unsigned long long jacobiSymbol(unsigned long long a, unsigned long long n) {
    int result = 1;
    while (a != 0) {
        if (a % 2 == 0) {
            if ((n % 8 == 3) || (n % 8 == 5))
                result = -result;
            a /= 2;
        } else {
            if ((a % 4 == 3) && (n % 4 == 3))
                result = -result;
            std::swap(a, n);
            a %= n;
        }
    }
    return n == 1 ? result : 0;
}

__global__ void verifyJacobiSymbols(unsigned long long* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = jacobiSymbol(numbers[idx], 2ULL * numbers[idx] + 1) == -1;
}

__device__ unsigned long long millerRabinTest(unsigned long long n, unsigned long long a) {
    if (n % 2 == 0 || a % n == 0) return 0;
    unsigned long long d = n - 1;
    int s = 0;
    while (d % 2 == 0) {
        d /= 2;
        ++s;
    }
    unsigned long long x = modularExponentiation(a, d, n);
    if (x == 1 || x == n - 1)
        return 1;
    for (int i = 1; i < s; ++i) {
        x = modularExponentiation(x, 2, n);
        if (x == n - 1)
            return 1;
    }
    return 0;
}

__global__ void performMillerRabinTests(unsigned long long* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = millerRabinTest(numbers[idx], 2);
}

__device__ unsigned long long carmichaelFunction(unsigned long long n) {
    if (n == 1) return 1;
    unsigned long long result = 1;
    for (int i = 2; i <= sqrt(n); ++i) {
        int k = 0;
        while (n % i == 0) {
            n /= i;
            k++;
        }
        if (k > 1)
            result *= pow(i, k - 1) * (i - 1);
        else
            result *= (i - 1);
    }
    if (n > 1)
        result *= (n - 1);
    return result;
}

__global__ void computeCarmichaelFunctions(unsigned long long* numbers, int size, unsigned long long* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = carmichaelFunction(numbers[idx]);
}

__device__ bool isStrongPseudoprime(unsigned long long n, unsigned long long a) {
    if (n % 2 == 0 || a % n == 0) return false;
    unsigned long long d = n - 1;
    int s = 0;
    while (d % 2 == 0) {
        d /= 2;
        ++s;
    }
    unsigned long long x = modularExponentiation(a, d, n);
    if (x == 1 || x == n - 1)
        return true;
    for (int i = 1; i < s; ++i) {
        x = modularExponentiation(x, 2, n);
        if (x == n - 1)
            return true;
    }
    return false;
}

__global__ void verifyStrongPseudoprimes(unsigned long long* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = isStrongPseudoprime(numbers[idx], 2);
}

__device__ unsigned long long pollardRhoFactorization(unsigned long long n, unsigned long long x, unsigned long long c) {
    unsigned long long d = 1;
    while (d == 1) {
        unsigned long long x1 = x, y1 = x;
        for (int i = 0; i < d; ++i)
            x1 = (modularExponentiation(x1, 2, n) + c) % n;
        unsigned long long k = 0;
        while (k < d && d == gcd(abs(x1 - y1), n)) {
            y1 = (modularExponentiation(y1, 2, n) + c) % n;
            if (++k == d)
                d *= 2;
        }
        x = x1;
    }
    return d;
}

__global__ void performPollardRhoFactorizations(unsigned long long* numbers, int size, unsigned long long* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = pollardRhoFactorization(numbers[idx], 2, 1);
}

__device__ unsigned long long extendedGCD(unsigned long long a, unsigned long long b, unsigned long long* x, unsigned long long* y) {
    if (a == 0) {
        *x = 0;
        *y = 1;
        return b;
    }
    unsigned long long x1, y1;
    unsigned long long gcd = extendedGCD(b % a, a, &x1, &y1);
    *x = y1 - (b / a) * x1;
    *y = x1;
    return gcd;
}

__global__ void computeExtendedGCDs(unsigned long long* numbersA, unsigned long long* numbersB, int size, unsigned long long* resultsX, unsigned long long* resultsY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        extendedGCD(numbersA[idx], numbersB[idx], &resultsX[idx], &resultsY[idx]);
}

__device__ bool isPrimitiveRoot(unsigned long long g, unsigned long long p) {
    for (unsigned long long i = 1; i < p - 1; ++i) {
        if (modularExponentiation(g, i, p) == 1)
            return false;
    }
    return true;
}

__global__ void verifyPrimitiveRoots(unsigned long long* numbersG, unsigned long long* numbersP, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = isPrimitiveRoot(numbersG[idx], numbersP[idx]);
}

__device__ unsigned long long discreteLogarithm(unsigned long long base, unsigned long long result, unsigned long long mod) {
    for (unsigned long long i = 0; i < mod - 1; ++i) {
        if (modularExponentiation(base, i, mod) == result)
            return i;
    }
    return -1;
}

__global__ void computeDiscreteLogarithms(unsigned long long* bases, unsigned long long* results, unsigned long long* mods, int size, unsigned long long* logResults) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        logResults[idx] = discreteLogarithm(bases[idx], results[idx], mods[idx]);
}

__device__ bool isCyclicGroup(unsigned long long mod) {
    for (unsigned long long i = 1; i < mod - 1; ++i) {
        if (gcd(i, mod - 1) == 1 && modularExponentiation(modularExponentiation(2, i, mod), (mod - 1) / i, mod) != 1)
            return false;
    }
    return true;
}

__global__ void verifyCyclicGroups(unsigned long long* mods, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = isCyclicGroup(mods[idx]);
}

int main() {
    // Example usage of the kernels
    return 0;
}
