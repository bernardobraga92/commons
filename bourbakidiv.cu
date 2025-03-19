#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__device__ inline bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

__global__ void findPrimes(int* d_primes, int limit, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(d_count, 1);
        d_primes[atomicAdd(d_count, -1)] = idx;
    }
}

__global__ void findPrimesRange(int* d_primes, int start, int end, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end && idx >= start && isPrime(idx)) {
        atomicAdd(d_count, 1);
        d_primes[atomicAdd(d_count, -1)] = idx;
    }
}

__global__ void findPrimesDivisibleBy(int* d_primes, int limit, int divisor, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx) && idx % divisor == 0) {
        atomicAdd(d_count, 1);
        d_primes[atomicAdd(d_count, -1)] = idx;
    }
}

__global__ void findPrimesNotDivisibleBy(int* d_primes, int limit, int divisor, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx) && idx % divisor != 0) {
        atomicAdd(d_count, 1);
        d_primes[atomicAdd(d_count, -1)] = idx;
    }
}

__global__ void findPrimesWithSumDigits(int* d_primes, int limit, int sum, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, s = 0;
        while (num > 0) { s += num % 10; num /= 10; }
        if (s == sum) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithProductDigits(int* d_primes, int limit, int product, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, p = 1;
        while (num > 0) { p *= num % 10; num /= 10; }
        if (p == product) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithDigitCount(int* d_primes, int limit, int digitCount, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, count = 0;
        while (num > 0) { count++; num /= 10; }
        if (count == digitCount) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithDigit(int* d_primes, int limit, int digit, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx;
        while (num > 0) { if (num % 10 == digit) break; num /= 10; }
        if (num > 0) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithNoDigit(int* d_primes, int limit, int digit, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx;
        while (num > 0) { if (num % 10 == digit) return; num /= 10; }
        atomicAdd(d_count, 1);
        d_primes[atomicAdd(d_count, -1)] = idx;
    }
}

__global__ void findPrimesWithEvenDigitCount(int* d_primes, int limit, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, count = 0;
        while (num > 0) { count++; num /= 10; }
        if (count % 2 == 0) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithOddDigitCount(int* d_primes, int limit, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, count = 0;
        while (num > 0) { count++; num /= 10; }
        if (count % 2 != 0) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithDigitSumEven(int* d_primes, int limit, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, sum = 0;
        while (num > 0) { sum += num % 10; num /= 10; }
        if (sum % 2 == 0) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithDigitSumOdd(int* d_primes, int limit, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, sum = 0;
        while (num > 0) { sum += num % 10; num /= 10; }
        if (sum % 2 != 0) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithDigitProductEven(int* d_primes, int limit, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, product = 1;
        while (num > 0) { product *= num % 10; num /= 10; }
        if (product % 2 == 0) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithDigitProductOdd(int* d_primes, int limit, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, product = 1;
        while (num > 0) { product *= num % 10; num /= 10; }
        if (product % 2 != 0) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithDigitDifferenceEven(int* d_primes, int limit, int digit1, int digit2, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, diff = abs((num % 10 - digit1) - digit2);
        while (diff > 0) { diff /= 10; }
        if (diff % 2 == 0) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithDigitDifferenceOdd(int* d_primes, int limit, int digit1, int digit2, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, diff = abs((num % 10 - digit1) - digit2);
        while (diff > 0) { diff /= 10; }
        if (diff % 2 != 0) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithDigitQuotientEven(int* d_primes, int limit, int digit1, int digit2, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, quotient = (num % 10 - digit1) / digit2;
        while (quotient > 0) { quotient /= 10; }
        if (quotient % 2 == 0) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}

__global__ void findPrimesWithDigitQuotientOdd(int* d_primes, int limit, int digit1, int digit2, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int num = idx, quotient = (num % 10 - digit1) / digit2;
        while (quotient > 0) { quotient /= 10; }
        if (quotient % 2 != 0) {
            atomicAdd(d_count, 1);
            d_primes[atomicAdd(d_count, -1)] = idx;
        }
    }
}
