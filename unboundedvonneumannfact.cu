#include <iostream>
#include <cuda_runtime.h>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findLargePrimes(int* d_primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < limit) {
        if (isPrime(idx)) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void sieveOfEratosthenes(int* d_primes, int limit) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 2) return;
    for (int i = tid; i <= limit; i += tid) {
        d_primes[i] = 0;
    }
}

__global__ void findPrimesInRange(int* d_primes, int start, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < start || idx > end) return;
    if (isPrime(idx)) {
        d_primes[idx] = idx;
    } else {
        d_primes[idx] = 0;
    }
}

__global__ void markNonPrimes(int* d_primes, int limit) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 2) return;
    for (int i = tid; i <= limit; i += tid) {
        d_primes[i] = 0;
    }
}

__global__ void findNextPrime(int* d_primes, int start) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < start) return;
    if (isPrime(idx)) {
        d_primes[idx] = idx;
    } else {
        d_primes[idx] = 0;
    }
}

__global__ void findPreviousPrime(int* d_primes, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > end) return;
    if (isPrime(idx)) {
        d_primes[idx] = idx;
    } else {
        d_primes[idx] = 0;
    }
}

__global__ void findPrimesDivisibleBy(int* d_primes, int limit, int divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && idx % divisor == 0 && isPrime(idx)) {
        d_primes[idx] = idx;
    } else {
        d_primes[idx] = 0;
    }
}

__global__ void findPrimesNotDivisibleBy(int* d_primes, int limit, int divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && idx % divisor != 0 && isPrime(idx)) {
        d_primes[idx] = idx;
    } else {
        d_primes[idx] = 0;
    }
}

__global__ void findPrimesWithDigitSum(int* d_primes, int limit, int sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = idx;
        int digitSum = 0;
        while (num > 0) {
            digitSum += num % 10;
            num /= 10;
        }
        if (digitSum == sum && isPrime(idx)) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
    }
}

__global__ void findPrimesWithEvenDigitCount(int* d_primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = idx;
        int digitCount = 0;
        while (num > 0) {
            ++digitCount;
            num /= 10;
        }
        if (digitCount % 2 == 0 && isPrime(idx)) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
    }
}

__global__ void findPrimesWithOddDigitCount(int* d_primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = idx;
        int digitCount = 0;
        while (num > 0) {
            ++digitCount;
            num /= 10;
        }
        if (digitCount % 2 != 0 && isPrime(idx)) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
    }
}

__global__ void findPrimesWithSpecificDigit(int* d_primes, int limit, int digit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = idx;
        bool hasDigit = false;
        while (num > 0) {
            if (num % 10 == digit) {
                hasDigit = true;
                break;
            }
            num /= 10;
        }
        if (hasDigit && isPrime(idx)) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
    }
}

__global__ void findPrimesWithoutSpecificDigit(int* d_primes, int limit, int digit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = idx;
        bool hasDigit = false;
        while (num > 0) {
            if (num % 10 == digit) {
                hasDigit = true;
                break;
            }
            num /= 10;
        }
        if (!hasDigit && isPrime(idx)) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
    }
}

__global__ void findPrimesWithPalindromeDigits(int* d_primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = idx;
        int reversedNum = 0;
        int original = num;
        while (num > 0) {
            reversedNum = reversedNum * 10 + num % 10;
            num /= 10;
        }
        if (original == reversedNum && isPrime(idx)) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
    }
}

__global__ void findPrimesWithNonPalindromeDigits(int* d_primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = idx;
        int reversedNum = 0;
        int original = num;
        while (num > 0) {
            reversedNum = reversedNum * 10 + num % 10;
            num /= 10;
        }
        if (original != reversedNum && isPrime(idx)) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
    }
}

__global__ void findPrimesWithSquareRootDigits(int* d_primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = sqrt(idx);
        if (num * num == idx && isPrime(idx)) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
    }
}

__global__ void findPrimesWithNonSquareRootDigits(int* d_primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = sqrt(idx);
        if (num * num != idx && isPrime(idx)) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
    }
}

int main() {
    const int limit = 1000000;
    int* h_primes = new int[limit];
    int* d_primes;

    cudaMalloc(&d_primes, limit * sizeof(int));
    cudaMemcpy(d_primes, h_primes, limit * sizeof(int), cudaMemcpyHostToDevice);

    findLargePrimes<<<128, 256>>>(d_primes, limit);
    // Add other kernel calls as needed

    cudaMemcpy(h_primes, d_primes, limit * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < limit; ++i) {
        if (h_primes[i] != 0) {
            printf("%d is a prime number\n", h_primes[i]);
        }
    }

    cudaFree(d_primes);
    delete[] h_primes;

    return 0;
}
