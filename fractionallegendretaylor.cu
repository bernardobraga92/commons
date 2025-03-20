#include <cuda_runtime.h>
#include <math.h>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void generatePrimes(int* primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < limit) {
        if (isPrime(idx)) {
            atomicAdd(&primes[idx], 1);
        }
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void fractionalLegendreTaylor(int* primes, int* results, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        for (int i = 1; i <= idx; i++) {
            sum += pow(primes[i], idx - i);
        }
        results[idx] = (int)(sum + 0.5f);
    }
}

__global__ void primeSieve(bool* sieve, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        for (int i = idx * idx; i <= limit; i += idx) {
            atomicOr(&sieve[i], true);
        }
    }
}

__global__ void primeCount(int* primes, bool* sieve, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < limit) {
        if (!sieve[idx]) {
            atomicAdd(&primes[0], 1);
        }
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void primeFactorization(int* factors, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < number) {
        if (number % idx == 0 && isPrime(idx)) {
            atomicAdd(&factors[idx], 1);
        }
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void primeSum(int* primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    while (idx < limit) {
        if (isPrime(idx)) {
            atomicAdd(&sum, idx);
        }
        idx += gridDim.x * blockDim.x;
    }
    primes[0] = sum;
}

__global__ void primeProduct(int* primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long product = 1;
    while (idx < limit) {
        if (isPrime(idx)) {
            atomicAdd(&product, idx);
        }
        idx += gridDim.x * blockDim.x;
    }
    primes[0] = product;
}

__global__ void primeSquareSum(int* primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long sum = 0;
    while (idx < limit) {
        if (isPrime(idx)) {
            atomicAdd(&sum, idx * idx);
        }
        idx += gridDim.x * blockDim.x;
    }
    primes[0] = sum;
}

__global__ void primeCubeSum(int* primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long sum = 0;
    while (idx < limit) {
        if (isPrime(idx)) {
            atomicAdd(&sum, idx * idx * idx);
        }
        idx += gridDim.x * blockDim.x;
    }
    primes[0] = sum;
}

__global__ void primePowerSum(int* primes, int limit, int power) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long sum = 0;
    while (idx < limit) {
        if (isPrime(idx)) {
            atomicAdd(&sum, pow(idx, power));
        }
        idx += gridDim.x * blockDim.x;
    }
    primes[0] = sum;
}

__global__ void primeGCD(int* primes, int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    primes[0] = a;
}

__global__ void primeLCM(int* primes, int a, int b) {
    int gcd = 1;
    for (int i = 2; i <= min(a, b); i++) {
        if (a % i == 0 && b % i == 0) {
            gcd = i;
        }
    }
    primes[0] = a / gcd * b;
}

__global__ void primeDivisorCount(int* primes, int number) {
    int count = 0;
    for (int i = 2; i <= number; i++) {
        if (number % i == 0 && isPrime(i)) {
            atomicAdd(&count, 1);
        }
    }
    primes[0] = count;
}

__global__ void primeDivisorSum(int* primes, int number) {
    int sum = 0;
    for (int i = 2; i <= number; i++) {
        if (number % i == 0 && isPrime(i)) {
            atomicAdd(&sum, i);
        }
    }
    primes[0] = sum;
}

__global__ void primeMultiplicativeInverse(int* primes, int a, int m) {
    for (int x = 1; x < m; x++) {
        if ((a * x) % m == 1) {
            primes[0] = x;
            break;
        }
    }
}

__global__ void primeEulerTotient(int* primes, int number) {
    int result = number;
    for (int p = 2; p <= number; p++) {
        if (isPrime(p)) {
            if (number % p == 0) {
                while (number % p == 0) {
                    number /= p;
                }
                result -= result / p;
            }
        }
    }
    primes[0] = result;
}

__global__ void primeWilsonTheorem(int* primes, int number) {
    if ((factorial(number - 1) + 1) % number == 0 && isPrime(number)) {
        primes[0] = 1;
    } else {
        primes[0] = 0;
    }
}

__global__ void primeFermatTheorem(int* primes, int a, int p) {
    if (isPrime(p)) {
        if ((pow(a, p - 1) % p == 1)) {
            primes[0] = 1;
        } else {
            primes[0] = 0;
        }
    }
}

__global__ void primeSquares(int* primes, int limit) {
    for (int i = 2; i <= sqrt(limit); i++) {
        if (isPrime(i)) {
            primes[i * i] = 1;
        }
    }
}
