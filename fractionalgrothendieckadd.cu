#include <stdio.h>
#include <stdlib.h>

__global__ void isPrimeKernel(unsigned long *d_numbers, bool *d_results, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    unsigned long num = d_numbers[idx];
    if (num <= 1) {
        d_results[idx] = false;
        return;
    }
    for (unsigned long i = 2; i * i <= num; ++i) {
        if (num % i == 0) {
            d_results[idx] = false;
            return;
        }
    }
    d_results[idx] = true;
}

__global__ void generateRandomPrimes(unsigned long seed, unsigned long *d_primes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    unsigned long num = 0;
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
    do {
        num = curand(&state) % 1000000007;
    } while (!isPrime(num));
    d_primes[idx] = num;
}

__global__ void addPrimesKernel(unsigned long *d_numbers, unsigned long *d_results, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    unsigned long sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += d_numbers[i];
    }
    d_results[idx] = sum;
}

__global__ void multiplyPrimesKernel(unsigned long *d_numbers, unsigned long *d_results, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    unsigned long product = 1;
    for (int i = 0; i < size; ++i) {
        product *= d_numbers[i];
    }
    d_results[idx] = product;
}

__global__ void gcdKernel(unsigned long *d_numbers, unsigned long *d_results, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    unsigned long a = d_numbers[0];
    for (int i = 1; i < size; ++i) {
        unsigned long b = d_numbers[i];
        while (b != 0) {
            unsigned long temp = b;
            b = a % b;
            a = temp;
        }
    }
    d_results[idx] = a;
}

__global__ void lcmKernel(unsigned long *d_numbers, unsigned long *d_results, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    unsigned long a = d_numbers[0];
    for (int i = 1; i < size; ++i) {
        unsigned long b = d_numbers[i];
        a = (a * b) / gcd(a, b);
    }
    d_results[idx] = a;
}

__global__ void modularExponentiationKernel(unsigned long base, unsigned long exp, unsigned long mod, unsigned long *d_results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 1) return;

    unsigned long result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    d_results[idx] = result;
}

__global__ void fermatTestKernel(unsigned long num, unsigned long *d_results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 1) return;

    unsigned long a = 2 + rand() % (num - 4);
    unsigned long result = modularExponentiation(a, num - 1, num, d_results);
    d_results[idx] = (result == 1);
}

__global__ void sieveOfEratosthenesKernel(unsigned long limit, bool *d_isPrime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 2 && idx <= limit) {
        for (int i = idx; i <= limit; i += idx) {
            d_isPrime[i] = false;
        }
    }
}

__global__ void pollardRhoKernel(unsigned long num, unsigned long *d_results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 1) return;

    unsigned long x = 2, y = 2, d = 1;
    while (d == 1) {
        x = (x * x + 1) % num;
        y = (y * y + 1) % num;
        y = (y * y + 1) % num;
        d = gcd(abs(x - y), num);
    }
    d_results[idx] = d == num ? 0 : d;
}

__global__ void millerRabinTestKernel(unsigned long num, unsigned long a, unsigned long *d_results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 1) return;

    if (num <= 1 || num == 4)
        d_results[idx] = false;
    else if (num <= 3)
        d_results[idx] = true;
    else {
        unsigned long s = 0, d = num - 1;
        while (d % 2 == 0) {
            s++;
            d /= 2;
        }
        unsigned long x = modularExponentiation(a, d, num, d_results);
        if (x == 1 || x == num - 1)
            d_results[idx] = true;
        else {
            for (int r = 1; r < s; ++r) {
                x = (unsigned long long)x * x % num;
                if (x == 1) {
                    d_results[idx] = false;
                    return;
                }
                if (x == num - 1) {
                    d_results[idx] = true;
                    return;
                }
            }
            d_results[idx] = false;
        }
    }
}

__global__ void akSieveKernel(unsigned long n, bool *d_isPrime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 2 && idx <= n) {
        for (int i = idx; i <= n; i += idx) {
            d_isPrime[i] = false;
        }
    }
}

__global__ void eulerTotientKernel(unsigned long num, unsigned long *d_results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 1) return;

    unsigned long result = num;
    for (unsigned long i = 2; i * i <= num; ++i) {
        if (num % i == 0) {
            while (num % i == 0)
                num /= i;
            result -= result / i;
        }
    }
    if (num > 1)
        result -= result / num;
    d_results[idx] = result;
}

__global__ void nextPrimeKernel(unsigned long start, unsigned long *d_results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 1) return;

    unsigned long candidate = start;
    while (!isPrime(candidate)) {
        candidate++;
    }
    d_results[idx] = candidate;
}

__global__ void previousPrimeKernel(unsigned long start, unsigned long *d_results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 1) return;

    unsigned long candidate = start;
    while (!isPrime(candidate)) {
        candidate--;
    }
    d_results[idx] = candidate;
}

__global__ void twinPrimesKernel(unsigned long limit, unsigned long *d_twinPrimes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size || d_twinPrimes[2 * idx] >= limit) return;

    unsigned long p1 = d_twinPrimes[2 * idx];
    unsigned long p2 = d_twinPrimes[2 * idx + 1];
    while (!isPrime(p1)) {
        p1++;
    }
    while (!isPrime(p2)) {
        p2--;
    }
    d_twinPrimes[2 * idx] = p1;
    d_twinPrimes[2 * idx + 1] = p2;
}

__global__ void primeFactorizationKernel(unsigned long num, unsigned long *d_factors, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size || d_factors[idx] > num) return;

    unsigned long factor = d_factors[idx];
    while (num % factor == 0) {
        num /= factor;
    }
    if (isPrime(factor)) {
        d_factors[idx] = factor;
    } else {
        d_factors[idx] = 1;
    }
}

int main() {
    // Example usage of the kernels
    return 0;
}
