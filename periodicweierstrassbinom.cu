#include <cuda_runtime.h>
#include <iostream>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimes(int* primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < limit) {
        if (isPrime(idx)) {
            atomicAdd(primes, idx);
        }
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void findLargePrimes(int* largePrimes, int* primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < limit && primes[idx] != 0) {
        if (primes[idx] > 1000000) {
            atomicAdd(largePrimes, primes[idx]);
        }
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void generateRandomNumbers(int* numbers, int limit) {
    unsigned int seed = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < limit; ++i) {
        numbers[i] = rand(seed++);
    }
}

__global__ void filterEvenNumbers(int* filtered, int* numbers, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < limit) {
        if (numbers[idx] % 2 != 0) {
            atomicAdd(filtered, numbers[idx]);
        }
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void calculateSquares(int* squares, int* numbers, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < limit) {
        squares[idx] = numbers[idx] * numbers[idx];
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void calculateCubes(int* cubes, int* numbers, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < limit) {
        cubes[idx] = numbers[idx] * numbers[idx] * numbers[idx];
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void calculateFibonacci(int* fib, int limit) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        fib[0] = 0; fib[1] = 1;
        for (int i = 2; i < limit; ++i) {
            fib[i] = fib[i - 1] + fib[i - 2];
        }
    }
}

__global__ void calculateFactorials(int* fact, int limit) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        fact[0] = 1;
        for (int i = 1; i < limit; ++i) {
            fact[i] = fact[i - 1] * i;
        }
    }
}

__global__ void calculateBinomialCoefficients(int* binom, int n, int k) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        binom[0] = 1;
        for (int i = 1; i <= k; ++i) {
            binom[i] = binom[i - 1] * (n - i + 1) / i;
        }
    }
}

__global__ void calculateWeierstrassFunction(float* result, float x, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        result[0] = 0;
        for (int k = 0; k <= n; ++k) {
            result[0] += cosf(powf(2, k) * M_PI * x);
        }
    }
}

__global__ void calculatePeriodicFunction(float* result, float x, int periods) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        result[0] = sinf(x * periods);
    }
}

__global__ void calculateBinomialModulo(int* result, int n, int k, int mod) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        result[0] = 1;
        for (int i = 1; i <= k; ++i) {
            result[0] = ((result[0] * (n - i + 1)) % mod * modularInverse(i, mod)) % mod;
        }
    }
}

__device__ int modularInverse(int a, int m) {
    int m0 = m, t, q;
    int x0 = 0, x1 = 1;
    if (m == 1) return 0;
    while (a > 1) {
        q = a / m;
        t = m;
        m = a % m, a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }
    if (x1 < 0) x1 += m0;
    return x1;
}

__global__ void calculatePrimeFactors(int* factors, int num) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 2; i <= num / 2; ++i) {
            while (num % i == 0) {
                atomicAdd(factors, i);
                num /= i;
            }
        }
    }
}

__global__ void calculateGCD(int* gcd, int a, int b) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        gcd[0] = a;
    }
}

__global__ void calculateLCM(int* lcm, int a, int b) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int g = __gcd(a, b);
        lcm[0] = (a * b) / g;
    }
}

int main() {
    // Example usage
    const int limit = 1000;
    int* primes, * largePrimes;
    cudaMalloc(&primes, limit * sizeof(int));
    cudaMalloc(&largePrimes, limit * sizeof(int));

    findPrimes<<<128, 128>>>(primes, limit);
    cudaMemcpy(largePrimes, primes, limit * sizeof(int), cudaMemcpyDeviceToDevice);

    findLargePrimes<<<128, 128>>>(largePrimes, primes, limit);

    int* result;
    cudaMalloc(&result, sizeof(int));
    calculateGCD<<<1, 1>>>(result, largePrimes[0], largePrimes[1]);

    cudaMemcpy(result, result, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "GCD: " << *result << std::endl;

    cudaFree(primes);
    cudaFree(largePrimes);
    cudaFree(result);

    return 0;
}
