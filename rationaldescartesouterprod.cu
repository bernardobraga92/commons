#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>

__device__ bool isPrime(unsigned int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (unsigned int i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

__global__ void findPrimes(unsigned int* primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (isPrime(idx)) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

__device__ unsigned int gcd(unsigned int a, unsigned int b) {
    while (b != 0) {
        unsigned int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__global__ void computeGCDs(unsigned int* numbers, unsigned int* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (unsigned int j = 0; j < size; ++j) {
            results[idx * size + j] = gcd(numbers[idx], numbers[j]);
        }
    }
}

__device__ unsigned int lcm(unsigned int a, unsigned int b) {
    return (a / gcd(a, b)) * b;
}

__global__ void computeLCMs(unsigned int* numbers, unsigned int* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (unsigned int j = 0; j < size; ++j) {
            results[idx * size + j] = lcm(numbers[idx], numbers[j]);
        }
    }
}

__global__ void computeRatios(unsigned int* numerators, unsigned int* denominators, float* ratios, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        ratios[idx] = static_cast<float>(numerators[idx]) / denominators[idx];
    }
}

__global__ void computeDescartesProduct(unsigned int* setA, unsigned int* setB, unsigned int* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        results[idx] = setA[idx / size] * setB[idx % size];
    }
}

__global__ void computeOuterProduct(unsigned int* vec1, unsigned int* vec2, unsigned int* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        results[idx] = vec1[idx / size] * vec2[idx % size];
    }
}

__global__ void computePrimePairs(unsigned int* primes, unsigned int limit, unsigned int* pairs) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx + 2)) {
        pairs[idx] = idx;
    } else {
        pairs[idx] = 0;
    }
}

__global__ void computeTwinPrimes(unsigned int* primes, unsigned int limit, unsigned int* twins) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx + 2)) {
        twins[idx] = idx;
    } else {
        twins[idx] = 0;
    }
}

__global__ void computePrimeSum(unsigned int* primes, unsigned int limit, unsigned long long* sum) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(sum, idx);
    }
}

__global__ void computePrimeProduct(unsigned int* primes, unsigned int limit, unsigned long long* product) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicMul(product, idx);
    }
}

__global__ void computeModularInverse(unsigned int a, unsigned int m, unsigned int* result) {
    unsigned int m0 = m;
    unsigned int x0 = 0, x1 = 1;
    if (m == 1) {
        *result = 0;
        return;
    }
    while (a > 1) {
        unsigned int q = a / m;
        unsigned int t = m;
        m = a % m;
        a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }
    if (x1 < 0) x1 += m0;
    *result = x1;
}

__global__ void computePrimeFibonacci(unsigned int* primes, unsigned int limit, unsigned int* fibs) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        unsigned int a = 0, b = 1, c = a + b;
        while (c <= idx) {
            a = b;
            b = c;
            c = a + b;
        }
        fibs[idx] = c == idx ? idx : 0;
    } else {
        fibs[idx] = 0;
    }
}

__global__ void computePrimePalindrome(unsigned int* primes, unsigned int limit, unsigned int* palindromes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        unsigned int num = idx;
        unsigned int reversed = 0;
        while (num > 0) {
            reversed = reversed * 10 + num % 10;
            num /= 10;
        }
        palindromes[idx] = (reversed == idx) ? idx : 0;
    } else {
        palindromes[idx] = 0;
    }
}

__global__ void computePrimeSquares(unsigned int* primes, unsigned int limit, unsigned long long* squares) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        squares[idx] = static_cast<unsigned long long>(idx) * idx;
    } else {
        squares[idx] = 0;
    }
}

__global__ void computePrimeCubes(unsigned int* primes, unsigned int limit, unsigned long long* cubes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        cubes[idx] = static_cast<unsigned long long>(idx) * idx * idx;
    } else {
        cubes[idx] = 0;
    }
}

int main() {
    const unsigned int limit = 1000;
    thrust::host_vector<unsigned int> h_primes(limit, 0);
    thrust::device_vector<unsigned int> d_primes = h_primes;

    findPrimes<<<(limit + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_primes.data()), limit);

    h_primes = d_primes;
    for (unsigned int i = 0; i < limit; ++i) {
        if (h_primes[i] != 0) {
            std::cout << "Prime: " << h_primes[i] << std::endl;
        }
    }

    return 0;
}
