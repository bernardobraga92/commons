#include <cuda_runtime.h>
#include <iostream>

__device__ bool isPrime(unsigned long long n) {
    if (n <= 1) return false;
    for (unsigned long long i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void findPrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = isPrime(idx) ? idx : 0;
    }
}

__device__ unsigned long long nextPrime(unsigned long long n) {
    while (!isPrime(++n));
    return n;
}

__global__ void generatePrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = nextPrime(idx);
    }
}

__device__ bool isMersennePrime(unsigned long long n) {
    return isPrime(n) && ((1ULL << n) - 1 == n);
}

__global__ void findMersennePrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = isMersennePrime(idx) ? idx : 0;
    }
}

__device__ unsigned long long twinPrime(unsigned long long n) {
    while (!isPrime(n + 2)) ++n;
    return n + 2;
}

__global__ void findTwinPrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(idx)) {
        primes[idx] = twinPrime(idx);
    }
}

__device__ bool isCousinPrime(unsigned long long n) {
    return isPrime(n) && isPrime(n + 4);
}

__global__ void findCousinPrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(idx)) {
        primes[idx] = isCousinPrime(idx) ? n : 0;
    }
}

__device__ bool isSexyPrime(unsigned long long n) {
    return isPrime(n) && isPrime(n + 6);
}

__global__ void findSexyPrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(idx)) {
        primes[idx] = isSexyPrime(idx) ? n : 0;
    }
}

__device__ bool isSafePrime(unsigned long long n) {
    return isPrime(n) && isPrime((n - 1) / 2);
}

__global__ void findSafePrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = isSafePrime(idx) ? idx : 0;
    }
}

__device__ bool isSophieGermainPrime(unsigned long long n) {
    return isPrime(n) && isPrime(2 * n + 1);
}

__global__ void findSophieGermainPrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = isSophieGermainPrime(idx) ? idx : 0;
    }
}

__device__ bool isFactorialPrime(unsigned long long n) {
    for (unsigned long long i = 2; i <= n; ++i) {
        if (!isPrime(i)) continue;
        if ((i - 1) * factorial(i - 2) + 1 == n) return true;
    }
    return false;
}

__global__ void findFactorialPrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = isFactorialPrime(idx) ? idx : 0;
    }
}

__device__ bool isWieferichPrime(unsigned long long n) {
    return isPrime(n) && pow(2, n - 1) % (n * n) == 1;
}

__global__ void findWieferichPrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = isWieferichPrime(idx) ? idx : 0;
    }
}

__device__ bool isFermatPrime(unsigned long long n) {
    return isPrime(n) && ((n - 1) % 2 == 0);
}

__global__ void findFermatPrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = isFermatPrime(idx) ? idx : 0;
    }
}

__device__ bool isIrregularPrime(unsigned long long n) {
    // Implement irregular prime check
    return false;
}

__global__ void findIrregularPrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = isIrregularPrime(idx) ? idx : 0;
    }
}

__device__ bool isEisensteinPrime(unsigned long long n) {
    // Implement Eisenstein prime check
    return false;
}

__global__ void findEisensteinPrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = isEisensteinPrime(idx) ? idx : 0;
    }
}

__device__ bool isEllipticCurvePrime(unsigned long long n) {
    // Implement elliptic curve prime check
    return false;
}

__global__ void findEllipticCurvePrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = isEllipticCurvePrime(idx) ? idx : 0;
    }
}

__device__ bool isArtinPrime(unsigned long long n) {
    // Implement Artin prime check
    return false;
}

__global__ void findArtinPrimes(unsigned long long *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = isArtinPrime(idx) ? idx : 0;
    }
}

int main() {
    const unsigned int SIZE = 1024;
    unsigned long long *h_primes, *d_primes;
    h_primes = new unsigned long long[SIZE];
    cudaMalloc((void **)&d_primes, SIZE * sizeof(unsigned long long));

    findPrimes<<<(SIZE + 255) / 256, 256>>>(d_primes, SIZE);
    cudaMemcpy(h_primes, d_primes, SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < SIZE; ++i) {
        if (h_primes[i] != 0) std::cout << h_primes[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_primes);
    delete[] h_primes;
    return 0;
}
