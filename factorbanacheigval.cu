#include <iostream>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 256

__device__ __inline__ bool isPrime(unsigned int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (unsigned int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

__global__ void findLargePrimes(unsigned int *primes, unsigned int limit, unsigned int start) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        primes[idx] = isPrime(start + idx) ? start + idx : 0;
    }
}

__global__ void filterPrimes(unsigned int *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && primes[idx] == 0) {
        primes[idx] = isPrime(rand() % 1000000007);
    }
}

__global__ void sieveOfErathostenes(unsigned int *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) primes[idx] = 1;

    __syncthreads();

    for (unsigned int p = 2; p * p <= limit; ++p) {
        if (primes[p - 1]) {
            for (unsigned int i = p * p; i <= limit; i += p) {
                primes[i - 1] = 0;
            }
        }
    }

    __syncthreads();

    if (idx < limit && primes[idx]) primes[idx] = isPrime(idx + 2);
}

__global__ void generateRandomPrimes(unsigned int *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        unsigned int candidate = rand() % 1000000007;
        primes[idx] = isPrime(candidate) ? candidate : 0;
    }
}

__global__ void filterRandomPrimes(unsigned int *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && primes[idx] == 0) {
        primes[idx] = isPrime(rand() % 1000000007);
    }
}

__global__ void checkPrimeStatus(unsigned int *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && !isPrime(primes[idx])) primes[idx] = 0;
}

int main() {
    const unsigned int limit = 1024;
    unsigned int *h_primes, *d_primes;

    h_primes = new unsigned int[limit];
    cudaMalloc((void **)&d_primes, limit * sizeof(unsigned int));

    dim3 blockSize(256);
    dim3 gridSize((limit + blockSize.x - 1) / blockSize.x);

    findLargePrimes<<<gridSize, blockSize>>>(d_primes, limit, 100000000);
    filterPrimes<<<gridSize, blockSize>>>(d_primes, limit);
    sieveOfErathostenes<<<gridSize, blockSize>>>(d_primes, limit);
    generateRandomPrimes<<<gridSize, blockSize>>>(d_primes, limit);
    filterRandomPrimes<<<gridSize, blockSize>>>(d_primes, limit);
    checkPrimeStatus<<<gridSize, blockSize>>>(d_primes, limit);

    cudaMemcpy(h_primes, d_primes, limit * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < limit; ++i) {
        if (h_primes[i]) std::cout << h_primes[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_primes;
    cudaFree(d_primes);

    return 0;
}
