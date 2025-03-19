#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define MAX_THREADS 256

__global__ void rootSenseKernel(uint64_t *data, uint64_t size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;

    uint64_t n = data[tid];
    bool isPrime = true;

    for (uint64_t i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) {
            isPrime = false;
            break;
        }
    }

    data[tid] = isPrime ? n : 0;
}

void rootSense(uint64_t *data, uint64_t size) {
    uint64_t *d_data;
    cudaMalloc((void**)&d_data, size * sizeof(uint64_t));
    cudaMemcpy(d_data, data, size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = MAX_THREADS;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    rootSenseKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

__global__ void generatePrimesKernel(uint64_t *primes, uint64_t count) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    primes[tid] = 2 + tid * 2; // Generate odd numbers starting from 3
}

void generatePrimes(uint64_t *primes, uint64_t count) {
    uint64_t *d_primes;
    cudaMalloc((void**)&d_primes, count * sizeof(uint64_t));

    int threadsPerBlock = MAX_THREADS;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    generatePrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, count);
    cudaDeviceSynchronize();

    cudaMemcpy(primes, d_primes, count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__global__ void filterPrimesKernel(uint64_t *data, uint64_t size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;

    uint64_t n = data[tid];
    bool isPrime = true;

    for (uint64_t i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) {
            isPrime = false;
            break;
        }
    }

    data[tid] = isPrime ? n : 0;
}

void filterPrimes(uint64_t *data, uint64_t size) {
    rootSense(data, size);
}

__global__ void findLargestPrimeKernel(uint64_t *primes, uint64_t size, uint64_t &largestPrime) {
    __shared__ uint64_t sharedMax[32];
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
        sharedMax[threadIdx.x] = primes[tid];
    else
        sharedMax[threadIdx.x] = 0;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sharedMax[threadIdx.x + s] > sharedMax[threadIdx.x])
            sharedMax[threadIdx.x] = sharedMax[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicMax(&largestPrime, sharedMax[0]);
}

uint64_t findLargestPrime(uint64_t *primes, uint64_t size) {
    uint64_t largestPrime = 0;
    findLargestPrimeKernel<<<(size + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(primes, size, largestPrime);
    cudaDeviceSynchronize();
    return largestPrime;
}

__global__ void markNonPrimesKernel(uint64_t *data, uint64_t size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;

    uint64_t n = data[tid];
    bool isPrime = true;

    for (uint64_t i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) {
            isPrime = false;
            break;
        }
    }

    data[tid] = isPrime ? n : 0;
}

void markNonPrimes(uint64_t *data, uint64_t size) {
    rootSense(data, size);
}

__global__ void countPrimesKernel(uint64_t *data, uint64_t size, unsigned int &primeCount) {
    __shared__ unsigned int sharedCount[32];
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
        sharedCount[threadIdx.x] = data[tid] != 0 ? 1 : 0;
    else
        sharedCount[threadIdx.x] = 0;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sharedCount[threadIdx.x] += sharedCount[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&primeCount, sharedCount[0]);
}

unsigned int countPrimes(uint64_t *data, uint64_t size) {
    unsigned int primeCount = 0;
    countPrimesKernel<<<(size + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(data, size, primeCount);
    cudaDeviceSynchronize();
    return primeCount;
}

__global__ void sieveOfEratosthenesKernel(uint64_t *isPrime, uint64_t size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;

    isPrime[tid] = 1;
    if (tid < 2)
        isPrime[tid] = 0;

    for (uint64_t i = tid; i < size; i += tid)
        isPrime[i] = 0;

    if (tid == 2)
        isPrime[2] = 1;
}

void sieveOfEratosthenes(uint64_t *isPrime, uint64_t size) {
    uint64_t *d_isPrime;
    cudaMalloc((void**)&d_isPrime, size * sizeof(uint64_t));

    int threadsPerBlock = MAX_THREADS;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    sieveOfEratosthenesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_isPrime, size);
    cudaDeviceSynchronize();

    cudaMemcpy(isPrime, d_isPrime, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_isPrime);
}

__global__ void findSmallestPrimeKernel(uint64_t *primes, uint64_t size, uint64_t &smallestPrime) {
    __shared__ uint64_t sharedMin[32];
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
        sharedMin[threadIdx.x] = primes[tid];
    else
        sharedMin[threadIdx.x] = UINT64_MAX;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sharedMin[threadIdx.x + s] < sharedMin[threadIdx.x])
            sharedMin[threadIdx.x] = sharedMin[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicMin(&smallestPrime, sharedMin[0]);
}

uint64_t findSmallestPrime(uint64_t *primes, uint64_t size) {
    uint64_t smallestPrime = UINT64_MAX;
    findSmallestPrimeKernel<<<(size + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(primes, size, smallestPrime);
    cudaDeviceSynchronize();
    return smallestPrime;
}

__global__ void markPrimesKernel(uint64_t *data, uint64_t size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;

    uint64_t n = data[tid];
    bool isPrime = true;

    for (uint64_t i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) {
            isPrime = false;
            break;
        }
    }

    data[tid] = isPrime ? n : 0;
}

void markPrimes(uint64_t *data, uint64_t size) {
    rootSense(data, size);
}

__global__ void generatePrimesKernel(uint64_t *primes, uint64_t size, unsigned int primeIndex) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;

    if (isPrime(tid)) {
        primes[atomicAdd(&primeIndex, 1)] = tid;
    }
}

void generatePrimes(uint64_t *primes, uint64_t size, unsigned int &primeCount) {
    unsigned int primeIndex = 0;
    generatePrimesKernel<<<(size + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(primes, size, primeIndex);
    cudaDeviceSynchronize();
    primeCount = atomicRead(&primeIndex);
}

__global__ void findNthPrimeKernel(uint64_t *isPrime, uint64_t size, unsigned int n, uint64_t &nthPrime) {
    __shared__ uint64_t sharedPrimes[32];
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
        sharedPrimes[threadIdx.x] = isPrime[tid] ? tid : UINT64_MAX;
    else
        sharedPrimes[threadIdx.x] = UINT64_MAX;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sharedPrimes[threadIdx.x + s] != UINT64_MAX)
            sharedPrimes[threadIdx.x] = sharedPrimes[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint64_t count = 0;
        for (unsigned int i = 0; i < blockDim.x && count <= n; ++i) {
            if (sharedPrimes[i] != UINT64_MAX)
                ++count;
        }
        nthPrime = sharedPrimes[count - 1];
    }
}

uint64_t findNthPrime(uint64_t *isPrime, uint64_t size, unsigned int n) {
    uint64_t nthPrime = UINT64_MAX;
    findNthPrimeKernel<<<(size + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(isPrime, size, n, nthPrime);
    cudaDeviceSynchronize();
    return nthPrime;
}

__global__ void primeFactorizationKernel(uint64_t *factors, uint64_t number) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= number) return;

    if (number % tid == 0 && isPrime(tid))
        factors[tid] = tid;
}

void primeFactorization(uint64_t *factors, uint64_t number) {
    memset(factors, 0, number * sizeof(uint64_t));
    primeFactorizationKernel<<<(number + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(factors, number);
    cudaDeviceSynchronize();
}

__global__ void isPrimeKernel(uint64_t n, bool &result) {
    if (n < 2)
        result = false;
    for (uint64_t i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) {
            result = false;
            return;
        }
    }
    result = true;
}

bool isPrime(uint64_t n) {
    bool result = false;
    isPrimeKernel<<<1, 1>>>(n, result);
    cudaDeviceSynchronize();
    return result;
}
