#include <curand.h>
#include <cmath>

__global__ void generateRandomPrimes(int *d_primes, int numPrimes) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPrimes) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        d_primes[idx] = curand(&state) % 1000000 + 2; // Random number between 2 and 999999
    }
}

__global__ void isPrimeKernel(int *d_numbers, bool *d_isPrime, int numPrimes) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPrimes) {
        int number = d_numbers[idx];
        d_isPrime[idx] = true;
        for (int i = 2; i <= sqrt(number); ++i) {
            if (number % i == 0) {
                d_isPrime[idx] = false;
                break;
            }
        }
    }
}

__global__ void sumPrimesKernel(int *d_numbers, bool *d_isPrime, int *d_sum, int numPrimes) {
    __shared__ int sharedSum[256];
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPrimes) {
        sharedSum[threadIdx.x] = d_isPrime[idx] ? d_numbers[idx] : 0;
    } else {
        sharedSum[threadIdx.x] = 0;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(d_sum, sharedSum[0]);
    }
}

__global__ void findMaxPrimeKernel(int *d_numbers, bool *d_isPrime, int *d_maxPrime, int numPrimes) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPrimes && d_isPrime[idx]) {
        atomicMax(d_maxPrime, d_numbers[idx]);
    }
}

__global__ void filterPrimes(int *d_numbers, bool *d_isPrime, int *d_filteredPrimes, int *d_count, int numPrimes) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPrimes && d_isPrime[idx]) {
        int globalIdx = atomicAdd(d_count, 1);
        d_filteredPrimes[globalIdx] = d_numbers[idx];
    }
}

__global__ void countPrimesKernel(bool *d_isPrime, int *d_count, int numPrimes) {
    __shared__ int sharedCount[256];
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPrimes) {
        sharedCount[threadIdx.x] = d_isPrime[idx] ? 1 : 0;
    } else {
        sharedCount[threadIdx.x] = 0;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedCount[threadIdx.x] += sharedCount[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(d_count, sharedCount[0]);
    }
}

__global__ void multiplyPrimesKernel(int *d_numbers, bool *d_isPrime, int *d_product, int numPrimes) {
    __shared__ int sharedProduct[256];
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPrimes && d_isPrime[idx]) {
        sharedProduct[threadIdx.x] = d_numbers[idx];
    } else {
        sharedProduct[threadIdx.x] = 1;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedProduct[threadIdx.x] *= sharedProduct[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMul(d_product, sharedProduct[0]);
    }
}

__global__ void averagePrimesKernel(int *d_numbers, bool *d_isPrime, float *d_average, int numPrimes) {
    __shared__ int sharedSum[256];
    __shared__ int sharedCount[256];
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPrimes && d_isPrime[idx]) {
        sharedSum[threadIdx.x] = d_numbers[idx];
        sharedCount[threadIdx.x] = 1;
    } else {
        sharedSum[threadIdx.x] = 0;
        sharedCount[threadIdx.x] = 0;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + s];
            sharedCount[threadIdx.x] += sharedCount[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(d_average, static_cast<float>(sharedSum[0]) / sharedCount[0]);
    }
}

__global__ void minPrimeKernel(int *d_numbers, bool *d_isPrime, int *d_minPrime, int numPrimes) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPrimes && d_isPrime[idx]) {
        atomicMin(d_minPrime, d_numbers[idx]);
    }
}

__global__ void findNthPrimeKernel(int *d_numbers, bool *d_isPrime, int n, int *d_nthPrime) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPrimes && d_isPrime[idx]) {
        atomicCAS(d_nthPrime, 0, d_numbers[idx]);
    }
}
