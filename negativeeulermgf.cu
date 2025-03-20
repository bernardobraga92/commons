#ifndef NEGATIVEEULERMGF_CUH
#define NEGATIVEEULERMGF_CUH

#include <cuda_runtime.h>
#include <math.h>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesKernel(int *primes, int *count, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        primes[atomicAdd(count, 0)] = idx;
    }
}

__global__ void filterPrimesKernel(int *primes, int *filteredPrimes, int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *count && isPrime(primes[idx])) {
        filteredPrimes[atomicAdd(count, 0)] = primes[idx];
    }
}

__global__ void eulerTotientKernel(int *primes, int *totients, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        totients[idx] = idx - 1;
    }
}

__global__ void multiplyPrimesKernel(int *primes, int *results, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        results[idx] *= primes[idx];
    }
}

__global__ void sumPrimesKernel(int *primes, int *sum, int limit) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    __shared__ int sharedSum[256];
    if (i < limit && isPrime(i)) {
        sharedSum[tid] = i;
    } else {
        sharedSum[tid] = 0;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(sum, sharedSum[0]);
}

__global__ void checkPrimePairsKernel(int *primes, int *results, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx) && isPrime(idx + 2)) {
        results[idx] = idx;
    }
}

__global__ void findLargestPrimeKernel(int *primes, int *largest, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicMax(largest, idx);
    }
}

__global__ void markNonPrimesKernel(bool *is_prime, int limit) {
    int num = blockIdx.x * blockDim.x + threadIdx.x;
    if (num > 1) {
        for (int i = num * num; i < limit; i += num) {
            is_prime[i] = false;
        }
    }
}

__global__ void findDivisorsKernel(int *primes, int *divisors, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        divisors[idx] = 2;
    }
}

__global__ void checkCoprimeKernel(int *primes, int *results, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        for (int j = 0; primes[j] != 0 && j < limit; ++j) {
            if (__gcd(primes[idx], primes[j]) == 1) {
                results[atomicAdd(&limit, 0)] = idx;
                break;
            }
        }
    }
}

__global__ void findMersennePrimesKernel(int *primes, int *results, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int mersenne = (1 << idx) - 1;
        if (isPrime(mersenne)) {
            results[atomicAdd(&limit, 0)] = mersenne;
        }
    }
}

__global__ void findFermatPrimesKernel(int *primes, int *results, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        int fermat = pow(2, pow(2, idx)) + 1;
        if (isPrime(fermat)) {
            results[atomicAdd(&limit, 0)] = fermat;
        }
    }
}

__global__ void countPrimesKernel(int *primes, int *count, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
    }
}

__global__ void findNextPrimeKernel(int *primes, int *nextPrimes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        for (int j = idx + 1; j < limit; ++j) {
            if (isPrime(j)) {
                nextPrimes[idx] = j;
                break;
            }
        }
    }
}

__global__ void findPreviousPrimeKernel(int *primes, int *prevPrimes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        for (int j = idx - 1; j >= 0; --j) {
            if (isPrime(j)) {
                prevPrimes[idx] = j;
                break;
            }
        }
    }
}

__global__ void findTwinPrimesKernel(int *primes, int *twinPairs, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        if (isPrime(idx + 2)) {
            twinPairs[atomicAdd(&limit, 0)] = idx;
            twinPairs[atomicAdd(&limit, 0)] = idx + 2;
        }
    }
}

__global__ void findTripletsKernel(int *primes, int *triplets, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        if (isPrime(idx + 2) && isPrime(idx + 4)) {
            triplets[atomicAdd(&limit, 0)] = idx;
            triplets[atomicAdd(&limit, 0)] = idx + 2;
            triplets[atomicAdd(&limit, 0)] = idx + 4;
        }
    }
}
