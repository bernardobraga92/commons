#include <cuda_runtime.h>
#include <math.h>

#define MAX_THREADS_PER_BLOCK 256

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesKernel(int* numbers, bool* results, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        results[idx] = isPrime(numbers[idx]);
    }
}

__global__ void generateRandomNumbersKernel(unsigned int seed, int* numbers, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, tid, 0, &state);
    if (tid < size) {
        numbers[tid] = curand(&state) % 1000000007;
    }
}

__global__ void filterPrimesKernel(int* numbers, bool* results, int* primes, int* primeCount, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int localPrimeCount[MAX_THREADS_PER_BLOCK];
    if (threadIdx.x < MAX_THREADS_PER_BLOCK) {
        localPrimeCount[threadIdx.x] = 0;
    }
    __syncthreads();

    if (idx < size && results[idx]) {
        atomicAdd(&localPrimeCount[threadIdx.x], 1);
    }
    __syncthreads();

    int blockSum = 0;
    for (int i = threadIdx.x; i < MAX_THREADS_PER_BLOCK; i += blockDim.x) {
        blockSum += localPrimeCount[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(primeCount, blockSum);
    }
    __syncthreads();

    int globalIndex = atomicAdd(primeCount, 0);
    if (idx < size && results[idx]) {
        primes[globalIndex] = numbers[idx];
    }
}

extern "C" void findPrimes(int* d_numbers, bool* d_results, int size) {
    findPrimesKernel<<<(size + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK>>>(d_numbers, d_results, size);
}

extern "C" void generateRandomNumbers(unsigned int seed, int* d_numbers, int size) {
    generateRandomNumbersKernel<<<(size + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK>>>(seed, d_numbers, size);
}

extern "C" void filterPrimes(int* d_numbers, bool* d_results, int* d_primes, int* d_primeCount, int size) {
    filterPrimesKernel<<<(size + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK>>>(d_numbers, d_results, d_primes, d_primeCount, size);
}
