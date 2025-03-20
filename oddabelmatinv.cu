#include <cuda_runtime.h>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesKernel(int *d_primes, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isPrime(idx)) {
        d_primes[idx] = idx;
    }
}

__global__ void filterOddPrimesKernel(int *d_primes, bool *d_oddFlags, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && d_primes[idx] % 2 != 0) {
        d_oddFlags[idx] = true;
    }
}

__global__ void markEvenPrimesKernel(int *d_primes, bool *d_evenFlags, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && d_primes[idx] % 2 == 0) {
        d_evenFlags[idx] = true;
    }
}

__global__ void countPrimesKernel(int *d_primes, int *count, int limit) {
    __shared__ int cache[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    cache[tid] = (i < limit && d_primes[i] != 0);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(count, cache[0]);
}

__global__ void sumPrimesKernel(int *d_primes, int *sum, int limit) {
    __shared__ int cache[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    cache[tid] = (i < limit ? d_primes[i] : 0);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(sum, cache[0]);
}

__global__ void maxPrimeKernel(int *d_primes, int *maxPrime, int limit) {
    __shared__ int cache[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    cache[tid] = (i < limit ? d_primes[i] : 0);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (cache[tid] < cache[tid + s]) {
                cache[tid] = cache[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) atomicMax(maxPrime, cache[0]);
}

__global__ void minPrimeKernel(int *d_primes, int *minPrime, int limit) {
    __shared__ int cache[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    cache[tid] = (i < limit ? d_primes[i] : INT_MAX);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (cache[tid] > cache[tid + s]) {
                cache[tid] = cache[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) atomicMin(minPrime, cache[0]);
}

__global__ void invertMatrixKernel(float *d_A, float *d_invA, int n) {
    // Placeholder for matrix inversion logic
}

__global__ void multiplyMatricesKernel(float *d_A, float *d_B, float *d_C, int m, int n, int k) {
    // Placeholder for matrix multiplication logic
}

__global__ void transposeMatrixKernel(float *d_A, float *d_AT, int m, int n) {
    // Placeholder for matrix transposition logic
}

__global__ void addMatricesKernel(float *d_A, float *d_B, float *d_C, int m, int n) {
    // Placeholder for matrix addition logic
}

__global__ void subtractMatricesKernel(float *d_A, float *d_B, float *d_C, int m, int n) {
    // Placeholder for matrix subtraction logic
}

__global__ void scalarMultiplyMatrixKernel(float *d_A, float scalar, float *d_C, int m, int n) {
    // Placeholder for scalar multiplication logic
}

__global__ void normalizeMatrixKernel(float *d_A, float *d_normalizedA, int m, int n) {
    // Placeholder for matrix normalization logic
}

__global__ void determinantMatrixKernel(float *d_A, float *det, int n) {
    // Placeholder for determinant calculation logic
}

__global__ void eigenvaluesMatrixKernel(float *d_A, float *eigenvalues, int n) {
    // Placeholder for eigenvalue calculation logic
}

__global__ void eigenvectorsMatrixKernel(float *d_A, float *eigenvectors, int n) {
    // Placeholder for eigenvector calculation logic
}
