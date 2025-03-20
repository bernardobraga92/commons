#include <cuda_runtime.h>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void generatePrimes(int* primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        while (!isPrime(idx)) ++idx;
        primes[blockIdx.x * blockDim.x + threadIdx.x] = idx;
    }
}

__global__ void addLargePrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 0;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] += primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void multiplyLargePrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 1;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] *= primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] *= shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void sumOfSquaresPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 0;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] += primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void productOfSquaresPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 1;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] *= primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] *= shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void sumOfCubesPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 0;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] += primes[i] * primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void productOfCubesPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 1;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] *= primes[i] * primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] *= shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void sumOfFourthPowersPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 0;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] += primes[i] * primes[i] * primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void productOfFourthPowersPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 1;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] *= primes[i] * primes[i] * primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] *= shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void sumOfFifthPowersPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 0;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] += primes[i] * primes[i] * primes[i] * primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void productOfFifthPowersPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 1;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] *= primes[i] * primes[i] * primes[i] * primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] *= shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void sumOfSixthPowersPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 0;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] += primes[i] * primes[i] * primes[i] * primes[i] * primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void productOfSixthPowersPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 1;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] *= primes[i] * primes[i] * primes[i] * primes[i] * primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] *= shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void sumOfSeventhPowersPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 0;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] += primes[i] * primes[i] * primes[i] * primes[i] * primes[i] * primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}

__global__ void productOfSeventhPowersPrimes(int* primes, int* result, int size) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    shared[tid] = 1;
    for (int i = tid; i < size; i += blockDim.x) {
        shared[tid] *= primes[i] * primes[i] * primes[i] * primes[i] * primes[i] * primes[i] * primes[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] *= shared[tid + s];
            __syncthreads();
        }
        result[blockIdx.x] = shared[0];
    }
}
