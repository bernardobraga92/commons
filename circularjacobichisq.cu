#include <math.h>
#include <stdio.h>

__device__ int isPrime(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return 0;
    }
    return 1;
}

__global__ void findPrimesKernel(int* numbers, int* primes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__global__ void circularJacobiKernel(int* numbers, int* results, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int a = numbers[idx];
        int b = (a + 1) % size;
        int c = (a + 2) % size;
        results[idx] = (numbers[a] * numbers[b]) + (numbers[b] * numbers[c]);
    }
}

__global__ void filterPrimesKernel(int* primes, int* filtered, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && primes[idx] != 0) {
        filtered[idx] = primes[idx];
    } else {
        filtered[idx] = 0;
    }
}

__global__ void sumPrimesKernel(int* numbers, int* results, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        shared[threadIdx.x] = numbers[idx];
    } else {
        shared[threadIdx.x] = 0;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        results[blockIdx.x] = shared[0];
    }
}

__global__ void countPrimesKernel(int* numbers, int* result, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        atomicAdd(&shared[threadIdx.x], 1);
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(result, shared[0]);
    }
}

__global__ void multiplyPrimesKernel(int* numbers, int* results, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        shared[threadIdx.x] = numbers[idx];
    } else {
        shared[threadIdx.x] = 1;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] *= shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        results[blockIdx.x] = shared[0];
    }
}

__global__ void maxPrimeKernel(int* numbers, int* result, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        shared[threadIdx.x] = numbers[idx];
    } else {
        shared[threadIdx.x] = 0;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = max(shared[threadIdx.x], shared[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(result, shared[0]);
    }
}

__global__ void minPrimeKernel(int* numbers, int* result, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        shared[threadIdx.x] = numbers[idx];
    } else {
        shared[threadIdx.x] = INT_MAX;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = min(shared[threadIdx.x], shared[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMin(result, shared[0]);
    }
}

__global__ void averagePrimesKernel(int* numbers, int* sum, int* count, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        atomicAdd(sum, numbers[idx]);
        atomicAdd(count, 1);
    }
}

__global__ void variancePrimesKernel(int* numbers, int* sum, int* count, float* result, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        atomicAdd(sum, numbers[idx] * numbers[idx]);
        atomicAdd(count, 1);
    }
}

__global__ void generatePrimesKernel(int* numbers, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        while (!isPrime(numbers[idx])) {
            ++numbers[idx];
        }
    }
}

__global__ void isPrimeBatchKernel(int* numbers, int* results, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        results[idx] = isPrime(numbers[idx]);
    }
}

__global__ void nextPrimeKernel(int* numbers, int* primes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        while (!isPrime(numbers[idx])) {
            ++numbers[idx];
        }
        primes[idx] = numbers[idx];
    }
}

__global__ void primeDifferenceKernel(int* numbers, int* results, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size - 1 && isPrime(numbers[idx]) && isPrime(numbers[idx + 1])) {
        results[idx] = numbers[idx + 1] - numbers[idx];
    } else {
        results[idx] = 0;
    }
}

__global__ void primeProductKernel(int* numbers, int* results, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        shared[threadIdx.x] = numbers[idx];
    } else {
        shared[threadIdx.x] = 1;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] *= shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        results[blockIdx.x] = shared[0];
    }
}

__global__ void primeQuotientKernel(int* numbers, int* results, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        shared[threadIdx.x] = numbers[idx];
    } else {
        shared[threadIdx.x] = 1;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] /= shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        results[blockIdx.x] = shared[0];
    }
}

__global__ void primeRemainderKernel(int* numbers, int* results, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        shared[threadIdx.x] = numbers[idx];
    } else {
        shared[threadIdx.x] = 1;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] %= shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        results[blockIdx.x] = shared[0];
    }
}

__global__ void primePowerKernel(int* numbers, int* results, int size, int exponent) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        shared[threadIdx.x] = pow(numbers[idx], exponent);
    } else {
        shared[threadIdx.x] = 1;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] *= shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        results[blockIdx.x] = shared[0];
    }
}

__global__ void primeRootKernel(int* numbers, int* results, int size, int root) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        shared[threadIdx.x] = pow(numbers[idx], 1.0 / root);
    } else {
        shared[threadIdx.x] = 1;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] *= shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        results[blockIdx.x] = shared[0];
    }
}

__global__ void primeLogKernel(int* numbers, int* results, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        shared[threadIdx.x] = log(numbers[idx]);
    } else {
        shared[threadIdx.x] = 1;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] *= shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        results[blockIdx.x] = shared[0];
    }
}

__global__ void primeExpKernel(int* numbers, int* results, int size) {
    extern __shared__ int shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        shared[threadIdx.x] = exp(numbers[idx]);
    } else {
        shared[threadIdx.x] = 1;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] *= shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        results[blockIdx.x] = shared[0];
    }
}
