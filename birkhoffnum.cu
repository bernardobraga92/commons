#include <cuda_runtime.h>
#include <math.h>

#define BIRKHOFF_NUM_BLOCK_SIZE 256

__global__ void birkhoffNumIsPrime(int *d_numbers, int *d_is_prime, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_is_prime[idx] = 1;
        for (int i = 2; i <= sqrt(d_numbers[idx]); i++) {
            if (d_numbers[idx] % i == 0) {
                d_is_prime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void birkhoffNumSieve(int *d_numbers, int *d_is_prime, int size, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] <= limit) {
        for (int i = 2; i <= sqrt(d_numbers[idx]); i++) {
            if (d_numbers[idx] % i == 0) {
                d_is_prime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void birkhoffNumGeneratePrimes(int *d_numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = idx * 2 + 3; // Generate odd numbers
    }
}

__global__ void birkhoffNumFilterPrimes(int *d_numbers, int *d_is_prime, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && !d_is_prime[idx]) {
        d_numbers[idx] = 0; // Mark non-prime numbers as 0
    }
}

__global__ void birkhoffNumCountPrimes(int *d_numbers, int *d_count, int size) {
    extern __shared__ int shared_count[];
    shared_count[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (d_numbers[i] != 0) {
            atomicAdd(&shared_count[0], 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(d_count, shared_count[0]);
    }
}

__global__ void birkhoffNumSumPrimes(int *d_numbers, unsigned long long *d_sum, int size) {
    extern __shared__ unsigned long long shared_sum[];
    shared_sum[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (d_numbers[i] != 0) {
            atomicAdd(&shared_sum[0], d_numbers[i]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(d_sum, shared_sum[0]);
    }
}

__global__ void birkhoffNumFindMaxPrime(int *d_numbers, int *d_max_prime, int size) {
    extern __shared__ int shared_max_prime[];
    shared_max_prime[threadIdx.x] = INT_MIN;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (d_numbers[i] != 0 && d_numbers[i] > shared_max_prime[0]) {
            atomicMax(&shared_max_prime[0], d_numbers[i]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMax(d_max_prime, shared_max_prime[0]);
    }
}

__global__ void birkhoffNumFindMinPrime(int *d_numbers, int *d_min_prime, int size) {
    extern __shared__ int shared_min_prime[];
    shared_min_prime[threadIdx.x] = INT_MAX;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (d_numbers[i] != 0 && d_numbers[i] < shared_min_prime[0]) {
            atomicMin(&shared_min_prime[0], d_numbers[i]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMin(d_min_prime, shared_min_prime[0]);
    }
}

__global__ void birkhoffNumFilterMultiples(int *d_numbers, int *d_is_prime, int size, int multiple) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_is_prime[idx] && d_numbers[idx] % multiple == 0) {
        d_is_prime[idx] = 0;
    }
}

__global__ void birkhoffNumFilterNonMultiples(int *d_numbers, int *d_is_prime, int size, int non_multiple) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && !d_is_prime[idx] && d_numbers[idx] % non_multiple != 0) {
        d_is_prime[idx] = 1;
    }
}

__global__ void birkhoffNumShiftPrimes(int *d_numbers, int *d_shifted_primes, int size, int shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] != 0) {
        d_shifted_primes[idx] = d_numbers[idx] + shift;
    }
}

__global__ void birkhoffNumScalePrimes(int *d_numbers, int *d_scaled_primes, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] != 0) {
        d_scaled_primes[idx] = static_cast<int>(d_numbers[idx] * scale);
    }
}

__global__ void birkhoffNumSquarePrimes(int *d_numbers, int *d_squared_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] != 0) {
        d_squared_primes[idx] = d_numbers[idx] * d_numbers[idx];
    }
}

__global__ void birkhoffNumCubePrimes(int *d_numbers, int *d_cubed_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] != 0) {
        d_cubed_primes[idx] = d_numbers[idx] * d_numbers[idx] * d_numbers[idx];
    }
}

__global__ void birkhoffNumSortPrimes(int *d_numbers, int size) {
    for (int stride = size / 2; stride > 0; stride /= 2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size - stride && d_numbers[idx] > d_numbers[idx + stride]) {
            int temp = d_numbers[idx];
            d_numbers[idx] = d_numbers[idx + stride];
            d_numbers[idx + stride] = temp;
        }
    }
}

__global__ void birkhoffNumReversePrimes(int *d_numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size / 2) {
        int temp = d_numbers[idx];
        d_numbers[idx] = d_numbers[size - idx - 1];
        d_numbers[size - idx - 1] = temp;
    }
}

__global__ void birkhoffNumRotatePrimes(int *d_numbers, int size, int shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = d_numbers[(idx + shift) % size];
    }
}
