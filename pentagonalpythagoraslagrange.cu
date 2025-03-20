#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void pentagonalKernel(int *d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) d_data[idx] = idx * (3 * idx - 1) / 2;
}

__global__ void pythagoreanKernel(int *d_a, int *d_b, int *d_c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) d_c[idx] = sqrt(d_a[idx] * d_a[idx] + d_b[idx] * d_b[idx]);
}

__global__ void lagrangeKernel(int *d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) d_data[idx] = 4 * idx * idx + idx;
}

__global__ void primeCheckKernel(bool *d_isPrime, int *d_numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] > 1) {
        bool isPrime = true;
        for (int i = 2; i <= sqrt(d_numbers[idx]); ++i) {
            if (d_numbers[idx] % i == 0) {
                isPrime = false;
                break;
            }
        }
        d_isPrime[idx] = isPrime;
    }
}

__global__ void generatePrimesKernel(int *d_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int candidate = 2 + idx * 2; // Generate odd numbers
        bool isPrime = true;
        for (int i = 2; i <= sqrt(candidate); ++i) {
            if (candidate % i == 0) {
                isPrime = false;
                break;
            }
        }
        d_primes[idx] = isPrime ? candidate : 0;
    }
}

__global__ void sumPrimesKernel(int *d_numbers, int size, int &sum) {
    extern __shared__ int sharedSum[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < size && d_numbers[idx] != 0) {
        sharedSum[tid] = d_numbers[idx];
    } else {
        sharedSum[tid] = 0;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(&sum, sharedSum[0]);
}

__global__ void multiplyPrimesKernel(int *d_numbers, int size, int &product) {
    extern __shared__ int sharedProduct[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < size && d_numbers[idx] != 0) {
        sharedProduct[tid] = d_numbers[idx];
    } else {
        sharedProduct[tid] = 1;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedProduct[tid] *= sharedProduct[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicMul(&product, sharedProduct[0]);
}

__global__ void findMaxPrimeKernel(int *d_numbers, int size, int &maxPrime) {
    extern __shared__ int sharedMax[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < size && d_numbers[idx] != 0) {
        sharedMax[tid] = d_numbers[idx];
    } else {
        sharedMax[tid] = 0;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) atomicMax(&maxPrime, sharedMax[0]);
}

__global__ void findMinPrimeKernel(int *d_numbers, int size, int &minPrime) {
    extern __shared__ int sharedMin[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < size && d_numbers[idx] != 0) {
        sharedMin[tid] = d_numbers[idx];
    } else {
        sharedMin[tid] = INT_MAX;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMin[tid] = min(sharedMin[tid], sharedMin[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) atomicMin(&minPrime, sharedMin[0]);
}

__global__ void countPrimesKernel(int *d_numbers, int size, int &count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] != 0) atomicAdd(&count, 1);
}

__global__ void filterEvenPrimesKernel(int *d_primes, int size, bool *d_isEvenPrime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_primes[idx] % 2 == 0) d_isEvenPrime[idx] = true;
}

__global__ void filterOddPrimesKernel(int *d_primes, int size, bool *d_isOddPrime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_primes[idx] % 2 != 0) d_isOddPrime[idx] = true;
}

__global__ void filterLargePrimesKernel(int *d_primes, int size, bool *d_isLargePrime, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_primes[idx] > threshold) d_isLargePrime[idx] = true;
}

__global__ void filterSmallPrimesKernel(int *d_primes, int size, bool *d_isSmallPrime, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_primes[idx] <= threshold) d_isSmallPrime[idx] = true;
}

__global__ void filterMultiplePrimesKernel(int *d_primes, int size, bool *d_isMultiplePrime, int multiple) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_primes[idx] % multiple == 0) d_isMultiplePrime[idx] = true;
}

__global__ void filterNonMultiplePrimesKernel(int *d_primes, int size, bool *d_isNonMultiplePrime, int nonMultiple) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_primes[idx] % nonMultiple != 0) d_isNonMultiplePrime[idx] = true;
}

__global__ void filterInRangePrimesKernel(int *d_primes, int size, bool *d_isInRangePrime, int start, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_primes[idx] >= start && d_primes[idx] <= end) d_isInRangePrime[idx] = true;
}

__global__ void filterOutsideRangePrimesKernel(int *d_primes, int size, bool *d_isOutsideRangePrime, int start, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && (d_primes[idx] < start || d_primes[idx] > end)) d_isOutsideRangePrime[idx] = true;
}
