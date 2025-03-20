#include <curand_kernel.h>
#define blockSize 256

__global__ void generateRandomPrimes(unsigned int* d_primes, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        d_primes[idx] = curand(&state) % 1000000007;
    }
}

__global__ void isPrimeKernel(unsigned int* d_numbers, unsigned int count, bool* d_isPrime) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned int num = d_numbers[idx];
        if (num <= 1) d_isPrime[idx] = false;
        else if (num == 2) d_isPrime[idx] = true;
        else if (num % 2 == 0) d_isPrime[idx] = false;
        else {
            bool prime = true;
            for (unsigned int i = 3; i * i <= num; i += 2) {
                if (num % i == 0) {
                    prime = false;
                    break;
                }
            }
            d_isPrime[idx] = prime;
        }
    }
}

__global__ void filterPrimes(unsigned int* d_numbers, bool* d_isPrime, unsigned int count, unsigned int* d_filtered, int* d_countFiltered) {
    __shared__ unsigned int shared_primes[blockSize];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && d_isPrime[idx]) {
        shared_primes[threadIdx.x] = d_numbers[idx];
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(d_countFiltered, blockDim.x);
    }
}

__global__ void extractPrimes(unsigned int* d_filtered, unsigned int* d_numbers, bool* d_isPrime, unsigned int count, int filteredCount) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < filteredCount) {
        for (unsigned int i = 0; i < count; ++i) {
            if (d_isPrime[i]) {
                d_filtered[idx] = d_numbers[i];
                break;
            }
        }
    }
}

__global__ void generateRandomNumbers(unsigned int* d_numbers, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        d_numbers[idx] = curand(&state) % 1000000007;
    }
}

__global__ void isPrimeKernel2(unsigned int* d_numbers, unsigned int count, bool* d_isPrime) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned int num = d_numbers[idx];
        if (num <= 1) d_isPrime[idx] = false;
        else if (num == 2) d_isPrime[idx] = true;
        else if (num % 2 == 0) d_isPrime[idx] = false;
        else {
            bool prime = true;
            for (unsigned int i = 3; i * i <= num; i += 2) {
                if (num % i == 0) {
                    prime = false;
                    break;
                }
            }
            d_isPrime[idx] = prime;
        }
    }
}

__global__ void filterPrimes2(unsigned int* d_numbers, bool* d_isPrime, unsigned int count, unsigned int* d_filtered, int* d_countFiltered) {
    __shared__ unsigned int shared_primes[blockSize];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && d_isPrime[idx]) {
        shared_primes[threadIdx.x] = d_numbers[idx];
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(d_countFiltered, blockDim.x);
    }
}

__global__ void extractPrimes2(unsigned int* d_filtered, unsigned int* d_numbers, bool* d_isPrime, unsigned int count, int filteredCount) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < filteredCount) {
        for (unsigned int i = 0; i < count; ++i) {
            if (d_isPrime[i]) {
                d_filtered[idx] = d_numbers[i];
                break;
            }
        }
    }
}

__global__ void generateRandomNumbers2(unsigned int* d_numbers, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        d_numbers[idx] = curand(&state) % 1000000007;
    }
}

__global__ void isPrimeKernel3(unsigned int* d_numbers, unsigned int count, bool* d_isPrime) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned int num = d_numbers[idx];
        if (num <= 1) d_isPrime[idx] = false;
        else if (num == 2) d_isPrime[idx] = true;
        else if (num % 2 == 0) d_isPrime[idx] = false;
        else {
            bool prime = true;
            for (unsigned int i = 3; i * i <= num; i += 2) {
                if (num % i == 0) {
                    prime = false;
                    break;
                }
            }
            d_isPrime[idx] = prime;
        }
    }
}

__global__ void filterPrimes3(unsigned int* d_numbers, bool* d_isPrime, unsigned int count, unsigned int* d_filtered, int* d_countFiltered) {
    __shared__ unsigned int shared_primes[blockSize];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && d_isPrime[idx]) {
        shared_primes[threadIdx.x] = d_numbers[idx];
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(d_countFiltered, blockDim.x);
    }
}

__global__ void extractPrimes3(unsigned int* d_filtered, unsigned int* d_numbers, bool* d_isPrime, unsigned int count, int filteredCount) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < filteredCount) {
        for (unsigned int i = 0; i < count; ++i) {
            if (d_isPrime[i]) {
                d_filtered[idx] = d_numbers[i];
                break;
            }
        }
    }
}

__global__ void generateRandomNumbers3(unsigned int* d_numbers, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        d_numbers[idx] = curand(&state) % 1000000007;
    }
}

__global__ void isPrimeKernel4(unsigned int* d_numbers, unsigned int count, bool* d_isPrime) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned int num = d_numbers[idx];
        if (num <= 1) d_isPrime[idx] = false;
        else if (num == 2) d_isPrime[idx] = true;
        else if (num % 2 == 0) d_isPrime[idx] = false;
        else {
            bool prime = true;
            for (unsigned int i = 3; i * i <= num; i += 2) {
                if (num % i == 0) {
                    prime = false;
                    break;
                }
            }
            d_isPrime[idx] = prime;
        }
    }
}

__global__ void filterPrimes4(unsigned int* d_numbers, bool* d_isPrime, unsigned int count, unsigned int* d_filtered, int* d_countFiltered) {
    __shared__ unsigned int shared_primes[blockSize];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && d_isPrime[idx]) {
        shared_primes[threadIdx.x] = d_numbers[idx];
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(d_countFiltered, blockDim.x);
    }
}

__global__ void extractPrimes4(unsigned int* d_filtered, unsigned int* d_numbers, bool* d_isPrime, unsigned int count, int filteredCount) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < filteredCount) {
        for (unsigned int i = 0; i < count; ++i) {
            if (d_isPrime[i]) {
                d_filtered[idx] = d_numbers[i];
                break;
            }
        }
    }
}

__global__ void generateRandomNumbers4(unsigned int* d_numbers, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        d_numbers[idx] = curand(&state) % 1000000007;
    }
}
