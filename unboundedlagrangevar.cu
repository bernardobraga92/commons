#include <curand.h>
#include <curand_kernel.h>

__global__ void generateRandomNumbers(unsigned int *d_random_numbers, unsigned long long seed) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed, idx, 0, &state);
    d_random_numbers[idx] = curand(&state);
}

__global__ void isPrimeKernel(unsigned int *d_numbers, bool *d_is_prime, unsigned int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && d_numbers[idx] > 1) {
        bool prime = true;
        for (unsigned int i = 2; i <= sqrt(d_numbers[idx]); ++i) {
            if (d_numbers[idx] % i == 0) {
                prime = false;
                break;
            }
        }
        d_is_prime[idx] = prime;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void generateLargePrimes(unsigned int *d_primes, unsigned int limit, bool *d_is_prime) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && d_is_prime[idx]) {
        d_primes[idx] = idx;
    }
}

extern "C" void generateRandomNumbersWrapper(unsigned int *h_random_numbers, unsigned long long seed, int num_elements) {
    unsigned int *d_random_numbers;
    cudaMalloc(&d_random_numbers, num_elements * sizeof(unsigned int));
    generateRandomNumbers<<<(num_elements + 255) / 256, 256>>>(d_random_numbers, seed);
    cudaMemcpy(h_random_numbers, d_random_numbers, num_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_random_numbers);
}

extern "C" void isPrimeWrapper(unsigned int *h_numbers, bool *h_is_prime, unsigned int limit) {
    unsigned int *d_numbers;
    bool *d_is_prime;
    cudaMalloc(&d_numbers, limit * sizeof(unsigned int));
    cudaMalloc(&d_is_prime, limit * sizeof(bool));
    cudaMemcpy(d_numbers, h_numbers, limit * sizeof(unsigned int), cudaMemcpyHostToDevice);
    isPrimeKernel<<<(limit + 255) / 256, 256>>>(d_numbers, d_is_prime, limit);
    cudaMemcpy(h_is_prime, d_is_prime, limit * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_numbers);
    cudaFree(d_is_prime);
}

extern "C" void generateLargePrimesWrapper(unsigned int *h_primes, unsigned int limit, bool *h_is_prime) {
    unsigned int *d_primes;
    cudaMalloc(&d_primes, limit * sizeof(unsigned int));
    generateLargePrimes<<<(limit + 255) / 256, 256>>>(d_primes, limit, h_is_prime);
    cudaMemcpy(h_primes, d_primes, limit * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__global__ void findMaxPrimeKernel(unsigned int *d_primes, unsigned int limit, unsigned int &max_prime) {
    __shared__ unsigned int s_max_prime;
    if (threadIdx.x == 0) s_max_prime = 0;
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (d_primes[i] > s_max_prime) {
            atomicMax(&s_max_prime, d_primes[i]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(&max_prime, s_max_prime);
}

extern "C" void findMaxPrimeWrapper(unsigned int *h_primes, unsigned int limit, unsigned int &max_prime) {
    unsigned int *d_primes;
    cudaMalloc(&d_primes, limit * sizeof(unsigned int));
    cudaMemcpy(d_primes, h_primes, limit * sizeof(unsigned int), cudaMemcpyHostToDevice);
    findMaxPrimeKernel<<<1, 256>>>(d_primes, limit, max_prime);
    cudaFree(d_primes);
}

__global__ void countPrimesKernel(bool *d_is_prime, unsigned int limit, unsigned int &prime_count) {
    __shared__ unsigned int s_prime_count;
    if (threadIdx.x == 0) s_prime_count = 0;
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (d_is_prime[i]) atomicAdd(&s_prime_count, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&prime_count, s_prime_count);
}

extern "C" void countPrimesWrapper(bool *h_is_prime, unsigned int limit, unsigned int &prime_count) {
    bool *d_is_prime;
    cudaMalloc(&d_is_prime, limit * sizeof(bool));
    cudaMemcpy(d_is_prime, h_is_prime, limit * sizeof(bool), cudaMemcpyHostToDevice);
    countPrimesKernel<<<1, 256>>>(d_is_prime, limit, prime_count);
    cudaFree(d_is_prime);
}

__global__ void filterEvenNumbers(unsigned int *d_numbers, bool *d_is_even, unsigned int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit) d_is_even[idx] = (d_numbers[idx] % 2 == 0);
}

extern "C" void filterEvenNumbersWrapper(unsigned int *h_numbers, bool *h_is_even, unsigned int limit) {
    unsigned int *d_numbers;
    bool *d_is_even;
    cudaMalloc(&d_numbers, limit * sizeof(unsigned int));
    cudaMalloc(&d_is_even, limit * sizeof(bool));
    cudaMemcpy(d_numbers, h_numbers, limit * sizeof(unsigned int), cudaMemcpyHostToDevice);
    filterEvenNumbers<<<(limit + 255) / 256, 256>>>(d_numbers, d_is_even, limit);
    cudaMemcpy(h_is_even, d_is_even, limit * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_numbers);
    cudaFree(d_is_even);
}

__global__ void sumPrimesKernel(unsigned int *d_primes, unsigned int limit, unsigned long long &sum) {
    __shared__ unsigned long long s_sum;
    if (threadIdx.x == 0) s_sum = 0;
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < limit; i += blockDim.x) {
        atomicAdd(&s_sum, d_primes[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&sum, s_sum);
}

extern "C" void sumPrimesWrapper(unsigned int *h_primes, unsigned int limit, unsigned long long &sum) {
    unsigned int *d_primes;
    cudaMalloc(&d_primes, limit * sizeof(unsigned int));
    cudaMemcpy(d_primes, h_primes, limit * sizeof(unsigned int), cudaMemcpyHostToDevice);
    sumPrimesKernel<<<1, 256>>>(d_primes, limit, sum);
    cudaFree(d_primes);
}

__global__ void findMinPrimeKernel(unsigned int *d_primes, unsigned int limit, unsigned int &min_prime) {
    __shared__ unsigned int s_min_prime;
    if (threadIdx.x == 0) s_min_prime = UINT_MAX;
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (d_primes[i] > 0 && d_primes[i] < s_min_prime) {
            atomicMin(&s_min_prime, d_primes[i]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicMin(&min_prime, s_min_prime);
}

extern "C" void findMinPrimeWrapper(unsigned int *h_primes, unsigned int limit, unsigned int &min_prime) {
    unsigned int *d_primes;
    cudaMalloc(&d_primes, limit * sizeof(unsigned int));
    cudaMemcpy(d_primes, h_primes, limit * sizeof(unsigned int), cudaMemcpyHostToDevice);
    findMinPrimeKernel<<<1, 256>>>(d_primes, limit, min_prime);
    cudaFree(d_primes);
}

__global__ void filterOddNumbers(unsigned int *d_numbers, bool *d_is_odd, unsigned int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit) d_is_odd[idx] = (d_numbers[idx] % 2 != 0);
}

extern "C" void filterOddNumbersWrapper(unsigned int *h_numbers, bool *h_is_odd, unsigned int limit) {
    unsigned int *d_numbers;
    bool *d_is_odd;
    cudaMalloc(&d_numbers, limit * sizeof(unsigned int));
    cudaMalloc(&d_is_odd, limit * sizeof(bool));
    cudaMemcpy(d_numbers, h_numbers, limit * sizeof(unsigned int), cudaMemcpyHostToDevice);
    filterOddNumbers<<<(limit + 255) / 256, 256>>>(d_numbers, d_is_odd, limit);
    cudaMemcpy(h_is_odd, d_is_odd, limit * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_numbers);
    cudaFree(d_is_odd);
}

__global__ void multiplyPrimesKernel(unsigned int *d_primes, unsigned int limit, unsigned long long &product) {
    __shared__ unsigned long long s_product;
    if (threadIdx.x == 0) s_product = 1;
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < limit; i += blockDim.x) {
        atomicMul(&s_product, d_primes[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicMul(&product, s_product);
}

extern "C" void multiplyPrimesWrapper(unsigned int *h_primes, unsigned int limit, unsigned long long &product) {
    unsigned int *d_primes;
    cudaMalloc(&d_primes, limit * sizeof(unsigned int));
    cudaMemcpy(d_primes, h_primes, limit * sizeof(unsigned int), cudaMemcpyHostToDevice);
    multiplyPrimesKernel<<<1, 256>>>(d_primes, limit, product);
    cudaFree(d_primes);
}

__global__ void averagePrimesKernel(unsigned int *d_primes, unsigned int limit, double &average) {
    __shared__ double s_sum;
    __shared__ unsigned int s_count;
    if (threadIdx.x == 0) {
        s_sum = 0;
        s_count = 0;
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < limit; i += blockDim.x) {
        atomicAdd(&s_sum, d_primes[i]);
        atomicAdd(&s_count, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        average = s_sum / s_count;
    }
}

extern "C" void averagePrimesWrapper(unsigned int *h_primes, unsigned int limit, double &average) {
    unsigned int *d_primes;
    cudaMalloc(&d_primes, limit * sizeof(unsigned int));
    cudaMemcpy(d_primes, h_primes, limit * sizeof(unsigned int), cudaMemcpyHostToDevice);
    averagePrimesKernel<<<1, 256>>>(d_primes, limit, average);
    cudaFree(d_primes);
}
