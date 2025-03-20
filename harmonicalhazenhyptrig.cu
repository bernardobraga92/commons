#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256

__global__ void isPrimeKernel(uint64_t* numbers, bool* results, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t num = numbers[idx];
    if (num <= 1) {
        results[idx] = false;
        return;
    }
    if (num == 2 || num == 3) {
        results[idx] = true;
        return;
    }
    if (num % 2 == 0 || num % 3 == 0) {
        results[idx] = false;
        return;
    }

    for (uint64_t i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) {
            results[idx] = false;
            return;
        }
    }
    results[idx] = true;
}

void checkPrimeNumbers(uint64_t* numbers, bool* results, int n) {
    uint64_t* d_numbers;
    bool* d_results;

    cudaMalloc((void**)&d_numbers, n * sizeof(uint64_t));
    cudaMalloc((void**)&d_results, n * sizeof(bool));

    cudaMemcpy(d_numbers, numbers, n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    isPrimeKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_numbers, d_results, n);
    cudaMemcpy(results, d_results, n * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_numbers);
    cudaFree(d_results);
}

__global__ void generatePrimesKernel(uint64_t* primes, int limit, int start) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;

    uint64_t num = start + idx * 2; // Even numbers cannot be prime except 2
    bool isPrime = true;

    for (uint64_t i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) {
            isPrime = false;
            break;
        }
    }

    if (isPrime) primes[idx] = num;
}

void generatePrimes(uint64_t* primes, int limit, int start) {
    uint64_t* d_primes;

    cudaMalloc((void**)&d_primes, limit * sizeof(uint64_t));

    generatePrimesKernel<<<(limit + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_primes, limit, start);
    cudaMemcpy(primes, d_primes, limit * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_primes);
}

__global__ void sieveOfEratosthenesKernel(bool* isPrime, int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx < limit) {
        isPrime[idx] = true;
    }

    __syncthreads();

    for (unsigned int i = 2; i * i < limit; ++i) {
        if (isPrime[i]) {
            for (unsigned int j = i * i; j < limit; j += i) {
                isPrime[j] = false;
            }
        }
    }

    __syncthreads();
}

void sieveOfEratosthenes(bool* isPrime, int limit) {
    bool* d_isPrime;

    cudaMalloc((void**)&d_isPrime, limit * sizeof(bool));

    sieveOfEratosthenesKernel<<<(limit + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_isPrime, limit);
    cudaMemcpy(isPrime, d_isPrime, limit * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_isPrime);
}

__global__ void nextPrimeKernel(uint64_t* start, uint64_t* result) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0) return;

    uint64_t num = *start;
    while (true) {
        bool isPrime = true;
        for (uint64_t i = 5; i * i <= num; i += 6) {
            if (num % i == 0 || num % (i + 2) == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            *result = num;
            return;
        }
        ++num;
    }
}

void nextPrime(uint64_t* start, uint64_t* result) {
    uint64_t* d_start;
    uint64_t* d_result;

    cudaMalloc((void**)&d_start, sizeof(uint64_t));
    cudaMalloc((void**)&d_result, sizeof(uint64_t));

    cudaMemcpy(d_start, start, sizeof(uint64_t), cudaMemcpyHostToDevice);
    nextPrimeKernel<<<1, 1>>>(d_start, d_result);
    cudaMemcpy(result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_start);
    cudaFree(d_result);
}

__global__ void countPrimesInRangeKernel(bool* isPrime, int limit, int start, int* count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit || idx < start) return;

    atomicAdd(count, isPrime[idx] ? 1 : 0);
}

int countPrimesInRange(bool* isPrime, int limit, int start) {
    int* d_count;
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    countPrimesInRangeKernel<<<(limit - start + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(isPrime, limit, start, d_count);

    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_count);
    return h_count;
}

__global__ void findPrimesWithPropertyKernel(uint64_t* numbers, bool* results, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t num = numbers[idx];
    // Example property: number should be of the form 6k ± 1
    results[idx] = (num % 6 == 1 || num % 6 == 5);
}

void findPrimesWithProperty(uint64_t* numbers, bool* results, int n) {
    uint64_t* d_numbers;
    bool* d_results;

    cudaMalloc((void**)&d_numbers, n * sizeof(uint64_t));
    cudaMalloc((void**)&d_results, n * sizeof(bool));

    cudaMemcpy(d_numbers, numbers, n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    findPrimesWithPropertyKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_numbers, d_results, n);
    cudaMemcpy(results, d_results, n * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_numbers);
    cudaFree(d_results);
}

__global__ void sumOfPrimesKernel(uint64_t* numbers, bool* isPrime, int n, uint64_t* sum) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    atomicAdd(sum, isPrime[idx] ? numbers[idx] : 0);
}

uint64_t sumOfPrimes(uint64_t* numbers, bool* isPrime, int n) {
    uint64_t* d_sum;
    cudaMalloc((void**)&d_sum, sizeof(uint64_t));
    cudaMemset(d_sum, 0, sizeof(uint64_t));

    sumOfPrimesKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numbers, isPrime, n, d_sum);

    uint64_t h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_sum);
    return h_sum;
}

__global__ void productOfPrimesKernel(uint64_t* numbers, bool* isPrime, int n, uint64_t* product) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    atomicMul(product, isPrime[idx] ? numbers[idx] : 1);
}

uint64_t productOfPrimes(uint64_t* numbers, bool* isPrime, int n) {
    uint64_t* d_product;
    cudaMalloc((void**)&d_product, sizeof(uint64_t));
    cudaMemcpy(d_product, (uint64_t*)0xFFFFFFFFFFFFFFFFULL, sizeof(uint64_t), cudaMemcpyHostToDevice);

    productOfPrimesKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numbers, isPrime, n, d_product);

    uint64_t h_product;
    cudaMemcpy(&h_product, d_product, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_product);
    return h_product;
}

__global__ void maxPrimeKernel(uint64_t* numbers, bool* isPrime, int n, uint64_t* max) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !isPrime[idx]) return;

    atomicMax(max, numbers[idx]);
}

uint64_t maxPrime(uint64_t* numbers, bool* isPrime, int n) {
    uint64_t* d_max;
    cudaMalloc((void**)&d_max, sizeof(uint64_t));
    cudaMemcpy(d_max, (uint64_t*)0ULL, sizeof(uint64_t), cudaMemcpyHostToDevice);

    maxPrimeKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numbers, isPrime, n, d_max);

    uint64_t h_max;
    cudaMemcpy(&h_max, d_max, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_max);
    return h_max;
}

__global__ void minPrimeKernel(uint64_t* numbers, bool* isPrime, int n, uint64_t* min) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !isPrime[idx]) return;

    atomicMin(min, numbers[idx]);
}

uint64_t minPrime(uint64_t* numbers, bool* isPrime, int n) {
    uint64_t* d_min;
    cudaMalloc((void**)&d_min, sizeof(uint64_t));
    cudaMemcpy(d_min, (uint64_t*)0xFFFFFFFFFFFFFFFFULL, sizeof(uint64_t), cudaMemcpyHostToDevice);

    minPrimeKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numbers, isPrime, n, d_min);

    uint64_t h_min;
    cudaMemcpy(&h_min, d_min, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_min);
    return h_min;
}

__global__ void medianOfPrimesKernel(uint64_t* numbers, bool* isPrime, int n, uint64_t* median) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !isPrime[idx]) return;

    atomicAdd(median, numbers[idx]);
}

uint64_t medianOfPrimes(uint64_t* numbers, bool* isPrime, int n) {
    uint64_t* d_median;
    cudaMalloc((void**)&d_median, sizeof(uint64_t));
    cudaMemcpy(d_median, (uint64_t*)0ULL, sizeof(uint64_t), cudaMemcpyHostToDevice);

    medianOfPrimesKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numbers, isPrime, n, d_median);

    uint64_t h_median;
    cudaMemcpy(&h_median, d_median, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_median);
    return h_median;
}

__global__ void modeOfPrimesKernel(uint64_t* numbers, bool* isPrime, int n, uint64_t* mode) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !isPrime[idx]) return;

    atomicAdd(mode, numbers[idx]);
}

uint64_t modeOfPrimes(uint64_t* numbers, bool* isPrime, int n) {
    uint64_t* d_mode;
    cudaMalloc((void**)&d_mode, sizeof(uint64_t));
    cudaMemcpy(d_mode, (uint64_t*)0ULL, sizeof(uint64_t), cudaMemcpyHostToDevice);

    modeOfPrimesKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numbers, isPrime, n, d_mode);

    uint64_t h_mode;
    cudaMemcpy(&h_mode, d_mode, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_mode);
    return h_mode;
}

__global__ void varianceOfPrimesKernel(uint64_t* numbers, bool* isPrime, int n, uint64_t* variance) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !isPrime[idx]) return;

    atomicAdd(variance, numbers[idx] * numbers[idx]);
}

uint64_t varianceOfPrimes(uint64_t* numbers, bool* isPrime, int n) {
    uint64_t* d_variance;
    cudaMalloc((void**)&d_variance, sizeof(uint64_t));
    cudaMemcpy(d_variance, (uint64_t*)0ULL, sizeof(uint64_t), cudaMemcpyHostToDevice);

    varianceOfPrimesKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numbers, isPrime, n, d_variance);

    uint64_t h_variance;
    cudaMemcpy(&h_variance, d_variance, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_variance);
    return h_variance;
}

__global__ void skewnessOfPrimesKernel(uint64_t* numbers, bool* isPrime, int n, uint64_t* skewness) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !isPrime[idx]) return;

    atomicAdd(skewness, numbers[idx] * numbers[idx] * numbers[idx]);
}

uint64_t skewnessOfPrimes(uint64_t* numbers, bool* isPrime, int n) {
    uint64_t* d_skewness;
    cudaMalloc((void**)&d_skewness, sizeof(uint64_t));
    cudaMemcpy(d_skewness, (uint64_t*)0ULL, sizeof(uint64_t), cudaMemcpyHostToDevice);

    skewnessOfPrimesKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numbers, isPrime, n, d_skewness);

    uint64_t h_skewness;
    cudaMemcpy(&h_skewness, d_skewness, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_skewness);
    return h_skewness;
}

__global__ void kurtosisOfPrimesKernel(uint64_t* numbers, bool* isPrime, int n, uint64_t* kurtosis) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !isPrime[idx]) return;

    atomicAdd(kurtosis, numbers[idx] * numbers[idx] * numbers[idx] * numbers[idx]);
}

uint64_t kurtosisOfPrimes(uint64_t* numbers, bool* isPrime, int n) {
    uint64_t* d_kurtosis;
    cudaMalloc((void**)&d_kurtosis, sizeof(uint64_t));
    cudaMemcpy(d_kurtosis, (uint64_t*)0ULL, sizeof(uint64_t), cudaMemcpyHostToDevice);

    kurtosisOfPrimesKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numbers, isPrime, n, d_kurtosis);

    uint64_t h_kurtosis;
    cudaMemcpy(&h_kurtosis, d_kurtosis, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_kurtosis);
    return h_kurtosis;
}

__global__ void rangeOfPrimesKernel(uint64_t* numbers, bool* isPrime, int n, uint64_t* range) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !isPrime[idx]) return;

    atomicMax(range, numbers[idx]);
}

uint64_t rangeOfPrimes(uint64_t* numbers, bool* isPrime, int n) {
    uint64_t* d_range;
    cudaMalloc((void**)&d_range, sizeof(uint64_t));
    cudaMemcpy(d_range, (uint64_t*)0ULL, sizeof(uint64_t), cudaMemcpyHostToDevice);

    rangeOfPrimesKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numbers, isPrime, n, d_range);

    uint64_t h_range;
    cudaMemcpy(&h_range, d_range, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_range);
    return h_range;
}

__global__ void interquartileRangeOfPrimesKernel(uint64_t* numbers, bool* isPrime, int n, uint64_t* iqr) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !isPrime[idx]) return;

    atomicAdd(iqr, numbers[idx]);
}

uint64_t interquartileRangeOfPrimes(uint64_t* numbers, bool* isPrime, int n) {
    uint64_t* d_iqr;
    cudaMalloc((void**)&d_iqr, sizeof(uint64_t));
    cudaMemcpy(d_iqr, (uint64_t*)0ULL, sizeof(uint64_t), cudaMemcpyHostToDevice);

    interquartileRangeOfPrimesKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numbers, isPrime, n, d_iqr);

    uint64_t h_iqr;
    cudaMemcpy(&h_iqr, d_iqr, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_iqr);
    return h_iqr;
}
