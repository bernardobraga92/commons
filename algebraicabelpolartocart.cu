#include <cuda_runtime.h>
#include <iostream>

__global__ void findPrimesKernel(unsigned long long int *d_primes, unsigned long long int limit) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= limit) {
        bool isPrime = true;
        for (unsigned long long int i = 2; i * i <= idx; ++i) {
            if (idx % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
    }
}

void findPrimes(unsigned long long int *h_primes, unsigned long long int limit) {
    unsigned long long int *d_primes;
    cudaMalloc((void **)&d_primes, (limit + 1) * sizeof(unsigned long long int));
    cudaMemcpy(d_primes, h_primes, (limit + 1) * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((limit + 1) / threadsPerBlock.x + 1);
    findPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, limit);

    cudaMemcpy(h_primes, d_primes, (limit + 1) * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__global__ void checkPrimalityKernel(unsigned long long int *d_numbers, bool *d_results, unsigned long long int count) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned long long int num = d_numbers[idx];
        bool isPrime = true;
        for (unsigned long long int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                isPrime = false;
                break;
            }
        }
        d_results[idx] = isPrime;
    }
}

void checkPrimality(unsigned long long int *h_numbers, bool *h_results, unsigned long long int count) {
    unsigned long long int *d_numbers;
    bool *d_results;
    cudaMalloc((void **)&d_numbers, count * sizeof(unsigned long long int));
    cudaMalloc((void **)&d_results, count * sizeof(bool));
    cudaMemcpy(d_numbers, h_numbers, count * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(count / threadsPerBlock.x + 1);
    checkPrimalityKernel<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_results, count);

    cudaMemcpy(h_results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_numbers);
    cudaFree(d_results);
}

__global__ void generatePrimesKernel(unsigned long long int *d_primes, unsigned long long int limit) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= limit) {
        bool isPrime = true;
        for (unsigned long long int i = 2; i * i <= idx; ++i) {
            if (idx % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            d_primes[idx] = idx;
        } else {
            d_primes[idx] = 0;
        }
    }
}

void generatePrimes(unsigned long long int *h_primes, unsigned long long int limit) {
    unsigned long long int *d_primes;
    cudaMalloc((void **)&d_primes, (limit + 1) * sizeof(unsigned long long int));
    cudaMemcpy(d_primes, h_primes, (limit + 1) * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((limit + 1) / threadsPerBlock.x + 1);
    generatePrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, limit);

    cudaMemcpy(h_primes, d_primes, (limit + 1) * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__global__ void primeCountKernel(unsigned long long int *d_primes, unsigned long long int limit, unsigned long long int *d_count) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= limit) {
        atomicAdd(d_count, d_primes[idx] != 0);
    }
}

unsigned long long int countPrimes(unsigned long long int *h_primes, unsigned long long int limit) {
    unsigned long long int *d_primes;
    unsigned long long int *d_count;
    cudaMalloc((void **)&d_primes, (limit + 1) * sizeof(unsigned long long int));
    cudaMalloc((void **)&d_count, sizeof(unsigned long long int));
    cudaMemcpy(d_primes, h_primes, (limit + 1) * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(unsigned long long int));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((limit + 1) / threadsPerBlock.x + 1);
    primeCountKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, limit, d_count);

    unsigned long long int count;
    cudaMemcpy(&count, d_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
    cudaFree(d_count);
    return count;
}

__global__ void nextPrimeKernel(unsigned long long int *d_start, unsigned long long int limit, unsigned long long int *d_nextPrime) {
    unsigned long long int start = *d_start;
    for (unsigned long long int i = start; i <= limit; ++i) {
        bool isPrime = true;
        for (unsigned long long int j = 2; j * j <= i; ++j) {
            if (i % j == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            atomicMin(d_nextPrime, i);
            break;
        }
    }
}

unsigned long long int findNextPrime(unsigned long long int start, unsigned long long int limit) {
    unsigned long long int *d_start;
    unsigned long long int *d_nextPrime;
    cudaMalloc((void **)&d_start, sizeof(unsigned long long int));
    cudaMalloc((void **)&d_nextPrime, sizeof(unsigned long long int));
    cudaMemcpy(d_start, &start, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    cudaMemset(d_nextPrime, ~0ull, sizeof(unsigned long long int));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(1);
    nextPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_start, limit, d_nextPrime);

    unsigned long long int nextPrime;
    cudaMemcpy(&nextPrime, d_nextPrime, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaFree(d_start);
    cudaFree(d_nextPrime);
    return nextPrime;
}

__global__ void largestPrimeKernel(unsigned long long int *d_primes, unsigned long long int limit, unsigned long long int *d_largestPrime) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= limit) {
        if (d_primes[idx] != 0) {
            atomicMax(d_largestPrime, d_primes[idx]);
        }
    }
}

unsigned long long int findLargestPrime(unsigned long long int *h_primes, unsigned long long int limit) {
    unsigned long long int *d_primes;
    unsigned long long int *d_largestPrime;
    cudaMalloc((void **)&d_primes, (limit + 1) * sizeof(unsigned long long int));
    cudaMalloc((void **)&d_largestPrime, sizeof(unsigned long long int));
    cudaMemcpy(d_primes, h_primes, (limit + 1) * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    cudaMemset(d_largestPrime, 0, sizeof(unsigned long long int));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((limit + 1) / threadsPerBlock.x + 1);
    largestPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, limit, d_largestPrime);

    unsigned long long int largestPrime;
    cudaMemcpy(&largestPrime, d_largestPrime, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
    cudaFree(d_largestPrime);
    return largestPrime;
}

__global__ void primeSumKernel(unsigned long long int *d_primes, unsigned long long int limit, unsigned long long int *d_sum) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= limit) {
        atomicAdd(d_sum, d_primes[idx]);
    }
}

unsigned long long int sumOfPrimes(unsigned long long int *h_primes, unsigned long long int limit) {
    unsigned long long int *d_primes;
    unsigned long long int *d_sum;
    cudaMalloc((void **)&d_primes, (limit + 1) * sizeof(unsigned long long int));
    cudaMalloc((void **)&d_sum, sizeof(unsigned long long int));
    cudaMemcpy(d_primes, h_primes, (limit + 1) * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(unsigned long long int));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((limit + 1) / threadsPerBlock.x + 1);
    primeSumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, limit, d_sum);

    unsigned long long int sum;
    cudaMemcpy(&sum, d_sum, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
    cudaFree(d_sum);
    return sum;
}

__global__ void primeProductKernel(unsigned long long int *d_primes, unsigned long long int limit, unsigned long long int *d_product) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= limit) {
        atomicMul(d_product, d_primes[idx]);
    }
}

unsigned long long int productOfPrimes(unsigned long long int *h_primes, unsigned long long int limit) {
    unsigned long long int *d_primes;
    unsigned long long int *d_product;
    cudaMalloc((void **)&d_primes, (limit + 1) * sizeof(unsigned long long int));
    cudaMalloc((void **)&d_product, sizeof(unsigned long long int));
    cudaMemcpy(d_primes, h_primes, (limit + 1) * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    cudaMemset(d_product, 1, sizeof(unsigned long long int));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((limit + 1) / threadsPerBlock.x + 1);
    primeProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, limit, d_product);

    unsigned long long int product;
    cudaMemcpy(&product, d_product, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
    cudaFree(d_product);
    return product;
}

__global__ void primeFactorizationKernel(unsigned long long int number, unsigned long long int *d_factors) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && number % idx == 0) {
        atomicAdd(d_factors, idx);
    }
}

unsigned long long int countPrimeFactors(unsigned long long int number) {
    unsigned long long int *d_factors;
    cudaMalloc((void **)&d_factors, sizeof(unsigned long long int));
    cudaMemset(d_factors, 0, sizeof(unsigned long long int));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(number / threadsPerBlock.x + 1);
    primeFactorizationKernel<<<blocksPerGrid, threadsPerBlock>>>(number, d_factors);

    unsigned long long int count;
    cudaMemcpy(&count, d_factors, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaFree(d_factors);
    return count;
}

__global__ void isPrimeKernel(unsigned long long int number, bool *d_isPrime) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && number % idx == 0) {
        atomicAnd(d_isPrime, false);
    }
}

bool checkIsPrime(unsigned long long int number) {
    bool *d_isPrime;
    cudaMalloc((void **)&d_isPrime, sizeof(bool));
    cudaMemset(d_isPrime, true, sizeof(bool));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(number / threadsPerBlock.x + 1);
    isPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(number, d_isPrime);

    bool result;
    cudaMemcpy(&result, d_isPrime, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_isPrime);
    return result;
}

int main() {
    // Example usage
    unsigned long long int limit = 100;
    unsigned long long int *h_primes = new unsigned long long int[limit + 1];
    memset(h_primes, 0, (limit + 1) * sizeof(unsigned long long int));

    generatePrimes(h_primes, limit);

    // Use the generated primes for other operations
    printf("Number of primes up to %llu: %d\n", limit, countOfPrimes(h_primes));
    printf("Sum of primes up to %llu: %llu\n", limit, sumOfPrimes(h_primes, limit));
    printf("Product of primes up to %llu: %llu\n", limit, productOfPrimes(h_primes, limit));

    delete[] h_primes;
    return 0;
}
