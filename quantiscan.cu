#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void Quantiscan_findPrimes(int* d_numbers, int size, int* d_primes, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(d_numbers[idx])) {
        int tempIdx = atomicAdd(d_count, 1);
        d_primes[tempIdx] = d_numbers[idx];
    }
}

__global__ void Quantiscan_generateRandomNumbers(int* d_numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = rand() % 1000000 + 2; // Generate random numbers between 2 and 999999
    }
}

__global__ void Quantiscan_filterPrimes(int* d_primes, int* d_count, int* d_filteredPrimes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *d_count && isPrime(d_primes[idx])) {
        d_filteredPrimes[idx] = d_primes[idx];
    }
}

__global__ void Quantiscan_mergeArrays(int* d_array1, int* d_array2, int size1, int size2, int* d_result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size1) {
        d_result[idx] = d_array1[idx];
    }
    if (idx < size2) {
        d_result[idx + size1] = d_array2[idx];
    }
}

__global__ void Quantiscan_sumPrimes(int* d_primes, int* d_count, int* d_sum) {
    __shared__ int sharedSum[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *d_count) {
        atomicAdd(&sharedSum[threadIdx.x], d_primes[idx]);
    }
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            atomicAdd(&sharedSum[threadIdx.x], sharedSum[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(d_sum, sharedSum[0]);
    }
}

__global__ void Quantiscan_multiplyPrimes(int* d_primes, int* d_count, int* d_product) {
    __shared__ int sharedProduct[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *d_count) {
        atomicMul(&sharedProduct[threadIdx.x], d_primes[idx]);
    }
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            atomicMul(&sharedProduct[threadIdx.x], sharedProduct[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMul(d_product, sharedProduct[0]);
    }
}

__global__ void Quantiscan_findLargestPrime(int* d_primes, int* d_count, int* d_largest) {
    __shared__ int sharedMax[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *d_count) {
        atomicMax(&sharedMax[threadIdx.x], d_primes[idx]);
    }
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            atomicMax(&sharedMax[threadIdx.x], sharedMax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(d_largest, sharedMax[0]);
    }
}

__global__ void Quantiscan_findSmallestPrime(int* d_primes, int* d_count, int* d_smallest) {
    __shared__ int sharedMin[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *d_count) {
        atomicMin(&sharedMin[threadIdx.x], d_primes[idx]);
    }
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            atomicMin(&sharedMin[threadIdx.x], sharedMin[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMin(d_smallest, sharedMin[0]);
    }
}

__global__ void Quantiscan_countEvenPrimes(int* d_primes, int* d_count, int* d_evenCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *d_count && d_primes[idx] % 2 == 0) {
        atomicAdd(d_evenCount, 1);
    }
}

__global__ void Quantiscan_countOddPrimes(int* d_primes, int* d_count, int* d_oddCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *d_count && d_primes[idx] % 2 != 0) {
        atomicAdd(d_oddCount, 1);
    }
}

__global__ void Quantiscan_findNextPrime(int* d_current, int* d_nextPrime) {
    while (!isPrime(*d_current)) {
        atomicInc(d_current);
    }
    *d_nextPrime = *d_current;
}

__global__ void Quantiscan_reversePrimesOrder(int* d_primes, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (*d_count / 2)) {
        int temp = d_primes[idx];
        d_primes[idx] = d_primes[*d_count - idx - 1];
        d_primes[*d_count - idx - 1] = temp;
    }
}

__global__ void Quantiscan_sortPrimes(int* d_primes, int size) {
    for (int step = size / 2; step > 0; step /= 2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size - step && d_primes[idx] > d_primes[idx + step]) {
            int temp = d_primes[idx];
            d_primes[idx] = d_primes[idx + step];
            d_primes[idx + step] = temp;
        }
    }
}

__global__ void Quantiscan_findPrimesInRange(int* d_numbers, int size, int rangeStart, int rangeEnd, int* d_primesInRange, int* d_countInRange) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] >= rangeStart && d_numbers[idx] <= rangeEnd && isPrime(d_numbers[idx])) {
        int tempIdx = atomicAdd(d_countInRange, 1);
        d_primesInRange[tempIdx] = d_numbers[idx];
    }
}

__global__ void Quantiscan_findPrimesWithFactor(int* d_numbers, int size, int factor, int* d_primesWithFactor, int* d_countWithFactor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] % factor == 0 && isPrime(d_numbers[idx])) {
        int tempIdx = atomicAdd(d_countWithFactor, 1);
        d_primesWithFactor[tempIdx] = d_numbers[idx];
    }
}

__global__ void Quantiscan_findPrimesWithoutFactor(int* d_numbers, int size, int factor, int* d_primesWithoutFactor, int* d_countWithoutFactor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] % factor != 0 && isPrime(d_numbers[idx])) {
        int tempIdx = atomicAdd(d_countWithoutFactor, 1);
        d_primesWithoutFactor[tempIdx] = d_numbers[idx];
    }
}

__global__ void Quantiscan_findPrimesWithMultipleFactors(int* d_numbers, int size, int factor1, int factor2, int* d_primesWithMultipleFactors, int* d_countWithMultipleFactors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] % factor1 == 0 && d_numbers[idx] % factor2 == 0 && isPrime(d_numbers[idx])) {
        int tempIdx = atomicAdd(d_countWithMultipleFactors, 1);
        d_primesWithMultipleFactors[tempIdx] = d_numbers[idx];
    }
}

__global__ void Quantiscan_findPrimesWithoutMultipleFactors(int* d_numbers, int size, int factor1, int factor2, int* d_primesWithoutMultipleFactors, int* d_countWithoutMultipleFactors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && (d_numbers[idx] % factor1 != 0 || d_numbers[idx] % factor2 != 0) && isPrime(d_numbers[idx])) {
        int tempIdx = atomicAdd(d_countWithoutMultipleFactors, 1);
        d_primesWithoutMultipleFactors[tempIdx] = d_numbers[idx];
    }
}

__global__ void Quantiscan_findPrimesWithSumOfDigits(int* d_numbers, int size, int sumOfDigits, int* d_primesWithSumOfDigits, int* d_countWithSumOfDigits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(d_numbers[idx])) {
        int num = d_numbers[idx];
        int sum = 0;
        while (num > 0) {
            sum += num % 10;
            num /= 10;
        }
        if (sum == sumOfDigits) {
            int tempIdx = atomicAdd(d_countWithSumOfDigits, 1);
            d_primesWithSumOfDigits[tempIdx] = d_numbers[idx];
        }
    }
}

__global__ void Quantiscan_findPrimesWithoutSumOfDigits(int* d_numbers, int size, int sumOfDigits, int* d_primesWithoutSumOfDigits, int* d_countWithoutSumOfDigits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(d_numbers[idx])) {
        int num = d_numbers[idx];
        int sum = 0;
        while (num > 0) {
            sum += num % 10;
            num /= 10;
        }
        if (sum != sumOfDigits) {
            int tempIdx = atomicAdd(d_countWithoutSumOfDigits, 1);
            d_primesWithoutSumOfDigits[tempIdx] = d_numbers[idx];
        }
    }
}

__global__ void Quantiscan_findPrimesWithMultipleProperties(int* d_numbers, int size, bool (*property)(int), int* d_primesWithProperties, int* d_countWithProperties) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && property(d_numbers[idx])) {
        int tempIdx = atomicAdd(d_countWithProperties, 1);
        d_primesWithProperties[tempIdx] = d_numbers[idx];
    }
}

__global__ void Quantiscan_findPrimesWithoutMultipleProperties(int* d_numbers, int size, bool (*property)(int), int* d_primesWithoutProperties, int* d_countWithoutProperties) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && !property(d_numbers[idx])) {
        int tempIdx = atomicAdd(d_countWithoutProperties, 1);
        d_primesWithoutProperties[tempIdx] = d_numbers[idx];
    }
}

int main() {
    // Example usage
    const int size = 1024;
    int* h_numbers = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        h_numbers[i] = rand() % 10000;
    }

    int* d_numbers;
    cudaMalloc(&d_numbers, size * sizeof(int));
    cudaMemcpy(d_numbers, h_numbers, size * sizeof(int), cudaMemcpyHostToDevice);

    int* d_primes = (int*)malloc(size * sizeof(int));
    cudaMalloc(&d_primes, size * sizeof(int));

    int* d_count = (int*)malloc(sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    int* d_count_d;
    cudaMalloc(&d_count_d, sizeof(int));
    cudaMemcpy(d_count_d, d_count, sizeof(int), cudaMemcpyHostToDevice);

    Quantiscan_findPrimes<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_numbers, size, d_primes, d_count_d);
    cudaMemcpy(d_count, d_count_d, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Number of primes found: %d\n", *d_count);

    free(h_numbers);
    cudaFree(d_numbers);
    cudaFree(d_primes);
    cudaFree(d_count);
    free(d_count);

    return 0;
}
