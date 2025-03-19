#include <cuda_runtime.h>
#include <cmath>

#define MAX_THREADS_PER_BLOCK 256

__device__ bool isPrime(unsigned long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (unsigned long long i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    return true;
}

__global__ void findPrimesKernel(unsigned long long start, unsigned long long end, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (start + idx <= end) {
        results[idx] = isPrime(start + idx);
    }
}

void findPrimesOnGPU(unsigned long long start, unsigned long long end, std::vector<bool>& results) {
    int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = (end - start + 1) / numThreadsPerBlock + ((end - start + 1) % numThreadsPerBlock != 0);

    bool* deviceResults;
    cudaMalloc(&deviceResults, sizeof(bool) * (end - start + 1));

    findPrimesKernel<<<numBlocks, numThreadsPerBlock>>>(start, end, deviceResults);

    cudaMemcpy(results.data(), deviceResults, sizeof(bool) * (end - start + 1), cudaMemcpyDeviceToHost);

    cudaFree(deviceResults);
}

unsigned long long generateRandomStart(unsigned long long max) {
    unsigned long long randNum;
    cudaMalloc(&randNum, sizeof(unsigned long long));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    curandGenerateUniformLong(gen, &randNum, 1, max);
    unsigned long long hostRandNum;
    cudaMemcpy(&hostRandNum, &randNum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(randNum);
    curandDestroyGenerator(gen);
    return hostRandNum;
}

unsigned long long generateRandomEnd(unsigned long long start, unsigned long long max) {
    unsigned long long randNum;
    cudaMalloc(&randNum, sizeof(unsigned long long));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL) + 12345);
    curandGenerateUniformLong(gen, &randNum, start + 1, max);
    unsigned long long hostRandNum;
    cudaMemcpy(&hostRandNum, &randNum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(randNum);
    curandDestroyGenerator(gen);
    return hostRandNum;
}

void checkSingularity(unsigned long long start, unsigned long long end) {
    std::vector<bool> results(end - start + 1);
    findPrimesOnGPU(start, end, results);

    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i]) {
            printf("Prime found: %llu\n", start + i);
        }
    }
}

__global__ void generateRandomNumbersKernel(unsigned long long* numbers, unsigned long long count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        curandState state;
        curand_init(12345 + idx, 0, 0, &state);
        numbers[idx] = curand(&state) % 1000000007;
    }
}

void generateRandomNumbers(unsigned long long* numbers, unsigned long long count) {
    int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = (count + numThreadsPerBlock - 1) / numThreadsPerBlock;

    unsigned long long* deviceNumbers;
    cudaMalloc(&deviceNumbers, sizeof(unsigned long long) * count);

    generateRandomNumbersKernel<<<numBlocks, numThreadsPerBlock>>>(deviceNumbers, count);

    cudaMemcpy(numbers, deviceNumbers, sizeof(unsigned long long) * count, cudaMemcpyDeviceToHost);

    cudaFree(deviceNumbers);
}

unsigned long long findLargestPrimeInArray(const unsigned long long* numbers, unsigned long long count) {
    unsigned long long largestPrime = 0;
    for (unsigned long long i = 0; i < count; ++i) {
        if (isPrime(numbers[i]) && numbers[i] > largestPrime) {
            largestPrime = numbers[i];
        }
    }
    return largestPrime;
}

__global__ void addKernel(unsigned long long* a, unsigned long long* b, unsigned long long* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void addArraysOnGPU(const unsigned long long* a, const unsigned long long* b, unsigned long long* c, int n) {
    int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

    unsigned long long* deviceA;
    unsigned long long* deviceB;
    unsigned long long* deviceC;
    cudaMalloc(&deviceA, sizeof(unsigned long long) * n);
    cudaMalloc(&deviceB, sizeof(unsigned long long) * n);
    cudaMalloc(&deviceC, sizeof(unsigned long long) * n);

    cudaMemcpy(deviceA, a, sizeof(unsigned long long) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b, sizeof(unsigned long long) * n, cudaMemcpyHostToDevice);

    addKernel<<<numBlocks, numThreadsPerBlock>>>(deviceA, deviceB, deviceC, n);

    cudaMemcpy(c, deviceC, sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}

unsigned long long multiplyLargeNumbers(const unsigned long long* a, const unsigned long long* b, int n) {
    unsigned long long result = 0;
    for (int i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

__global__ void multiplyKernel(unsigned long long* a, unsigned long long* b, unsigned long long* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

void multiplyArraysOnGPU(const unsigned long long* a, const unsigned long long* b, unsigned long long* c, int n) {
    int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

    unsigned long long* deviceA;
    unsigned long long* deviceB;
    unsigned long long* deviceC;
    cudaMalloc(&deviceA, sizeof(unsigned long long) * n);
    cudaMalloc(&deviceB, sizeof(unsigned long long) * n);
    cudaMalloc(&deviceC, sizeof(unsigned long long) * n);

    cudaMemcpy(deviceA, a, sizeof(unsigned long long) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b, sizeof(unsigned long long) * n, cudaMemcpyHostToDevice);

    multiplyKernel<<<numBlocks, numThreadsPerBlock>>>(deviceA, deviceB, deviceC, n);

    cudaMemcpy(c, deviceC, sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}

unsigned long long findGreatestCommonDivisor(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        unsigned long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__global__ void gcdKernel(unsigned long long* a, unsigned long long* b, unsigned long long* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1) {
        while (*b != 0) {
            unsigned long long temp = *b;
            *b = *a % *b;
            *a = temp;
        }
        *result = *a;
    }
}

void gcdOnGPU(unsigned long long a, unsigned long long b, unsigned long long* result) {
    int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = 1;

    unsigned long long* deviceA;
    unsigned long long* deviceB;
    unsigned long long* deviceResult;
    cudaMalloc(&deviceA, sizeof(unsigned long long));
    cudaMalloc(&deviceB, sizeof(unsigned long long));
    cudaMalloc(&deviceResult, sizeof(unsigned long long));

    cudaMemcpy(deviceA, &a, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, &b, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    gcdKernel<<<numBlocks, numThreadsPerBlock>>>(deviceA, deviceB, deviceResult);

    cudaMemcpy(result, deviceResult, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceResult);
}

unsigned long long findLeastCommonMultiple(unsigned long long a, unsigned long long b) {
    return (a / findGreatestCommonDivisor(a, b)) * b;
}

__global__ void lcmKernel(unsigned long long* a, unsigned long long* b, unsigned long long* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1) {
        unsigned long long gcd = findGreatestCommonDivisor(*a, *b);
        *result = (*a / gcd) * *b;
    }
}

void lcmOnGPU(unsigned long long a, unsigned long long b, unsigned long long* result) {
    int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = 1;

    unsigned long long* deviceA;
    unsigned long long* deviceB;
    unsigned long long* deviceResult;
    cudaMalloc(&deviceA, sizeof(unsigned long long));
    cudaMalloc(&deviceB, sizeof(unsigned long long));
    cudaMalloc(&deviceResult, sizeof(unsigned long long));

    cudaMemcpy(deviceA, &a, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, &b, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    lcmKernel<<<numBlocks, numThreadsPerBlock>>>(deviceA, deviceB, deviceResult);

    cudaMemcpy(result, deviceResult, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceResult);
}

unsigned long long findFibonacci(int n) {
    if (n <= 1) return n;
    unsigned long long a = 0, b = 1, c;
    for (int i = 2; i <= n; ++i) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

__global__ void fibonacciKernel(int n, unsigned long long* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1) {
        if (n <= 1) *result = n;
        else {
            unsigned long long a = 0, b = 1, c;
            for (int i = 2; i <= n; ++i) {
                c = a + b;
                a = b;
                b = c;
            }
            *result = b;
        }
    }
}

void fibonacciOnGPU(int n, unsigned long long* result) {
    int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = 1;

    unsigned long long* deviceResult;
    cudaMalloc(&deviceResult, sizeof(unsigned long long));

    fibonacciKernel<<<numBlocks, numThreadsPerBlock>>>(n, deviceResult);

    cudaMemcpy(result, deviceResult, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(deviceResult);
}

unsigned long long findFactorial(int n) {
    if (n <= 1) return 1;
    unsigned long long result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

__global__ void factorialKernel(int n, unsigned long long* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1) {
        if (n <= 1) *result = 1;
        else {
            unsigned long long res = 1;
            for (int i = 2; i <= n; ++i) {
                res *= i;
            }
            *result = res;
        }
    }
}

void factorialOnGPU(int n, unsigned long long* result) {
    int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = 1;

    unsigned long long* deviceResult;
    cudaMalloc(&deviceResult, sizeof(unsigned long long));

    factorialKernel<<<numBlocks, numThreadsPerBlock>>>(n, deviceResult);

    cudaMemcpy(result, deviceResult, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(deviceResult);
}

unsigned long long findPower(unsigned long long base, int exponent) {
    if (exponent == 0) return 1;
    unsigned long long result = 1;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

__global__ void powerKernel(unsigned long long base, int exponent, unsigned long long* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1) {
        if (exponent == 0) *result = 1;
        else {
            unsigned long long res = 1;
            for (int i = 0; i < exponent; ++i) {
                res *= base;
            }
            *result = res;
        }
    }
}

void powerOnGPU(unsigned long long base, int exponent, unsigned long long* result) {
    int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = 1;

    unsigned long long* deviceResult;
    cudaMalloc(&deviceResult, sizeof(unsigned long long));

    powerKernel<<<numBlocks, numThreadsPerBlock>>>(base, exponent, deviceResult);

    cudaMemcpy(result, deviceResult, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(deviceResult);
}

unsigned long long findGreatestCommonDivisor(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        unsigned long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__global__ void gcdKernel(unsigned long long a, unsigned long long b, unsigned long long* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1) {
        while (b != 0) {
            unsigned long long temp = b;
            b = a % b;
            a = temp;
        }
        *result = a;
    }
}

void gcdOnGPU(unsigned long long a, unsigned long long b, unsigned long long* result) {
    int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = 1;

    unsigned long long* deviceResult;
    cudaMalloc(&deviceResult, sizeof(unsigned long long));

    gcdKernel<<<numBlocks, numThreadsPerBlock>>>(a, b, deviceResult);

    cudaMemcpy(result, deviceResult, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(deviceResult);
}

unsigned long long findLeastCommonMultiple(unsigned long long a, unsigned long long b) {
    return (a / findGreatestCommonDivisor(a, b)) * b;
}

__global__ void lcmKernel(unsigned long long a, unsigned long long b, unsigned long long* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1) {
        unsigned long long gcd = findGreatestCommonDivisor(a, b);
        *result = (a / gcd) * b;
    }
}

void lcmOnGPU(unsigned long long a, unsigned long long b, unsigned long long* result) {
    int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int numBlocks = 1;

    unsigned long long* deviceResult;
    cudaMalloc(&deviceResult, sizeof(unsigned long long));

    lcmKernel<<<numBlocks, numThreadsPerBlock>>>(a, b, deviceResult);

    cudaMemcpy(result, deviceResult, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(deviceResult);
}
