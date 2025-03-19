#include <cuda_runtime.h>
#include <math.h>

#define MAX_THREADS_PER_BLOCK 256

__device__ int isPrime(int n) {
    if (n <= 1) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    for (int i = 3; i <= sqrt(n); i += 2)
        if (n % i == 0) return 0;
    return 1;
}

__device__ int isSchoenfliesPrime(int n) {
    return isPrime(n) && (n % 6 == 5 || n % 6 == 1);
}

__global__ void findPrimesInRange(int *d_primes, int start, int end, int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end - start) return;
    int num = start + idx;
    if (isSchoenfliesPrime(num)) atomicAdd(count, 1);
}

__global__ void generateRandomPrimes(int *d_primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    d_primes[idx] = rand() % (count * 1000000) + 2; // Ensure the numbers are reasonably large
}

__global__ void filterPrimes(int *d_input, int *d_output, int inputCount, int *outputCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inputCount) return;
    if (isSchoenfliesPrime(d_input[idx])) {
        int outIdx = atomicAdd(outputCount, 1);
        d_output[outIdx] = d_input[idx];
    }
}

__global__ void sieveOfErathostenes(int *d_primes, int n, int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_primes[idx] = 1;
    __syncthreads();
    for (int i = idx; i < n; i += n)
        d_primes[i] = 0;
    __syncthreads();
    if (d_primes[idx] == 1 && isSchoenfliesPrime(idx + 2)) atomicAdd(count, 1);
}

__global__ void checkPrimalityBatch(int *d_numbers, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    d_numbers[idx] = isSchoenfliesPrime(d_numbers[idx]);
}

__global__ void findNextSchoenfliesPrime(int *d_primes, int start, int *foundPrime) {
    for (int num = start; num < INT_MAX; ++num) {
        if (isSchoenfliesPrime(num)) {
            atomicExch(foundPrime, num);
            break;
        }
    }
}

__global__ void findPreviousSchoenfliesPrime(int *d_primes, int end, int *foundPrime) {
    for (int num = end; num > 2; --num) {
        if (isSchoenfliesPrime(num)) {
            atomicExch(foundPrime, num);
            break;
        }
    }
}

__global__ void countSchoenfliesPrimesInRange(int *d_primes, int start, int end, int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end - start) return;
    int num = start + idx;
    if (isSchoenfliesPrime(num)) atomicAdd(count, 1);
}

__global__ void sumSchoenfliesPrimesInRange(int *d_primes, int start, int end, int *sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localSum = 0;
    for (int i = idx; i < end - start; i += gridDim.x * blockDim.x)
        if (isSchoenfliesPrime(start + i)) localSum += start + i;
    atomicAdd(sum, localSum);
}

__global__ void findMaxSchoenfliesPrimeInRange(int *d_primes, int start, int end, int *maxPrime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end - start) return;
    int num = start + idx;
    if (isSchoenfliesPrime(num) && num > atomicMax(maxPrime, 0)) atomicExch(maxPrime, num);
}

__global__ void findMinSchoenfliesPrimeInRange(int *d_primes, int start, int end, int *minPrime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end - start) return;
    int num = start + idx;
    if (isSchoenfliesPrime(num) && num < atomicMin(minPrime, INT_MAX)) atomicExch(minPrime, num);
}

__global__ void shufflePrimes(int *d_primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int swapIdx = rand() % count;
    int temp = d_primes[idx];
    d_primes[idx] = d_primes[swapIdx];
    d_primes[swapIdx] = temp;
}

__global__ void reversePrimes(int *d_primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count / 2) return;
    int temp = d_primes[idx];
    d_primes[idx] = d_primes[count - idx - 1];
    d_primes[count - idx - 1] = temp;
}

__global__ void multiplyPrimes(int *d_primes, int count, int multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    d_primes[idx] *= multiplier;
}

__global__ void addOffsetToPrimes(int *d_primes, int count, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    d_primes[idx] += offset;
}

__global__ void findNthSchoenfliesPrime(int n, int *foundPrime) {
    for (int num = 2; num < INT_MAX; ++num) {
        if (isSchoenfliesPrime(num)) {
            --n;
            if (n == 0) {
                atomicExch(foundPrime, num);
                break;
            }
        }
    }
}

__global__ void findNthPreviousSchoenfliesPrime(int n, int *foundPrime) {
    for (int num = INT_MAX - 1; num > 2; --num) {
        if (isSchoenfliesPrime(num)) {
            --n;
            if (n == 0) {
                atomicExch(foundPrime, num);
                break;
            }
        }
    }
}

__global__ void findPrimesWithSum(int targetSum, int *d_primes, int count, int *foundPrime) {
    for (int i = 0; i < count - 1; ++i) {
        if (isSchoenfliesPrime(d_primes[i])) {
            for (int j = i + 1; j < count; ++j) {
                if (isSchoenfliesPrime(d_primes[j]) && d_primes[i] + d_primes[j] == targetSum) {
                    atomicExch(foundPrime, d_primes[i]);
                    break;
                }
            }
        }
    }
}

__global__ void findPrimesWithProduct(int targetProduct, int *d_primes, int count, int *foundPrime) {
    for (int i = 0; i < count - 1; ++i) {
        if (isSchoenfliesPrime(d_primes[i])) {
            for (int j = i + 1; j < count; ++j) {
                if (isSchoenfliesPrime(d_primes[j]) && d_primes[i] * d_primes[j] == targetProduct) {
                    atomicExch(foundPrime, d_primes[i]);
                    break;
                }
            }
        }
    }
}

__global__ void findPrimesWithDifference(int targetDifference, int *d_primes, int count, int *foundPrime) {
    for (int i = 0; i < count - 1; ++i) {
        if (isSchoenfliesPrime(d_primes[i])) {
            for (int j = i + 1; j < count; ++j) {
                if (isSchoenfliesPrime(d_primes[j]) && abs(d_primes[i] - d_primes[j]) == targetDifference) {
                    atomicExch(foundPrime, d_primes[i]);
                    break;
                }
            }
        }
    }
}

__global__ void findPrimesWithQuotient(int targetQuotient, int *d_primes, int count, int *foundPrime) {
    for (int i = 0; i < count - 1; ++i) {
        if (isSchoenfliesPrime(d_primes[i])) {
            for (int j = i + 1; j < count; ++j) {
                if (isSchoenfliesPrime(d_primes[j]) && d_primes[i] / d_primes[j] == targetQuotient) {
                    atomicExch(foundPrime, d_primes[i]);
                    break;
                }
            }
        }
    }
}
