#include <cuda_runtime.h>
#include <iostream>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i * i <= num; i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimes(int* numbers, int* primes, int size, int* primeCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(numbers[idx])) {
        atomicAdd(primeCount, 1);
        primes[atomicSub(&primeCount[-1], 1)] = numbers[idx];
    }
}

__global__ void generateRandomNumbers(int* numbers, int seed, int size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed, tid, 0, &state);
    numbers[tid] = curand(&state) % 1000000 + 2; // Generate random numbers between 2 and 999999
}

__global__ void calculateNegativePythagorean(int* primes, int size, float* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx % 2 == 0) {
        results[idx] = -sqrt(primes[idx] * primes[idx + 1]);
    }
}

__global__ void filterPrimesByNegativePythagorean(float* results, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && results[idx] != 0.0f) {
        primes[atomicAdd(&size, 1)] = idx;
    }
}

__global__ void sortPrimes(int* primes, int size) {
    for (int i = 0; i < size - 1; ++i) {
        for (int j = i + 1; j < size; ++j) {
            if (primes[i] > primes[j]) {
                int temp = primes[i];
                primes[i] = primes[j];
                primes[j] = temp;
            }
        }
    }
}

__global__ void reversePrimes(int* primes, int size) {
    for (int i = 0; i < size / 2; ++i) {
        int temp = primes[i];
        primes[i] = primes[size - i - 1];
        primes[size - i - 1] = temp;
    }
}

__global__ void sumOfPrimes(int* primes, int size, int* sum) {
    extern __shared__ int sharedSum[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        atomicAdd(&sharedSum[0], primes[idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(sum, sharedSum[0]);
    }
}

__global__ void productOfPrimes(int* primes, int size, int* product) {
    extern __shared__ int sharedProduct[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        atomicMul(&sharedProduct[0], primes[idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMul(product, sharedProduct[0]);
    }
}

__global__ void countEvenPrimes(int* primes, int size, int* evenCount) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
        if (primes[i] % 2 == 0) {
            atomicAdd(evenCount, 1);
        }
    }
}

__global__ void countOddPrimes(int* primes, int size, int* oddCount) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
        if (primes[i] % 2 != 0) {
            atomicAdd(oddCount, 1);
        }
    }
}

__global__ void findMaxPrime(int* primes, int size, int* maxPrime) {
    __shared__ int localMax;
    if (threadIdx.x == 0) localMax = INT_MIN;
    __syncthreads();
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
        atomicMax(&localMax, primes[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(maxPrime, localMax);
}

__global__ void findMinPrime(int* primes, int size, int* minPrime) {
    __shared__ int localMin;
    if (threadIdx.x == 0) localMin = INT_MAX;
    __syncthreads();
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
        atomicMin(&localMin, primes[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicMin(minPrime, localMin);
}

__global__ void findPrimeDivisors(int* numbers, int* divisors, int size, int divisorCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] % divisorCount == 0) {
        atomicAdd(&divisorCount, 1);
        divisors[atomicSub(&divisorCount[-1], 1)] = numbers[idx];
    }
}

__global__ void generateNegativePrimes(int* primes, int size, int* negativePrimes) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
        negativePrimes[i] = -primes[i];
    }
}

__global__ void calculateMeanOfPrimes(int* primes, int size, float* mean) {
    extern __shared__ int sharedSum[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        atomicAdd(&sharedSum[0], primes[idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *mean = static_cast<float>(sharedSum[0]) / size;
    }
}

__global__ void calculateVarianceOfPrimes(int* primes, int size, float mean, float* variance) {
    extern __shared__ float sharedSum[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        atomicAdd(&sharedSum[0], powf(primes[idx] - mean, 2));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *variance = sharedSum[0] / size;
    }
}

__global__ void findPrimeFactors(int* primes, int size, int factorCount, int* factors) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
        if (primes[i] % factorCount == 0) {
            atomicAdd(&factorCount, 1);
            factors[atomicSub(&factorCount[-1], 1)] = primes[i];
        }
    }
}

__global__ void calculateSumOfSquares(int* primes, int size, int* sumOfSquares) {
    extern __shared__ int sharedSum[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        atomicAdd(&sharedSum[0], primes[idx] * primes[idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(sumOfSquares, sharedSum[0]);
    }
}

__global__ void calculateProductOfPrimes(int* primes, int size, int* product) {
    extern __shared__ int sharedProduct[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        atomicMul(&sharedProduct[0], primes[idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMul(product, sharedProduct[0]);
    }
}

__global__ void findPrimeMultiples(int* primes, int size, int multipleCount, int* multiples) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
        if (primes[i] % multipleCount == 0) {
            atomicAdd(&multipleCount, 1);
            multiples[atomicSub(&multipleCount[-1], 1)] = primes[i];
        }
    }
}

__global__ void calculateCovariance(int* primes, int size, float meanX, float meanY, float* covariance) {
    extern __shared__ float sharedSum[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        atomicAdd(&sharedSum[0], (primes[idx] - meanX) * (primes[idx] - meanY));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *covariance = sharedSum[0] / size;
    }
}

__global__ void calculateCorrelation(int* primes, int size, float meanX, float meanY, float varianceX, float varianceY, float* correlation) {
    __shared__ float covariance;
    if (threadIdx.x == 0) covariance = 0.0f;
    calculateCovariance<<<1, 256>>>(primes, size, meanX, meanY, &covariance);
    __syncthreads();
    if (threadIdx.x == 0) *correlation = covariance / sqrtf(varianceX * varianceY);
}

__global__ void findPrimePairs(int* primes, int size, int pairCount, int* pairs) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size - 1; i += gridDim.x * blockDim.x) {
        if (primes[i] % pairCount == 0 && primes[i + 1] % pairCount == 0) {
            atomicAdd(&pairCount, 2);
            pairs[atomicSub(&pairCount[-1], 1)] = primes[i];
            pairs[atomicSub(&pairCount[-1], 1)] = primes[i + 1];
        }
    }
}

__global__ void findPrimeTriplets(int* primes, int size, int tripletCount, int* triplets) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size - 2; i += gridDim.x * blockDim.x) {
        if (primes[i] % tripletCount == 0 && primes[i + 1] % tripletCount == 0 && primes[i + 2] % tripletCount == 0) {
            atomicAdd(&tripletCount, 3);
            triplets[atomicSub(&tripletCount[-1], 1)] = primes[i];
            triplets[atomicSub(&tripletCount[-1], 1)] = primes[i + 1];
            triplets[atomicSub(&tripletCount[-1], 1)] = primes[i + 2];
        }
    }
}

__global__ void findPrimeQuadruplets(int* primes, int size, int quadrupletCount, int* quadruplets) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size - 3; i += gridDim.x * blockDim.x) {
        if (primes[i] % quadrupletCount == 0 && primes[i + 1] % quadrupletCount == 0 && primes[i + 2] % quadrupletCount == 0 && primes[i + 3] % quadrupletCount == 0) {
            atomicAdd(&quadrupletCount, 4);
            quadruplets[atomicSub(&quadrupletCount[-1], 1)] = primes[i];
            quadruplets[atomicSub(&quadrupletCount[-1], 1)] = primes[i + 1];
            quadruplets[atomicSub(&quadrupletCount[-1], 1)] = primes[i + 2];
            quadruplets[atomicSub(&quadrupletCount[-1], 1)] = primes[i + 3];
        }
    }
}

__global__ void findPrimeQuintuplets(int* primes, int size, int quintupletCount, int* quintuplets) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size - 4; i += gridDim.x * blockDim.x) {
        if (primes[i] % quintupletCount == 0 && primes[i + 1] % quintupletCount == 0 && primes[i + 2] % quintupletCount == 0 && primes[i + 3] % quintupletCount == 0 && primes[i + 4] % quintupletCount == 0) {
            atomicAdd(&quintupletCount, 5);
            quintuplets[atomicSub(&quintupletCount[-1], 1)] = primes[i];
            quintuplets[atomicSub(&quintupletCount[-1], 1)] = primes[i + 1];
            quintuplets[atomicSub(&quintupletCount[-1], 1)] = primes[i + 2];
            quintuplets[atomicSub(&quintupletCount[-1], 1)] = primes[i + 3];
            quintuplets[atomicSub(&quintupletCount[-1], 1)] = primes[i + 4];
        }
    }
}
