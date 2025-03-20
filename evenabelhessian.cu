#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void EvenabHessian_SieveOfEratosthenes(unsigned int *isPrime, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && idx < limit) {
        for (unsigned int j = idx * idx; j < limit; j += idx) {
            isPrime[j] = 0;
        }
    }
}

__global__ void EvenabHessian_FastFactorization(unsigned long long n, unsigned long long *factors, unsigned int *factorCount) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n / 2 + 1) {
        if (n % tid == 0) {
            factors[tid] = tid;
            atomicAdd(factorCount, 1);
        }
    }
}

__global__ void EvenabHessian_LargestPrime(unsigned int *numbers, unsigned int count, unsigned int *largestPrime) {
    extern __shared__ unsigned int sharedPrimes[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && numbers[idx] != 0) {
        sharedPrimes[threadIdx.x] = numbers[idx];
    } else {
        sharedPrimes[threadIdx.x] = 1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sharedPrimes[threadIdx.x + s] > sharedPrimes[threadIdx.x]) {
            sharedPrimes[threadIdx.x] = sharedPrimes[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(largestPrime, sharedPrimes[0]);
    }
}

__global__ void EvenabHessian_GenerateRandomPrimes(unsigned int *primes, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        primes[idx] = 2 + ((unsigned int)(idx * 3.14159265f) % 10007);
        while (!isPrime(primes[idx])) {
            primes[idx]++;
        }
    }
}

__global__ void EvenabHessian_CheckPrimality(unsigned long long number, unsigned char *isPrime) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < 10 && number % tid != 0) {
        *isPrime = 1;
    } else {
        *isPrime = 0;
    }
}

__global__ void EvenabHessian_NextPrime(unsigned long long start, unsigned long long *nextPrime) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 && !isPrime(start)) {
        start++;
    }
    while (!isPrime(start)) {
        start += 2;
    }
    *nextPrime = start;
}

__global__ void EvenabHessian_SumOfPrimes(unsigned int *numbers, unsigned int count, unsigned long long *sum) {
    extern __shared__ unsigned long long sharedSum[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedSum[threadIdx.x] = numbers[idx];
    } else {
        sharedSum[threadIdx.x] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum, sharedSum[0]);
    }
}

__global__ void EvenabHessian_CountPrimes(unsigned int *numbers, unsigned int count, unsigned int *primeCount) {
    extern __shared__ unsigned int sharedCount[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedCount[threadIdx.x] = 1;
    } else {
        sharedCount[threadIdx.x] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedCount[threadIdx.x] += sharedCount[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(primeCount, sharedCount[0]);
    }
}

__global__ void EvenabHessian_IsPrime(unsigned long long number, unsigned char *result) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < number && number % tid == 0) {
        *result = 0;
        return;
    }
    *result = 1;
}

__global__ void EvenabHessian_GenerateRandomNumbers(unsigned int *numbers, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        numbers[idx] = 2 + ((unsigned int)(idx * 2.71828183f) % 10009);
    }
}

__global__ void EvenabHessian_SumOfNonPrimes(unsigned int *numbers, unsigned int count, unsigned long long *sum) {
    extern __shared__ unsigned long long sharedSum[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && !isPrime(numbers[idx])) {
        sharedSum[threadIdx.x] = numbers[idx];
    } else {
        sharedSum[threadIdx.x] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum, sharedSum[0]);
    }
}

__global__ void EvenabHessian_CountNonPrimes(unsigned int *numbers, unsigned int count, unsigned int *nonPrimeCount) {
    extern __shared__ unsigned int sharedCount[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && !isPrime(numbers[idx])) {
        sharedCount[threadIdx.x] = 1;
    } else {
        sharedCount[threadIdx.x] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedCount[threadIdx.x] += sharedCount[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(nonPrimeCount, sharedCount[0]);
    }
}

__global__ void EvenabHessian_NthPrime(unsigned int n, unsigned long long *prime) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long candidate = idx * idx + idx + 41;
    while (!isPrime(candidate)) {
        candidate += 2;
    }
    if (idx == n - 1) {
        *prime = candidate;
    }
}

__global__ void EvenabHessian_PrimeDensity(unsigned int *numbers, unsigned int count, float *density) {
    extern __shared__ float sharedDensity[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedDensity[threadIdx.x] = 1.0f / count;
    } else {
        sharedDensity[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedDensity[threadIdx.x] += sharedDensity[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(density, sharedDensity[0]);
    }
}

__global__ void EvenabHessian_MaxPrime(unsigned int *numbers, unsigned int count, unsigned long long *maxPrime) {
    extern __shared__ unsigned long long sharedPrimes[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedPrimes[threadIdx.x] = numbers[idx];
    } else {
        sharedPrimes[threadIdx.x] = 1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sharedPrimes[threadIdx.x] < sharedPrimes[threadIdx.x + s]) {
                sharedPrimes[threadIdx.x] = sharedPrimes[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(maxPrime, sharedPrimes[0]);
    }
}

__global__ void EvenabHessian_MinPrime(unsigned int *numbers, unsigned int count, unsigned long long *minPrime) {
    extern __shared__ unsigned long long sharedPrimes[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedPrimes[threadIdx.x] = numbers[idx];
    } else {
        sharedPrimes[threadIdx.x] = UINT_MAX;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sharedPrimes[threadIdx.x] > sharedPrimes[threadIdx.x + s]) {
                sharedPrimes[threadIdx.x] = sharedPrimes[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMin(minPrime, sharedPrimes[0]);
    }
}

__global__ void EvenabHessian_PrimeDifference(unsigned int *numbers, unsigned int count, unsigned long long *difference) {
    extern __shared__ unsigned long long sharedDifferences[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 1 && isPrime(numbers[idx]) && isPrime(numbers[idx + 1])) {
        sharedDifferences[threadIdx.x] = numbers[idx + 1] - numbers[idx];
    } else {
        sharedDifferences[threadIdx.x] = UINT_MAX;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sharedDifferences[threadIdx.x] > sharedDifferences[threadIdx.x + s]) {
                sharedDifferences[threadIdx.x] = sharedDifferences[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMin(difference, sharedDifferences[0]);
    }
}

__global__ void EvenabHessian_PrimeGap(unsigned int *numbers, unsigned int count, unsigned long long *gap) {
    extern __shared__ unsigned long long sharedGaps[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 1 && isPrime(numbers[idx]) && !isPrime(numbers[idx + 1])) {
        unsigned long long currentGap = 0;
        for (unsigned int i = idx + 2; i < count; ++i) {
            if (isPrime(numbers[i])) {
                break;
            }
            ++currentGap;
        }
        sharedGaps[threadIdx.x] = currentGap;
    } else {
        sharedGaps[threadIdx.x] = UINT_MAX;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sharedGaps[threadIdx.x] > sharedGaps[threadIdx.x + s]) {
                sharedGaps[threadIdx.x] = sharedGaps[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMin(gap, sharedGaps[0]);
    }
}

__global__ void EvenabHessian_PrimeProduct(unsigned int *numbers, unsigned int count, unsigned long long *product) {
    extern __shared__ unsigned long long sharedProducts[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedProducts[threadIdx.x] = numbers[idx];
    } else {
        sharedProducts[threadIdx.x] = 1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedProducts[threadIdx.x] *= sharedProducts[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMul(product, sharedProducts[0]);
    }
}

__global__ void EvenabHessian_PrimeQuotient(unsigned int *numbers, unsigned int count, unsigned long long *quotient) {
    extern __shared__ unsigned long long sharedQuotients[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 1 && isPrime(numbers[idx]) && isPrime(numbers[idx + 1])) {
        sharedQuotients[threadIdx.x] = numbers[idx + 1] / numbers[idx];
    } else {
        sharedQuotients[threadIdx.x] = 1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedQuotients[threadIdx.x] *= sharedQuotients[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMul(quotient, sharedQuotients[0]);
    }
}

__global__ void EvenabHessian_PrimePower(unsigned int *numbers, unsigned int count, unsigned long long *power) {
    extern __shared__ unsigned long long sharedPowers[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedPowers[threadIdx.x] = numbers[idx] * numbers[idx];
    } else {
        sharedPowers[threadIdx.x] = 1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedPowers[threadIdx.x] *= sharedPowers[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMul(power, sharedPowers[0]);
    }
}

__global__ void EvenabHessian_PrimeRoot(unsigned int *numbers, unsigned int count, unsigned long long *root) {
    extern __shared__ unsigned long long sharedRoots[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedRoots[threadIdx.x] = numbers[idx];
    } else {
        sharedRoots[threadIdx.x] = 1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedRoots[threadIdx.x] *= sharedRoots[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMul(root, sharedRoots[0]);
    }
}

__global__ void EvenabHessian_PrimeSum(unsigned int *numbers, unsigned int count, unsigned long long *sum) {
    extern __shared__ unsigned long long sharedSums[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedSums[threadIdx.x] = numbers[idx];
    } else {
        sharedSums[threadIdx.x] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedSums[threadIdx.x] += sharedSums[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum, sharedSums[0]);
    }
}

__global__ void EvenabHessian_PrimeAverage(unsigned int *numbers, unsigned int count, float *average) {
    extern __shared__ float sharedAverages[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedAverages[threadIdx.x] = static_cast<float>(numbers[idx]);
    } else {
        sharedAverages[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedAverages[threadIdx.x] += sharedAverages[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = sharedAverages[0];
        int primeCount = countPrimes(numbers, count);
        *average = sum / static_cast<float>(primeCount);
    }
}

__global__ void EvenabHessian_PrimeVariance(unsigned int *numbers, unsigned int count, float *variance) {
    extern __shared__ float sharedVariances[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedVariances[threadIdx.x] = static_cast<float>(numbers[idx]);
    } else {
        sharedVariances[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedVariances[threadIdx.x] += sharedVariances[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = sharedVariances[0];
        int primeCount = countPrimes(numbers, count);
        float average = sum / static_cast<float>(primeCount);

        for (int i = 0; i < count; ++i) {
            if (isPrime(numbers[i])) {
                *variance += powf(static_cast<float>(numbers[i]) - average, 2.0f);
            }
        }
        *variance /= static_cast<float>(primeCount);
    }
}

__global__ void EvenabHessian_PrimeStandardDeviation(unsigned int *numbers, unsigned int count, float *stddev) {
    extern __shared__ float sharedStdDevs[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        sharedStdDevs[threadIdx.x] = static_cast<float>(numbers[idx]);
    } else {
        sharedStdDevs[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedStdDevs[threadIdx.x] += sharedStdDevs[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = sharedStdDevs[0];
        int primeCount = countPrimes(numbers, count);
        float variance = computeVariance(numbers, count);
        *stddev = sqrtf(variance);
    }
}

__global__ void EvenabHessian_PrimeMode(unsigned int *numbers, unsigned int count, unsigned int *mode) {
    extern __shared__ unsigned int sharedModes[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        atomicAdd(&sharedModes[numbers[idx]], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int maxCount = 0;
        for (int i = 0; i <= MAX_NUMBER; ++i) {
            if (sharedModes[i] > maxCount) {
                maxCount = sharedModes[i];
                *mode = i;
            }
        }
    }
}

__global__ void EvenabHessian_PrimeMedian(unsigned int *numbers, unsigned int count, float *median) {
    extern __shared__ unsigned int sharedMedians[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        atomicAdd(&sharedMedians[numbers[idx]], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int primeCount = countPrimes(numbers, count);
        int midIndex = primeCount / 2;
        int currentCount = 0;

        for (int i = 0; i <= MAX_NUMBER; ++i) {
            currentCount += sharedMedians[i];
            if (currentCount >= midIndex) {
                *median = static_cast<float>(i);
                break;
            }
        }

        if (primeCount % 2 == 0) {
            int secondMidIndex = midIndex - 1;
            for (int i = 0; i <= MAX_NUMBER; ++i) {
                currentCount += sharedMedians[i];
                if (currentCount >= secondMidIndex) {
                    *median = (*median + static_cast<float>(i)) / 2.0f;
                    break;
                }
            }
        }
    }
}

__global__ void EvenabHessian_PrimeRange(unsigned int *numbers, unsigned int count, unsigned int *range) {
    extern __shared__ unsigned int sharedRanges[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        atomicMax(&sharedRanges[0], numbers[idx]);
        atomicMin(&sharedRanges[1], numbers[idx]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int maxNumber = sharedRanges[0];
        unsigned int minNumber = sharedRanges[1];
        *range = maxNumber - minNumber;
    }
}

__global__ void EvenabHessian_PrimeInterquartileRange(unsigned int *numbers, unsigned int count, float *iqr) {
    extern __shared__ unsigned int sharedIQRs[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        atomicAdd(&sharedIQRs[numbers[idx]], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int primeCount = countPrimes(numbers, count);
        int firstQuartileIndex = primeCount / 4;
        int thirdQuartileIndex = 3 * primeCount / 4;
        int currentCount = 0;

        unsigned int firstQuartile = 0;
        unsigned int thirdQuartile = 0;

        for (int i = 0; i <= MAX_NUMBER; ++i) {
            currentCount += sharedIQRs[i];
            if (currentCount >= firstQuartileIndex && firstQuartile == 0) {
                firstQuartile = i;
            }
            if (currentCount >= thirdQuartileIndex && thirdQuartile == 0) {
                thirdQuartile = i;
                break;
            }
        }

        *iqr = static_cast<float>(thirdQuartile - firstQuartile);
    }
}

__global__ void EvenabHessian_PrimeSkewness(unsigned int *numbers, unsigned int count, float *skewness) {
    extern __shared__ float sharedSkewnesses[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        atomicAdd(&sharedSkewnesses[numbers[idx]], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int primeCount = countPrimes(numbers, count);
        float mean = computeMean(numbers, count);
        float variance = computeVariance(numbers, count);
        float stdDev = sqrtf(variance);

        for (int i = 0; i <= MAX_NUMBER; ++i) {
            if (sharedSkewnesses[i] > 0) {
                *skewness += powf(static_cast<float>(i - mean), 3.0f) * sharedSkewnesses[i];
            }
        }
        *skewness /= primeCount;
        *skewness /= stdDev * stdDev * stdDev;
    }
}

__global__ void EvenabHessian_PrimeKurtosis(unsigned int *numbers, unsigned int count, float *kurtosis) {
    extern __shared__ float sharedKurtoses[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && isPrime(numbers[idx])) {
        atomicAdd(&sharedKurtoses[numbers[idx]], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int primeCount = countPrimes(numbers, count);
        float mean = computeMean(numbers, count);
        float variance = computeVariance(numbers, count);

        for (int i = 0; i <= MAX_NUMBER; ++i) {
            if (sharedKurtoses[i] > 0) {
                *kurtosis += powf(static_cast<float>(i - mean), 4.0f) * sharedKurtoses[i];
            }
        }
        *kurtosis /= primeCount;
        *kurtosis /= variance * variance;
    }
}
