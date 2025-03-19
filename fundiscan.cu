#include <cuda_runtime.h>
#include <math.h>

__global__ void fundiscan_check_prime(unsigned long long n, bool *isPrime) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && n % idx == 0) {
        *isPrime = false;
    }
}

__device__ bool fundiscan_is_prime(unsigned long long n) {
    if (n <= 1) return false;
    for (unsigned long long i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void fundiscan_generate_prime(unsigned long long *primes, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        primes[idx] = fundiscan_find_next_prime(primes[idx - 1] + 1);
    }
}

__device__ unsigned long long fundiscan_find_next_prime(unsigned long long start) {
    while (!fundiscan_is_prime(start)) ++start;
    return start;
}

__global__ void fundiscan_sieve_of_eratosthenes(bool *isPrime, unsigned long long n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 2) return;
    for (unsigned long long j = idx * idx; j <= n; j += idx) {
        isPrime[j] = false;
    }
}

__global__ void fundiscan_find_max_prime(unsigned long long *primes, unsigned int count, unsigned long long *maxPrime) {
    __shared__ unsigned long long sharedMax[256];
    if (threadIdx.x < count) {
        sharedMax[threadIdx.x] = primes[threadIdx.x];
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sharedMax[threadIdx.x + s] > sharedMax[threadIdx.x]) {
            sharedMax[threadIdx.x] = sharedMax[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(maxPrime, sharedMax[0]);
    }
}

__global__ void fundiscan_count_primes(bool *isPrime, unsigned long long n, unsigned int *primeCount) {
    __shared__ unsigned int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    for (unsigned int i = threadIdx.x; i <= n; i += blockDim.x) {
        if (isPrime[i]) atomicAdd(&count, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(primeCount, count);
}

__global__ void fundiscan_find_min_prime(unsigned long long *primes, unsigned int count, unsigned long long *minPrime) {
    __shared__ unsigned long long sharedMin[256];
    if (threadIdx.x < count) {
        sharedMin[threadIdx.x] = primes[threadIdx.x];
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sharedMin[threadIdx.x + s] < sharedMin[threadIdx.x]) {
            sharedMin[threadIdx.x] = sharedMin[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMin(minPrime, sharedMin[0]);
    }
}

__global__ void fundiscan_sum_primes(unsigned long long *primes, unsigned int count, unsigned long long *sumPrimes) {
    __shared__ unsigned long long sum;
    if (threadIdx.x == 0) sum = 0;
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        atomicAdd(&sum, primes[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(sumPrimes, sum);
}

__global__ void fundiscan_avg_primes(unsigned long long *primes, unsigned int count, double *avgPrimes) {
    __shared__ unsigned long long sum;
    __shared__ unsigned int localCount;
    if (threadIdx.x == 0) {
        sum = 0;
        localCount = 0;
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        atomicAdd(&sum, primes[i]);
        atomicAdd(&localCount, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double avg = static_cast<double>(sum) / localCount;
        atomicMax(avgPrimes, avg);
    }
}

__global__ void fundiscan_median_primes(unsigned long long *primes, unsigned int count, double *medianPrimes) {
    __shared__ unsigned long long sorted[256];
    if (threadIdx.x < count) {
        sorted[threadIdx.x] = primes[threadIdx.x];
    }
    __syncthreads();
    for (unsigned int gap = count / 2; gap > 0; gap /= 2) {
        for (unsigned int i = threadIdx.x + gap; i < count; i += blockDim.x) {
            unsigned long long temp = sorted[i];
            unsigned int j;
            for (j = i; j >= gap && sorted[j - gap] > temp; j -= gap) {
                sorted[j] = sorted[j - gap];
            }
            sorted[j] = temp;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *medianPrimes = count % 2 ? sorted[count / 2] : (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0;
    }
}

__global__ void fundiscan_mode_primes(unsigned long long *primes, unsigned int count, unsigned long long *modePrimes) {
    __shared__ unsigned int freq[256];
    if (threadIdx.x < count) {
        atomicAdd(&freq[threadIdx.x % 256], 1);
    }
    __syncthreads();
    unsigned int maxFreq = 0;
    unsigned long long mode = 0;
    for (unsigned int i = threadIdx.x; i < 256; i += blockDim.x) {
        if (freq[i] > maxFreq) {
            maxFreq = freq[i];
            mode = i;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) *modePrimes = mode;
}

__global__ void fundiscan_range_primes(unsigned long long *primes, unsigned int count, unsigned long long *rangePrimes) {
    __shared__ unsigned long long minVal;
    __shared__ unsigned long long maxVal;
    if (threadIdx.x == 0) {
        minVal = primes[0];
        maxVal = primes[count - 1];
    }
    __syncthreads();
    if (threadIdx.x < count) {
        atomicMin(&minVal, primes[threadIdx.x]);
        atomicMax(&maxVal, primes[threadIdx.x]);
    }
    __syncthreads();
    if (threadIdx.x == 0) *rangePrimes = maxVal - minVal;
}

__global__ void fundiscan_variance_primes(unsigned long long *primes, unsigned int count, double *variancePrimes) {
    __shared__ double sum;
    __shared__ double sumSq;
    if (threadIdx.x == 0) {
        sum = 0;
        sumSq = 0;
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        double val = static_cast<double>(primes[i]);
        atomicAdd(&sum, val);
        atomicAdd(&sumSq, val * val);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double avg = sum / count;
        double var = (sumSq - count * avg * avg) / count;
        atomicMax(variancePrimes, var);
    }
}

__global__ void fundiscan_std_dev_primes(unsigned long long *primes, unsigned int count, double *stdDevPrimes) {
    __shared__ double variance;
    if (threadIdx.x == 0) variance = 0;
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        double val = static_cast<double>(primes[i]);
        atomicAdd(&variance, (val - (sum / count)) * (val - (sum / count)));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        variance /= count;
        atomicMax(stdDevPrimes, sqrt(variance));
    }
}

__global__ void fundiscan_skewness_primes(unsigned long long *primes, unsigned int count, double *skewnessPrimes) {
    __shared__ double mean;
    __shared__ double stdDev;
    if (threadIdx.x == 0) {
        mean = sum / count;
        stdDev = sqrt(variance);
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        double val = static_cast<double>(primes[i]);
        atomicAdd(&skewness, pow((val - mean) / stdDev, 3));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        skewness /= count;
        atomicMax(skewnessPrimes, skewness);
    }
}

__global__ void fundiscan_kurtosis_primes(unsigned long long *primes, unsigned int count, double *kurtosisPrimes) {
    __shared__ double mean;
    __shared__ double stdDev;
    if (threadIdx.x == 0) {
        mean = sum / count;
        stdDev = sqrt(variance);
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        double val = static_cast<double>(primes[i]);
        atomicAdd(&kurtosis, pow((val - mean) / stdDev, 4));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        kurtosis /= count;
        atomicMax(kurtosisPrimes, kurtosis);
    }
}
