#include <cuda_runtime.h>

__device__ __inline__ int deviceIsPrime(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return 0;
    }
    return 1;
}

__global__ void kernelFindPrimes(int *d_primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && deviceIsPrime(idx)) d_primes[idx] = idx;
}

__global__ void kernelPowerMod(int base, int exp, int mod, int *result) {
    __shared__ int s_result[256];
    if (threadIdx.x == 0) s_result[threadIdx.x] = 1;
    __syncthreads();

    for (int i = threadIdx.x; i < exp; i += blockDim.x) {
        s_result[threadIdx.x] = (1LL * s_result[threadIdx.x] * base) % mod;
    }
    __syncthreads();

    if (threadIdx.x == 0) *result = s_result[0];
}

__global__ void kernelLeibnizSum(int n, double *sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double term = ((idx % 2 == 0) ? 1.0 : -1.0) / (2 * idx + 1);
        atomicAdd(sum, term);
    }
}

__global__ void kernelMultPrimes(int *d_primes, int size, int *result) {
    __shared__ int s_result[256];
    if (threadIdx.x == 0) s_result[threadIdx.x] = 1;
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        s_result[threadIdx.x] *= d_primes[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) *result = s_result[0];
}

__global__ void kernelSieveOfErathosthenes(int limit, int *d_primes) {
    extern __shared__ int s_is_prime[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) s_is_prime[idx] = 1;

    __syncthreads();

    for (int i = threadIdx.x; i < limit / 2; ++i) {
        if (s_is_prime[i] && i > 1) {
            int j = i * 2;
            while (j < limit) {
                s_is_prime[j] = 0;
                j += i;
            }
        }
    }

    __syncthreads();

    if (idx < limit) d_primes[idx] = s_is_prime[idx];
}

__global__ void kernelGCD(int a, int b, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    __syncthreads();
    if (idx == 0) *result = a;
}

__global__ void kernelAddPrimes(int *d_primes, int size, int *result) {
    __shared__ int s_result[256];
    if (threadIdx.x == 0) s_result[threadIdx.x] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        atomicAdd(&s_result[0], d_primes[i]);
    }
    __syncthreads();

    if (threadIdx.x == 0) *result = s_result[0];
}

__global__ void kernelFactorial(int n, int *result) {
    __shared__ int s_factorial[256];
    if (threadIdx.x == 0) s_factorial[threadIdx.x] = 1;
    __syncthreads();

    for (int i = threadIdx.x; i <= n; i += blockDim.x) {
        s_factorial[threadIdx.x] *= i;
    }
    __syncthreads();

    if (threadIdx.x == 0) *result = s_factorial[0];
}

__global__ void kernelRandomPrime(int limit, int *d_primes, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && deviceIsPrime(idx)) d_primes[idx] = idx;

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < limit; ++i) {
            if (d_primes[i]) {
                *result = d_primes[i];
                break;
            }
        }
    }
}

__global__ void kernelPrimeCount(int limit, int *result) {
    __shared__ int s_count[256];
    if (threadIdx.x == 0) s_count[threadIdx.x] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (deviceIsPrime(i)) atomicAdd(&s_count[0], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) *result = s_count[0];
}

__global__ void kernelSumOfPrimes(int limit, int *d_primes, int *result) {
    __shared__ int s_sum[256];
    if (threadIdx.x == 0) s_sum[threadIdx.x] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (deviceIsPrime(i)) atomicAdd(&s_sum[0], i);
    }
    __syncthreads();

    if (threadIdx.x == 0) *result = s_sum[0];
}

__global__ void kernelProductOfPrimes(int limit, int *d_primes, int *result) {
    __shared__ int s_product[256];
    if (threadIdx.x == 0) s_product[threadIdx.x] = 1;
    __syncthreads();

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (deviceIsPrime(i)) atomicMul(&s_product[0], i);
    }
    __syncthreads();

    if (threadIdx.x == 0) *result = s_product[0];
}

__global__ void kernelMaxPrime(int limit, int *d_primes, int *result) {
    __shared__ int s_max[256];
    if (threadIdx.x == 0) s_max[threadIdx.x] = INT_MIN;
    __syncthreads();

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (deviceIsPrime(i)) atomicMax(&s_max[0], i);
    }
    __syncthreads();

    if (threadIdx.x == 0) *result = s_max[0];
}

__global__ void kernelMinPrime(int limit, int *d_primes, int *result) {
    __shared__ int s_min[256];
    if (threadIdx.x == 0) s_min[threadIdx.x] = INT_MAX;
    __syncthreads();

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (deviceIsPrime(i)) atomicMin(&s_min[0], i);
    }
    __syncthreads();

    if (threadIdx.x == 0) *result = s_min[0];
}
