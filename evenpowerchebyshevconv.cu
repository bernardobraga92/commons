#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void evenPowerChebyshevConv(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (int)(pow(data[idx], 2)) % 1000003;
    }
}

__global__ void primeCheckKernel(unsigned long long *data, bool *isPrime, int size) {
    unsigned long long num = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (num < size) {
        isPrime[num] = true;
        for (unsigned long long i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                isPrime[num] = false;
                break;
            }
        }
    }
}

__global__ void generatePrimes(unsigned long long *primes, int count) {
    unsigned long long num = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (num < count) {
        primes[num] = num;
    }
}

__global__ void sieveOfErathostenes(bool *isPrime, int size) {
    for (int p = 2; p * p < size; ++p) {
        if (isPrime[p]) {
            for (int i = p * p; i < size; i += p) {
                isPrime[i] = false;
            }
        }
    }
}

__global__ void sumPrimesKernel(unsigned long long *primes, unsigned long long *sum, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i]) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, cache[0]);
    }
}

__global__ void multiplyPrimesKernel(unsigned long long *primes, unsigned long long *result, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i]) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 1;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] *= cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMul(result, cache[0]);
    }
}

__global__ void countPrimesKernel(unsigned long long *primes, unsigned int *count, int size) {
    __shared__ unsigned int cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i]) {
        cache[tid] = 1;
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(count, cache[0]);
    }
}

__global__ void maxPrimeKernel(unsigned long long *primes, unsigned long long *maxPrime, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i]) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (cache[tid] < cache[tid + s]) {
                cache[tid] = cache[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(maxPrime, cache[0]);
    }
}

__global__ void minPrimeKernel(unsigned long long *primes, unsigned long long *minPrime, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i]) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = ULONG_MAX;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (cache[tid] > cache[tid + s]) {
                cache[tid] = cache[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(minPrime, cache[0]);
    }
}

__global__ void evenPowerKernel(unsigned long long *data, int size) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = pow(data[idx], 2);
    }
}

__global__ void chebyshevConvolutionKernel(float *data, float *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = cos(acos(data[idx]) * 0.5);
    }
}

__global__ void primeProductModKernel(unsigned long long *primes, unsigned long long *result, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i]) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 1;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] *= cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMul(result, cache[0]);
    }
}

__global__ void primeSumModKernel(unsigned long long *primes, unsigned long long *result, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i]) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, cache[0]);
    }
}

__global__ void primeCountModKernel(unsigned long long *primes, unsigned int *count, int size) {
    __shared__ unsigned int cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i]) {
        cache[tid] = 1;
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(count, cache[0]);
    }
}

__global__ void primeMaxModKernel(unsigned long long *primes, unsigned long long *maxPrime, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i]) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (cache[tid] < cache[tid + s]) {
                cache[tid] = cache[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(maxPrime, cache[0]);
    }
}

__global__ void primeMinModKernel(unsigned long long *primes, unsigned long long *minPrime, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i]) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = ULONG_MAX;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (cache[tid] > cache[tid + s]) {
                cache[tid] = cache[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(minPrime, cache[0]);
    }
}

__global__ void primeEvenProductKernel(unsigned long long *primes, unsigned long long *result, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i] % 2 == 0) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 1;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] *= cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMul(result, cache[0]);
    }
}

__global__ void primeOddProductKernel(unsigned long long *primes, unsigned long long *result, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i] % 2 != 0) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 1;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] *= cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMul(result, cache[0]);
    }
}

__global__ void primeEvenSumKernel(unsigned long long *primes, unsigned long long *result, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i] % 2 == 0) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, cache[0]);
    }
}

__global__ void primeOddSumKernel(unsigned long long *primes, unsigned long long *result, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i] % 2 != 0) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, cache[0]);
    }
}

__global__ void primeEvenCountKernel(unsigned long long *primes, unsigned int *result, int size) {
    __shared__ unsigned int cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i] % 2 == 0) {
        cache[tid] = 1;
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, cache[0]);
    }
}

__global__ void primeOddCountKernel(unsigned long long *primes, unsigned int *result, int size) {
    __shared__ unsigned int cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i] % 2 != 0) {
        cache[tid] = 1;
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, cache[0]);
    }
}

__global__ void primeEvenMaxKernel(unsigned long long *primes, unsigned long long *result, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i] % 2 == 0) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && cache[tid] < cache[tid + s]) {
            cache[tid] = cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(result, cache[0]);
    }
}

__global__ void primeOddMaxKernel(unsigned long long *primes, unsigned long long *result, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i] % 2 != 0) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && cache[tid] < cache[tid + s]) {
            cache[tid] = cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(result, cache[0]);
    }
}

__global__ void primeEvenMinKernel(unsigned long long *primes, unsigned long long *result, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i] % 2 == 0) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = ULONG_MAX;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && cache[tid] > cache[tid + s]) {
            cache[tid] = cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(result, cache[0]);
    }
}

__global__ void primeOddMinKernel(unsigned long long *primes, unsigned long long *result, int size) {
    __shared__ unsigned long long cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && primes[i] % 2 != 0) {
        cache[tid] = primes[i];
    } else {
        cache[tid] = ULONG_MAX;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && cache[tid] > cache[tid + s]) {
            cache[tid] = cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(result, cache[0]);
    }
}
