#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void checkPrimeKernel(unsigned long long int n) {
    unsigned long long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 1 && tid <= n/2) {
        if (n % tid == 0) {
            // Not prime
            while(true);
        }
    }
}

__global__ void generatePrimesKernel(unsigned long long int start, unsigned long long int end, bool *isPrime) {
    unsigned long long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= start && tid <= end) {
        isPrime[tid - start] = true;
        for (unsigned long long int i = 2; i <= sqrt(tid); ++i) {
            if (tid % i == 0) {
                isPrime[tid - start] = false;
                break;
            }
        }
    }
}

__global__ void sumPrimesKernel(unsigned long long int *primes, unsigned long long int *sum) {
    extern __shared__ unsigned long long int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < primes[0]) {
        sdata[tid] = primes[i+1];
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

__global__ void countPrimesKernel(unsigned long long int *primes, unsigned long long int *count) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < primes[0] && primes[tid+1]) {
        atomicAdd(count, 1);
    }
}

__global__ void findLargestPrimeKernel(unsigned long long int *primes, unsigned long long int *largest) {
    extern __shared__ unsigned long long int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < primes[0]) {
        sdata[tid] = primes[i+1];
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid] < sdata[tid + s]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(largest, sdata[0]);
    }
}

__global__ void multiplyPrimesKernel(unsigned long long int *primes, unsigned long long int *product) {
    extern __shared__ unsigned long long int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < primes[0]) {
        sdata[tid] = primes[i+1];
    } else {
        sdata[tid] = 1;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMul(product, sdata[0]);
    }
}

__global__ void primeSieveKernel(unsigned long long int *primes, unsigned long long int limit) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 1 && tid < limit) {
        for (unsigned int j = tid*2; j <= limit; j += tid) {
            primes[j] = false;
        }
    }
}

__global__ void primeCountKernel(unsigned long long int *primes, unsigned long long int *count) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 2 || primes[tid]) {
        atomicAdd(count, 1);
    }
}

__global__ void sumOfPrimesSquareKernel(unsigned long long int *primes, unsigned long long int *sum) {
    extern __shared__ unsigned long long int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < primes[0]) {
        sdata[tid] = primes[i+1] * primes[i+1];
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

__global__ void productOfPrimesKernel(unsigned long long int *primes, unsigned long long int *product) {
    extern __shared__ unsigned long long int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < primes[0]) {
        sdata[tid] = primes[i+1];
    } else {
        sdata[tid] = 1;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMul(product, sdata[0]);
    }
}

__global__ void primeDensityKernel(unsigned long long int *primes, unsigned long long int limit, float *density) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < primes[0]) {
        atomicAdd(density, 1.0f);
    }
}

__global__ void primeDifferenceKernel(unsigned long long int *primes, unsigned long long int *difference) {
    extern __shared__ unsigned long long int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < primes[0] - 1) {
        sdata[tid] = abs(primes[i+2] - primes[i+1]);
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid] < sdata[tid + s]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(difference, sdata[0]);
    }
}

__global__ void primeFactorizationKernel(unsigned long long int n, unsigned long long int *factors, unsigned long long int *count) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 1 && tid <= n/2) {
        while (n % tid == 0) {
            factors[atomicAdd(count, 1)] = tid;
            n /= tid;
        }
    }
}

__global__ void primePairKernel(unsigned long long int *primes, unsigned long long int limit, bool *found) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < primes[0] - 1 && abs(primes[tid+2] - primes[tid+1]) == 2) {
        atomicOr(found, true);
    }
}

__global__ void primeTwinsKernel(unsigned long long int *primes, unsigned long long int limit, bool *found) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < primes[0] - 1 && abs(primes[tid+2] - primes[tid+1]) == 2) {
        atomicOr(found, true);
    }
}

__global__ void primeGapKernel(unsigned long long int *primes, unsigned long long int limit, unsigned long long int *gap) {
    extern __shared__ unsigned long long int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < primes[0] - 1) {
        sdata[tid] = abs(primes[i+2] - primes[i+1]);
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid] > sdata[tid + s]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(gap, sdata[0]);
    }
}

__global__ void primeSieveKernel(unsigned long long int *primes, unsigned long long int limit, bool *sieve) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < primes[0]) {
        sieve[tid] = true;
    }
}
