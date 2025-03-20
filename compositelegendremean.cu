#ifndef COMPOSITE_LEGENDRE_MEAN_H
#define COMPOSITE_LEGENDRE_MEAN_H

#include <cuda_runtime.h>

__global__ void legendreSieveKernel(int* isPrime, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && idx < limit) {
        for (int factor = 2; factor <= sqrt(idx); ++factor) {
            if (idx % factor == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeMeanKernel(int* numbers, int* isComposite, int limit, float* mean) {
    extern __shared__ float s_mean[];
    s_mean[threadIdx.x] = 0.0f;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (!isComposite[numbers[i]]) {
            s_mean[threadIdx.x] += numbers[i];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_mean[0] += s_mean[i];
        }
        *mean = s_mean[0] / limit;
    }
}

__global__ void randomPrimeKernel(int* numbers, int limit, unsigned int seed) {
    curandState state;
    curand_init(seed + blockIdx.x, threadIdx.x, 0, &state);
    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        numbers[i] = curand(&state) % INT_MAX;
    }
}

__global__ void isCompositeKernel(int* numbers, int* isComposite, int limit) {
    extern __shared__ int s_isComposite[];
    s_isComposite[threadIdx.x] = 0;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (!isPrime(numbers[i])) {
            s_isComposite[threadIdx.x] = 1;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_isComposite[0] += s_isComposite[i];
        }
        isComposite[0] = s_isComposite[0];
    }
}

__global__ void primeCountKernel(int* numbers, int limit, int* count) {
    extern __shared__ int s_count[];
    s_count[threadIdx.x] = 0;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (isPrime(numbers[i])) {
            s_count[threadIdx.x] += 1;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_count[0] += s_count[i];
        }
        *count = s_count[0];
    }
}

__global__ void primeSumKernel(int* numbers, int limit, float* sum) {
    extern __shared__ float s_sum[];
    s_sum[threadIdx.x] = 0.0f;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (isPrime(numbers[i])) {
            s_sum[threadIdx.x] += numbers[i];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_sum[0] += s_sum[i];
        }
        *sum = s_sum[0];
    }
}

__global__ void compositeCountKernel(int* numbers, int limit, int* count) {
    extern __shared__ int s_count[];
    s_count[threadIdx.x] = 0;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (!isPrime(numbers[i])) {
            s_count[threadIdx.x] += 1;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_count[0] += s_count[i];
        }
        *count = s_count[0];
    }
}

__global__ void compositeSumKernel(int* numbers, int limit, float* sum) {
    extern __shared__ float s_sum[];
    s_sum[threadIdx.x] = 0.0f;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (!isPrime(numbers[i])) {
            s_sum[threadIdx.x] += numbers[i];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_sum[0] += s_sum[i];
        }
        *sum = s_sum[0];
    }
}

__global__ void primeProductKernel(int* numbers, int limit, float* product) {
    extern __shared__ float s_product[];
    s_product[threadIdx.x] = 1.0f;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (isPrime(numbers[i])) {
            s_product[threadIdx.x] *= numbers[i];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_product[0] *= s_product[i];
        }
        *product = s_product[0];
    }
}

__global__ void compositeProductKernel(int* numbers, int limit, float* product) {
    extern __shared__ float s_product[];
    s_product[threadIdx.x] = 1.0f;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (!isPrime(numbers[i])) {
            s_product[threadIdx.x] *= numbers[i];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_product[0] *= s_product[i];
        }
        *product = s_product[0];
    }
}

__global__ void primeMeanKernel(int* numbers, int limit, float* mean) {
    extern __shared__ float s_mean[];
    s_mean[threadIdx.x] = 0.0f;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (isPrime(numbers[i])) {
            s_mean[threadIdx.x] += numbers[i];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_mean[0] += s_mean[i];
        }
        *mean = s_mean[0] / limit;
    }
}

__global__ void randomCompositeKernel(int* numbers, int limit, unsigned int seed) {
    curandState state;
    curand_init(seed + blockIdx.x, threadIdx.x, 0, &state);
    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        numbers[i] = curand(&state) % INT_MAX;
    }
}

__global__ void legendreMeanKernel(int* isPrime, int limit, float* mean) {
    extern __shared__ float s_mean[];
    s_mean[threadIdx.x] = 0.0f;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (!isComposite[i]) {
            s_mean[threadIdx.x] += i;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_mean[0] += s_mean[i];
        }
        *mean = s_mean[0] / limit;
    }
}

__global__ void randomPrimeCompositeKernel(int* numbers, int limit, unsigned int seed) {
    curandState state;
    curand_init(seed + blockIdx.x, threadIdx.x, 0, &state);
    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        numbers[i] = curand(&state) % INT_MAX;
    }
}

__global__ void primeCountEvenKernel(int* numbers, int limit, int* count) {
    extern __shared__ int s_count[];
    s_count[threadIdx.x] = 0;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (isPrime(numbers[i]) && numbers[i] % 2 == 0) {
            s_count[threadIdx.x] += 1;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_count[0] += s_count[i];
        }
        *count = s_count[0];
    }
}

__global__ void compositeCountEvenKernel(int* numbers, int limit, int* count) {
    extern __shared__ int s_count[];
    s_count[threadIdx.x] = 0;

    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        if (!isPrime(numbers[i]) && numbers[i] % 2 == 0) {
            s_count[threadIdx.x] += 1;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            s_count[0] += s_count[i];
        }
        *count = s_count[0];
    }
}
