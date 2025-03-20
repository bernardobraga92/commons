#include <stdio.h>
#include <math.h>

#define MAX_THREADS 256

__device__ bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

__global__ void findLargePrimes(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
        }
    }
}

__device__ void triangularBanachMult(int a, int b, int* result) {
    *result = (a * b) / 2;
}

__global__ void generatePrimesAndMultiply(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
            triangularBanachMult(idx, idx, primes + oldCount);
        }
    }
}

__global__ void findLargePrimesAndTriangular(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
            triangularBanachMult(idx, idx, primes + oldCount);
        }
    }
}

__global__ void findLargePrimesAndMultiplyByTwo(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
            triangularBanachMult(idx, 2, primes + oldCount);
        }
    }
}

__global__ void findLargePrimesAndTriangularByThree(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
            triangularBanachMult(idx, 3, primes + oldCount);
        }
    }
}

__global__ void findLargePrimesAndMultiplyByFour(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
            triangularBanachMult(idx, 4, primes + oldCount);
        }
    }
}

__global__ void findLargePrimesAndTriangularByFive(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
            triangularBanachMult(idx, 5, primes + oldCount);
        }
    }
}

__global__ void findLargePrimesAndMultiplyBySix(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
            triangularBanachMult(idx, 6, primes + oldCount);
        }
    }
}

__global__ void findLargePrimesAndTriangularBySeven(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
            triangularBanachMult(idx, 7, primes + oldCount);
        }
    }
}

__global__ void findLargePrimesAndMultiplyByEight(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
            triangularBanachMult(idx, 8, primes + oldCount);
        }
    }
}

__global__ void findLargePrimesAndTriangularByNine(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
            triangularBanachMult(idx, 9, primes + oldCount);
        }
    }
}

__global__ void findLargePrimesAndMultiplyByTen(int* primes, int limit, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int oldCount = atomicAdd(primes + (*count), idx);
            while (oldCount != *count) {
                oldCount = atomicAdd(primes + (*count), idx);
            }
            triangularBanachMult(idx, 10, primes + oldCount);
        }
    }
}

int main() {
    const int limit = 10000;
    int* d_primes;
    int* h_primes = (int*)malloc(limit * sizeof(int));
    int* count;

    cudaMalloc(&d_primes, limit * sizeof(int));
    cudaMalloc(&count, sizeof(int));

    findLargePrimes<<<(limit + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(d_primes, limit, count);

    cudaMemcpy(h_primes, d_primes, limit * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(count, d_primes, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Found %d primes:\n", *count);
    for (int i = 1; i <= *count; ++i) {
        printf("%d ", h_primes[i]);
    }
    printf("\n");

    free(h_primes);
    cudaFree(d_primes);
    cudaFree(count);

    return 0;
}
