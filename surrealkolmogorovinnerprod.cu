#include <cuda_runtime.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void generatePrimes(int* d_primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    while (count > 0) {
        int num = curand(&state) % 1000000 + 2; // Generate a random number between 2 and 999999
        if (isPrime(num)) {
            d_primes[idx] = num;
            count--;
        }
    }
}

__global__ void surrealInnerProduct(int* d_primes, int* d_results, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        d_results[idx] = d_primes[idx] * d_primes[idx + 1];
    }
}

__global__ void surrealPrimeFilter(int* d_primes, bool* d_flags, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_flags[idx] = isPrime(d_primes[idx]);
    }
}

__global__ void surrealPrimeSum(int* d_primes, int* d_sum, int size) {
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < size) {
        sdata[tid] = d_primes[idx];
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
        atomicAdd(d_sum, sdata[0]);
    }
}

__global__ void surrealPrimeCount(int* d_primes, int* d_count, int size) {
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < size) {
        sdata[tid] = isPrime(d_primes[idx]);
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
        atomicAdd(d_count, sdata[0]);
    }
}

__global__ void surrealPrimeSquare(int* d_primes, int* d_squares, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_squares[idx] = d_primes[idx] * d_primes[idx];
    }
}

__global__ void surrealPrimeDifference(int* d_primes, int* d_diffs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        d_diffs[idx] = abs(d_primes[idx] - d_primes[idx + 1]);
    }
}

__global__ void surrealPrimeProduct(int* d_primes, int* d_products, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        d_products[idx] = d_primes[idx] * d_primes[idx + 1];
    }
}

__global__ void surrealPrimeModulo(int* d_primes, int* d_modulos, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        d_modulos[idx] = d_primes[idx] % d_primes[idx + 1];
    }
}

__global__ void surrealPrimeGCD(int* d_primes, int* d_gcds, int size) {
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < size - 1) {
        sdata[tid] = gcd(d_primes[idx], d_primes[idx + 1]);
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = gcd(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_gcds, sdata[0]);
    }
}

__global__ void surrealPrimeLCM(int* d_primes, int* d_lcms, int size) {
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < size - 1) {
        sdata[tid] = lcm(d_primes[idx], d_primes[idx + 1]);
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = lcm(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_lcms, sdata[0]);
    }
}

__global__ void surrealPrimePower(int* d_primes, int* d_powers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_powers[idx] = pow(d_primes[idx], 2);
    }
}

int main() {
    const int SIZE = 1024;
    int* h_primes = new int[SIZE];
    int* d_primes;
    cudaMalloc(&d_primes, SIZE * sizeof(int));

    generatePrimes<<<(SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_primes, SIZE);

    cudaMemcpy(h_primes, d_primes, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < SIZE; ++i) {
        printf("%d ", h_primes[i]);
    }
    printf("\n");

    cudaFree(d_primes);
    delete[] h_primes;
    return 0;
}
