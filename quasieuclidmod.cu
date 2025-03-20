#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void generateRandomPrimes(int *d_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    curandState state;
    curand_init(clock64(), idx, 0, &state);

    int candidate;
    do {
        candidate = curand(&state) % 1000000 + 2;
    } while (!isPrime(candidate));

    d_primes[idx] = candidate;
}

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void filterPrimes(const int *d_primes, int *d_filteredPrimes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_filteredPrimes[idx] = isPrime(d_primes[idx]) ? d_primes[idx] : 0;
}

__global__ void sieveOfEratosthenes(int *d_sieve, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    if (idx == 0 || idx == 1) d_sieve[idx] = 0;
    else d_sieve[idx] = 1;

    __syncthreads();

    for (int i = 2; i * i <= size; ++i) {
        if (d_sieve[i]) {
            for (int j = i * i; j < size; j += i) {
                d_sieve[j] = 0;
            }
        }
    }
}

__global__ void findMaxPrime(const int *d_primes, int *d_maxPrime, int size) {
    extern __shared__ int shared[];
    if (threadIdx.x == 0) shared[0] = 0;

    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        atomicMax(&shared[0], d_primes[i]);
    }

    __syncthreads();

    if (threadIdx.x == 0) *d_maxPrime = shared[0];
}

__global__ void addOneToPrimes(int *d_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] += isPrime(d_primes[idx]) ? 1 : 0;
}

__global__ void multiplyByTwo(int *d_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] *= 2;
}

__global__ void squarePrimes(int *d_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] = isPrime(d_primes[idx]) ? d_primes[idx] * d_primes[idx] : 0;
}

__global__ void subtractOne(int *d_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx]--;
}

__global__ void findMinPrime(const int *d_primes, int *d_minPrime, int size) {
    extern __shared__ int shared[];
    if (threadIdx.x == 0) shared[0] = INT_MAX;

    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        atomicMin(&shared[0], d_primes[i]);
    }

    __syncthreads();

    if (threadIdx.x == 0) *d_minPrime = shared[0];
}

__global__ void moduloPrimes(int *d_primes, int mod, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] = isPrime(d_primes[idx]) ? d_primes[idx] % mod : 0;
}

__global__ void countPrimes(const int *d_primes, int *d_count, int size) {
    extern __shared__ int shared[];
    if (threadIdx.x == 0) shared[0] = 0;

    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        atomicAdd(&shared[0], isPrime(d_primes[i]));
    }

    __syncthreads();

    if (threadIdx.x == 0) *d_count = shared[0];
}

__global__ void doublePrimes(int *d_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] *= 2;
}

__global__ void divideByTwo(int *d_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] /= 2;
}

__global__ void togglePrimeBits(int *d_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] = isPrime(d_primes[idx]) ? ~d_primes[idx] : 0;
}

__global__ void incrementPrimes(int *d_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] += isPrime(d_primes[idx]);
}

__global__ void decrementPrimes(int *d_primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] -= isPrime(d_primes[idx]);
}

__global__ void XORPrimes(int *d_primes, int value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] ^= isPrime(d_primes[idx]) ? value : 0;
}

__global__ void ORPrimes(int *d_primes, int value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] |= isPrime(d_primes[idx]) ? value : 0;
}

__global__ void ANDPrimes(int *d_primes, int value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] &= isPrime(d_primes[idx]) ? value : 0;
}

__global__ void shiftLeftPrimes(int *d_primes, int shift, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] <<= shift;
}

__global__ void shiftRightPrimes(int *d_primes, int shift, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_primes[idx] >>= shift;
}

int main() {
    // Example usage
    const int size = 1024;
    int *h_primes = new int[size];
    int *d_primes, *d_filteredPrimes;

    cudaMalloc(&d_primes, size * sizeof(int));
    cudaMalloc(&d_filteredPrimes, size * sizeof(int));

    generateRandomPrimes<<<(size + 255) / 256, 256>>>(d_primes, size);
    cudaMemcpy(h_primes, d_primes, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Further processing with other kernels...

    cudaFree(d_primes);
    cudaFree(d_filteredPrimes);
    delete[] h_primes;

    return 0;
}
