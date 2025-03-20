#include <cuda_runtime.h>
#include <math.h>

__device__ bool isPrime(unsigned long long n) {
    if (n <= 1) return false;
    for (unsigned long long i = 2; i * i <= n; ++i)
        if (n % i == 0) return false;
    return true;
}

__global__ void findPrimesKernel(unsigned long long *d_primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_primes[idx] = isPrime(idx + 2) ? idx + 2 : 0;
    }
}

__global__ void generateRandomPrimes(unsigned long long *d_randomPrimes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long candidate = rand() % (1ULL << 63);
        d_randomPrimes[idx] = isPrime(candidate) ? candidate : 0;
    }
}

__global__ void sumOfPrimesKernel(unsigned long long *d_primes, unsigned int size, unsigned long long *d_sum) {
    extern __shared__ unsigned long long sdata[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    if (idx < size)
        sdata[tid] = d_primes[idx];
    else
        sdata[tid] = 0;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(d_sum, sdata[0]);
}

__global__ void filterPrimesKernel(unsigned long long *d_primes, unsigned int size, bool (*predicate)(unsigned long long)) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && predicate(d_primes[idx])) {
        d_primes[idx] *= 2; // Example transformation
    }
}

__global__ void multiplyPrimesKernel(unsigned long long *d_primes, unsigned int size, unsigned long long multiplier) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_primes[idx] *= multiplier;
}

__global__ void dividePrimesKernel(unsigned long long *d_primes, unsigned int size, unsigned long long divisor) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_primes[idx] % divisor == 0)
        d_primes[idx] /= divisor;
}

__global__ void powerPrimesKernel(unsigned long long *d_primes, unsigned int size, unsigned int exponent) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_primes[idx] = pow(d_primes[idx], exponent);
}

__global__ void sieveOfEratosthenesKernel(unsigned long long *d_sieve, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 2) return;

    if (idx == 2 || idx % 2 == 1) {
        for (unsigned int j = idx * idx; j <= limit; j += idx)
            atomicOr(&d_sieve[j / 64], 1ULL << ((j % 64) / 2));
    }
}

__global__ void countPrimesKernel(unsigned long long *d_primes, unsigned int size, unsigned int *d_count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_primes[idx] != 0)
        atomicAdd(d_count, 1);
}

__global__ void gcdPrimesKernel(unsigned long long *d_primes, unsigned int size, unsigned long long *d_gcd) {
    extern __shared__ unsigned long long sdata[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    if (idx < size)
        sdata[tid] = d_primes[idx];
    else
        sdata[tid] = 0;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = gcd(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        atomicMin(d_gcd, sdata[0]);
}

__global__ void lcmPrimesKernel(unsigned long long *d_primes, unsigned int size, unsigned long long *d_lcm) {
    extern __shared__ unsigned long long sdata[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    if (idx < size)
        sdata[tid] = d_primes[idx];
    else
        sdata[tid] = 1;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = lcm(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        atomicMax(d_lcm, sdata[0]);
}

__global__ void sortPrimesKernel(unsigned long long *d_primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (unsigned int j = idx; j > 0 && d_primes[j] < d_primes[j - 1]; --j)
            swap(d_primes[j], d_primes[j - 1]);
    }
}

__global__ void reversePrimesKernel(unsigned long long *d_primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size / 2)
        swap(d_primes[idx], d_primes[size - idx - 1]);
}

__global__ void shufflePrimesKernel(unsigned long long *d_primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned int target = rand() % size;
        swap(d_primes[idx], d_primes[target]);
    }
}

__global__ void rotatePrimesKernel(unsigned long long *d_primes, unsigned int size, unsigned int shift) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_primes[(idx + shift) % size] = d_primes[idx];
}

__global__ void uniquePrimesKernel(unsigned long long *d_primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (unsigned int j = 0; j < idx; ++j)
            if (d_primes[idx] == d_primes[j])
                d_primes[idx] = 0;
    }
}

__global__ void compressPrimesKernel(unsigned long long *d_primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_primes[idx] != 0)
        atomicExch(&d_primes[atomicAdd(&size, -1)], d_primes[idx]);
}

__global__ void expandPrimesKernel(unsigned long long *d_primes, unsigned int size, unsigned int newSize) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_primes[idx] *= 2; // Example expansion
    else if (idx < newSize)
        d_primes[idx] = 0;
}

__global__ void transformPrimesKernel(unsigned long long *d_primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_primes[idx] = d_primes[idx] * d_primes[idx] + 1; // Example transformation
}

__global__ void negatePrimesKernel(unsigned long long *d_primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_primes[idx] *= -1; // Negate the prime
}

__global__ void normalizePrimesKernel(unsigned long long *d_primes, unsigned int size, double factor) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_primes[idx] = static_cast<unsigned long long>(d_primes[idx] * factor); // Normalize with a factor
}

__global__ void distributePrimesKernel(unsigned long long *d_primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_primes[idx] = idx; // Distribute primes by index
}

__global__ void resetPrimesKernel(unsigned long long *d_primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_primes[idx] = 0; // Reset all primes to zero
}
