#include <cuda_runtime.h>
#include <math.h>

__device__ bool isPrime(unsigned long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (unsigned long long i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    return true;
}

__global__ void findPrimes(unsigned long long* d_primes, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        d_primes[idx] = idx * 2 + 3; // Generate odd numbers starting from 3
        while (!isPrime(d_primes[idx])) {
            d_primes[idx]++;
        }
    }
}

__global__ void generateHermite(unsigned long long* d_hermite, unsigned int num_elements) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        double x = 2.0 * idx / num_elements - 1.0; // Map to [-1, 1]
        double h_0 = 1.0;
        double h_1 = 2.0 * x;
        for (int n = 2; n <= idx; n++) {
            double h_n = 2.0 * x * h_1 - 2.0 * (n - 1) * h_0;
            h_0 = h_1;
            h_1 = h_n;
        }
        d_hermite[idx] = static_cast<unsigned long long>(h_1);
    }
}

__global__ void sqrtPrimes(unsigned long long* d_primes, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        double sqrt_val = sqrt(static_cast<double>(d_primes[idx]));
        d_primes[idx] = static_cast<unsigned long long>(sqrt_val);
    }
}

__global__ void circularShift(unsigned long long* d_data, unsigned int shift, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned int new_idx = (idx + shift) % size;
        d_data[new_idx] = d_data[idx];
    }
}

__global__ void sumPrimes(unsigned long long* d_primes, unsigned long long* d_sum, unsigned int num_primes) {
    extern __shared__ unsigned long long sdata[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    if (idx < num_primes) {
        sdata[tid] = d_primes[idx];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(d_sum, sdata[0]);
}

__global__ void productPrimes(unsigned long long* d_primes, unsigned long long* d_product, unsigned int num_primes) {
    extern __shared__ unsigned long long sdata[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    if (idx < num_primes) {
        sdata[tid] = d_primes[idx];
    } else {
        sdata[tid] = 1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] *= sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicMul(d_product, sdata[0]);
}

__global__ void filterPrimes(unsigned long long* d_primes, unsigned long long* d_filtered, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes && isPrime(d_primes[idx])) {
        d_filtered[idx] = d_primes[idx];
    } else {
        d_filtered[idx] = 0;
    }
}

__global__ void squarePrimes(unsigned long long* d_primes, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        d_primes[idx] *= d_primes[idx];
    }
}

__global__ void negatePrimes(unsigned long long* d_primes, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        d_primes[idx] = -d_primes[idx];
    }
}

__global__ void shiftAndAdd(unsigned long long* d_data, unsigned long long value, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] += value;
    }
}

__global__ void multiplyByPi(unsigned long long* d_primes, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        double pi = 3.141592653589793;
        d_primes[idx] = static_cast<unsigned long long>(d_primes[idx] * pi);
    }
}

__global__ void moduloPrimes(unsigned long long* d_primes, unsigned int mod, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        d_primes[idx] %= mod;
    }
}

__global__ void incrementPrimes(unsigned long long* d_primes, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        d_primes[idx]++;
    }
}

__global__ void decrementPrimes(unsigned long long* d_primes, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        d_primes[idx]--;
    }
}

__global__ void xorPrimes(unsigned long long* d_primes, unsigned long long value, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        d_primes[idx] ^= value;
    }
}

__global__ void orPrimes(unsigned long long* d_primes, unsigned long long value, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        d_primes[idx] |= value;
    }
}

__global__ void andPrimes(unsigned long long* d_primes, unsigned long long value, unsigned int num_primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_primes) {
        d_primes[idx] &= value;
    }
}
