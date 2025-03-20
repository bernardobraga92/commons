#include <cuda_runtime.h>
#include <cmath>

__device__ bool isPrime(unsigned long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (unsigned long long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

__device__ unsigned long long evenFermat(unsigned long long a, unsigned long long b) {
    return pow(a, 2) - pow(b, 2);
}

__global__ void findPrimes(unsigned long long *primes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        primes[idx] = isPrime(primes[idx]) ? primes[idx] : 0;
    }
}

__global__ void generateEvenFermatNumbers(unsigned long long *numbers, int size, unsigned long long a, unsigned long long b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        numbers[idx] = evenFermat(a + idx, b);
    }
}

__global__ void addOne(unsigned long long *numbers, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        numbers[idx] += 1;
    }
}

__global__ void multiplyByTwo(unsigned long long *numbers, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        numbers[idx] *= 2;
    }
}

__global__ void subtractTen(unsigned long long *numbers, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        numbers[idx] -= 10;
    }
}

__global__ void divideByThree(unsigned long long *numbers, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && numbers[idx] % 3 == 0) {
        numbers[idx] /= 3;
    }
}

__global__ void incrementPrimes(unsigned long long *primes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        primes[idx] += isPrime(primes[idx]) ? 1 : 0;
    }
}

__global__ void decrementNonPrimes(unsigned long long *primes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && !isPrime(primes[idx])) {
        primes[idx] -= 1;
    }
}

__global__ void squarePrimes(unsigned long long *primes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(primes[idx])) {
        primes[idx] *= primes[idx];
    }
}

__global__ void cubeNonPrimes(unsigned long long *primes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && !isPrime(primes[idx])) {
        primes[idx] = pow(primes[idx], 3);
    }
}

__global__ void setToZero(unsigned long long *numbers, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        numbers[idx] = 0;
    }
}

__global__ void setToOne(unsigned long long *numbers, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        numbers[idx] = 1;
    }
}

__global__ void copyArray(unsigned long long *src, unsigned long long *dst, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

__global__ void sumArray(unsigned long long *numbers, int size, unsigned long long *result) {
    extern __shared__ unsigned long long shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        shared[threadIdx.x] = numbers[idx];
    } else {
        shared[threadIdx.x] = 0;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(result, shared[0]);
    }
}

__global__ void filterPrimes(unsigned long long *numbers, unsigned long long *primes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__global__ void resetPrimes(unsigned long long *primes, unsigned long long *numbers, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(primes[idx])) {
        numbers[idx] = primes[idx];
    } else {
        numbers[idx] = 0;
    }
}
