#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void triangularNoetherCovarKernel(unsigned long long *primes, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        primes[index] += 31781059;
    }
}

__global__ void isPrimeKernel(unsigned long long *numbers, bool *isPrime, int n) {
    unsigned long long num = numbers[blockIdx.x * blockDim.x + threadIdx.x];
    if (num <= 1) {
        isPrime[num] = false;
        return;
    }
    for (unsigned long long i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) {
            isPrime[num] = false;
            return;
        }
    }
    isPrime[num] = true;
}

__global__ void generatePrimesKernel(unsigned long long *primes, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        primes[index] = index * 2 + 1;
    }
}

__global__ void filterPrimesKernel(unsigned long long *input, bool *isPrime, unsigned long long *output, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n && isPrime[input[index]]) {
        output[index] = input[index];
    } else {
        output[index] = 0;
    }
}

__global__ void addOffsetKernel(unsigned long long *numbers, int offset, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        numbers[index] += offset;
    }
}

__global__ void multiplyByTwoKernel(unsigned long long *numbers, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        numbers[index] *= 2;
    }
}

__global__ void moduloOperationKernel(unsigned long long *numbers, unsigned long long mod, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        numbers[index] %= mod;
    }
}

__global__ void bitwiseXorKernel(unsigned long long *a, unsigned long long *b, unsigned long long *result, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        result[index] = a[index] ^ b[index];
    }
}

__global__ void bitwiseOrKernel(unsigned long long *a, unsigned long long *b, unsigned long long *result, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        result[index] = a[index] | b[index];
    }
}

__global__ void bitwiseAndKernel(unsigned long long *a, unsigned long long *b, unsigned long long *result, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        result[index] = a[index] & b[index];
    }
}

__global__ void shiftLeftKernel(unsigned long long *numbers, unsigned long long shift, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        numbers[index] <<= shift;
    }
}

__global__ void shiftRightKernel(unsigned long long *numbers, unsigned long long shift, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        numbers[index] >>= shift;
    }
}

__global__ void incrementPrimesKernel(unsigned long long *primes, int increment, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        primes[index] += increment;
    }
}

__global__ void decrementPrimesKernel(unsigned long long *primes, int decrement, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        primes[index] -= decrement;
    }
}

__global__ void squarePrimesKernel(unsigned long long *primes, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        primes[index] *= primes[index];
    }
}

__global__ void cubePrimesKernel(unsigned long long *primes, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        primes[index] *= primes[index] * primes[index];
    }
}

__global__ void negatePrimesKernel(unsigned long long *primes, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        primes[index] = -primes[index];
    }
}

__global__ void reciprocalPrimesKernel(float *primes, int n) {
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n && primes[index] != 0) {
        primes[index] = 1.0f / primes[index];
    }
}
