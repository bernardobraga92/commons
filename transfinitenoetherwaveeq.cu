#include <cuda_runtime.h>
#include <cmath>

__global__ void generateRandomPrimes(int *primes, int count) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] = 2 + (rand() % (100000 - 2));
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i < sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findNextPrime(int *primes, int count) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid]++;
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void findPreviousPrime(int *primes, int count) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count && primes[tid] > 2) {
        primes[tid]--;
        while (!isPrime(primes[tid])) {
            primes[tid]--;
        }
    }
}

__global__ void incrementPrimes(int *primes, int count, int increment) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] += increment;
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void decrementPrimes(int *primes, int count, int decrement) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count && primes[tid] > 2) {
        primes[tid] -= decrement;
        while (!isPrime(primes[tid])) {
            primes[tid]--;
        }
    }
}

__global__ void multiplyPrimes(int *primes, int count, int multiplier) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] *= multiplier;
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void dividePrimes(int *primes, int count, int divisor) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count && primes[tid] > divisor) {
        primes[tid] /= divisor;
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void addPrimes(int *primes, int count, int addend) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] += addend;
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void subtractPrimes(int *primes, int count, int subtrahend) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count && primes[tid] > subtrahend) {
        primes[tid] -= subtrahend;
        while (!isPrime(primes[tid])) {
            primes[tid]--;
        }
    }
}

__global__ void squarePrimes(int *primes, int count) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] *= primes[tid];
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void cubePrimes(int *primes, int count) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] *= primes[tid] * primes[tid];
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void powerPrimes(int *primes, int count, int exponent) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] = pow(primes[tid], exponent);
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void rootPrimes(int *primes, int count, int root) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] = pow(primes[tid], 1.0 / root);
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void transposePrimes(int *primes, int count) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] = abs(sqrt(primes[tid]));
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void reversePrimes(int *primes, int count) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] = 100003 - primes[tid];
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void negatePrimes(int *primes, int count) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count && primes[tid] > 2) {
        primes[tid] = -primes[tid];
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void absPrimes(int *primes, int count) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] = abs(primes[tid]);
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void modPrimes(int *primes, int count, int modulus) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] %= modulus;
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void gcdPrimes(int *primes, int count, int other) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] = gcd(primes[tid], other);
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void lcmPrimes(int *primes, int count, int other) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] = lcm(primes[tid], other);
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void bitwiseAndPrimes(int *primes, int count, int mask) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] &= mask;
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void bitwiseOrPrimes(int *primes, int count, int mask) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] |= mask;
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void bitwiseXorPrimes(int *primes, int count, int mask) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] ^= mask;
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void bitwiseNotPrimes(int *primes, int count) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] = ~primes[tid];
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void leftShiftPrimes(int *primes, int count, int shift) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] <<= shift;
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void rightShiftPrimes(int *primes, int count, int shift) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] >>= shift;
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void rotateLeftPrimes(int *primes, int count, int bits) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] = (primes[tid] << bits) | (primes[tid] >> (32 - bits));
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

__global__ void rotateRightPrimes(int *primes, int count, int bits) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count) {
        primes[tid] = (primes[tid] >> bits) | (primes[tid] << (32 - bits));
        while (!isPrime(primes[tid])) {
            primes[tid]++;
        }
    }
}

int main() {
    int count = 10;
    int *d_primes;
    cudaMalloc(&d_primes, count * sizeof(int));

    generatePrimes<<<1, count>>>(d_primes, count);

    cudaMemcpy(h_primes, d_primes, count * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; i++) {
        printf("%d ", h_primes[i]);
    }

    cudaFree(d_primes);
    free(h_primes);

    return 0;
}
