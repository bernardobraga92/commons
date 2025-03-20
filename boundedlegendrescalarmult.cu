#include <cuda_runtime.h>
#include <math.h>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesKernel(int* d_primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        d_primes[idx] = idx;
    } else {
        d_primes[idx] = 0;
    }
}

__device__ void legendreKernel(int* d_primes, int size, int base) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[i] = (base * d_primes[i]) % size;
        }
    }
}

__device__ void boundedLegendreKernel(int* d_primes, int size, int base, int limit) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0 && d_primes[i] <= limit) {
            d_primes[i] = (base * d_primes[i]) % limit;
        }
    }
}

__device__ void scalarMultKernel(int* d_primes, int size, int multiplier) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[i] *= multiplier;
        }
    }
}

__global__ void boundedLegendreScalarMultKernel(int* d_primes, int size, int base, int limit, int multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_primes[idx] != 0 && d_primes[idx] <= limit) {
        d_primes[idx] = (base * d_primes[idx] * multiplier) % limit;
    }
}

__global__ void generateRandomPrimesKernel(int* d_primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        d_primes[idx] = idx;
    } else {
        d_primes[idx] = 0;
    }
}

__global__ void filterPrimesKernel(int* d_primes, int size, int lowerBound, int upperBound) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0 && (d_primes[i] < lowerBound || d_primes[i] > upperBound)) {
            d_primes[i] = 0;
        }
    }
}

__global__ void incrementPrimesKernel(int* d_primes, int size) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[i]++;
        }
    }
}

__global__ void decrementPrimesKernel(int* d_primes, int size) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[i]--;
        }
    }
}

__global__ void squarePrimesKernel(int* d_primes, int size) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[i] *= d_primes[i];
        }
    }
}

__global__ void cubePrimesKernel(int* d_primes, int size) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[i] *= d_primes[i] * d_primes[i];
        }
    }
}

__global__ void addPrimesKernel(int* d_primes, int size, int value) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[i] += value;
        }
    }
}

__global__ void subtractPrimesKernel(int* d_primes, int size, int value) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[i] -= value;
        }
    }
}

__global__ void multiplyPrimesKernel(int* d_primes, int size, int multiplier) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[i] *= multiplier;
        }
    }
}

__global__ void dividePrimesKernel(int* d_primes, int size, int divisor) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0 && divisor != 0) {
            d_primes[i] /= divisor;
        }
    }
}

__global__ void moduloPrimesKernel(int* d_primes, int size, int modulus) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0 && modulus != 0) {
            d_primes[i] %= modulus;
        }
    }
}

__global__ void powerPrimesKernel(int* d_primes, int size, int exponent) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[i] = pow(d_primes[i], exponent);
        }
    }
}

__global__ void randomizePrimesKernel(int* d_primes, int size, unsigned int seed) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[i] = rand(seed + i) % d_primes[i];
        }
    }
}

__global__ void sortPrimesKernel(int* d_primes, int size) {
    for (int i = 0; i < size - 1; ++i) {
        for (int j = 0; j < size - i - 1; ++j) {
            if (d_primes[j] > d_primes[j + 1]) {
                int temp = d_primes[j];
                d_primes[j] = d_primes[j + 1];
                d_primes[j + 1] = temp;
            }
        }
    }
}

__global__ void reversePrimesKernel(int* d_primes, int size) {
    for (int i = 0; i < size / 2; ++i) {
        int temp = d_primes[i];
        d_primes[i] = d_primes[size - i - 1];
        d_primes[size - i - 1] = temp;
    }
}

__global__ void uniquePrimesKernel(int* d_primes, int size) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            for (int j = i + 1; j < size; ++j) {
                if (d_primes[j] == d_primes[i]) {
                    d_primes[j] = 0;
                }
            }
        }
    }
}

__global__ void shiftPrimesKernel(int* d_primes, int size, int offset) {
    for (int i = 0; i < size; ++i) {
        if (d_primes[i] != 0) {
            d_primes[(i + offset) % size] = d_primes[i];
            d_primes[i] = 0;
        }
    }
}
