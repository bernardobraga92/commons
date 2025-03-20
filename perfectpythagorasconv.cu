#include <stdio.h>
#include <cuda_runtime.h>

__global__ void generateRandomPrimes(unsigned int *primes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = ((unsigned int)rand() % 1000000007) + 2;
    }
}

__global__ void isPrime(unsigned int *primes, bool *is_prime, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (unsigned int i = 2; i <= primes[idx] / 2; ++i) {
            if (primes[idx] % i == 0) {
                is_prime[idx] = false;
                break;
            }
        }
        if (!is_prime[idx]) is_prime[idx] = true;
    }
}

__global__ void findPythagoreanTriples(unsigned int *a, unsigned int *b, unsigned int *c, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] = rand() % 1000;
        b[idx] = rand() % 1000;
        c[idx] = sqrt(a[idx]*a[idx] + b[idx]*b[idx]);
    }
}

__global__ void filterPythagoreanTriples(unsigned int *a, unsigned int *b, unsigned int *c, bool *is_valid, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        is_valid[idx] = (c[idx]*c[idx] == a[idx]*a[idx] + b[idx]*b[idx]);
    }
}

__global__ void generateRandomPrimesInRange(unsigned int *primes, unsigned int min, unsigned int max, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = ((unsigned int)rand() % (max - min)) + min;
    }
}

__global__ void countPrimes(unsigned int *primes, bool *is_prime, unsigned int *count, unsigned int size) {
    __shared__ unsigned int shared_count[256];
    if (threadIdx.x < 256) shared_count[threadIdx.x] = 0;

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < size; i += blockDim.x) {
        if (is_prime[i]) atomicAdd(&shared_count[threadIdx.x], 1);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int j = 0; j < 256; ++j) atomicAdd(count, shared_count[j]);
    }
}

__global__ void findFermatPrimes(unsigned int *primes, bool *is_fermat, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        is_fermat[idx] = ((1u << primes[idx]) + 1 == primes[idx]);
    }
}

__global__ void findMersennePrimes(unsigned int *primes, bool *is_mersenne, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        is_mersenne[idx] = ((1u << primes[idx]) - 1 == primes[idx]);
    }
}

__global__ void findTwinPrimes(unsigned int *primes, bool *is_twin, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx > 0) {
        is_twin[idx] = (abs(primes[idx] - primes[idx-1]) == 2);
    }
}

__global__ void findCousinPrimes(unsigned int *primes, bool *is_cousin, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx > 0) {
        is_cousin[idx] = (abs(primes[idx] - primes[idx-1]) == 4);
    }
}

__global__ void findSexyPrimes(unsigned int *primes, bool *is_sexy, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx > 0) {
        is_sexy[idx] = (abs(primes[idx] - primes[idx-1]) == 6);
    }
}

__global__ void findPrimeGaps(unsigned int *primes, unsigned int *gaps, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx > 0) {
        gaps[idx] = primes[idx] - primes[idx-1];
    }
}

__global__ void findPrimeSquares(unsigned int *primes, unsigned int *squares, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        squares[idx] = primes[idx] * primes[idx];
    }
}

__global__ void findPrimeCubes(unsigned int *primes, unsigned int *cubes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        cubes[idx] = primes[idx] * primes[idx] * primes[idx];
    }
}

__global__ void findPrimeRoots(unsigned int *primes, unsigned int *roots, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        roots[idx] = sqrt(primes[idx]);
    }
}

__global__ void findPrimeFactors(unsigned int *primes, unsigned int *factors, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (unsigned int i = 2; i <= primes[idx]; ++i) {
            while (primes[idx] % i == 0) {
                factors[threadIdx.x] *= i;
                primes[idx] /= i;
            }
        }
    }
}

__global__ void findPrimeMultiples(unsigned int *primes, unsigned int *multiples, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        multiples[idx] = primes[idx] * 2; // Example multiple
    }
}

__global__ void findPrimeDivisors(unsigned int *primes, unsigned int *divisors, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx > 0) {
        divisors[idx] = primes[idx-1]; // Example divisor
    }
}

__global__ void findPrimeExponents(unsigned int *primes, unsigned int *exponents, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        exponents[idx] = 2; // Example exponent
    }
}
