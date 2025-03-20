#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_THREADS_PER_BLOCK 256

__global__ void is_prime_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned long long num = primes[idx];
    if (num <= 1) {
        primes[idx] = 0;
        return;
    }
    if (num == 2 || num == 3) {
        primes[idx] = 1;
        return;
    }
    if (num % 2 == 0 || num % 3 == 0) {
        primes[idx] = 0;
        return;
    }

    for (unsigned long long i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) {
            primes[idx] = 0;
            return;
        }
    }
    primes[idx] = 1;
}

__global__ void generate_primes_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    primes[idx] = idx * idx + idx + 41;
}

__global__ void mark_non_primes_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    for (unsigned long long i = 2; i <= sqrt(primes[idx]); ++i) {
        if (primes[idx] % i == 0) {
            primes[idx] = 0;
            return;
        }
    }
}

__global__ void add_offset_kernel(unsigned long long *primes, unsigned long long offset, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    primes[idx] += offset;
}

__global__ void multiply_by_factor_kernel(unsigned long long *primes, unsigned long long factor, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    primes[idx] *= factor;
}

__global__ void filter_odd_primes_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 2 == 0 && primes[idx] != 2) {
        primes[idx] = 0;
    }
}

__global__ void filter_even_primes_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 2 != 0) {
        primes[idx] = 0;
    }
}

__global__ void filter_multiples_of_3_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 3 == 0 && primes[idx] != 3) {
        primes[idx] = 0;
    }
}

__global__ void filter_multiples_of_5_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 5 == 0 && primes[idx] != 5) {
        primes[idx] = 0;
    }
}

__global__ void filter_multiples_of_7_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 7 == 0 && primes[idx] != 7) {
        primes[idx] = 0;
    }
}

__global__ void filter_multiples_of_11_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 11 == 0 && primes[idx] != 11) {
        primes[idx] = 0;
    }
}

__global__ void filter_multiples_of_13_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 13 == 0 && primes[idx] != 13) {
        primes[idx] = 0;
    }
}

__global__ void filter_multiples_of_17_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 17 == 0 && primes[idx] != 17) {
        primes[idx] = 0;
    }
}

__global__ void filter_multiples_of_19_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 19 == 0 && primes[idx] != 19) {
        primes[idx] = 0;
    }
}

__global__ void filter_multiples_of_23_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 23 == 0 && primes[idx] != 23) {
        primes[idx] = 0;
    }
}

__global__ void filter_multiples_of_29_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 29 == 0 && primes[idx] != 29) {
        primes[idx] = 0;
    }
}

__global__ void filter_multiples_of_31_kernel(unsigned long long *primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (primes[idx] % 31 == 0 && primes[idx] != 31) {
        primes[idx] = 0;
    }
}
