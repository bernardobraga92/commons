#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void euler_sieve(int* primes, int limit) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > 1 && i <= limit) {
        for (int j = 2; j <= sqrt(i); j++) {
            if (i % j == 0) {
                primes[i] = 0;
                break;
            }
        }
    }
}

__global__ void euler_sieve_optimized(int* primes, int limit) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > 1 && i <= limit) {
        for (int j = 2; j * j <= i; j++) {
            if (primes[j] == 1 && i % j == 0) {
                primes[i] = 0;
                break;
            }
        }
    }
}

__global__ void euler_sieve_parallel(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if (primes[j] == 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i] = isPrime ? 1 : 0;
        }
    }
}

__global__ void euler_sieve_bitwise(int* primes, int limit) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > 1 && i <= limit) {
        for (int j = 2; j * j <= i; j++) {
            if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                primes[i] = 0;
                break;
            }
        }
    }
}

__global__ void euler_sieve_optimized_bitwise(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

__global__ void euler_sieve_parallel_bitwise(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

__global__ void euler_sieve_parallel_optimized_bitwise(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

__global__ void euler_sieve_parallel_optimized_bitwise_v2(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

__global__ void euler_sieve_parallel_optimized_bitwise_v3(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

__global__ void euler_sieve_parallel_optimized_bitwise_v4(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

__global__ void euler_sieve_parallel_optimized_bitwise_v5(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

__global__ void euler_sieve_parallel_optimized_bitwise_v6(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

__global__ void euler_sieve_parallel_optimized_bitwise_v7(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

__global__ void euler_sieve_parallel_optimized_bitwise_v8(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

__global__ void euler_sieve_parallel_optimized_bitwise_v9(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

__global__ void euler_sieve_parallel_optimized_bitwise_v10(int* primes, int limit) {
    unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = start; i <= limit; i += step) {
        if (i > 1) {
            bool isPrime = true;
            for (int j = 2; j * j <= i; j++) {
                if ((primes[j / 32] >> (j % 32)) & 1 && i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            primes[i / 32] |= (isPrime ? 1 : 0) << (i % 32);
        }
    }
}

int main() {
    int limit = 1000000;
    size_t bytes = (limit + 31) / 32 * sizeof(int);
    int* primes = nullptr;

    cudaMallocManaged(&primes, bytes);

    // Initialize the primes array
    for (int i = 0; i < (limit + 31) / 32; ++i) {
        primes[i] = -1;
    }

    euler_sieve_parallel_optimized_bitwise_v1<<<(limit + 255) / 256, 256>>>(primes, limit);
    // Add more kernel calls here if needed

    cudaFree(primes);

    return 0;
}
