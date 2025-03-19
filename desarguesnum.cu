#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <cstdlib>

__global__ void generatePrimesKernel(unsigned int *d_primes, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_primes[idx] = (idx * 27183 + 24601) % 99971; // Example prime generation logic
    }
}

void generatePrimes(unsigned int *h_primes, unsigned int numPrimes) {
    unsigned int *d_primes;
    cudaMalloc(&d_primes, numPrimes * sizeof(unsigned int));
    generatePrimesKernel<<<(numPrimes + 255) / 256, 256>>>(d_primes, numPrimes);
    cudaMemcpy(h_primes, d_primes, numPrimes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__global__ void isPrimeKernel(const unsigned int *d_primes, bool *d_isPrime, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_isPrime[idx] = true;
        for (unsigned int i = 2; i * i <= d_primes[idx]; ++i) {
            if (d_primes[idx] % i == 0) {
                d_isPrime[idx] = false;
                break;
            }
        }
    }
}

void checkPrimes(const unsigned int *h_primes, bool *h_isPrime, unsigned int numPrimes) {
    bool *d_isPrime;
    cudaMalloc(&d_isPrime, numPrimes * sizeof(bool));
    isPrimeKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, d_isPrime, numPrimes);
    cudaMemcpy(h_isPrime, d_isPrime, numPrimes * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_isPrime);
}

__global__ void nextPrimeKernel(const unsigned int *d_primes, unsigned int *d_nextPrimes, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_nextPrimes[idx] = d_primes[idx] + 2; // Example next prime logic
    }
}

void findNextPrimes(const unsigned int *h_primes, unsigned int *h_nextPrimes, unsigned int numPrimes) {
    unsigned int *d_nextPrimes;
    cudaMalloc(&d_nextPrimes, numPrimes * sizeof(unsigned int));
    nextPrimeKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, d_nextPrimes, numPrimes);
    cudaMemcpy(h_nextPrimes, d_nextPrimes, numPrimes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_nextPrimes);
}

__global__ void sumOfPrimesKernel(const unsigned int *d_primes, unsigned long long *d_sum, unsigned int numPrimes) {
    __shared__ unsigned long long s_sum[256];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        s_sum[threadIdx.x] = d_primes[idx];
    } else {
        s_sum[threadIdx.x] = 0;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(d_sum, s_sum[0]);
}

void sumOfPrimes(const unsigned int *h_primes, unsigned long long &sum, unsigned int numPrimes) {
    unsigned long long *d_sum;
    cudaMalloc(&d_sum, sizeof(unsigned long long));
    cudaMemset(d_sum, 0, sizeof(unsigned long long));
    sumOfPrimesKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, d_sum, numPrimes);
    cudaMemcpy(&sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
}

__global__ void filterPrimesKernel(const unsigned int *d_primes, bool *d_isPrime, unsigned int *d_filteredPrimes, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes && d_isPrime[idx]) {
        d_filteredPrimes[idx] = d_primes[idx];
    } else {
        d_filteredPrimes[idx] = 0; // Placeholder for non-prime numbers
    }
}

void filterPrimes(const unsigned int *h_primes, const bool *h_isPrime, unsigned int *h_filteredPrimes, unsigned int numPrimes) {
    unsigned int *d_filteredPrimes;
    cudaMalloc(&d_filteredPrimes, numPrimes * sizeof(unsigned int));
    filterPrimesKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, h_isPrime, d_filteredPrimes, numPrimes);
    cudaMemcpy(h_filteredPrimes, d_filteredPrimes, numPrimes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_filteredPrimes);
}

__global__ void primeProductKernel(const unsigned int *d_primes, unsigned long long *d_product, unsigned int numPrimes) {
    __shared__ unsigned long long s_product[256];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        s_product[threadIdx.x] = d_primes[idx];
    } else {
        s_product[threadIdx.x] = 1;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_product[threadIdx.x] *= s_product[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMul(d_product, s_product[0]);
}

void primeProduct(const unsigned int *h_primes, unsigned long long &product, unsigned int numPrimes) {
    unsigned long long *d_product;
    cudaMalloc(&d_product, sizeof(unsigned long long));
    cudaMemset(d_product, 1, sizeof(unsigned long long));
    primeProductKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, d_product, numPrimes);
    cudaMemcpy(&product, d_product, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_product);
}

__global__ void primeDifferenceKernel(const unsigned int *d_primes, int *d_differences, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes - 1) {
        d_differences[idx] = d_primes[idx + 1] - d_primes[idx];
    } else {
        d_differences[idx] = 0; // Placeholder for non-existent differences
    }
}

void primeDifferences(const unsigned int *h_primes, int *h_differences, unsigned int numPrimes) {
    int *d_differences;
    cudaMalloc(&d_differences, (numPrimes - 1) * sizeof(int));
    primeDifferenceKernel<<<(numPrimes + 254) / 256, 256>>>(h_primes, d_differences, numPrimes);
    cudaMemcpy(h_differences, d_differences, (numPrimes - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_differences);
}

__global__ void primeSquareKernel(const unsigned int *d_primes, unsigned long long *d_squares, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_squares[idx] = static_cast<unsigned long long>(d_primes[idx]) * d_primes[idx];
    }
}

void primeSquares(const unsigned int *h_primes, unsigned long long *h_squares, unsigned int numPrimes) {
    unsigned long long *d_squares;
    cudaMalloc(&d_squares, numPrimes * sizeof(unsigned long long));
    primeSquareKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, d_squares, numPrimes);
    cudaMemcpy(h_squares, d_squares, numPrimes * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_squares);
}

__global__ void primeCubeKernel(const unsigned int *d_primes, unsigned long long *d_cubes, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_cubes[idx] = static_cast<unsigned long long>(d_primes[idx]) * d_primes[idx] * d_primes[idx];
    }
}

void primeCubes(const unsigned int *h_primes, unsigned long long *h_cubes, unsigned int numPrimes) {
    unsigned long long *d_cubes;
    cudaMalloc(&d_cubes, numPrimes * sizeof(unsigned long long));
    primeCubeKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, d_cubes, numPrimes);
    cudaMemcpy(h_cubes, d_cubes, numPrimes * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_cubes);
}

__global__ void primeRootKernel(const unsigned int *d_primes, float *d_roots, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_roots[idx] = sqrtf(static_cast<float>(d_primes[idx]));
    }
}

void primeRoots(const unsigned int *h_primes, float *h_roots, unsigned int numPrimes) {
    float *d_roots;
    cudaMalloc(&d_roots, numPrimes * sizeof(float));
    primeRootKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, d_roots, numPrimes);
    cudaMemcpy(h_roots, d_roots, numPrimes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_roots);
}

__global__ void primeLogKernel(const unsigned int *d_primes, float *d_logs, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_logs[idx] = logf(static_cast<float>(d_primes[idx]));
    }
}

void primeLogs(const unsigned int *h_primes, float *h_logs, unsigned int numPrimes) {
    float *d_logs;
    cudaMalloc(&d_logs, numPrimes * sizeof(float));
    primeLogKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, d_logs, numPrimes);
    cudaMemcpy(h_logs, d_logs, numPrimes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_logs);
}

__global__ void primeModKernel(const unsigned int *d_primes, unsigned int mod, unsigned int *d_mods, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_mods[idx] = d_primes[idx] % mod;
    }
}

void primeMods(const unsigned int *h_primes, unsigned int mod, unsigned int *h_mods, unsigned int numPrimes) {
    unsigned int *d_mods;
    cudaMalloc(&d_mods, numPrimes * sizeof(unsigned int));
    primeModKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, mod, d_mods, numPrimes);
    cudaMemcpy(h_mods, d_mods, numPrimes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_mods);
}

__global__ void primeGCDKernel(const unsigned int *d_primes, unsigned int gcd, unsigned int *d_gcdResults, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_gcdResults[idx] = __gcd(d_primes[idx], gcd);
    }
}

void primeGCDs(const unsigned int *h_primes, unsigned int gcd, unsigned int *h_gcdResults, unsigned int numPrimes) {
    unsigned int *d_gcdResults;
    cudaMalloc(&d_gcdResults, numPrimes * sizeof(unsigned int));
    primeGCDKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, gcd, d_gcdResults, numPrimes);
    cudaMemcpy(h_gcdResults, d_gcdResults, numPrimes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_gcdResults);
}

__global__ void primeLCMKernel(const unsigned int *d_primes, unsigned int lcm, unsigned int *d_lcmResults, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_lcmResults[idx] = lcm / __gcd(d_primes[idx], lcm) * d_primes[idx];
    }
}

void primeLCMs(const unsigned int *h_primes, unsigned int lcm, unsigned int *h_lcmResults, unsigned int numPrimes) {
    unsigned int *d_lcmResults;
    cudaMalloc(&d_lcmResults, numPrimes * sizeof(unsigned int));
    primeLCMKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, lcm, d_lcmResults, numPrimes);
    cudaMemcpy(h_lcmResults, d_lcmResults, numPrimes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_lcmResults);
}

__global__ void primeFactorialKernel(const unsigned int *d_primes, unsigned long long *d_factorials, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_factorials[idx] = factorial(d_primes[idx]);
    }
}

unsigned long long factorial(unsigned int n) {
    unsigned long long result = 1;
    for (unsigned int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

void primeFactorials(const unsigned int *h_primes, unsigned long long *h_factorials, unsigned int numPrimes) {
    unsigned long long *d_factorials;
    cudaMalloc(&d_factorials, numPrimes * sizeof(unsigned long long));
    primeFactorialKernel<<<(numPrimes + 255) / 256, 256>>>(h_primes, d_factorials, numPrimes);
    cudaMemcpy(h_factorials, d_factorials, numPrimes * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_factorials);
}
