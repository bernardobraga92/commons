#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <cmath>

__global__ void generateRandomPrimes(int* primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        thrust::minstd_rand rng(idx);
        thrust::uniform_int_distribution<int> dist(2, 1000000);
        primes[idx] = dist(rng);
    }
}

__global__ void isPrime(int* numbers, int* results, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && numbers[idx] > 1) {
        bool prime = true;
        for (int i = 2; i <= sqrt(numbers[idx]); ++i) {
            if (numbers[idx] % i == 0) {
                prime = false;
                break;
            }
        }
        results[idx] = prime ? 1 : 0;
    } else {
        results[idx] = 0;
    }
}

__global__ void multiplyPrimes(int* primes, int* results, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1) {
        results[idx] = primes[idx] * primes[idx + 1];
    } else {
        results[idx] = 0;
    }
}

__global__ void sumPrimes(int* primes, int* result, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int shared_sum[];
    shared_sum[threadIdx.x] = 0;
    if (idx < n) {
        shared_sum[threadIdx.x] += primes[idx];
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(result, shared_sum[0]);
    }
}

__global__ void findMaxPrime(int* primes, int* max_prime, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int shared_max[];
    shared_max[threadIdx.x] = 0;
    if (idx < n) {
        shared_max[threadIdx.x] = primes[idx];
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_max[threadIdx.x + s] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(max_prime, shared_max[0]);
    }
}

__global__ void findMinPrime(int* primes, int* min_prime, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int shared_min[];
    shared_min[threadIdx.x] = 2147483647; // INT_MAX
    if (idx < n) {
        shared_min[threadIdx.x] = primes[idx];
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_min[threadIdx.x + s] < shared_min[threadIdx.x]) {
                shared_min[threadIdx.x] = shared_min[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMin(min_prime, shared_min[0]);
    }
}

__global__ void findEvenPrimes(int* primes, int* even_primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && primes[idx] % 2 == 0) {
        even_primes[threadIdx.x] = primes[idx];
    } else {
        even_primes[threadIdx.x] = 0;
    }
}

__global__ void findOddPrimes(int* primes, int* odd_primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && primes[idx] % 2 != 0) {
        odd_primes[threadIdx.x] = primes[idx];
    } else {
        odd_primes[threadIdx.x] = 0;
    }
}

__global__ void findPrimeSquares(int* primes, int* prime_squares, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        prime_squares[idx] = primes[idx] * primes[idx];
    } else {
        prime_squares[threadIdx.x] = 0;
    }
}

__global__ void findPrimeCubes(int* primes, int* prime_cubes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        prime_cubes[idx] = primes[idx] * primes[idx] * primes[idx];
    } else {
        prime_cubes[threadIdx.x] = 0;
    }
}

__global__ void findPrimeRoots(int* primes, int* prime_roots, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && primes[idx] > 1) {
        prime_roots[idx] = sqrt(primes[idx]);
    } else {
        prime_roots[threadIdx.x] = 0;
    }
}

__global__ void findPrimeLogarithms(int* primes, float* prime_logs, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && primes[idx] > 1) {
        prime_logs[idx] = log(primes[idx]);
    } else {
        prime_logs[threadIdx.x] = 0.0f;
    }
}

__global__ void findPrimeFactors(int* primes, int* factors, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && primes[idx] > 1) {
        for (int i = 2; i <= sqrt(primes[idx]); ++i) {
            if (primes[idx] % i == 0) {
                factors[threadIdx.x] = i;
                break;
            }
        }
    } else {
        factors[threadIdx.x] = 0;
    }
}

__global__ void findPrimeMultiples(int* primes, int* multiples, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        multiples[idx] = primes[idx] * 2; // Example multiple
    } else {
        multiples[threadIdx.x] = 0;
    }
}

__global__ void findPrimeDifferences(int* primes, int* differences, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1) {
        differences[idx] = primes[idx + 1] - primes[idx];
    } else {
        differences[threadIdx.x] = 0;
    }
}

__global__ void findPrimeDivisors(int* primes, int* divisors, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && primes[idx] > 1) {
        for (int i = 2; i <= sqrt(primes[idx]); ++i) {
            if (primes[idx] % i == 0) {
                divisors[threadIdx.x] = i;
                break;
            }
        }
    } else {
        divisors[threadIdx.x] = 0;
    }
}

__global__ void findPrimeProducts(int* primes, int* products, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1) {
        products[idx] = primes[idx] * primes[idx + 1];
    } else {
        products[threadIdx.x] = 0;
    }
}

__global__ void findPrimeRatios(int* primes, float* ratios, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1 && primes[idx] > 1 && primes[idx + 1] > 1) {
        ratios[idx] = static_cast<float>(primes[idx]) / primes[idx + 1];
    } else {
        ratios[threadIdx.x] = 0.0f;
    }
}

__global__ void findPrimePowers(int* primes, int* powers, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        powers[idx] = primes[idx] * primes[idx]; // Example power
    } else {
        powers[threadIdx.x] = 0;
    }
}

__global__ void findPrimeComposites(int* primes, int* composites, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        composites[idx] = primes[idx] * primes[idx]; // Example composite
    } else {
        composites[threadIdx.x] = 0;
    }
}

__global__ void findPrimeTwinPrimes(int* primes, int* twin_primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1 && abs(primes[idx] - primes[idx + 1]) == 2) {
        twin_primes[idx] = primes[idx];
    } else {
        twin_primes[threadIdx.x] = 0;
    }
}

__global__ void findPrimeCousinPrimes(int* primes, int* cousin_primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1 && abs(primes[idx] - primes[idx + 1]) == 4) {
        cousin_primes[idx] = primes[idx];
    } else {
        cousin_primes[threadIdx.x] = 0;
    }
}

int main() {
    // Example usage
    int n = 10;
    int *h_primes, *d_primes;
    h_primes = new int[n];

    // Initialize h_primes with some prime numbers
    h_primes[0] = 2; h_primes[1] = 3; h_primes[2] = 5; h_primes[3] = 7; 
    h_primes[4] = 11; h_primes[5] = 13; h_primes[6] = 17; h_primes[7] = 19;
    h_primes[8] = 23; h_primes[9] = 29;

    cudaMalloc(&d_primes, n * sizeof(int));
    cudaMemcpy(d_primes, h_primes, n * sizeof(int), cudaMemcpyHostToDevice);

    // Example kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    findPrimeSquares<<<blocksPerGrid, threadsPerBlock>>>(d_primes, d_primes, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_primes, d_primes, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the result
    for (int i = 0; i < n; ++i) {
        printf("Prime Square %d: %d\n", i, h_primes[i]);
    }

    cudaFree(d_primes);
    delete[] h_primes;

    return 0;
}
