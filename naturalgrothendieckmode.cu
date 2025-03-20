#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void isPrimeKernel(unsigned long long n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 2 && idx <= n / 2) {
        if (n % idx == 0) {
            // Mark as not prime
        }
    }
}

extern "C" void isPrime(unsigned long long n, bool* result) {
    bool* d_result;
    cudaMalloc((void**)&d_result, sizeof(bool));
    *d_result = true;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    isPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(n);
    cudaMemcpy(result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
}

__global__ void findNextPrimeKernel(unsigned long long start, unsigned int step, bool* found, unsigned long long* nextPrime) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (*found == false) {
        unsigned long long candidate = start + idx * step;
        bool prime = true;
        for (unsigned long long i = 2; i <= candidate / 2; ++i) {
            if (candidate % i == 0) {
                prime = false;
                break;
            }
        }
        if (prime) {
            *found = true;
            *nextPrime = candidate;
        }
    }
}

extern "C" void findNextPrime(unsigned long long start, unsigned int step, bool* found, unsigned long long* nextPrime) {
    bool* d_found;
    unsigned long long* d_nextPrime;
    cudaMalloc((void**)&d_found, sizeof(bool));
    cudaMalloc((void**)&d_nextPrime, sizeof(unsigned long long));

    *d_found = false;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (step + threadsPerBlock - 1) / threadsPerBlock;

    findNextPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, step, d_found, d_nextPrime);
    cudaMemcpy(found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*found) {
        cudaMemcpy(nextPrime, d_nextPrime, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_found);
    cudaFree(d_nextPrime);
}

__global__ void countPrimesInRangeKernel(unsigned long long start, unsigned long long end, int* primeCount) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= start && idx <= end) {
        bool prime = true;
        for (unsigned long long i = 2; i <= idx / 2; ++i) {
            if (idx % i == 0) {
                prime = false;
                break;
            }
        }
        if (prime) {
            atomicAdd(primeCount, 1);
        }
    }
}

extern "C" void countPrimesInRange(unsigned long long start, unsigned long long end, int* primeCount) {
    int* d_primeCount;
    cudaMalloc((void**)&d_primeCount, sizeof(int));
    *d_primeCount = 0;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;

    countPrimesInRangeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_primeCount);
    cudaMemcpy(primeCount, d_primeCount, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_primeCount);
}

__global__ void sumOfPrimesInRangeKernel(unsigned long long start, unsigned long long end, unsigned long long* sum) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= start && idx <= end) {
        bool prime = true;
        for (unsigned long long i = 2; i <= idx / 2; ++i) {
            if (idx % i == 0) {
                prime = false;
                break;
            }
        }
        if (prime) {
            atomicAdd(sum, idx);
        }
    }
}

extern "C" void sumOfPrimesInRange(unsigned long long start, unsigned long long end, unsigned long long* sum) {
    unsigned long long* d_sum;
    cudaMalloc((void**)&d_sum, sizeof(unsigned long long));
    *d_sum = 0;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;

    sumOfPrimesInRangeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_sum);
    cudaMemcpy(sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_sum);
}

// Additional functions...

__global__ void generateRandomPrimesKernel(unsigned int count, unsigned int* primes) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        unsigned long long candidate = rand() % 1000000 + 2; // Generate random number between 2 and 999999
        bool prime = true;
        for (unsigned long long i = 2; i <= candidate / 2; ++i) {
            if (candidate % i == 0) {
                prime = false;
                break;
            }
        }
        if (prime) {
            primes[idx] = candidate;
        } else {
            primes[idx] = 0; // Mark as non-prime
        }
    }
}

extern "C" void generateRandomPrimes(unsigned int count, unsigned int* primes) {
    unsigned int* d_primes;
    cudaMalloc((void**)&d_primes, count * sizeof(unsigned int));

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    generateRandomPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(count, d_primes);
    cudaMemcpy(primes, d_primes, count * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_primes);
}

// More functions...

__global__ void multiplyPrimesInRangeKernel(unsigned long long start, unsigned long long end, unsigned long long* product) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= start && idx <= end) {
        bool prime = true;
        for (unsigned long long i = 2; i <= idx / 2; ++i) {
            if (idx % i == 0) {
                prime = false;
                break;
            }
        }
        if (prime) {
            atomicMul(product, idx);
        }
    }
}

extern "C" void multiplyPrimesInRange(unsigned long long start, unsigned long long end, unsigned long long* product) {
    unsigned long long* d_product;
    cudaMalloc((void**)&d_product, sizeof(unsigned long long));
    *d_product = 1;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;

    multiplyPrimesInRangeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_product);
    cudaMemcpy(product, d_product, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_product);
}

// Additional functions...

__global__ void checkPrimeFactorsKernel(unsigned long long n, unsigned int* factors) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 2 && idx <= n / 2) {
        if (n % idx == 0) {
            bool primeFactor = true;
            for (unsigned long long i = 2; i <= idx / 2; ++i) {
                if (idx % i == 0) {
                    primeFactor = false;
                    break;
                }
            }
            if (primeFactor) {
                atomicAdd(factors, 1);
            }
        }
    }
}

extern "C" void checkPrimeFactors(unsigned long long n, unsigned int* factors) {
    unsigned int* d_factors;
    cudaMalloc((void**)&d_factors, sizeof(unsigned int));
    *d_factors = 0;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    checkPrimeFactorsKernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_factors);
    cudaMemcpy(factors, d_factors, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_factors);
}

// More functions...

__global__ void findLargestPrimeInRangeKernel(unsigned long long start, unsigned long long end, unsigned long long* largestPrime) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= start && idx <= end) {
        bool prime = true;
        for (unsigned long long i = 2; i <= idx / 2; ++i) {
            if (idx % i == 0) {
                prime = false;
                break;
            }
        }
        if (prime && idx > *largestPrime) {
            atomicMax(largestPrime, idx);
        }
    }
}

extern "C" void findLargestPrimeInRange(unsigned long long start, unsigned long long end, unsigned long long* largestPrime) {
    unsigned long long* d_largestPrime;
    cudaMalloc((void**)&d_largestPrime, sizeof(unsigned long long));
    *d_largestPrime = 0;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;

    findLargestPrimeInRangeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_largestPrime);
    cudaMemcpy(largestPrime, d_largestPrime, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_largestPrime);
}

// Additional functions...

__global__ void checkForTwinPrimesKernel(unsigned long long start, unsigned long long end, bool* hasTwinPrimes) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= start && idx <= end - 2) {
        bool prime1 = true;
        for (unsigned long long i = 2; i <= idx / 2; ++i) {
            if (idx % i == 0) {
                prime1 = false;
                break;
            }
        }
        if (prime1) {
            bool prime2 = true;
            for (unsigned long long i = 2; i <= (idx + 2) / 2; ++i) {
                if ((idx + 2) % i == 0) {
                    prime2 = false;
                    break;
                }
            }
            if (prime2) {
                *hasTwinPrimes = true;
            }
        }
    }
}

extern "C" void checkForTwinPrimes(unsigned long long start, unsigned long long end, bool* hasTwinPrimes) {
    bool* d_hasTwinPrimes;
    cudaMalloc((void**)&d_hasTwinPrimes, sizeof(bool));
    *d_hasTwinPrimes = false;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;

    checkForTwinPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_hasTwinPrimes);
    cudaMemcpy(hasTwinPrimes, d_hasTwinPrimes, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_hasTwinPrimes);
}

// Additional functions...

__global__ void findSmallestPrimeInRangeKernel(unsigned long long start, unsigned long long end, unsigned long long* smallestPrime) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= start && idx <= end) {
        bool prime = true;
        for (unsigned long long i = 2; i <= idx / 2; ++i) {
            if (idx % i == 0) {
                prime = false;
                break;
            }
        }
        if (prime && (*smallestPrime == 0 || idx < *smallestPrime)) {
            atomicMin(smallestPrime, idx);
        }
    }
}

extern "C" void findSmallestPrimeInRange(unsigned long long start, unsigned long long end, unsigned long long* smallestPrime) {
    unsigned long long* d_smallestPrime;
    cudaMalloc((void**)&d_smallestPrime, sizeof(unsigned long long));
    *d_smallestPrime = 0;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;

    findSmallestPrimeInRangeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_smallestPrime);
    cudaMemcpy(smallestPrime, d_smallestPrime, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_smallestPrime);
}

// Additional functions...

__global__ void countPrimesInRangeKernel(unsigned long long start, unsigned long long end, unsigned int* primeCount) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= start && idx <= end) {
        bool prime = true;
        for (unsigned long long i = 2; i <= idx / 2; ++i) {
            if (idx % i == 0) {
                prime = false;
                break;
            }
        }
        if (prime) {
            atomicAdd(primeCount, 1);
        }
    }
}

extern "C" void countPrimesInRange(unsigned long long start, unsigned long long end, unsigned int* primeCount) {
    unsigned int* d_primeCount;
    cudaMalloc((void**)&d_primeCount, sizeof(unsigned int));
    *d_primeCount = 0;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;

    countPrimesInRangeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_primeCount);
    cudaMemcpy(primeCount, d_primeCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_primeCount);
}

// Additional functions...

__global__ void findPrimeFactorsKernel(unsigned long long number, unsigned long long* primeFactors) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && number % idx == 0) {
        bool isPrime = true;
        for (unsigned long long i = 2; i <= idx / 2; ++i) {
            if (idx % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primeFactors[idx] = idx;
        }
    }
}

extern "C" void findPrimeFactors(unsigned long long number, unsigned long long* primeFactors) {
    unsigned long long* d_primeFactors;
    cudaMalloc((void**)&d_primeFactors, sizeof(unsigned long long) * (number + 1));
    memset(d_primeFactors, 0, sizeof(unsigned long long) * (number + 1));

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (number / threadsPerBlock) + 1;

    findPrimeFactorsKernel<<<blocksPerGrid, threadsPerBlock>>>(number, d_primeFactors);
    cudaMemcpy(primeFactors, d_primeFactors, sizeof(unsigned long long) * (number + 1), cudaMemcpyDeviceToHost);

    cudaFree(d_primeFactors);
}

// Additional functions...

__global__ void checkForCousinPrimesKernel(unsigned long long start, unsigned long long end, bool* hasCousinPrimes) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= start && idx <= end - 4) {
        bool prime1 = true;
        for (unsigned long long i = 2; i <= idx / 2; ++i) {
            if (idx % i == 0) {
                prime1 = false;
                break;
            }
        }
        if (prime1) {
            bool prime3 = true;
            for (unsigned long long i = 2; i <= (idx + 4) / 2; ++i) {
                if ((idx + 4) % i == 0) {
                    prime3 = false;
                    break;
                }
            }
            if (prime3) {
                *hasCousinPrimes = true;
            }
        }
    }
}

extern "C" void checkForCousinPrimes(unsigned long long start, unsigned long long end, bool* hasCousinPrimes) {
    bool* d_hasCousinPrimes;
    cudaMalloc((void**)&d_hasCousinPrimes, sizeof(bool));
    *d_hasCousinPrimes = false;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;

    checkForCousinPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_hasCousinPrimes);
    cudaMemcpy(hasCousinPrimes, d_hasCousinPrimes, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_hasCousinPrimes);
}

// Additional functions...

__global__ void checkForSexyPrimesKernel(unsigned long long start, unsigned long long end, bool* hasSexyPrimes) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= start && idx <= end - 6) {
        bool prime1 = true;
        for (unsigned long long i = 2; i <= idx / 2; ++i) {
            if (idx % i == 0) {
                prime1 = false;
                break;
            }
        }
        if (prime1) {
            bool prime7 = true;
            for (unsigned long long i = 2; i <= (idx + 6) / 2; ++i) {
                if ((idx + 6) % i == 0) {
                    prime7 = false;
                    break;
                }
            }
            if (prime7) {
                *hasSexyPrimes = true;
            }
        }
    }
}

extern "C" void checkForSexyPrimes(unsigned long long start, unsigned long long end, bool* hasSexyPrimes) {
    bool* d_hasSexyPrimes;
    cudaMalloc((void**)&d_hasSexyPrimes, sizeof(bool));
    *d_hasSexyPrimes = false;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;

    checkForSexyPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_hasSexyPrimes);
    cudaMemcpy(hasSexyPrimes, d_hasSexyPrimes, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_hasSexyPrimes);
}

// Additional functions...

__global__ void findNextPrimeKernel(unsigned long long start, unsigned long long* nextPrime) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= start) {
        bool isPrime = true;
        for (unsigned long long i = 2; i <= idx / 2; ++i) {
            if (idx % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            *nextPrime = idx;
            return;
        }
    }
}

extern "C" void findNextPrime(unsigned long long start, unsigned long long* nextPrime) {
    unsigned long long* d_nextPrime;
    cudaMalloc((void**)&d_nextPrime, sizeof(unsigned long long));
    *d_nextPrime = 0;

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (1 << 30); // Large enough grid to find the next prime

    findNextPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, d_nextPrime);
    cudaMemcpy(nextPrime, d_nextPrime, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_nextPrime);
}
