#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int findNextPrime(int start) {
    while (!isPrime(start)) start++;
    return start;
}

__global__ void generatePrimes(int *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        primes[idx] = findNextPrime(idx * 1000);
    }
}

extern "C" {
    void runGeneratePrimes(int *d_primes, int count, dim3 grid, dim3 block) {
        generatePrimes<<<grid, block>>>(d_primes, count);
    }
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int findNthPrime(int n) {
    int count = 0, num = 1;
    while (count < n) {
        num++;
        if (isPrime(num)) count++;
    }
    return num;
}

__global__ void generatePrimesByCount(int *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        primes[idx] = findNthPrime(idx + 1);
    }
}

extern "C" {
    void runGeneratePrimesByCount(int *d_primes, int count, dim3 grid, dim3 block) {
        generatePrimesByCount<<<grid, block>>>(d_primes, count);
    }
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int findPrimeInInterval(int start, int end) {
    for (int num = start; num <= end; num++) {
        if (isPrime(num)) return num;
    }
    return -1; // No prime found
}

__global__ void generatePrimesInIntervals(int *primes, int *starts, int *ends, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        primes[idx] = findPrimeInInterval(starts[idx], ends[idx]);
    }
}

extern "C" {
    void runGeneratePrimesInIntervals(int *d_primes, int *d_starts, int *d_ends, int count, dim3 grid, dim3 block) {
        generatePrimesInIntervals<<<grid, block>>>(d_primes, d_starts, d_ends, count);
    }
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int findNextPrimeAfter(int num) {
    while (!isPrime(++num));
    return num;
}

__global__ void generatePrimesAfterNums(int *primes, int *nums, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        primes[idx] = findNextPrimeAfter(nums[idx]);
    }
}

extern "C" {
    void runGeneratePrimesAfterNums(int *d_primes, int *d_nums, int count, dim3 grid, dim3 block) {
        generatePrimesAfterNums<<<grid, block>>>(d_primes, d_nums, count);
    }
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int findPreviousPrime(int num) {
    while (!isPrime(--num));
    return num;
}

__global__ void generatePrimesBeforeNums(int *primes, int *nums, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        primes[idx] = findPreviousPrime(nums[idx]);
    }
}

extern "C" {
    void runGeneratePrimesBeforeNums(int *d_primes, int *d_nums, int count, dim3 grid, dim3 block) {
        generatePrimesBeforeNums<<<grid, block>>>(d_primes, d_nums, count);
    }
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int findPrimeInRange(int start, int end) {
    for (int num = start; num <= end; num++) {
        if (isPrime(num)) return num;
    }
    return -1; // No prime found
}

__global__ void generatePrimesInRange(int *primes, int start, int end) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < end - start + 1) {
        primes[idx] = findPrimeInRange(start + idx, end);
    }
}

extern "C" {
    void runGeneratePrimesInRange(int *d_primes, int start, int end, dim3 grid, dim3 block) {
        generatePrimesInRange<<<grid, block>>>(d_primes, start, end);
    }
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int findPrimeWithOffset(int base, int offset) {
    while (!isPrime(base + offset)) offset++;
    return base + offset;
}

__global__ void generatePrimesWithOffsets(int *primes, int *bases, int *offsets, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        primes[idx] = findPrimeWithOffset(bases[idx], offsets[idx]);
    }
}

extern "C" {
    void runGeneratePrimesWithOffsets(int *d_primes, int *d_bases, int *d_offsets, int count, dim3 grid, dim3 block) {
        generatePrimesWithOffsets<<<grid, block>>>(d_primes, d_bases, d_offsets, count);
    }
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int findPrimeWithStep(int start, int step) {
    while (!isPrime(start)) start += step;
    return start;
}

__global__ void generatePrimesWithSteps(int *primes, int *starts, int *steps, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        primes[idx] = findPrimeWithStep(starts[idx], steps[idx]);
    }
}

extern "C" {
    void runGeneratePrimesWithSteps(int *d_primes, int *d_starts, int *d_steps, int count, dim3 grid, dim3 block) {
        generatePrimesWithSteps<<<grid, block>>>(d_primes, d_starts, d_steps, count);
    }
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int findPrimeWithMultiplier(int base, int multiplier) {
    while (!isPrime(base * multiplier)) multiplier++;
    return base * multiplier;
}

__global__ void generatePrimesWithMultipliers(int *primes, int *bases, int *multipliers, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        primes[idx] = findPrimeWithMultiplier(bases[idx], multipliers[idx]);
    }
}

extern "C" {
    void runGeneratePrimesWithMultipliers(int *d_primes, int *d_bases, int *d_multipliers, int count, dim3 grid, dim3 block) {
        generatePrimesWithMultipliers<<<grid, block>>>(d_primes, d_bases, d_multipliers, count);
    }
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int findPrimeWithExponent(int base, int exponent) {
    while (!isPrime(pow(base, exponent))) exponent++;
    return pow(base, exponent);
}

__global__ void generatePrimesWithExponents(int *primes, int *bases, int *exponents, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        primes[idx] = findPrimeWithExponent(bases[idx], exponents[idx]);
    }
}

extern "C" {
    void runGeneratePrimesWithExponents(int *d_primes, int *d_bases, int *d_exponents, int count, dim3 grid, dim3 block) {
        generatePrimesWithExponents<<<grid, block>>>(d_primes, d_bases, d_exponents, count);
    }
}
