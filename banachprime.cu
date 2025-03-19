#ifndef BANACHPRIME_H
#define BANACHPRIME_H

#include <cuda_runtime.h>
#include <cmath>

__device__ inline bool banachPrime_isPrime(long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (long long i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    return true;
}

__device__ inline long long banachPrime_nextPrime(long long start) {
    while (!banachPrime_isPrime(start))
        ++start;
    return start;
}

__global__ void banachPrime_findPrimesKernel(long long *primes, long long start, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count)
        primes[idx] = banachPrime_nextPrime(start + idx);
}

extern "C" void banachPrime_findPrimes(long long *d_primes, long long start, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, start, count);
}

__device__ inline bool banachPrime_isLargePrime(long long n) {
    return banachPrime_isPrime(n) && n > 1000000;
}

__global__ void banachPrime_findLargePrimesKernel(long long *primes, long long start, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = banachPrime_nextPrime(start + idx);
        while (!banachPrime_isLargePrime(candidate))
            candidate = banachPrime_nextPrime(candidate + 1);
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findLargePrimes(long long *d_primes, long long start, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findLargePrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, start, count);
}

__device__ inline bool banachPrime_isSophieGermainPrime(long long n) {
    return banachPrime_isPrime(n) && banachPrime_isPrime(2 * n + 1);
}

__global__ void banachPrime_findSophieGermainPrimesKernel(long long *primes, long long start, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = banachPrime_nextPrime(start + idx);
        while (!banachPrime_isSophieGermainPrime(candidate))
            candidate = banachPrime_nextPrime(candidate + 1);
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findSophieGermainPrimes(long long *d_primes, long long start, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findSophieGermainPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, start, count);
}

__device__ inline bool banachPrime_isTwinPrime(long long n) {
    return banachPrime_isPrime(n) && (banachPrime_isPrime(n - 2) || banachPrime_isPrime(n + 2));
}

__global__ void banachPrime_findTwinPrimesKernel(long long *primes, long long start, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = banachPrime_nextPrime(start + idx);
        while (!banachPrime_isTwinPrime(candidate))
            candidate = banachPrime_nextPrime(candidate + 1);
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findTwinPrimes(long long *d_primes, long long start, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findTwinPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, start, count);
}

__device__ inline bool banachPrime_isMersennePrime(long long n) {
    if ((n & (n - 1)) != 0 || n < 2)
        return false;
    long long mersenne = (1LL << n) - 1;
    for (long long i = 2; i * i <= mersenne; ++i)
        if (mersenne % i == 0)
            return false;
    return true;
}

__global__ void banachPrime_findMersennePrimesKernel(long long *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = 2 + idx;
        while (!banachPrime_isMersennePrime(candidate))
            ++candidate;
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findMersennePrimes(long long *d_primes, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findMersennePrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, count);
}

__device__ inline bool banachPrime_isFermatPrime(long long n) {
    if (n < 3 || (n & (n - 1)) != 0)
        return false;
    long long fermat = (1LL << n) + 1;
    for (long long i = 2; i * i <= fermat; ++i)
        if (fermat % i == 0)
            return false;
    return true;
}

__global__ void banachPrime_findFermatPrimesKernel(long long *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = 2 + idx;
        while (!banachPrime_isFermatPrime(candidate))
            ++candidate;
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findFermatPrimes(long long *d_primes, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findFermatPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, count);
}

__device__ inline bool banachPrime_isCircularPrime(long long n) {
    if (!banachPrime_isPrime(n))
        return false;
    long long original = n;
    while (n > 0) {
        long long digit = n % 10;
        n /= 10;
        long long rotated = digit * static_cast<long long>(pow(10, log10(original))) + n;
        if (!banachPrime_isPrime(rotated))
            return false;
    }
    return true;
}

__global__ void banachPrime_findCircularPrimesKernel(long long *primes, long long start, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = banachPrime_nextPrime(start + idx);
        while (!banachPrime_isCircularPrime(candidate))
            candidate = banachPrime_nextPrime(candidate + 1);
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findCircularPrimes(long long *d_primes, long long start, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findCircularPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, start, count);
}

__device__ inline bool banachPrime_isPythagoreanPrime(long long n) {
    if ((n - 1) % 4 != 0)
        return false;
    return banachPrime_isPrime(n);
}

__global__ void banachPrime_findPythagoreanPrimesKernel(long long *primes, long long start, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = banachPrime_nextPrime(start + idx);
        while (!banachPrime_isPythagoreanPrime(candidate))
            candidate = banachPrime_nextPrime(candidate + 1);
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findPythagoreanPrimes(long long *d_primes, long long start, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findPythagoreanPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, start, count);
}

__device__ inline bool banachPrime_isSafePrime(long long n) {
    return banachPrime_isPrime(n) && banachPrime_isPrime((n - 1) / 2);
}

__global__ void banachPrime_findSafePrimesKernel(long long *primes, long long start, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = banachPrime_nextPrime(start + idx);
        while (!banachPrime_isSafePrime(candidate))
            candidate = banachPrime_nextPrime(candidate + 1);
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findSafePrimes(long long *d_primes, long long start, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findSafePrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, start, count);
}

__device__ inline bool banachPrime_isEmirp(long long n) {
    if (!banachPrime_isPrime(n))
        return false;
    long long reversed = 0;
    long long original = n;
    while (n > 0) {
        reversed = reversed * 10 + n % 10;
        n /= 10;
    }
    return banachPrime_isPrime(reversed) && original != reversed;
}

__global__ void banachPrime_findEmirpsKernel(long long *primes, long long start, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = banachPrime_nextPrime(start + idx);
        while (!banachPrime_isEmirp(candidate))
            candidate = banachPrime_nextPrime(candidate + 1);
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findEmirps(long long *d_primes, long long start, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findEmirpsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, start, count);
}

__device__ inline bool banachPrime_isStrongPrime(long long n) {
    if (!banachPrime_isPrime(n))
        return false;
    long long prev = banachPrime_nextPrime(n - 2);
    long long next = banachPrime_nextPrime(n + 2);
    return n > (prev + next) / 2;
}

__global__ void banachPrime_findStrongPrimesKernel(long long *primes, long long start, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = banachPrime_nextPrime(start + idx);
        while (!banachPrime_isStrongPrime(candidate))
            candidate = banachPrime_nextPrime(candidate + 1);
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findStrongPrimes(long long *d_primes, long long start, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findStrongPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, start, count);
}

__device__ inline bool banachPrime_isWeakPrime(long long n) {
    if (!banachPrime_isPrime(n))
        return false;
    long long prev = banachPrime_nextPrime(n - 2);
    long long next = banachPrime_nextPrime(n + 2);
    return n < (prev + next) / 2;
}

__global__ void banachPrime_findWeakPrimesKernel(long long *primes, long long start, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = banachPrime_nextPrime(start + idx);
        while (!banachPrime_isWeakPrime(candidate))
            candidate = banachPrime_nextPrime(candidate + 1);
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findWeakPrimes(long long *d_primes, long long start, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findWeakPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, start, count);
}

__device__ inline bool banachPrime_isSuperPrime(long long n) {
    if (!banachPrime_isPrime(n))
        return false;
    long long position = 0;
    for (long long i = 2; i <= n; ++i)
        if (banachPrime_isPrime(i))
            ++position;
    return banachPrime_isPrime(position);
}

__global__ void banachPrime_findSuperPrimesKernel(long long *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = 2 + idx;
        while (!banachPrime_isSuperPrime(candidate))
            ++candidate;
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findSuperPrimes(long long *d_primes, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findSuperPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, count);
}

__device__ inline bool banachPrime_isTwinPrime(long long n) {
    if (!banachPrime_isPrime(n))
        return false;
    return banachPrime_isPrime(n - 2) || banachPrime_isPrime(n + 2);
}

__global__ void banachPrime_findTwinPrimesKernel(long long *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = 3 + idx * 2;
        while (!banachPrime_isTwinPrime(candidate))
            candidate += 2;
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findTwinPrimes(long long *d_primes, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findTwinPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, count);
}

__device__ inline bool banachPrime_isLucasCarmichael(long long n) {
    if (!banachPrime_isPrime(n))
        return false;
    for (long long i = 2; i < n; ++i)
        if (pow(i, n - 1, n) != 1)
            return false;
    return true;
}

__global__ void banachPrime_findLucasCarmichaelPrimesKernel(long long *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = 2 + idx;
        while (!banachPrime_isLucasCarmichael(candidate))
            ++candidate;
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findLucasCarmichaelPrimes(long long *d_primes, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findLucasCarmichaelPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, count);
}

__device__ inline bool banachPrime_isSemiPrime(long long n) {
    if (!banachPrime_isPrime(n))
        return false;
    for (long long i = 2; i * i <= n; ++i)
        if (n % i == 0 && banachPrime_isPrime(i) && banachPrime_isPrime(n / i))
            return true;
    return false;
}

__global__ void banachPrime_findSemiPrimesKernel(long long *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = 4 + idx;
        while (!banachPrime_isSemiPrime(candidate))
            ++candidate;
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findSemiPrimes(long long *d_primes, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findSemiPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, count);
}

__device__ inline bool banachPrime_isAliquotSequenceContainsOne(long long n) {
    if (n <= 1)
        return false;
    while (n != 1) {
        long long sum = 0;
        for (long long i = 1; i < n; ++i)
            if (n % i == 0)
                sum += i;
        if (sum >= n || sum == 0)
            return false;
        n = sum;
    }
    return true;
}

__global__ void banachPrime_findAliquotSequencesContainingOneKernel(long long *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = 2 + idx;
        while (!banachPrime_isAliquotSequenceContainsOne(candidate))
            ++candidate;
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findAliquotSequencesContainingOne(long long *d_primes, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findAliquotSequencesContainingOneKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, count);
}

__device__ inline bool banachPrime_isAliquotSequenceContainsZero(long long n) {
    if (n <= 1)
        return false;
    while (n != 0) {
        long long sum = 0;
        for (long long i = 1; i < n; ++i)
            if (n % i == 0)
                sum += i;
        if (sum >= n || sum == 0)
            return false;
        n = sum;
    }
    return true;
}

__global__ void banachPrime_findAliquotSequencesContainingZeroKernel(long long *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        long long candidate = 2 + idx;
        while (!banachPrime_isAliquotSequenceContainsZero(candidate))
            ++candidate;
        primes[idx] = candidate;
    }
}

extern "C" void banachPrime_findAliquotSequencesContainingZero(long long *d_primes, int count, int threadsPerBlock, int blocksPerGrid) {
    banachPrime_findAliquotSequencesContainingZeroKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, count);
}
