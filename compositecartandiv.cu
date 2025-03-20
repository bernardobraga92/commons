#include <cmath>
#include <cstdlib>

__device__ bool isPrime(unsigned long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (unsigned long i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

__device__ unsigned long generateRandomPrime(unsigned long min, unsigned long max) {
    unsigned long num;
    do {
        num = min + static_cast<unsigned long>(rand()) % (max - min);
    } while (!isPrime(num));
    return num;
}

__global__ void findPrimesKernel(unsigned long *d_primes, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_primes[idx] = generateRandomPrime(2, 100000);
}

__global__ void multiplyPrimesKernel(unsigned long *d_primes, unsigned long *d_results, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_results[idx] = d_primes[idx] * d_primes[(idx + 1) % numPrimes];
}

__global__ void addPrimesKernel(unsigned long *d_primes, unsigned long *d_results, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_results[idx] = d_primes[idx] + d_primes[(idx + 1) % numPrimes];
}

__global__ void subtractPrimesKernel(unsigned long *d_primes, unsigned long *d_results, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_results[idx] = d_primes[idx] > d_primes[(idx + 1) % numPrimes] ? d_primes[idx] - d_primes[(idx + 1) % numPrimes] : 0;
}

__global__ void dividePrimesKernel(unsigned long *d_primes, unsigned long *d_results, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes && d_primes[(idx + 1) % numPrimes] != 0)
        d_results[idx] = d_primes[idx] / d_primes[(idx + 1) % numPrimes];
}

__global__ void squarePrimesKernel(unsigned long *d_primes, unsigned long *d_results, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_results[idx] = d_primes[idx] * d_primes[idx];
}

__global__ void cubePrimesKernel(unsigned long *d_primes, unsigned long *d_results, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_results[idx] = d_primes[idx] * d_primes[idx] * d_primes[idx];
}

__global__ void checkCompositeCartAndDivKernel(unsigned long *d_primes, bool *d_compositeResults, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_compositeResults[idx] = !isPrime(d_primes[idx] * d_primes[(idx + 1) % numPrimes]);
}

__global__ void findNextPrimeKernel(unsigned long *d_primes, unsigned long *d_nextPrimes, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_nextPrimes[idx] = generateRandomPrime(d_primes[idx], 200000);
}

__global__ void findPreviousPrimeKernel(unsigned long *d_primes, unsigned long *d_previousPrimes, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_previousPrimes[idx] = generateRandomPrime(2, d_primes[idx]);
}

__global__ void findTwinPrimesKernel(unsigned long *d_primes, bool *d_twinResults, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_twinResults[idx] = isPrime(d_primes[idx] + 2);
}

__global__ void findCousinPrimesKernel(unsigned long *d_primes, bool *d_cousinResults, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_cousinResults[idx] = isPrime(d_primes[idx] + 4);
}

__global__ void findSexyPrimesKernel(unsigned long *d_primes, bool *d_sexyResults, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_sexyResults[idx] = isPrime(d_primes[idx] + 6);
}

__global__ void findPrimeGapsKernel(unsigned long *d_primes, unsigned long *d_primeGaps, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_primeGaps[idx] = d_primes[(idx + 1) % numPrimes] - d_primes[idx];
}

__global__ void findPrimeSquaresKernel(unsigned long *d_primes, unsigned long *d_primeSquares, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_primeSquares[idx] = d_primes[idx] * d_primes[idx];
}

__global__ void findPrimeCubesKernel(unsigned long *d_primes, unsigned long *d_primeCubes, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_primeCubes[idx] = d_primes[idx] * d_primes[idx] * d_primes[idx];
}

__global__ void findPrimeFourthPowersKernel(unsigned long *d_primes, unsigned long *d_primeFourthPowers, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_primeFourthPowers[idx] = d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx];
}

__global__ void findPrimeFifthPowersKernel(unsigned long *d_primes, unsigned long *d_primeFifthPowers, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_primeFifthPowers[idx] = d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx];
}

__global__ void findPrimeSixthPowersKernel(unsigned long *d_primes, unsigned long *d_primeSixthPowers, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_primeSixthPowers[idx] = d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx];
}

__global__ void findPrimeSeventhPowersKernel(unsigned long *d_primes, unsigned long *d_primeSeventhPowers, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_primeSeventhPowers[idx] = d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx];
}

__global__ void findPrimeEighthPowersKernel(unsigned long *d_primes, unsigned long *d_primeEighthPowers, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_primeEighthPowers[idx] = d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx];
}

__global__ void findPrimeNinthPowersKernel(unsigned long *d_primes, unsigned long *d_primeNinthPowers, unsigned long numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes)
        d_primeNinthPowers[idx] = d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx] * d_primes[idx];
}
