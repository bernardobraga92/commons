#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256

__device__ __inline__ bool isPrime(unsigned int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (unsigned int i = 3; i * i <= num; i += 2)
        if (num % i == 0) return false;
    return true;
}

__global__ void generateRandomPrimes(unsigned int *primes, unsigned int count, unsigned long long seed) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
    while (idx < count) {
        unsigned int num = curand(&state) % 1000000007;
        if (isPrime(num)) {
            primes[idx] = num;
            idx += blockDim.x * gridDim.x;
        }
    }
}

__global__ void findLargestPrime(unsigned int *primes, unsigned int count, unsigned int *largestPrime) {
    extern __shared__ unsigned int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < count)
        sdata[tid] = primes[i];
    else
        sdata[tid] = 0;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicMax(largestPrime, sdata[0]);
}

__global__ void filterPrimes(unsigned int *primes, unsigned int count, unsigned int threshold, unsigned int *filteredPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && primes[idx] > threshold)
        filteredPrimes[idx] = primes[idx];
}

__global__ void sumPrimes(unsigned int *primes, unsigned int count, unsigned long long *sum) {
    extern __shared__ unsigned long long sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < count)
        sdata[tid] = primes[i];
    else
        sdata[tid] = 0;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(sum, sdata[0]);
}

__global__ void multiplyPrimes(unsigned int *primes, unsigned int count, unsigned long long *product) {
    extern __shared__ unsigned long long sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < count)
        sdata[tid] = primes[i];
    else
        sdata[tid] = 1;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] *= sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicMul(product, sdata[0]);
}

__global__ void countPrimes(unsigned int *primes, unsigned int count, unsigned int *countResult) {
    extern __shared__ unsigned int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < count)
        sdata[tid] = 1;
    else
        sdata[tid] = 0;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(countResult, sdata[0]);
}

__global__ void calculatePrimeGaps(unsigned int *primes, unsigned int count, unsigned int *gaps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 1)
        gaps[idx] = primes[idx + 1] - primes[idx];
}

__global__ void findSmallestPrime(unsigned int *primes, unsigned int count, unsigned int *smallestPrime) {
    extern __shared__ unsigned int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < count)
        sdata[tid] = primes[i];
    else
        sdata[tid] = UINT_MAX;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] < sdata[tid])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicMin(smallestPrime, sdata[0]);
}

__global__ void checkForTwinPrimes(unsigned int *primes, unsigned int count, bool *hasTwinPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 1 && primes[idx] + 2 == primes[idx + 1])
        atomicOr(hasTwinPrimes, true);
}

__global__ void calculatePrimeSquares(unsigned int *primes, unsigned int count, unsigned long long *squares) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        squares[idx] = (unsigned long long)primes[idx] * primes[idx];
}

__global__ void calculatePrimeCubes(unsigned int *primes, unsigned int count, unsigned long long *cubes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        cubes[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForCousinPrimes(unsigned int *primes, unsigned int count, bool *hasCousinPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 1 && abs(primes[idx] - primes[idx + 1]) == 4)
        atomicOr(hasCousinPrimes, true);
}

__global__ void checkForSexyPrimes(unsigned int *primes, unsigned int count, bool *hasSexyPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 1 && abs(primes[idx] - primes[idx + 1]) == 6)
        atomicOr(hasSexyPrimes, true);
}

__global__ void calculatePrimeFourthPowers(unsigned int *primes, unsigned int count, unsigned long long *fourthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        fourthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void calculatePrimeFifthPowers(unsigned int *primes, unsigned int count, unsigned long long *fifthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        fifthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimeTuples(unsigned int *primes, unsigned int count, bool *hasPrimeTuples) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 2 && primes[idx] + 4 == primes[idx + 1] && primes[idx] + 6 == primes[idx + 2])
        atomicOr(hasPrimeTuples, true);
}

__global__ void calculatePrimeSixthPowers(unsigned int *primes, unsigned int count, unsigned long long *sixthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        sixthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimeClusters(unsigned int *primes, unsigned int count, bool *hasPrimeClusters) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 4 && primes[idx] + 2 == primes[idx + 1] && primes[idx] + 6 == primes[idx + 2] &&
        primes[idx] + 8 == primes[idx + 3] && primes[idx] + 12 == primes[idx + 4])
        atomicOr(hasPrimeClusters, true);
}

__global__ void calculatePrimeSeventhPowers(unsigned int *primes, unsigned int count, unsigned long long *seventhPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        seventhPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                              primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns(unsigned int *primes, unsigned int count, bool *hasPrimePatterns) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 2 && primes[idx] + 4 == primes[idx + 1] && primes[idx] + 6 == primes[idx + 2] &&
        primes[idx] + 8 == primes[idx + 3])
        atomicOr(hasPrimePatterns, true);
}

__global__ void calculatePrimeEighthPowers(unsigned int *primes, unsigned int count, unsigned long long *eighthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        eighthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                             primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimeSeries(unsigned int *primes, unsigned int count, bool *hasPrimeSeries) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 2 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3])
        atomicOr(hasPrimeSeries, true);
}

__global__ void calculatePrimeNinthPowers(unsigned int *primes, unsigned int count, unsigned long long *ninthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        ninthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                            primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns2(unsigned int *primes, unsigned int count, bool *hasPrimePatterns2) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 4 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4])
        atomicOr(hasPrimePatterns2, true);
}

__global__ void calculatePrimeTenthPowers(unsigned int *primes, unsigned int count, unsigned long long *tenthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        tenthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                            primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns3(unsigned int *primes, unsigned int count, bool *hasPrimePatterns3) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 6 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5])
        atomicOr(hasPrimePatterns3, true);
}

__global__ void calculatePrimeEleventhPowers(unsigned int *primes, unsigned int count, unsigned long long *eleventhPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        eleventhPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                               primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns4(unsigned int *primes, unsigned int count, bool *hasPrimePatterns4) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 8 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7])
        atomicOr(hasPrimePatterns4, true);
}

__global__ void calculatePrimeTwelfthPowers(unsigned int *primes, unsigned int count, unsigned long long *twelfthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        twelfthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                               primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns5(unsigned int *primes, unsigned int count, bool *hasPrimePatterns5) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 10 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8])
        atomicOr(hasPrimePatterns5, true);
}

__global__ void calculatePrimeThirteenthPowers(unsigned int *primes, unsigned int count, unsigned long long *thirteenthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        thirteenthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                 primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns6(unsigned int *primes, unsigned int count, bool *hasPrimePatterns6) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 12 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9])
        atomicOr(hasPrimePatterns6, true);
}

__global__ void calculatePrimeFourteenthPowers(unsigned int *primes, unsigned int count, unsigned long long *fourteenthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        fourteenthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                  primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns7(unsigned int *primes, unsigned int count, bool *hasPrimePatterns7) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 14 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10])
        atomicOr(hasPrimePatterns7, true);
}

__global__ void calculatePrimeFifteenthPowers(unsigned int *primes, unsigned int count, unsigned long long *fifteenthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        fifteenthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                  primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns8(unsigned int *primes, unsigned int count, bool *hasPrimePatterns8) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 16 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11])
        atomicOr(hasPrimePatterns8, true);
}

__global__ void calculatePrimeSixteenthPowers(unsigned int *primes, unsigned int count, unsigned long long *sixteenthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        sixteenthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                  primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns9(unsigned int *primes, unsigned int count, bool *hasPrimePatterns9) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 18 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12])
        atomicOr(hasPrimePatterns9, true);
}

__global__ void calculatePrimeSeventeenthPowers(unsigned int *primes, unsigned int count, unsigned long long *seventeenthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        seventeenthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns10(unsigned int *primes, unsigned int count, bool *hasPrimePatterns10) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 20 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13])
        atomicOr(hasPrimePatterns10, true);
}

__global__ void calculatePrimeEighteenthPowers(unsigned int *primes, unsigned int count, unsigned long long *eighteenthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        eighteenthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns11(unsigned int *primes, unsigned int count, bool *hasPrimePatterns11) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 22 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14])
        atomicOr(hasPrimePatterns11, true);
}

__global__ void calculatePrimeNineteenthPowers(unsigned int *primes, unsigned int count, unsigned long long *nineteenthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        nineteenthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns12(unsigned int *primes, unsigned int count, bool *hasPrimePatterns12) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 24 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15])
        atomicOr(hasPrimePatterns12, true);
}

__global__ void calculatePrimeTwentiethPowers(unsigned int *primes, unsigned int count, unsigned long long *twentiethPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        twentiethPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns13(unsigned int *primes, unsigned int count, bool *hasPrimePatterns13) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 26 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16])
        atomicOr(hasPrimePatterns13, true);
}

__global__ void calculatePrimeTwentyFirstPowers(unsigned int *primes, unsigned int count, unsigned long long *twentyFirstPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        twentyFirstPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns14(unsigned int *primes, unsigned int count, bool *hasPrimePatterns14) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 28 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17])
        atomicOr(hasPrimePatterns14, true);
}

__global__ void calculatePrimeTwentySecondPowers(unsigned int *primes, unsigned int count, unsigned long long *twentySecondPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        twentySecondPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns15(unsigned int *primes, unsigned int count, bool *hasPrimePatterns15) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 30 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17] && primes[idx] + 42 == primes[idx + 18])
        atomicOr(hasPrimePatterns15, true);
}

__global__ void calculatePrimeTwentyThirdPowers(unsigned int *primes, unsigned int count, unsigned long long *twentyThirdPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        twentyThirdPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns16(unsigned int *primes, unsigned int count, bool *hasPrimePatterns16) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 32 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17] && primes[idx] + 42 == primes[idx + 18] && primes[idx] + 44 == primes[idx + 19])
        atomicOr(hasPrimePatterns16, true);
}

__global__ void calculatePrimeTwentyFourthPowers(unsigned int *primes, unsigned int count, unsigned long long *twentyFourthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        twentyFourthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns17(unsigned int *primes, unsigned int count, bool *hasPrimePatterns17) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 34 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17] && primes[idx] + 42 == primes[idx + 18] && primes[idx] + 44 == primes[idx + 19] && primes[idx] + 46 == primes[idx + 20])
        atomicOr(hasPrimePatterns17, true);
}

__global__ void calculatePrimeTwentyFifthPowers(unsigned int *primes, unsigned int count, unsigned long long *twentyFifthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        twentyFifthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns18(unsigned int *primes, unsigned int count, bool *hasPrimePatterns18) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 36 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17] && primes[idx] + 42 == primes[idx + 18] && primes[idx] + 44 == primes[idx + 19] && primes[idx] + 46 == primes[idx + 20] && primes[idx] + 48 == primes[idx + 21])
        atomicOr(hasPrimePatterns18, true);
}

__global__ void calculatePrimeTwentySixthPowers(unsigned int *primes, unsigned int count, unsigned long long *twentySixthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        twentySixthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns19(unsigned int *primes, unsigned int count, bool *hasPrimePatterns19) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 38 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17] && primes[idx] + 42 == primes[idx + 18] && primes[idx] + 44 == primes[idx + 19] && primes[idx] + 46 == primes[idx + 20] && primes[idx] + 48 == primes[idx + 21] && primes[idx] + 50 == primes[idx + 22])
        atomicOr(hasPrimePatterns19, true);
}

__global__ void calculatePrimeTwentySeventhPowers(unsigned int *primes, unsigned int count, unsigned long long *twentySeventhPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        twentySeventhPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns20(unsigned int *primes, unsigned int count, bool *hasPrimePatterns20) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 40 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17] && primes[idx] + 42 == primes[idx + 18] && primes[idx] + 44 == primes[idx + 19] && primes[idx] + 46 == primes[idx + 20] && primes[idx] + 48 == primes[idx + 21] && primes[idx] + 50 == primes[idx + 22] && primes[idx] + 52 == primes[idx + 23])
        atomicOr(hasPrimePatterns20, true);
}

__global__ void calculatePrimeTwentyEighthPowers(unsigned int *primes, unsigned int count, unsigned long long *twentyEighthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        twentyEighthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns21(unsigned int *primes, unsigned int count, bool *hasPrimePatterns21) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 42 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17] && primes[idx] + 42 == primes[idx + 18] && primes[idx] + 44 == primes[idx + 19] && primes[idx] + 46 == primes[idx + 20] && primes[idx] + 48 == primes[idx + 21] && primes[idx] + 50 == primes[idx + 22] && primes[idx] + 52 == primes[idx + 23] && primes[idx] + 54 == primes[idx + 24])
        atomicOr(hasPrimePatterns21, true);
}

__global__ void calculatePrimeTwentyNinthPowers(unsigned int *primes, unsigned int count, unsigned long long *twentyNinthPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        twentyNinthPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns22(unsigned int *primes, unsigned int count, bool *hasPrimePatterns22) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 44 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17] && primes[idx] + 42 == primes[idx + 18] && primes[idx] + 44 == primes[idx + 19] && primes[idx] + 46 == primes[idx + 20] && primes[idx] + 48 == primes[idx + 21] && primes[idx] + 50 == primes[idx + 22] && primes[idx] + 52 == primes[idx + 23] && primes[idx] + 54 == primes[idx + 24] && primes[idx] + 56 == primes[idx + 25])
        atomicOr(hasPrimePatterns22, true);
}

__global__ void calculatePrimeThirtiethPowers(unsigned int *primes, unsigned int count, unsigned long long *thirtyPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        thirtyPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns23(unsigned int *primes, unsigned int count, bool *hasPrimePatterns23) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 46 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17] && primes[idx] + 42 == primes[idx + 18] && primes[idx] + 44 == primes[idx + 19] && primes[idx] + 46 == primes[idx + 20] && primes[idx] + 48 == primes[idx + 21] && primes[idx] + 50 == primes[idx + 22] && primes[idx] + 52 == primes[idx + 23] && primes[idx] + 54 == primes[idx + 24] && primes[idx] + 56 == primes[idx + 25] && primes[idx] + 58 == primes[idx + 26])
        atomicOr(hasPrimePatterns23, true);
}

__global__ void calculatePrimeThirtyFirstPowers(unsigned int *primes, unsigned int count, unsigned long long *thirtyOnePowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        thirtyOnePowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns24(unsigned int *primes, unsigned int count, bool *hasPrimePatterns24) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 48 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17] && primes[idx] + 42 == primes[idx + 18] && primes[idx] + 44 == primes[idx + 19] && primes[idx] + 46 == primes[idx + 20] && primes[idx] + 48 == primes[idx + 21] && primes[idx] + 50 == primes[idx + 22] && primes[idx] + 52 == primes[idx + 23] && primes[idx] + 54 == primes[idx + 24] && primes[idx] + 56 == primes[idx + 25] && primes[idx] + 58 == primes[idx + 26] && primes[idx] + 60 == primes[idx + 27])
        atomicOr(hasPrimePatterns24, true);
}

__global__ void calculatePrimeThirtySecondPowers(unsigned int *primes, unsigned int count, unsigned long long *thirtyTwoPowers) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        thirtyTwoPowers[idx] = (unsigned long long)primes[idx] * primes[idx] * primes[idx] * primes[idx] *
                                   primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx] * primes[idx];
}

__global__ void checkForPrimePatterns25(unsigned int *primes, unsigned int count, bool *hasPrimePatterns25) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 50 && primes[idx] + 6 == primes[idx + 1] && primes[idx] + 8 == primes[idx + 2] &&
        primes[idx] + 10 == primes[idx + 3] && primes[idx] + 14 == primes[idx + 4] && primes[idx] + 16 == primes[idx + 5] &&
        primes[idx] + 18 == primes[idx + 6] && primes[idx] + 20 == primes[idx + 7] && primes[idx] + 22 == primes[idx + 8] && primes[idx] + 24 == primes[idx + 9] && primes[idx] + 26 == primes[idx + 10] && primes[idx] + 28 == primes[idx + 11] && primes[idx] + 30 == primes[idx + 12] && primes[idx] + 32 == primes[idx + 13] && primes[idx] + 34 == primes[idx + 14] && primes[idx] + 36 == primes[idx + 15] && primes[idx] + 38 == primes[idx + 16] && primes[idx] + 40 == primes[idx + 17] && primes[idx] + 42 == primes[idx + 18] && primes[idx] + 44 == primes[idx + 19] && primes[idx] + 46 == primes[idx + 20] && primes[idx] + 48 == primes[idx + 21] && primes[idx] + 50 == primes[idx + 22] && primes[idx] + 52 == primes[idx + 23] && primes[idx] + 54 == primes[idx + 24] && primes[idx] + 56 == primes[idx + 25] && primes[idx] + 58 == primes[idx + 26] && primes[idx] + 60 == primes[idx + 27] && primes[idx] + 62 == primes[idx + 28])
        atomicOr(hasPrimePatterns25, true);
}

// Function to check if a number is prime
bool is_prime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

// Function to generate a list of primes up to a given limit
std::vector<int> generate_primes(int limit) {
    std::vector<int> primes;
    for (int i = 2; i <= limit; ++i) {
        if (is_prime(i)) {
            primes.push_back(i);
        }
    }
    return primes;
}

// Main function to execute the CUDA kernels
int main() {
    // Define constants and allocate memory
    int limit = 1000;
    std::vector<int> primes = generate_primes(limit);
    int count = primes.size();
    unsigned long long *powers;
    bool *has_patterns;

    // Allocate device memory
    cudaMalloc((void**)&powers, count * sizeof(unsigned long long));
    cudaMalloc((void**)&has_patterns, 25 * sizeof(bool));

    // Copy data to device
    cudaMemcpy(powers, &primes[0], count * sizeof(int), cudaMemcpyHostToDevice);

    // Launch CUDA kernels
    calculate_prime_powers<<<(count + 255) / 256, 256>>>(powers, count);
    check_for_patterns<<<25, 1>>>(powers, count, has_patterns);

    // Copy results back to host
    cudaMemcpy(&primes[0], powers, count * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&has_patterns[0], has_patterns, 25 * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(powers);
    cudaFree(has_patterns);

    // Output results
    for (int i = 0; i < count; ++i) {
        std::cout << primes[i] << ": " << powers[i] << std::endl;
    }

    return 0;
}
