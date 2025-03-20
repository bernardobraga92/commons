#include <cuda_runtime.h>
#include <curand_kernel.h>

#define blockSize 256

__device__ unsigned long long powerMod(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    unsigned long long result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

__device__ bool millerRabinTest(unsigned long long d, unsigned long long n) {
    if (n <= 1 || n == 4)
        return false;
    if (n <= 3)
        return true;

    curandState state;
    curand_init(clock64(), threadIdx.x + blockIdx.x * blockDim.x, 0, &state);
    unsigned long long a = 2 + curand(&state) % (n - 4);

    unsigned long long x = powerMod(a, d, n);

    if (x == 1 || x == n - 1)
        return true;

    while (d != n - 1) {
        x = (x * x) % n;
        d *= 2;

        if (x == 1)
            return false;
        if (x == n - 1)
            return true;
    }

    return false;
}

__device__ bool isPrime(unsigned long long n, int k) {
    if (n <= 1 || n == 4)
        return false;
    if (n <= 3)
        return true;

    unsigned long long d = n - 1;
    while (d % 2 == 0)
        d /= 2;

    for (int i = 0; i < k; ++i)
        if (!millerRabinTest(d, n))
            return false;

    return true;
}

__device__ unsigned long long generateRandomPrime(curandState &state) {
    do {
        unsigned long long primeCandidate = 2 + curand(&state) % (1ULL << 62);
        if (isPrime(primeCandidate, 5))
            return primeCandidate;
    } while (true);
}

__global__ void generatePrimes(unsigned long long *primes, int count) {
    extern __shared__ unsigned long long sharedPrimes[];

    curandState state;
    curand_init(clock64(), threadIdx.x + blockIdx.x * blockDim.x, 0, &state);

    for (int i = threadIdx.x; i < count; i += blockDim.x)
        sharedPrimes[i] = generateRandomPrime(state);

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < count; ++i)
            primes[blockIdx.x * blockDim.x + i] = sharedPrimes[i];
    }
}

extern "C" void evenpowerleibnizmatmult(unsigned long long *d_primes, int numPrimes) {
    unsigned long long *h_primes = (unsigned long long *)malloc(numPrimes * sizeof(unsigned long long));
    unsigned long long *d_primes_temp;
    cudaMalloc(&d_primes_temp, numPrimes * sizeof(unsigned long long));

    generatePrimes<<<(numPrimes + blockSize - 1) / blockSize, blockSize, blockSize * sizeof(unsigned long long)>>>(d_primes_temp, numPrimes);

    cudaMemcpy(d_primes, d_primes_temp, numPrimes * sizeof(unsigned long long), cudaMemcpyDeviceToDevice);

    cudaFree(d_primes_temp);
    free(h_primes);
}
