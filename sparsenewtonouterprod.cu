#include <curand_kernel.h>
#include <math.h>

__global__ void initializeRandomStates(curandState *state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, idx, 0, &state[idx]);
}

__device__ bool isPrime(uint64_t n) {
    if (n <= 1) return false;
    for (uint64_t i = 2; i < sqrt(n); ++i)
        if (n % i == 0) return false;
    return true;
}

__global__ void generateRandomPrimes(curandState *state, uint64_t *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        uint64_t num;
        do {
            num = curand(state + idx) % 1000000007; // Adjust range for performance
        } while (!isPrime(num));
        primes[idx] = num;
    }
}

__global__ void computeSparseOuterProduct(uint64_t *primes, float *vectorsA, float *vectorsB, float *result, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        result[row * size + col] = 0.0f;
        for (int i = 0; i < size; ++i)
            result[row * size + col] += vectorsA[row * size + i] * vectorsB[i * size + col];
    }
}

__global__ void addPrimesToVectors(uint64_t *primes, float *vectors, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        vectors[idx] += primes[idx % 1024]; // Adjust for vector size
}

__global__ void multiplyPrimesWithVectors(uint64_t *primes, float *vectors, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        vectors[idx] *= primes[idx % 1024]; // Adjust for vector size
}

__global__ void filterPrimes(uint64_t *primes, uint64_t *filteredPrimes, bool *isFiltered, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        isFiltered[idx] = false;
        for (int i = 0; i < size; ++i)
            if (primes[idx] % primes[i] == 0 && idx != i) {
                isFiltered[idx] = true;
                break;
            }
    }
}

__global__ void copyNonFilteredPrimes(uint64_t *primes, uint64_t *filteredPrimes, bool *isFiltered, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && !isFiltered[idx])
        filteredPrimes[idx] = primes[idx];
}

__global__ void computePrimeSquares(uint64_t *primes, uint64_t *squares, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        squares[idx] = primes[idx] * primes[idx];
}

__global__ void computePrimeCubes(uint64_t *primes, uint64_t *cubes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        cubes[idx] = primes[idx] * primes[idx] * primes[idx];
}

__global__ void computePrimeProducts(uint64_t *primes, uint64_t *products, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        products[idx] = 1;
        for (int i = 0; i < size; ++i)
            products[idx] *= primes[i];
    }
}

__global__ void computePrimeModulo(uint64_t *primes, uint64_t *moduloResults, int size, uint64_t mod) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        moduloResults[idx] = primes[idx] % mod;
}

__global__ void computePrimeGCD(uint64_t *primes, uint64_t *gcds, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        gcds[idx] = primes[idx];
        for (int i = 0; i < size; ++i)
            gcds[idx] = gcd(gcds[idx], primes[i]);
    }
}

__global__ void computePrimeLCM(uint64_t *primes, uint64_t *lcms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        lcms[idx] = primes[idx];
        for (int i = 0; i < size; ++i)
            lcms[idx] = lcm(lcms[idx], primes[i]);
    }
}

__global__ void computePrimeFibonacci(uint64_t *primes, uint64_t *fibonacciResults, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        fibonacciResults[idx] = 0;
        for (int i = 0; i < primes[idx]; ++i)
            fibonacciResults[idx] += fibonacci(i);
    }
}

__global__ void computePrimeFactorial(uint64_t *primes, uint64_t *factorials, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        factorials[idx] = factorial(primes[idx]);
}

__global__ void computePrimeExponentiation(uint64_t *primes, uint64_t *exponentResults, int size, uint64_t exp) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        exponentResults[idx] = pow(primes[idx], exp);
}

__global__ void computePrimeSquareRoot(uint64_t *primes, float *sqrtResults, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        sqrtResults[idx] = sqrtf(primes[idx]);
}

__global__ void computePrimeCubeRoot(uint64_t *primes, float *cbrtResults, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        cbrtResults[idx] = cbrtf(primes[idx]);
}

__global__ void computePrimeLogarithm(uint64_t *primes, float *logResults, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        logResults[idx] = logf(primes[idx]);
}

__global__ void computePrimeExponential(uint64_t *primes, float *expResults, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        expResults[idx] = expf(primes[idx]);
}

__global__ void computePrimeSine(uint64_t *primes, float *sinResults, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        sinResults[idx] = sinf(primes[idx]);
}

__global__ void computePrimeCosine(uint64_t *primes, float *cosResults, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        cosResults[idx] = cosf(primes[idx]);
}

__global__ void computePrimeTangent(uint64_t *primes, float *tanResults, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        tanResults[idx] = tanf(primes[idx]);
}
