#include <curand_kernel.h>

__device__ uint64_t modularExponentiation(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

__device__ bool isProbablyPrime(uint64_t n, int k) {
    if (n <= 1 || n == 4)
        return false;
    if (n <= 3)
        return true;

    uint64_t d = n - 1;
    while (d % 2 == 0)
        d /= 2;

    for (int i = 0; i < k; i++) {
        curandState state;
        curand_init(clock(), threadIdx.x, 0, &state);
        uint64_t a = 2 + curand(&state) % (n - 4);

        uint64_t x = modularExponentiation(a, d, n);
        if (x == 1 || x == n - 1)
            continue;

        bool flag = false;
        while (d != n - 1) {
            x = (x * x) % n;
            d *= 2;

            if (x == 1)
                return false;
            if (x == n - 1) {
                flag = true;
                break;
            }
        }
        if (!flag)
            return false;
    }
    return true;
}

__device__ uint64_t findRandomPrime(uint64_t start, curandState *state) {
    while (true) {
        uint64_t candidate = start + curand(state) % 1000;
        if (isProbablyPrime(candidate, 5))
            return candidate;
    }
}

__global__ void generatePrimes(uint64_t *primes, int numPrimes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        curandState state;
        curand_init(clock(), idx, 0, &state);
        primes[idx] = findRandomPrime(1000000, &state);
    }
}

extern "C" void unboundedjacobitaylor(uint64_t *primes, int numPrimes) {
    generatePrimes<<<(numPrimes + 255) / 256, 256>>>(primes, numPrimes);
}
