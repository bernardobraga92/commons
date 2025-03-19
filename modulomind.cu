#include <cuda_runtime.h>
#include <cmath>

__device__ bool isPrime(uint64_t num) {
    if (num <= 1) return false;
    if (num <= 3) return true;
    if (num % 2 == 0 || num % 3 == 0) return false;
    for (uint64_t i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) return false;
    }
    return true;
}

__device__ uint64_t nextPrime(uint64_t start) {
    while (!isPrime(start)) ++start;
    return start;
}

__global__ void findPrimes(uint64_t *primes, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) primes[idx] = nextPrime(idx * 1000000ULL);
}

__device__ uint64_t modInverse(uint64_t a, uint64_t m) {
    for (uint64_t x = 1; x < m; ++x) {
        if ((a * x) % m == 1) return x;
    }
    return -1;
}

__global__ void computeModInverses(uint64_t *primes, uint64_t *inverses, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) inverses[idx] = modInverse(primes[idx], primes[idx] - 1);
}

__device__ bool isCoprime(uint64_t a, uint64_t b) {
    return gcd(a, b) == 1;
}

__device__ uint64_t gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__global__ void checkCoprimePairs(uint64_t *primes, bool *coprimeResults, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && idx > 0) coprimeResults[idx] = isCoprime(primes[idx], primes[idx - 1]);
}

__device__ uint64_t powerMod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1) result = (result * base) % mod;
        exp >>= 1;
        base = (base * base) % mod;
    }
    return result;
}

__global__ void computePowerMods(uint64_t *primes, uint64_t *powerResults, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) powerResults[idx] = powerMod(primes[idx], primes[idx] - 1, primes[idx]);
}

__device__ bool millerRabinTest(uint64_t d, uint64_t n) {
    uint64_t a = 2 + rand() % (n - 4);
    uint64_t x = powerMod(a, d, n);
    if (x == 1 || x == n - 1) return true;
    while (d != n - 1) {
        x = (x * x) % n;
        d *= 2;
        if (x == 1) return false;
        if (x == n - 1) return true;
    }
    return false;
}

__device__ bool isPrimeMR(uint64_t n, int k) {
    if (n <= 1 || n == 4) return false;
    if (n <= 3) return true;
    uint64_t d = n - 1;
    while (d % 2 == 0) d /= 2;
    for (int i = 0; i < k; ++i)
        if (!millerRabinTest(d, n)) return false;
    return true;
}

__global__ void findPrimesMR(uint64_t *primes, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) primes[idx] = nextPrime(idx * 1000000ULL);
}

__device__ uint64_t factorialMod(uint64_t num, uint64_t mod) {
    uint64_t result = 1;
    for (uint64_t i = 2; i <= num; ++i)
        result = (result * i) % mod;
    return result;
}

__global__ void computeFactorialsMod(uint64_t *primes, uint64_t *factorialResults, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) factorialResults[idx] = factorialMod(primes[idx], primes[idx]);
}

__device__ uint64_t chineseRemainder(uint64_t a1, uint64_t m1, uint64_t a2, uint64_t m2) {
    uint64_t M = m1 * m2;
    uint64_t Mi = M / m1;
    uint64_t y = modInverse(Mi, m1);
    return (a1 * Mi * y + a2) % M;
}

__global__ void computeChineseRemainders(uint64_t *primes, uint64_t *remainders, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && idx > 0) remainders[idx] = chineseRemainder(primes[idx - 1], primes[idx - 1], primes[idx], primes[idx]);
}

__device__ uint64_t pollardRho(uint64_t n, uint64_t c) {
    if (n % 2 == 0) return 2;
    uint64_t x = 2, y = 2, d = 1;
    while (d == 1) {
        x = (powerMod(x, 2, n) + c) % n;
        y = (powerMod(powerMod(y, 2, n), 2, n) + c) % n;
        d = gcd(abs(x - y), n);
    }
    return d == n ? -1 : d;
}

__global__ void findFactorsPR(uint64_t *primes, uint64_t *factors, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) factors[idx] = pollardRho(primes[idx], 1);
}

__device__ bool isWilsonPrime(uint64_t p) {
    return factorialMod(p - 1, p * p) == p - 1;
}

__global__ void findWilsonPrimes(uint64_t *primes, bool *wilsonResults, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) wilsonResults[idx] = isWilsonPrime(primes[idx]);
}

__device__ uint64_t legendreSymbol(uint64_t a, uint64_t p) {
    return powerMod(a, (p - 1) / 2, p);
}

__global__ void computeLegendreSymbols(uint64_t *primes, int64_t *legendreResults, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && idx > 0) legendreResults[idx] = legendreSymbol(primes[idx - 1], primes[idx]);
}

__device__ uint64_t jacobiSymbol(uint64_t a, uint64_t n) {
    int result = 1;
    while (a != 0) {
        if (a % 2 == 1) {
            if ((n % 8 == 3 || n % 8 == 5)) result *= -1;
        }
        if (n % 4 == 3 && a % 4 == 3) result *= -1;
        std::swap(a, n);
        a %= n;
    }
    return n == 1 ? result : 0;
}

__global__ void computeJacobiSymbols(uint64_t *primes, int64_t *jacobiResults, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && idx > 0) jacobiResults[idx] = jacobiSymbol(primes[idx - 1], primes[idx]);
}

__device__ uint64_t eulerTotient(uint64_t n) {
    uint64_t result = n;
    for (uint64_t i = 2; i * i <= n; ++i)
        if (n % i == 0) {
            while (n % i == 0) n /= i;
            result -= result / i;
        }
    if (n > 1) result -= result / n;
    return result;
}

__global__ void computeEulerTotients(uint64_t *primes, uint64_t *totientResults, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) totientResults[idx] = eulerTotient(primes[idx]);
}

__device__ bool isSemiPrime(uint64_t n) {
    for (uint64_t i = 2; i * i <= n; ++i)
        if (n % i == 0) return true;
    return false;
}

__global__ void findSemiPrimes(uint64_t *primes, bool *semiPrimeResults, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) semiPrimeResults[idx] = isSemiPrime(primes[idx]);
}
