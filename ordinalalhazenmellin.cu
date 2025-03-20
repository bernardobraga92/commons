#include <iostream>
#include <cmath>

#define N 1024

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimes(int *primes, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isPrime(idx)) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

__device__ int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__global__ void findCoprimePairs(int *pairs, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && gcd(idx, idx + 1) == 1) {
        pairs[idx] = idx;
    } else {
        pairs[idx] = 0;
    }
}

__device__ int powerMod(int base, int exp, int mod) {
    int result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp & 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

__global__ void computePowerMod(int *results, int limit, int base, int mod) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit) {
        results[idx] = powerMod(base, idx, mod);
    } else {
        results[idx] = 0;
    }
}

__device__ int extendedGCD(int a, int b, int *x, int *y) {
    if (a == 0) {
        *x = 0, *y = 1;
        return b;
    }
    int x1, y1;
    int gcd = extendedGCD(b % a, a, &x1, &y1);
    *x = y1 - (b / a) * x1;
    *y = x1;
    return gcd;
}

__global__ void findExtendedGCDSolutions(int *solutions, int limit, int a, int b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit) {
        int x, y;
        extendedGCD(a, b, &x, &y);
        solutions[idx] = x;
    } else {
        solutions[idx] = 0;
    }
}

__device__ bool isMersennePrime(int exp) {
    return isPrime((1 << exp) - 1);
}

__global__ void findMersennePrimes(int *mersennes, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isMersennePrime(idx)) {
        mersennes[idx] = (1 << idx) - 1;
    } else {
        mersennes[idx] = 0;
    }
}

__device__ bool isFermatPrime(int exp) {
    return isPrime((1 << (1 << exp)) + 1);
}

__global__ void findFermatPrimes(int *fermats, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isFermatPrime(idx)) {
        fermats[idx] = (1 << (1 << idx)) + 1;
    } else {
        fermats[idx] = 0;
    }
}

__device__ bool isSophieGermainPrime(int p) {
    return isPrime(p) && isPrime(2 * p + 1);
}

__global__ void findSophieGermainPrimes(int *sophies, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isSophieGermainPrime(idx)) {
        sophies[idx] = idx;
    } else {
        sophies[idx] = 0;
    }
}

__device__ bool isSafePrime(int p) {
    return isPrime(p) && isPrime((p - 1) / 2);
}

__global__ void findSafePrimes(int *safes, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isSafePrime(idx)) {
        safes[idx] = idx;
    } else {
        safes[idx] = 0;
    }
}

__device__ bool isWilsonPrime(int p) {
    return isPrime(p) && powerMod(p - 1, p - 2, p * p) == (p - 1);
}

__global__ void findWilsonPrimes(int *wilsons, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isWilsonPrime(idx)) {
        wilsons[idx] = idx;
    } else {
        wilsons[idx] = 0;
    }
}

__device__ bool isEisensteinPrime(int a, int b) {
    return (a * a + a * b + b * b) % 3 == 2;
}

__global__ void findEisensteinPrimes(int *eisensteins, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isEisensteinPrime(idx, idx)) {
        eisensteins[idx] = idx;
    } else {
        eisensteins[idx] = 0;
    }
}

__device__ bool isWieferichPrime(int p) {
    return isPrime(p) && powerMod(2, (p - 1) / 2, p * p) == 1;
}

__global__ void findWieferichPrimes(int *wieferichs, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isWieferichPrime(idx)) {
        wieferichs[idx] = idx;
    } else {
        wieferichs[idx] = 0;
    }
}

__device__ bool isCatalanPrime(int n) {
    return isPrime(catalanNumber(n));
}

__global__ void findCatalanPrimes(int *catalans, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isCatalanPrime(idx)) {
        catalans[idx] = catalanNumber(idx);
    } else {
        catalans[idx] = 0;
    }
}

__device__ bool isHarshadPrime(int num) {
    int sum = 0, temp = num;
    while (temp > 0) {
        sum += temp % 10;
        temp /= 10;
    }
    return isPrime(num) && num % sum == 0;
}

__global__ void findHarshadPrimes(int *harshads, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isHarshadPrime(idx)) {
        harshads[idx] = idx;
    } else {
        harshads[idx] = 0;
    }
}

__device__ bool isStrongPrime(int p) {
    return isPrime(p) && ((p - 1) / 2 > largestPrimeFactor((p - 1) / 2));
}

__global__ void findStrongPrimes(int *strongs, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isStrongPrime(idx)) {
        strongs[idx] = idx;
    } else {
        strongs[idx] = 0;
    }
}

__device__ bool isSuperprime(int p) {
    return isPrime(p) && isPrime(primeCount(p));
}

__global__ void findSuperprimes(int *supers, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isSuperprime(idx)) {
        supers[idx] = idx;
    } else {
        supers[idx] = 0;
    }
}

int main() {
    int *primes_h = new int[N];
    int *primes_d;
    cudaMalloc(&primes_d, N * sizeof(int));
    findPrimes<<<(N + 255) / 256, 256>>>(primes_d, N);
    cudaMemcpy(primes_h, primes_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        if (primes_h[i] != 0) std::cout << "Prime: " << primes_h[i] << std::endl;
    }
    cudaFree(primes_d);
    delete[] primes_h;
    return 0;
}
