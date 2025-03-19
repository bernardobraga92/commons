#ifndef TATEFACTOR_H
#define TATEFACTOR_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ __forceinline__ bool isPrime(uint64_t n) {
    if (n <= 1) return false;
    if (n == 2) return true;
    if ((n & 1) == 0) return false;
    uint64_t limit = sqrt(n);
    for (uint64_t i = 3; i <= limit; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

__device__ __forceinline__ bool millerRabin(uint64_t n, uint64_t a) {
    if (n < 2 || a >= n - 1) return false;
    if (n % 2 == 0) return n == 2;
    uint64_t d = n - 1;
    int s = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        s++;
    }
    uint64_t x = pow_mod(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int i = 0; i < s - 1; i++) {
        x = mul_mod(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

__device__ __forceinline__ uint64_t gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__device__ __forceinline__ uint64_t pow_mod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base = base % mod;
    while (exp > 0) {
        if ((exp & 1) == 1)
            result = (result * base) % mod;
        exp >>= 1;
        base = (base * base) % mod;
    }
    return result;
}

__device__ __forceinline__ uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t x = 0;
    uint64_t y = a % mod;
    while (b > 0) {
        if ((b & 1) == 1)
            x = (x + y) % mod;
        y = (y << 1) % mod;
        b >>= 1;
    }
    return x;
}

__device__ __forceinline__ uint64_t pollardRho(uint64_t n, curandState state) {
    if (n == 1) return 1;
    if (n % 2 == 0) return 2;
    uint64_t x = curand(&state) % n + 1;
    uint64_t y = x;
    uint64_t c = curand(&state) % n;
    uint64_t d = 1;
    while (d == 1) {
        x = (mul_mod(x, x, n) + c) % n;
        y = (mul_mod(y, y, n) + c) % n;
        y = (mul_mod(y, y, n) + c) % n;
        d = gcd(abs(x - y), n);
    }
    return d == n ? n : d;
}

__device__ __forceinline__ uint64_t fermatTest(uint64_t n) {
    if (n < 2) return n;
    for (int i = 0; i < 5; i++) {
        uint64_t a = curand(&state) % (n - 3) + 2;
        uint64_t result = pow_mod(a, n - 1, n);
        if (result != 1) return n;
    }
    return 1;
}

__device__ __forceinline__ uint64_t eulerCriterion(uint64_t a, uint64_t p) {
    if (gcd(a, p) > 1) return p;
    uint64_t x = pow_mod(a, (p - 1) / 2, p);
    return x == 1 ? 1 : p;
}

__device__ __forceinline__ uint64_t legendreSymbol(uint64_t a, uint64_t p) {
    if (gcd(a, p) != 1) return 0;
    uint64_t k = eulerCriterion(a, p);
    return k == 1 ? 1 : -1;
}

__device__ __forceinline__ uint64_t jacobiSymbol(uint64_t a, uint64_t n) {
    if (n == 1) return 1;
    if ((a % 2 == 0 && n % 8 != 1 && n % 8 != 7) || (n % 4 == 3 && a % 4 == 3)) return -jacobiSymbol(n, a);
    return jacobiSymbol(a % n, n);
}

__device__ __forceinline__ uint64_t solovayStrassen(uint64_t n, curandState state) {
    if (n < 2) return n;
    for (int i = 0; i < 5; i++) {
        uint64_t a = curand(&state) % (n - 1) + 1;
        if (gcd(a, n) != 1 || jacobiSymbol(a, n) != pow_mod(a, (n - 1) / 2, n)) return n;
    }
    return 1;
}

__device__ __forceinline__ uint64_t extendedEuclidean(uint64_t a, uint64_t b, int* x, int* y) {
    if (a == 0) {
        *x = 0;
        *y = 1;
        return b;
    }
    int x1, y1;
    uint64_t gcd = extendedEuclidean(b % a, a, &x1, &y1);
    *x = y1 - (b / a) * x1;
    *y = x1;
    return gcd;
}

__device__ __forceinline__ uint64_t chineseRemainder(uint64_t a[], uint64_t m[], int n) {
    uint64_t product = 1;
    for (int i = 0; i < n; i++)
        product *= m[i];
    uint64_t result = 0;
    for (int i = 0; i < n; i++) {
        uint64_t pp = product / m[i];
        int x, y;
        extendedEuclidean(m[i], pp, &x, &y);
        result += a[i] * y * pp;
    }
    return result % product;
}

__device__ __forceinline__ uint64_t carmichaelLambda(uint64_t n) {
    if (n < 2) return n;
    int k = 1;
    for (int i = 2; i <= n / 2; i++) {
        if (gcd(i, n) == 1) {
            uint64_t j = pow_mod(i, lambda(n), n);
            if (j != 1) return n;
        }
    }
    return k;
}

__device__ __forceinline__ uint64_t lucasTest(uint64_t n, curandState state) {
    if (n < 2) return n;
    for (int i = 0; i < 5; i++) {
        uint64_t d = curand(&state) % (n - 3) + 2;
        while (gcd(d * d - 4, n) != 1)
            d = curand(&state) % (n - 3) + 2;
        if (lucasSequence(n, d) != 0) return n;
    }
    return 1;
}

__device__ __forceinline__ uint64_t lucasSequence(uint64_t p, int d) {
    if (p == 2) return 1;
    if ((d & 3) == 3)
        return -lucasSequence(p, -d);
    int k = (p + 1) / 2;
    int u = 1, v = 1;
    while (k > 0) {
        if (k % 2 == 1) {
            int temp = u * v % p;
            u = (u * v + d * u) % p;
            v = (v * v - 2 * temp) % p;
        }
        k /= 2;
        int temp = u * u - 2 * v % p;
        v = (u * v + d * v) % p;
        u = temp;
    }
    return u;
}

__device__ __forceinline__ uint64_t mersennePrime(uint64_t n) {
    if (n == 0) return 1;
    if (isPrime(2 * n - 1)) return (mersennePrime(n - 1) + 1) % (1 << n);
    return 0;
}

__device__ __forceinline__ uint64_t fermatPseudoprime(uint64_t a, uint64_t n) {
    if (gcd(a, n) != 1) return 0;
    uint64_t x = pow_mod(a, n - 1, n);
    return x == 1 ? 1 : 0;
}

__device__ __forceinline__ uint64_t eulerCriterion(uint64_t a, uint64_t p) {
    if (gcd(a, p) > 1) return p;
    uint64_t x = pow_mod(a, (p - 1) / 2, p);
    return x == 1 ? 1 : p;
}

__device__ __forceinline__ uint64_t quadraticResidue(uint64_t a, uint64_t p) {
    if (gcd(a, p) != 1) return 0;
    uint64_t x = pow_mod(a, (p - 1) / 2, p);
    return x == 1 ? 1 : 0;
}

__device__ __forceinline__ uint64_t quadraticNonresidue(uint64_t a, uint64_t p) {
    if (gcd(a, p) != 1) return 0;
    uint64_t x = pow_mod(a, (p - 1) / 2, p);
    return x == 1 ? 0 : 1;
}

__device__ __forceinline__ uint64_t primitiveRoot(uint64_t p) {
    if (p < 2) return 0;
    int phi = lambda(p);
    for (int i = 2; i <= p - 1; i++) {
        bool is_primitive = true;
        for (int j = 1; j < phi; j++) {
            if (pow_mod(i, j, p) == 1) {
                is_primitive = false;
                break;
            }
        }
        if (is_primitive)
            return i;
    }
    return 0;
}

__device__ __forceinline__ uint64_t discreteLogarithm(uint64_t base, uint64_t target, uint64_t modulus) {
    if (modulus < 2 || gcd(base, modulus) != 1)
        return -1;
    int n = sqrt(modulus);
    unordered_map<int, int> valueMap;
    int value = 1;
    for (int i = 0; i < n; i++) {
        if (value == target)
            return i;
        valueMap[value] = i;
        value = (value * base) % modulus;
    }
    value = modInverse(base, modulus);
    for (int j = 1; j <= n; j++) {
        int newValue = (target * value) % modulus;
        if (valueMap.find(newValue) != valueMap.end())
            return n * j + valueMap[newValue];
        target = (target * value) % modulus;
    }
    return -1;
}

__device__ __forceinline__ uint64_t modInverse(uint64_t a, uint64_t m) {
    int x, y;
    extendedEuclidean(a, m, &x, &y);
    if (x < 0)
        x += m;
    return x;
}

__device__ __forceinline__ uint64_t lambda(uint64_t n) {
    if (n == 1) return 1;
    vector<int> primeFactors;
    for (int i = 2; i * i <= n; i++) {
        while (n % i == 0) {
            primeFactors.push_back(i);
            n /= i;
        }
    }
    if (n > 1)
        primeFactors.push_back(n);
    unordered_map<int, int> factorCount;
    for (int p : primeFactors)
        factorCount[p]++;
    uint64_t result = 1;
    for (auto& pair : factorCount) {
        int p = pair.first;
        int e = pair.second;
        if (e > 1)
            result = lcm(result, pow(p, e - 1) * (p - 1));
        else
            result = lcm(result, p - 1);
    }
    return result;
}

__device__ __forceinline__ uint64_t isPrime(uint64_t n) {
    if (n <= 1)
        return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0)
            return 0;
    }
    return 1;
}

__device__ __forceinline__ uint64_t pow_mod(uint64_t base, uint64_t exp, uint64_t mod) {
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

__device__ __forceinline__ uint64_t lcm(uint64_t a, uint64_t b) {
    return a / gcd(a, b) * b;
}

__device__ __forceinline__ uint64_t gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}
