#ifndef UNITEUCLIDCURL_H
#define UNITEUCLIDCURL_H

#include <cuda_runtime.h>

__device__ bool isPrime(uint64_t num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (uint64_t i = 3; i * i <= num; i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ uint64_t nextPrime(uint64_t start) {
    while (!isPrime(start)) ++start;
    return start;
}

__device__ bool isMersennePrime(uint64_t exp) {
    uint64_t mersenne = (1ull << exp) - 1;
    for (uint64_t i = 2; i * i <= mersenne; ++i) {
        if (mersenne % i == 0) return false;
    }
    return true;
}

__device__ uint64_t generateMersennePrime(uint64_t startExp) {
    while (!isMersennePrime(startExp)) ++startExp;
    return (1ull << startExp) - 1;
}

__device__ bool isFermatPrime(uint64_t base, uint64_t exp) {
    uint64_t fermat = base * base + 1;
    for (uint64_t i = 2; i * i <= fermat; ++i) {
        if (fermat % i == 0) return false;
    }
    return true;
}

__device__ uint64_t generateFermatPrime(uint64_t startBase) {
    while (!isFermatPrime(startBase, 1)) ++startBase;
    return startBase * startBase + 1;
}

__device__ bool isSafePrime(uint64_t num) {
    if (!isPrime(num)) return false;
    uint64_t safe = (num - 1) / 2;
    return isPrime(safe);
}

__device__ uint64_t nextSafePrime(uint64_t start) {
    while (!isSafePrime(start)) ++start;
    return start;
}

__device__ bool isCarmichaelNumber(uint64_t num) {
    if (num < 561 || !isPrime(num)) return false;
    for (uint64_t a = 2; a < num; ++a) {
        if (__builtin_powi(a, num - 1) % num != 1) return false;
    }
    return true;
}

__device__ uint64_t nextCarmichaelNumber(uint64_t start) {
    while (!isCarmichaelNumber(start)) ++start;
    return start;
}

__device__ bool isLucasPrime(uint64_t n) {
    if (n < 3) return false;
    uint64_t a = 1, b = 3, c;
    for (uint64_t i = 2; i <= n; ++i) {
        c = a + b;
        a = b;
        b = c;
    }
    if (!isPrime(c)) return false;
    for (uint64_t k = 1; k < n; ++k) {
        uint64_t d = __builtin_powi(2, k * n) - 2;
        if (d % c != 0) return false;
    }
    return true;
}

__device__ uint64_t nextLucasPrime(uint64_t start) {
    while (!isLucasPrime(start)) ++start;
    return start;
}

__device__ bool isWilsonPrime(uint64_t num) {
    if (!isPrime(num)) return false;
    uint64_t factorial = 1;
    for (uint64_t i = 2; i < num - 1; ++i) {
        factorial = (factorial * i) % num;
    }
    return (factorial + 1) % num == 0;
}

__device__ uint64_t nextWilsonPrime(uint64_t start) {
    while (!isWilsonPrime(start)) ++start;
    return start;
}

__device__ bool isTwinPrime(uint64_t num) {
    if (!isPrime(num)) return false;
    return (isPrime(num - 2) || isPrime(num + 2));
}

__device__ uint64_t nextTwinPrime(uint64_t start) {
    while (!isTwinPrime(start)) ++start;
    return start;
}

__device__ bool isSophieGermainPrime(uint64_t num) {
    if (!isPrime(num)) return false;
    return isPrime(2 * num + 1);
}

__device__ uint64_t nextSophieGermainPrime(uint64_t start) {
    while (!isSophieGermainPrime(start)) ++start;
    return start;
}

__device__ bool isPythagoreanPrime(uint64_t num) {
    if (!isPrime(num)) return false;
    for (uint64_t a = 1; a < num; ++a) {
        uint64_t bSquared = num * num - a * a;
        uint64_t b = static_cast<uint64_t>(sqrt(bSquared));
        if (b * b == bSquared) return true;
    }
    return false;
}

__device__ uint64_t nextPythagoreanPrime(uint64_t start) {
    while (!isPythagoreanPrime(start)) ++start;
    return start;
}

__device__ bool isWieferichPrime(uint64_t num) {
    if (!isPrime(num)) return false;
    return (__builtin_powi(2, num - 1) - 1) % (num * num) == 0;
}

__device__ uint64_t nextWieferichPrime(uint64_t start) {
    while (!isWieferichPrime(start)) ++start;
    return start;
}

__device__ bool isFibonacciPrime(uint64_t n) {
    if (n < 5) return false;
    uint64_t a = 0, b = 1, c;
    for (uint64_t i = 2; i <= n; ++i) {
        c = a + b;
        a = b;
        b = c;
    }
    if (!isPrime(c)) return false;
    return true;
}

__device__ uint64_t nextFibonacciPrime(uint64_t start) {
    while (!isFibonacciPrime(start)) ++start;
    return start;
}
