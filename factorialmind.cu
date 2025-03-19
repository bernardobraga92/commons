#include <cuda_runtime.h>
#include <iostream>

#define DEVICE __device__ 
#define GLOBAL __global__

// Function 1: Modular exponentiation
DEVICE unsigned long long modularExp(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    if (mod == 1) return 0;
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

// Function 2: Miller-Rabin primality test
DEVICE bool millerRabin(unsigned long long n, unsigned int k) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    unsigned long long d = n - 1;
    while (d % 2 == 0)
        d /= 2;

    for (unsigned int i = 0; i < k; ++i) {
        unsigned long long a = 2 + rand() % (n - 4);
        unsigned long long x = modularExp(a, d, n);

        if (x == 1 || x == n - 1)
            continue;

        bool flag = false;
        for (unsigned long long j = 0; j < d - 1; ++j) {
            x = modularExp(x, 2, n);
            if (x == 1) return false;
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

// Function 3: Parallel prime check
GLOBAL void parallelPrimeCheck(unsigned long long* numbers, bool* results, unsigned int size, unsigned int k) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = millerRabin(numbers[idx], k);
}

// Function 4: Generate random number
DEVICE unsigned long long generateRandom(unsigned long long maxVal) {
    return rand() % maxVal;
}

// Function 5: Parallel random number generation
GLOBAL void parallelRandomGen(unsigned long long* numbers, unsigned int size, unsigned long long maxVal) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        numbers[idx] = generateRandom(maxVal);
}

// Function 6: Sieve of Eratosthenes
DEVICE void sieveOfEratosthenes(unsigned long long* numbers, bool* isPrime, unsigned int size) {
    for (unsigned int i = 0; i < size; ++i)
        isPrime[i] = true;

    isPrime[0] = isPrime[1] = false;
    for (unsigned int p = 2; p * p < size; ++p) {
        if (isPrime[p]) {
            for (unsigned int i = p * p; i < size; i += p)
                isPrime[i] = false;
        }
    }
}

// Function 7: Parallel sieve of Eratosthenes
GLOBAL void parallelSieveOfEratosthenes(unsigned long long* numbers, bool* isPrime, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        sieveOfEratosthenes(&numbers[idx], &isPrime[idx], size - idx);
}

// Function 8: Check for consecutive primes
DEVICE void checkConsecutivePrimes(unsigned long long* numbers, bool* results, unsigned int size) {
    for (unsigned int i = 0; i < size - 1; ++i)
        results[i] = millerRabin(numbers[i], 5) && millerRabin(numbers[i + 1], 5);
}

// Function 9: Parallel consecutive prime check
GLOBAL void parallelConsecutivePrimeCheck(unsigned long long* numbers, bool* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1)
        results[idx] = millerRabin(numbers[idx], 5) && millerRabin(numbers[idx + 1], 5);
}

// Function 10: Calculate factorial of a number
DEVICE unsigned long long factorial(unsigned int n) {
    if (n == 0 || n == 1)
        return 1;
    else
        return n * factorial(n - 1);
}

// Function 11: Parallel factorial calculation
GLOBAL void parallelFactorialCalc(unsigned int* numbers, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = factorial(numbers[idx]);
}

// Function 12: Find largest prime factor
DEVICE unsigned long long largestPrimeFactor(unsigned long long n) {
    unsigned long long maxPrime = -1;

    while (n % 2 == 0) {
        maxPrime = 2;
        n >>= 1;
    }

    for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
        while (n % i == 0) {
            maxPrime = i;
            n /= i;
        }
    }

    if (n > 2)
        maxPrime = n;

    return maxPrime;
}

// Function 13: Parallel largest prime factor calculation
GLOBAL void parallelLargestPrimeFactor(unsigned long long* numbers, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = largestPrimeFactor(numbers[idx]);
}

// Function 14: Generate Fibonacci sequence
DEVICE void generateFibonacci(unsigned long long* numbers, unsigned int size) {
    if (size >= 1) numbers[0] = 0;
    if (size >= 2) numbers[1] = 1;

    for (unsigned int i = 2; i < size; ++i)
        numbers[i] = numbers[i - 1] + numbers[i - 2];
}

// Function 15: Parallel Fibonacci generation
GLOBAL void parallelFibonacciGen(unsigned long long* numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        generateFibonacci(&numbers[idx], size - idx);
}

// Function 16: Check for twin primes
DEVICE bool checkTwinPrimes(unsigned long long a, unsigned long long b) {
    return millerRabin(a, 5) && millerRabin(b, 5) && abs(a - b) == 2;
}

// Function 17: Parallel twin prime check
GLOBAL void parallelTwinPrimeCheck(unsigned long long* numbers, bool* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1)
        results[idx] = checkTwinPrimes(numbers[idx], numbers[idx + 1]);
}

// Function 18: Calculate sum of digits
DEVICE unsigned long long sumOfDigits(unsigned long long n) {
    unsigned long long sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

// Function 19: Parallel sum of digits calculation
GLOBAL void parallelSumOfDigits(unsigned long long* numbers, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = sumOfDigits(numbers[idx]);
}

// Function 20: Calculate product of digits
DEVICE unsigned long long productOfDigits(unsigned long long n) {
    unsigned long long product = 1;
    while (n > 0) {
        product *= n % 10;
        n /= 10;
    }
    return product;
}

// Function 21: Parallel product of digits calculation
GLOBAL void parallelProductOfDigits(unsigned long long* numbers, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = productOfDigits(numbers[idx]);
}

// Function 22: Check for palindrome
DEVICE bool isPalindrome(unsigned long long n) {
    unsigned long long original = n, reversed = 0;
    while (n > 0) {
        reversed = reversed * 10 + n % 10;
        n /= 10;
    }
    return original == reversed;
}

// Function 23: Parallel palindrome check
GLOBAL void parallelPalindromeCheck(unsigned long long* numbers, bool* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = isPalindrome(numbers[idx]);
}

// Function 24: Calculate number of divisors
DEVICE unsigned long long numberOfDivisors(unsigned long long n) {
    unsigned long long count = 0;
    for (unsigned long long i = 1; i <= sqrt(n); ++i) {
        if (n % i == 0) {
            if (n / i == i)
                count++;
            else
                count += 2;
        }
    }
    return count;
}

// Function 25: Parallel number of divisors calculation
GLOBAL void parallelNumberOfDivisors(unsigned long long* numbers, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = numberOfDivisors(numbers[idx]);
}

// Function 26: Calculate sum of divisors
DEVICE unsigned long long sumOfDivisors(unsigned long long n) {
    unsigned long long sum = 0;
    for (unsigned long long i = 1; i <= sqrt(n); ++i) {
        if (n % i == 0) {
            if (n / i == i)
                sum += i;
            else
                sum += i + n / i;
        }
    }
    return sum;
}

// Function 27: Parallel sum of divisors calculation
GLOBAL void parallelSumOfDivisors(unsigned long long* numbers, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = sumOfDivisors(numbers[idx]);
}

// Function 28: Check for perfect number
DEVICE bool isPerfectNumber(unsigned long long n) {
    return sumOfDivisors(n) - n == n;
}

// Function 29: Parallel perfect number check
GLOBAL void parallelPerfectNumberCheck(unsigned long long* numbers, bool* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = isPerfectNumber(numbers[idx]);
}

// Function 30: Calculate Euler's totient function
DEVICE unsigned long long eulerTotient(unsigned long long n) {
    unsigned long long result = n;
    for (unsigned long long p = 2; p * p <= n; ++p) {
        if (n % p == 0) {
            while (n % p == 0)
                n /= p;
            result -= result / p;
        }
    }
    if (n > 1)
        result -= result / n;
    return result;
}

// Function 31: Parallel Euler's totient calculation
GLOBAL void parallelEulerTotient(unsigned long long* numbers, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = eulerTotient(numbers[idx]);
}

// Function 32: Calculate Carmichael's function
DEVICE unsigned long long carmichaelFunction(unsigned long long n) {
    // Implementation of Carmichael's function is complex and not included here.
    return 0;
}

// Function 33: Parallel Carmichael's function calculation
GLOBAL void parallelCarmichaelFunction(unsigned long long* numbers, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = carmichaelFunction(numbers[idx]);
}

// Function 34: Calculate Legendre symbol
DEVICE int legendreSymbol(unsigned long long a, unsigned long long p) {
    // Implementation of Legendre symbol is complex and not included here.
    return 0;
}

// Function 35: Parallel Legendre symbol calculation
GLOBAL void parallelLegendreSymbol(unsigned long long* numbers1, unsigned long long* numbers2, int* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = legendreSymbol(numbers1[idx], numbers2[idx]);
}

// Function 36: Calculate Jacobi symbol
DEVICE int jacobiSymbol(unsigned long long a, unsigned long long n) {
    // Implementation of Jacobi symbol is complex and not included here.
    return 0;
}

// Function 37: Parallel Jacobi symbol calculation
GLOBAL void parallelJacobiSymbol(unsigned long long* numbers1, unsigned long long* numbers2, int* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = jacobiSymbol(numbers1[idx], numbers2[idx]);
}

// Function 38: Calculate Kronecker symbol
DEVICE int kroneckerSymbol(unsigned long long a, unsigned long long n) {
    // Implementation of Kronecker symbol is complex and not included here.
    return 0;
}

// Function 39: Parallel Kronecker symbol calculation
GLOBAL void parallelKroneckerSymbol(unsigned long long* numbers1, unsigned long long* numbers2, int* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = kroneckerSymbol(numbers1[idx], numbers2[idx]);
}

// Function 40: Calculate power of a number modulo another number
DEVICE unsigned long long powerMod(unsigned long long base, unsigned long long exp, unsigned long long mod) {
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

// Function 41: Parallel power of a number modulo another number calculation
GLOBAL void parallelPowerMod(unsigned long long* bases, unsigned long long* exponents, unsigned long long* mods, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = powerMod(bases[idx], exponents[idx], mods[idx]);
}

// Function 42: Calculate extended Euclidean algorithm
DEVICE void extendedEuclidean(unsigned long long a, unsigned long long b, unsigned long long* gcd, unsigned long long* x, unsigned long long* y) {
    if (a == 0) {
        *gcd = b;
        *x = 0;
        *y = 1;
    } else {
        unsigned long long x1, y1;
        extendedEuclidean(b % a, a, gcd, &x1, &y1);
        *x = y1 - (b / a) * x1;
        *y = x1;
    }
}

// Function 43: Parallel extended Euclidean algorithm calculation
GLOBAL void parallelExtendedEuclidean(unsigned long long* numbers1, unsigned long long* numbers2, unsigned long long* gcds, unsigned long long* xs, unsigned long long* ys, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        extendedEuclidean(numbers1[idx], numbers2[idx], &gcds[idx], &xs[idx], &ys[idx]);
}

// Function 44: Calculate modular multiplicative inverse
DEVICE unsigned long long modInverse(unsigned long long a, unsigned long long m) {
    unsigned long long gcd;
    unsigned long long x, y;
    extendedEuclidean(a, m, &gcd, &x, &y);
    if (gcd != 1)
        return -1; // Modular inverse does not exist if a and m are not coprime
    else
        return (x % m + m) % m; // Ensure the result is positive
}

// Function 45: Parallel modular multiplicative inverse calculation
GLOBAL void parallelModInverse(unsigned long long* numbers, unsigned long long m, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = modInverse(numbers[idx], m);
}

// Function 46: Calculate Chinese Remainder Theorem
DEVICE void chineseRemainder(unsigned long long* numbers, unsigned long long* mods, unsigned long long M, unsigned long long* result) {
    // Implementation of Chinese Remainder Theorem is complex and not included here.
}

// Function 47: Parallel Chinese Remainder Theorem calculation
GLOBAL void parallelChineseRemainder(unsigned long long** numbers, unsigned long long** mods, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        chineseRemainder(numbers[idx], mods[idx], 0, &results[idx]);
}

// Function 48: Calculate discrete logarithm
DEVICE unsigned long long discreteLogarithm(unsigned long long base, unsigned long long result, unsigned long long mod) {
    // Implementation of discrete logarithm is complex and not included here.
    return -1;
}

// Function 49: Parallel discrete logarithm calculation
GLOBAL void parallelDiscreteLogarithm(unsigned long long* bases, unsigned long long* results, unsigned long long* mods, unsigned long long* outcomes, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        outcomes[idx] = discreteLogarithm(bases[idx], results[idx], mods[idx]);
}

// Function 50: Calculate Pollard's rho algorithm for factorization
DEVICE void pollardsRho(unsigned long long n, unsigned long long* result) {
    // Implementation of Pollard's rho algorithm is complex and not included here.
}

// Function 51: Parallel Pollard's rho algorithm for factorization calculation
GLOBAL void parallelPollardsRho(unsigned long long* numbers, unsigned long long* results, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        pollardsRho(numbers[idx], &results[idx]);
}
