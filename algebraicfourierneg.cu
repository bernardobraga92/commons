#include <cuda_runtime.h>
#include <cmath>

__device__ int isPrime(int num) {
    if (num <= 1) return 0;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return 0;
    }
    return 1;
}

__global__ void generatePrimes(int* primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        primes[idx] = isPrime(idx + 2);
    }
}

__device__ unsigned long long fastExp(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    unsigned long long result = 1;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return result;
}

__global__ void checkFermatPrimality(int* results, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = idx + 2;
        if (num == 2 || num == 3)
            results[idx] = 1;
        else if (num % 2 == 0)
            results[idx] = 0;
        else
            results[idx] = fastExp(2, num - 1, num) == 1;
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

__global__ void checkMillerRabinPrimality(int* results, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = idx + 2;
        if (num == 2 || num == 3)
            results[idx] = 1;
        else if (num % 2 == 0)
            results[idx] = 0;
        else {
            int d = num - 1;
            int s = 0;
            while ((d & 1) == 0) {
                d >>= 1;
                ++s;
            }
            for (int i = 0; i < 5; ++i) {
                unsigned long long a = rand() % (num - 3) + 2;
                if (gcd(a, num) != 1)
                    continue;
                unsigned long long x = fastExp(a, d, num);
                if (x == 1 || x == num - 1)
                    continue;
                for (int j = 0; j < s - 1; ++j) {
                    x = fastExp(x, 2, num);
                    if (x == num - 1)
                        break;
                }
                if (x != num - 1)
                    results[idx] = 0;
            }
        }
    }
}

__device__ int isSophieGermainPrime(int num) {
    return isPrime(num) && isPrime(2 * num + 1);
}

__global__ void findSophieGermainPrimes(int* primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        primes[idx] = isSophieGermainPrime(idx + 2);
    }
}

__device__ unsigned long long modInverse(unsigned long long a, unsigned long long m) {
    for (unsigned long long x = 1; x < m; ++x)
        if ((a * x) % m == 1)
            return x;
    return -1;
}

__global__ void checkWilsonTheorem(int* results, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = idx + 2;
        results[idx] = (fastExp(num - 1, num - 2, num) == (num - 1)) && isPrime(num);
    }
}

__device__ unsigned long long modPow(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    unsigned long long result = 1;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return result;
}

__global__ void checkEulerCriterion(int* results, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int num = idx + 2;
        if (num == 2)
            results[idx] = 1;
        else if (num % 4 != 3 && isPrime(num))
            results[idx] = modPow(2, (num - 1) / 2, num) == 1;
        else
            results[idx] = 0;
    }
}

__device__ int findTwinPrimes(int* primes, int limit) {
    for (int i = 0; i < limit - 1; ++i) {
        if (isPrime(i + 2) && isPrime(i + 3))
            return i;
    }
    return -1;
}

__global__ void findCousinPrimes(int* primes, int limit) {
    for (int i = 0; i < limit - 4; ++i) {
        if (isPrime(i + 2) && isPrime(i + 5))
            return i;
    }
    return -1;
}

__device__ int findSexyPrimes(int* primes, int limit) {
    for (int i = 0; i < limit - 6; ++i) {
        if (isPrime(i + 2) && isPrime(i + 7))
            return i;
    }
    return -1;
}

__device__ unsigned long long findFermatPseudoprime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = i + 2;
        if (fastExp(2, num - 1, num) == 1 && !isPrime(num))
            return num;
    }
    return -1;
}

__global__ void findCarmichaelNumbers(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = i + 2;
        if (num % 2 == 1 && isPrime(num)) {
            bool carmichael = true;
            for (int b = 2; b < num - 1; ++b) {
                if (gcd(b, num) == 1 && fastExp(b, num - 1, num) != 1) {
                    carmichael = false;
                    break;
                }
            }
            if (carmichael)
                return i;
        }
    }
    return -1;
}

__device__ unsigned long long findSemiPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        for (int j = i + 1; j < limit; ++j) {
            if (isPrime(i + 2) && isPrime(j + 2))
                return (i + 2) * (j + 2);
        }
    }
    return -1;
}

__device__ int findPerfectPower(int* results, int limit) {
    for (int a = 2; a <= sqrt(limit); ++a) {
        for (int b = 2; b < log2(limit); ++b) {
            int num = pow(a, b);
            if (num <= limit && isPrime(num))
                return num;
        }
    }
    return -1;
}

__device__ unsigned long long findStrongPseudoprime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = i + 2;
        if (!isPrime(num)) {
            bool strong = true;
            for (int a = 2; a < num - 1; ++a) {
                unsigned long long d = num - 1;
                int s = 0;
                while ((d & 1) == 0) {
                    d >>= 1;
                    ++s;
                }
                unsigned long long x = fastExp(a, d, num);
                if (x == 1 || x == num - 1)
                    continue;
                for (int j = 0; j < s - 1; ++j) {
                    x = fastExp(x, 2, num);
                    if (x == num - 1)
                        break;
                }
                if (x != num - 1) {
                    strong = false;
                    break;
                }
            }
            if (strong)
                return num;
        }
    }
    return -1;
}

__global__ void findLucasPseudoprime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = i + 2;
        if (!isPrime(num)) {
            int p = 3;
            int q = -1;
            unsigned long long D = p * p - 4 * q;
            unsigned long long n = (modInverse(2, num) * (p + sqrt(D))) % num;
            for (int j = 0; j < log2(num); ++j) {
                if (n == 1)
                    return num;
                n = (n * n - 2) % num;
            }
        }
    }
    return -1;
}

__device__ int findWieferichPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = i + 2;
        if (isPrime(num)) {
            unsigned long long powResult = modPow(2, num - 1, num * num);
            if (powResult == 1)
                return num;
        }
    }
    return -1;
}

__device__ unsigned long long findSafePrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = i + 2;
        if (isPrime(num) && isPrime((num - 1) / 2))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSophieGermainPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = i + 2;
        if (isPrime(num) && isPrime(2 * num + 1))
            return num;
    }
    return -1;
}

__device__ unsigned long long findMersennePrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = pow(2, i + 2) - 1;
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findFibonacciPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = fibonacci(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findLucasPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = lucas(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findMersenneNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = pow(2, i + 2) - 1;
        return num;
    }
    return -1;
}

__device__ unsigned long long findEulerPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = euler(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findNivenPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = niven(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findHarshadPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = harshad(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSelfDivisiblePrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = selfDivisible(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findPalindromicPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = palindromic(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findEmirp(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = emirp(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findCircularPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = circular(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findPermutablePrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = permutable(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findTruncatablePrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = truncatable(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findLeftTruncatablePrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = leftTruncatable(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findRightTruncatablePrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = rightTruncatable(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findTitanicPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = titanic(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findColossallyAbundantPrime(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = colossallyAbundant(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findHyperPerfectNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = hyperPerfect(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findHarmonicNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = harmonic(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findAmicableNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = amicable(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findPerfectNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = perfect(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findAbundantNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = abundant(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findDeficientNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = deficient(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findAlmostPerfectNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = almostPerfect(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findQuasiperfectNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = quasiperfect(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSemiPerfectNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = semiPerfect(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSublimeNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = sublime(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findPerfectDigitSumNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = perfectDigitSum(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSelfDescriptiveNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = selfDescriptive(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findHarshadNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = harshad(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findNivenNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = niven(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findStrongHarshadNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = strongHarshad(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findStrongNivenNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = strongNiven(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandachePrimordialNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandachePrimordial(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandachePrimorialNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandachePrimorial(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSequenceNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSequence(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandachePalindromeNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandachePalindrome(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheOddNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheOdd(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEvenNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEven(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSquareNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSquare(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheCubeNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheCube(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFourthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFourthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFifthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFifthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventhPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventhPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEighthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEighthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheNinthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheNinthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTenthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTenthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEleventhPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEleventhPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTwelfthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTwelfthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheThirteenthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheThirteenthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFourteenthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFourteenthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFifteenthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFifteenthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixteenthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixteenthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventeenthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventeenthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEighteenthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEighteenthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheNineteenthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheNineteenthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTwentiethPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTwentiethPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTwentyFirstPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTwentyFirstPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTwentySecondPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTwentySecondPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTwentyThirdPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTwentyThirdPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTwentyFourthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTwentyFourthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTwentyFifthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTwentyFifthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTwentySixthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTwentySixthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTwentySeventhPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTwentySeventhPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTwentyEighthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTwentyEighthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheTwentyNinthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheTwentyNinthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheThirtiethPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheThirtiethPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheThirtyFirstPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheThirtyFirstPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheThirtySecondPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheThirtySecondPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheThirtyThirdPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheThirtyThirdPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheThirtyFourthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheThirtyFourthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheThirtyFifthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheThirtyFifthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheThirtySixthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheThirtySixthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheThirtySeventhPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheThirtySeventhPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheThirtyEighthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheThirtyEighthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheThirtyNinthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheThirtyNinthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFortiethPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFortiethPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFortyFirstPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFortyFirstPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFortySecondPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFortySecondPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFortyThirdPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFortyThirdPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFortyFourthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFortyFourthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFortyFifthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFortyFifthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFortySixthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFortySixthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFortySeventhPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFortySeventhPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFortyEighthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFortyEighthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFortyNinthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFortyNinthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFiftiethPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFiftiethPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFiftyFirstPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFiftyFirstPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFiftySecondPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFiftySecondPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFiftyThirdPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFiftyThirdPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFiftyFourthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFiftyFourthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFiftyFifthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFiftyFifthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFiftySixthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFiftySixthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFiftySeventhPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFiftySeventhPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFiftyEighthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFiftyEighthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheFiftyNinthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheFiftyNinthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixtiethPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixtiethPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixtyFirstPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixtyFirstPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixtySecondPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixtySecondPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixtyThirdPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixtyThirdPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixtyFourthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixtyFourthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixtyFifthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixtyFifthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixtySixthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixtySixthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixtySeventhPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixtySeventhPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixtyEighthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixtyEighthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSixtyNinthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSixtyNinthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventiethPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventiethPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventyFirstPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventyFirstPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventySecondPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventySecondPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventyThirdPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventyThirdPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventyFourthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventyFourthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventyFifthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventyFifthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventySixthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventySixthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventySeventhPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventySeventhPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventyEighthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventyEighthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheSeventyNinthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheSeventyNinthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEightiethPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEightiethPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEightyFirstPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEightyFirstPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEightySecondPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEightySecondPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEightyThirdPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEightyThirdPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEightyFourthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEightyFourthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEightyFifthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEightyFifthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEightySixthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEightySixthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEightySeventhPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEightySeventhPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEightyEighthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEightyEighthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheEightyNinthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheEightyNinthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheNinetiethPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheNinetiethPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheNinetyFirstPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheNinetyFirstPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheNinetySecondPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheNinetySecondPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheNinetyThirdPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheNinetyThirdPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheNinetyFourthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheNinetyFourthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheNinetyFifthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheNinetyFifthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}

__device__ unsigned long long findSmarandacheHundredthPowerNumber(int* results, int limit) {
    for (int i = 0; i < limit; ++i) {
        int num = smarandacheHundredthPower(i + 2);
        if (isPrime(num))
            return num;
    }
    return -1;
}
