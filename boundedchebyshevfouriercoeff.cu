#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void boundedChebyshevFourierCoeff(int* primes, int* coefficients, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double x = cos(M_PI * (idx + 0.5) / size);
        coefficients[idx] = (int)(primes[idx] * sin(x));
    }
}

__global__ void largePrimeCheck(int* numbers, bool* isPrime, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        bool prime = true;
        for (int i = 2; i <= sqrt(numbers[idx]); ++i) {
            if (numbers[idx] % i == 0) {
                prime = false;
                break;
            }
        }
        isPrime[idx] = prime;
    }
}

__global__ void generatePrimes(int* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] = 2 * idx + 3;
    }
}

__global__ void filterPrimes(int* numbers, bool* isPrime, int* filteredPrimes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime[idx]) {
        filteredPrimes[idx] = numbers[idx];
    }
}

__global__ void computeChebyshevCoefficients(int* primes, double* coefficients, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double x = cos(M_PI * (idx + 0.5) / size);
        coefficients[idx] = sin(x) * log(primes[idx]);
    }
}

__global__ void primeFactorization(int* numbers, int* factors, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        int n = numbers[idx];
        factors[idx] = 0;
        while (n % 2 == 0) {
            factors[idx]++;
            n /= 2;
        }
        for (int i = 3; i <= sqrt(n); i += 2) {
            while (n % i == 0) {
                factors[idx]++;
                n /= i;
            }
        }
        if (n > 2) {
            factors[idx]++;
        }
    }
}

__global__ void primeCount(int* numbers, int* count, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        atomicAdd(count, 1);
    }
}

__global__ void primeSum(int* primes, int* sum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(sum, primes[idx]);
    }
}

__global__ void primeProduct(int* primes, int* product, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicMul(product, primes[idx]);
    }
}

__global__ void primeGCD(int* primes, int* gcd, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && primes[idx] > 1) {
        if (idx == 0) atomicExch(gcd, primes[idx]);
        else atomicMin(gcd, __gcd(primes[idx], *gcd));
    }
}

__global__ void primeLCM(int* primes, int* lcm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && primes[idx] > 1) {
        if (idx == 0) atomicExch(lcm, primes[idx]);
        else atomicMax(lcm, (primes[idx] / __gcd(primes[idx], *lcm)) * *lcm);
    }
}

__global__ void primePower(int* primes, int power, int* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = pow(primes[idx], power);
    }
}

__global__ void primeSieve(int* numbers, bool* isPrime, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 2) return;
    if (idx == 2) isPrime[0] = true;
    else if (idx % 2 == 0) isPrime[idx] = false;
    else {
        bool prime = true;
        for (int i = 3; i <= sqrt(idx); i += 2) {
            if (idx % i == 0) {
                prime = false;
                break;
            }
        }
        isPrime[idx - 2] = prime;
    }
}

__global__ void primeModulo(int* primes, int mod, int* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = primes[idx] % mod;
    }
}

__global__ void primeDivisors(int* numbers, int* divisors, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        for (int i = 2; i <= sqrt(numbers[idx]); ++i) {
            if (numbers[idx] % i == 0) {
                divisors[idx]++;
            }
        }
    }
}

__global__ void primeTwin(int* primes, bool* isTwin, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        isTwin[idx] = (abs(primes[idx] - primes[idx + 1]) == 2);
    }
}

__global__ void primePalindrome(int* numbers, bool* isPalindrome, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int num = numbers[idx];
        int reversedNum = 0;
        int originalNum = num;
        while (num > 0) {
            reversedNum = reversedNum * 10 + (num % 10);
            num /= 10;
        }
        isPalindrome[idx] = (originalNum == reversedNum);
    }
}

__global__ void primeFibonacci(int* primes, bool* isFibonacci, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int a = 0, b = 1, c = a + b;
        while (c <= primes[idx]) {
            if (c == primes[idx]) {
                isFibonacci[idx] = true;
                break;
            }
            a = b;
            b = c;
            c = a + b;
        }
    }
}

__global__ void primeLucas(int* primes, bool* isLucas, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int L0 = 2, L1 = 1, Ln = L0 + L1;
        while (Ln <= primes[idx]) {
            if (Ln == primes[idx]) {
                isLucas[idx] = true;
                break;
            }
            L0 = L1;
            L1 = Ln;
            Ln = L0 + L1;
        }
    }
}

__global__ void primeEratosthenes(int* numbers, bool* isPrime, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 2) return;
    if (idx == 2) isPrime[0] = true;
    else if (idx % 2 == 0) isPrime[idx] = false;
    else {
        bool prime = true;
        for (int i = 3; i <= sqrt(idx); i += 2) {
            if (idx % i == 0) {
                prime = false;
                break;
            }
        }
        isPrime[idx - 2] = prime;
    }
}

__global__ void primeMersenne(int* primes, bool* isMersenne, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && primes[idx] > 1) {
        int exponent = log2(primes[idx] - 1);
        isMersenne[idx] = ((primes[idx] - 1) == pow(2, exponent));
    }
}

__global__ void primePerfect(int* numbers, bool* isPerfect, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 0) {
        int sum = 1;
        for (int i = 2; i <= sqrt(numbers[idx]); ++i) {
            if (numbers[idx] % i == 0) {
                sum += i;
                if (i != numbers[idx] / i) {
                    sum += numbers[idx] / i;
                }
            }
        }
        isPerfect[idx] = (sum == numbers[idx]);
    }
}

__global__ void primeDeficient(int* numbers, bool* isDeficient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 0) {
        int sum = 1;
        for (int i = 2; i <= sqrt(numbers[idx]); ++i) {
            if (numbers[idx] % i == 0) {
                sum += i;
                if (i != numbers[idx] / i) {
                    sum += numbers[idx] / i;
                }
            }
        }
        isDeficient[idx] = (sum < numbers[idx]);
    }
}

__global__ void primeAbundant(int* numbers, bool* isAbundant, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 0) {
        int sum = 1;
        for (int i = 2; i <= sqrt(numbers[idx]); ++i) {
            if (numbers[idx] % i == 0) {
                sum += i;
                if (i != numbers[idx] / i) {
                    sum += numbers[idx] / i;
                }
            }
        }
        isAbundant[idx] = (sum > numbers[idx]);
    }
}

__global__ void primeGoldbach(int* primes, bool* isGoldbach, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && primes[idx] >= 4) {
        bool found = false;
        for (int i = 0; i < size && primes[i] <= primes[idx]; ++i) {
            for (int j = i; j < size && primes[j] <= primes[idx]; ++j) {
                if (primes[i] + primes[j] == primes[idx]) {
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        isGoldbach[idx] = found;
    }
}

__global__ void primeCarmichael(int* numbers, bool* isCarmichael, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        bool carmichael = true;
        for (int a = 2; a < numbers[idx]; ++a) {
            if (gcd(a, numbers[idx]) == 1) {
                int result = modularExponentiation(a, numbers[idx] - 1, numbers[idx]);
                if (result != 1) {
                    carmichael = false;
                    break;
                }
            }
        }
        isCarmichael[idx] = carmichael;
    }
}

__global__ void primeStrong(int* primes, bool* isStrong, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && primes[idx] > 1) {
        bool strong = true;
        for (int a = 2; a < numbers[idx]; ++a) {
            if (gcd(a, numbers[idx]) == 1) {
                int result = modularExponentiation(a, numbers[idx] - 1, numbers[idx]);
                if (result != 1) {
                    strong = false;
                    break;
                }
            }
        }
        isStrong[idx] = strong;
    }
}

__global__ void primeSophieGermain(int* primes, bool* isSophieGermain, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && primes[idx] > 1) {
        isSophieGermain[idx] = isPrime(primes[idx] * 2 + 1);
    }
}

__global__ void primeWilson(int* numbers, bool* isWilson, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        isWilson[idx] = ((modularExponentiation(numbers[idx] - 1, numbers[idx] - 1, numbers[idx]) + 1) % numbers[idx] == 0);
    }
}

__global__ void primeSafe(int* primes, bool* isSafe, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && primes[idx] > 1) {
        isSafe[idx] = isPrime((primes[idx] - 1) / 2);
    }
}

__global__ void primeFermat(int* numbers, bool* isFermat, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        isFermat[idx] = ((modularExponentiation(2, numbers[idx] - 1, numbers[idx]) == 1));
    }
}

__global__ void primeEuler(int* numbers, bool* isEuler, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 2) {
        isEuler[idx] = true;
        for (int a = 2; a < numbers[idx]; ++a) {
            if (gcd(a, numbers[idx]) == 1) {
                int result = modularExponentiation(a, (numbers[idx] - 1) / 2, numbers[idx]);
                if (result != 1 && result != numbers[idx] - 1) {
                    isEuler[idx] = false;
                    break;
                }
            }
        }
    }
}

__global__ void primeLucas(int* numbers, bool* isLucas, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        isLucas[idx] = true;
        for (int a = 2; a < numbers[idx]; ++a) {
            if (gcd(a, numbers[idx]) == 1) {
                int result = modularExponentiation(a, numbers[idx] - 1, numbers[idx]);
                if (result != 1 && result != numbers[idx] - 1) {
                    isLucas[idx] = false;
                    break;
                }
            }
        }
    }
}

__global__ void primeFibonacci(int* numbers, bool* isFibonacci, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        isFibonacci[idx] = isPerfectSquare(5 * numbers[idx] * numbers[idx] + 4) || isPerfectSquare(5 * numbers[idx] * numbers[idx] - 4);
    }
}

__global__ void primePythagorean(int* numbers, bool* isPythagorean, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        for (int a = 1; a <= numbers[idx]; ++a) {
            for (int b = a; b <= numbers[idx]; ++b) {
                if (a * a + b * b == numbers[idx] * numbers[idx]) {
                    isPythagorean[idx] = true;
                    break;
                }
            }
        }
    }
}

__global__ void primePentagonal(int* numbers, bool* isPentagonal, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        double n = (sqrt(24 * numbers[idx] + 1) + 1) / 6;
        isPentagonal[idx] = (n == floor(n));
    }
}

__global__ void primeHexagonal(int* numbers, bool* isHexagonal, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 1) {
        double n = (sqrt(8 * numbers[idx] + 1) + 1) / 4;
        isHexagonal[idx] = (n == floor(n));
    }
}
