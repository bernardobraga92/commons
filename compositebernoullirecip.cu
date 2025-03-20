#include <math.h>
#include <curand_kernel.h>

__global__ void bernoulliCoefficientKernel(float *coefficients, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx <= n) {
        float sum = 1.0f;
        for (int k = 1; k < idx; ++k) {
            sum += coefficients[k];
        }
        coefficients[idx] = -sum / idx;
    }
}

__global__ void generatePrimesKernel(unsigned long long *primes, unsigned long long limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && idx < limit) {
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(idx); ++j) {
            if (idx % j == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[idx] = idx;
        } else {
            primes[idx] = 0;
        }
    }
}

__global__ void randomLargePrimeKernel(unsigned long long *largePrimes, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        unsigned long long primeCandidate = curand(state + idx) % 982451653 + 1000000007; // Large prime range
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(primeCandidate); ++j) {
            if (primeCandidate % j == 0) {
                isPrime = false;
                break;
            }
        }
        largePrimes[idx] = isPrime ? primeCandidate : 0;
    }
}

__global__ void checkPrimalityKernel(unsigned long long *numbers, bool *isPrimeArray, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long number = numbers[idx];
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(number); ++j) {
            if (number % j == 0) {
                isPrime = false;
                break;
            }
        }
        isPrimeArray[idx] = isPrime;
    }
}

__global__ void generateFermatNumbersKernel(unsigned long long *fermatNumbers, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fermatNumbers[idx] = (1ULL << (1ULL << idx)) + 1;
    }
}

__global__ void checkFermatPrimalityKernel(unsigned long long *fermatNumbers, bool *isPrimeArray, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long number = fermatNumbers[idx];
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(number); ++j) {
            if (number % j == 0) {
                isPrime = false;
                break;
            }
        }
        isPrimeArray[idx] = isPrime;
    }
}

__global__ void sieveOfErathostenesKernel(bool *isPrime, unsigned long long limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 2) {
        isPrime[idx] = true; // 2 is the only even prime
    } else if (idx > 1 && idx % 2 != 0) {
        bool prime = true;
        for (unsigned long long j = 3; j <= sqrt(idx); j += 2) {
            if (idx % j == 0) {
                prime = false;
                break;
            }
        }
        isPrime[idx] = prime;
    }
}

__global__ void randomMersennePrimesKernel(unsigned long long *mersennePrimes, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        unsigned long long exponent = curand(state + idx) % 61 + 1; // Known Mersenne prime exponents
        unsigned long long candidate = (1ULL << exponent) - 1;
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(candidate); ++j) {
            if (candidate % j == 0) {
                isPrime = false;
                break;
            }
        }
        mersennePrimes[idx] = isPrime ? candidate : 0;
    }
}

__global__ void generateLucasNumbersKernel(unsigned long long *lucasNumbers, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        lucasNumbers[idx] = 2;
    } else if (idx == 1) {
        lucasNumbers[idx] = 1;
    } else if (idx < n) {
        lucasNumbers[idx] = lucasNumbers[idx - 1] + lucasNumbers[idx - 2];
    }
}

__global__ void checkLucasPrimalityKernel(unsigned long long *lucasNumbers, bool *isPrimeArray, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long number = lucasNumbers[idx];
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(number); ++j) {
            if (number % j == 0) {
                isPrime = false;
                break;
            }
        }
        isPrimeArray[idx] = isPrime;
    }
}

__global__ void randomTwinPrimesKernel(unsigned long long *twinPrimes, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        unsigned long long candidate = curand(state + idx) % 982451653 + 1000000007; // Large prime range
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(candidate); ++j) {
            if (candidate % j == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime && ((candidate - 2 > 1 && ((candidate - 2) % 2 != 0 || (candidate - 2) == 3)) || 
                       (candidate + 2 < 982451653 && ((candidate + 2) % 2 != 0 || (candidate + 2) == 3)))) {
            twinPrimes[idx] = candidate;
        } else {
            twinPrimes[idx] = 0;
        }
    }
}

__global__ void checkTwinPrimalityKernel(unsigned long long *twinPrimes, bool *isPrimeArray, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long number = twinPrimes[idx];
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(number); ++j) {
            if (number % j == 0) {
                isPrime = false;
                break;
            }
        }
        isPrimeArray[idx] = isPrime;
    }
}

__global__ void randomSophieGermainPrimesKernel(unsigned long long *sophieGermainPrimes, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        unsigned long long candidate = curand(state + idx) % 982451653 + 1000000007; // Large prime range
        bool isPrime = true, isSophieGermain = false;
        for (unsigned long long j = 2; j <= sqrt(candidate); ++j) {
            if (candidate % j == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            unsigned long long sophieGermainCandidate = 2 * candidate + 1;
            bool isSophieGermainPrime = true;
            for (unsigned long long j = 2; j <= sqrt(sophieGermainCandidate); ++j) {
                if (sophieGermainCandidate % j == 0) {
                    isSophieGermainPrime = false;
                    break;
                }
            }
            sophieGermainPrimes[idx] = isSophieGermainPrime ? candidate : 0;
        } else {
            sophieGermainPrimes[idx] = 0;
        }
    }
}

__global__ void checkSophieGermainPrimalityKernel(unsigned long long *sophieGermainPrimes, bool *isPrimeArray, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long number = sophieGermainPrimes[idx];
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(number); ++j) {
            if (number % j == 0) {
                isPrime = false;
                break;
            }
        }
        isPrimeArray[idx] = isPrime;
    }
}

__global__ void randomCullenPrimesKernel(unsigned long long *cullenPrimes, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        unsigned long long exponent = curand(state + idx) % 61 + 1; // Known Cullen prime exponents
        unsigned long long candidate = exponent * pow(2, exponent) + 1;
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(candidate); ++j) {
            if (candidate % j == 0) {
                isPrime = false;
                break;
            }
        }
        cullenPrimes[idx] = isPrime ? candidate : 0;
    }
}

__global__ void checkCullenPrimalityKernel(unsigned long long *cullenPrimes, bool *isPrimeArray, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long number = cullenPrimes[idx];
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(number); ++j) {
            if (number % j == 0) {
                isPrime = false;
                break;
            }
        }
        isPrimeArray[idx] = isPrime;
    }
}

__global__ void randomWoodallPrimesKernel(unsigned long long *woodallPrimes, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        unsigned long long exponent = curand(state + idx) % 61 + 1; // Known Woodall prime exponents
        unsigned long long candidate = exponent * pow(2, exponent) - 1;
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(candidate); ++j) {
            if (candidate % j == 0) {
                isPrime = false;
                break;
            }
        }
        woodallPrimes[idx] = isPrime ? candidate : 0;
    }
}

__global__ void checkWoodallPrimalityKernel(unsigned long long *woodallPrimes, bool *isPrimeArray, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long number = woodallPrimes[idx];
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(number); ++j) {
            if (number % j == 0) {
                isPrime = false;
                break;
            }
        }
        isPrimeArray[idx] = isPrime;
    }
}

__global__ void randomFibonacciPrimesKernel(unsigned long long *fibonacciPrimes, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        unsigned long long a = 0, b = 1, fib = 0;
        for (int i = 0; i <= idx; ++i) {
            fib = a + b;
            a = b;
            b = fib;
        }
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(fib); ++j) {
            if (fib % j == 0) {
                isPrime = false;
                break;
            }
        }
        fibonacciPrimes[idx] = isPrime ? fib : 0;
    }
}

__global__ void checkFibonacciPrimalityKernel(unsigned long long *fibonacciPrimes, bool *isPrimeArray, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long number = fibonacciPrimes[idx];
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(number); ++j) {
            if (number % j == 0) {
                isPrime = false;
                break;
            }
        }
        isPrimeArray[idx] = isPrime;
    }
}

__global__ void randomPythagoreanPrimesKernel(unsigned long long *pythagoreanPrimes, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        unsigned long long candidate = idx * idx + 1;
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(candidate); ++j) {
            if (candidate % j == 0) {
                isPrime = false;
                break;
            }
        }
        pythagoreanPrimes[idx] = isPrime ? candidate : 0;
    }
}

__global__ void checkPythagoreanPrimalityKernel(unsigned long long *pythagoreanPrimes, bool *isPrimeArray, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long number = pythagoreanPrimes[idx];
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(number); ++j) {
            if (number % j == 0) {
                isPrime = false;
                break;
            }
        }
        isPrimeArray[idx] = isPrime;
    }
}

__global__ void randomMersennePrimesKernel(unsigned long long *mersennePrimes, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        unsigned long long exponent = curand(state + idx) % 61 + 1; // Known Mersenne prime exponents
        unsigned long long candidate = pow(2, exponent) - 1;
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(candidate); ++j) {
            if (candidate % j == 0) {
                isPrime = false;
                break;
            }
        }
        mersennePrimes[idx] = isPrime ? candidate : 0;
    }
}

__global__ void checkMersennePrimalityKernel(unsigned long long *mersennePrimes, bool *isPrimeArray, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long number = mersennePrimes[idx];
        bool isPrime = true;
        for (unsigned long long j = 2; j <= sqrt(number); ++j) {
            if (number % j == 0) {
                isPrime = false;
                break;
            }
        }
        isPrimeArray[idx] = isPrime;
    }
}
