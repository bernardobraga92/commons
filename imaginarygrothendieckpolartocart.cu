#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void findPrimeCandidates(unsigned long *candidates, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) candidates[idx] = idx * 2 + 1;
}

__global__ void checkPrimality(unsigned long *candidates, bool *isPrime, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && candidates[idx] > 1) {
        isPrime[idx] = true;
        for (unsigned long i = 2; i * i <= candidates[idx]; ++i) {
            if (candidates[idx] % i == 0) {
                isPrime[idx] = false;
                break;
            }
        }
    }
}

__global__ void sieveOfEratosthenes(unsigned int *isComposite, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && idx > 1) {
        for (unsigned long i = idx * idx; i < limit; i += idx) {
            isComposite[i] = 1;
        }
    }
}

__global__ void fermatPrimalityTest(unsigned long *candidates, bool *isPrime, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && candidates[idx] > 3) {
        unsigned long a = 2;
        unsigned long result = modPow(a, candidates[idx] - 1, candidates[idx]);
        isPrime[idx] = (result == 1);
    }
}

__global__ void millerRabinPrimalityTest(unsigned long *candidates, bool *isPrime, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && candidates[idx] > 3) {
        unsigned long d = candidates[idx] - 1;
        int s = 0;
        while (d % 2 == 0) {
            d /= 2;
            ++s;
        }
        isPrime[idx] = true;
        for (int k = 0; k < 5 && isPrime[idx]; ++k) {
            unsigned long a = 2 + rand() % (candidates[idx] - 4);
            unsigned long x = modPow(a, d, candidates[idx]);
            if (x == 1 || x == candidates[idx] - 1) continue;
            bool composite = true;
            for (int r = 0; r < s - 1 && composite; ++r) {
                x = modPow(x, 2, candidates[idx]);
                if (x == candidates[idx] - 1) {
                    composite = false;
                    break;
                }
            }
            if (composite) isPrime[idx] = false;
        }
    }
}

__global__ void generateRandomPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        primes[idx] = 2 + rand() % (limit - 1);
        bool isPrime = true;
        for (unsigned long i = 2; i * i <= primes[idx]; ++i) {
            if (primes[idx] % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (!isPrime) generateRandomPrimes<<<1, limit>>>(primes, limit);
    }
}

__global__ void pollardRho(unsigned long *numbers, unsigned long *factors, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && numbers[idx] > 1) {
        unsigned long x = 2;
        unsigned long y = 2;
        unsigned long d = 1;
        while (d == 1) {
            x = (x * x + 1) % numbers[idx];
            y = (y * y + 1) % numbers[idx];
            y = (y * y + 1) % numbers[idx];
            d = __gcd(abs(x - y), numbers[idx]);
        }
        factors[idx] = d == numbers[idx] ? 0 : d;
    }
}

__global__ void findFermatPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        primes[idx] = pow(2, idx + 1) + 1;
        bool isPrime = true;
        for (unsigned long i = 2; i * i <= primes[idx]; ++i) {
            if (primes[idx] % i == 0) {
                isPrime = false;
                break;
            }
        }
    }
}

__global__ void findMersennePrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        primes[idx] = (1ul << idx) - 1;
        bool isPrime = true;
        for (unsigned long i = 2; i * i <= primes[idx]; ++i) {
            if (primes[idx] % i == 0) {
                isPrime = false;
                break;
            }
        }
    }
}

__global__ void findTwinPrimes(unsigned long *primes, unsigned long *twin, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit - 1 && primes[idx] > 1 && primes[idx + 1] > 1) {
        twin[idx] = abs(primes[idx] - primes[idx + 1]) == 2 ? primes[idx] : 0;
    }
}

__global__ void findCousinPrimes(unsigned long *primes, unsigned long *cousin, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit - 1 && primes[idx] > 1 && primes[idx + 1] > 1) {
        cousin[idx] = abs(primes[idx] - primes[idx + 1]) == 4 ? primes[idx] : 0;
    }
}

__global__ void findSexyPrimes(unsigned long *primes, unsigned long *sexy, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit - 1 && primes[idx] > 1 && primes[idx + 1] > 1) {
        sexy[idx] = abs(primes[idx] - primes[idx + 1]) == 6 ? primes[idx] : 0;
    }
}

__global__ void findSafePrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && primes[idx] > 3) {
        unsigned long q = (primes[idx] - 1) / 2;
        bool isPrimeQ = true;
        for (unsigned long i = 2; i * i <= q; ++i) {
            if (q % i == 0) {
                isPrimeQ = false;
                break;
            }
        }
        primes[idx] = isPrimeQ ? primes[idx] : 0;
    }
}

__global__ void findSophieGermainPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && primes[idx] > 1) {
        unsigned long p2 = 2 * primes[idx] + 1;
        bool isPrimeP2 = true;
        for (unsigned long i = 2; i * i <= p2; ++i) {
            if (p2 % i == 0) {
                isPrimeP2 = false;
                break;
            }
        }
        primes[idx] = isPrimeP2 ? primes[idx] : 0;
    }
}

__global__ void findCircularPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && primes[idx] > 1) {
        unsigned long num = primes[idx];
        bool isCircularPrime = true;
        for (int i = 0; i < log10(num); ++i) {
            if (!isPrimeNumber(num)) {
                isCircularPrime = false;
                break;
            }
            num = rotateRight(num);
        }
        primes[idx] = isCircularPrime ? primes[idx] : 0;
    }
}

__global__ void findRepunitPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        unsigned long repunit = (pow(10, idx + 1) - 1) / 9;
        bool isPrime = true;
        for (unsigned long i = 2; i * i <= repunit; ++i) {
            if (repunit % i == 0) {
                isPrime = false;
                break;
            }
        }
        primes[idx] = isPrime ? repunit : 0;
    }
}

__global__ void findPalindromicPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && primes[idx] > 1) {
        bool isPalindrome = true;
        for (int i = 0; i < log10(primes[idx]) / 2; ++i) {
            if (getDigit(primes[idx], i) != getDigit(primes[idx], log10(primes[idx]) - i - 1)) {
                isPalindrome = false;
                break;
            }
        }
        primes[idx] = isPalindrome ? primes[idx] : 0;
    }
}

__global__ void findEmirpPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && primes[idx] > 1) {
        unsigned long reversedNum = reverseNumber(primes[idx]);
        bool isEmirp = isPrimeNumber(reversedNum) && primes[idx] != reversedNum;
        primes[idx] = isEmirp ? primes[idx] : 0;
    }
}

__global__ void findWieferichPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && pow(2, primes[idx]) - 1 % pow(primes[idx], 2) == 0) {
        primes[idx] = primes[idx];
    } else {
        primes[idx] = 0;
    }
}

__global__ void findWoodallPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - idx - 1)) {
        primes[idx] = pow(2, idx + 1) - idx - 1;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findWilliamsPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findXylophonePrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 1)) {
        primes[idx] = pow(2, idx + 1) - 1;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findYggdrasilPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findZetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPhiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findChiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findPsiPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findOmegaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findThetaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findRhoPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findSigmaPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findTauPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}

__global__ void findUpsilonPrimes(unsigned long *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrimeNumber(pow(2, idx + 1) - 3)) {
        primes[idx] = pow(2, idx + 1) - 3;
    } else {
        primes[idx] = 0;
    }
}
