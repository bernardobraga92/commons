#ifndef DIVISORRAMANUJANOUTERPROD_H
#define DIVISORRAMANUJANOUTERPROD_H

__global__ void isPrimeKernel(unsigned long long n, bool *isPrime) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= n/2) {
        if (n % idx == 0) {
            isPrime[0] = false;
        }
    }
}

__global__ void generatePrimesKernel(unsigned long long start, unsigned long long end, bool *isPrimeArray) {
    unsigned long long num = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (num <= end) {
        bool isPrime = true;
        for (unsigned long long i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0) {
                isPrime = false;
                break;
            }
        }
        isPrimeArray[num - start] = isPrime;
    }
}

__global__ void divisorCountKernel(unsigned long long n, unsigned long long *divisorCount) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1 && idx <= n) {
        if (n % idx == 0) {
            atomicAdd(divisorCount, 1);
        }
    }
}

__global__ void ramanujanDivisorsKernel(unsigned long long n, unsigned long long *divisors) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1 && idx <= n/2) {
        if (n % idx == 0) {
            divisors[idx] = idx;
        }
    }
}

__global__ void primeSumKernel(unsigned long long start, unsigned long long end, unsigned long long *primeSum) {
    unsigned long long num = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (num <= end) {
        bool isPrime = true;
        for (unsigned long long i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) atomicAdd(primeSum, num);
    }
}

__global__ void twinPrimesKernel(unsigned long long start, unsigned long long end, bool *twinPrimesArray) {
    unsigned long long num = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (num <= end - 2) {
        bool isPrime1 = true, isPrime2 = true;
        for (unsigned long long i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0) {
                isPrime1 = false;
                break;
            }
        }
        for (unsigned long long i = 2; i <= sqrt(num + 2); ++i) {
            if ((num + 2) % i == 0) {
                isPrime2 = false;
                break;
            }
        }
        twinPrimesArray[num - start] = (isPrime1 && isPrime2);
    }
}

__global__ void largestFactorKernel(unsigned long long n, unsigned long long *largestFactor) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1 && idx <= n/2) {
        if (n % idx == 0) {
            atomicMax(largestFactor, idx);
        }
    }
}

__global__ void primeCountKernel(unsigned long long start, unsigned long long end, unsigned long long *primeCount) {
    unsigned long long num = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (num <= end) {
        bool isPrime = true;
        for (unsigned long long i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) atomicAdd(primeCount, 1);
    }
}

__global__ void smallestFactorKernel(unsigned long long n, unsigned long long *smallestFactor) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= n/2) {
        if (n % idx == 0) {
            atomicMin(smallestFactor, idx);
        }
    }
}

__global__ void primeProductKernel(unsigned long long start, unsigned long long end, unsigned long long *primeProduct) {
    unsigned long long num = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (num <= end) {
        bool isPrime = true;
        for (unsigned long long i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) atomicMul(primeProduct, num);
    }
}

__global__ void primeFactorCountKernel(unsigned long long n, unsigned long long *primeFactorCount) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= n/2) {
        if (n % idx == 0 && isPrime(idx)) {
            atomicAdd(primeFactorCount, 1);
        }
    }
}

__global__ void primeDifferenceKernel(unsigned long long start, unsigned long long end, unsigned long long *primeDifferences) {
    unsigned long long num = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (num <= end - 2) {
        bool isPrime1 = true, isPrime2 = true;
        for (unsigned long long i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0) {
                isPrime1 = false;
                break;
            }
        }
        for (unsigned long long i = 2; i <= sqrt(num + 2); ++i) {
            if ((num + 2) % i == 0) {
                isPrime2 = false;
                break;
            }
        }
        if (isPrime1 && isPrime2) atomicAdd(primeDifferences, num + 2 - num);
    }
}

__global__ void primeSquareKernel(unsigned long long n, unsigned long long *primeSquare) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= n/2) {
        if (n % idx == 0 && isPrime(idx)) {
            atomicAdd(primeSquare, idx * idx);
        }
    }
}

__global__ void primeCubeKernel(unsigned long long n, unsigned long long *primeCube) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= n/2) {
        if (n % idx == 0 && isPrime(idx)) {
            atomicAdd(primeCube, idx * idx * idx);
        }
    }
}

__global__ void primeSquareRootKernel(unsigned long long n, unsigned long long *primeSquareRoot) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= n/2) {
        if (n % idx == 0 && isPrime(idx)) {
            atomicAdd(primeSquareRoot, sqrt(idx));
        }
    }
}

__global__ void primeCubeRootKernel(unsigned long long n, unsigned long long *primeCubeRoot) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= n/2) {
        if (n % idx == 0 && isPrime(idx)) {
            atomicAdd(primeCubeRoot, cbrt(idx));
        }
    }
}

__global__ void primeFactorSumKernel(unsigned long long n, unsigned long long *primeFactorSum) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= n/2) {
        if (n % idx == 0 && isPrime(idx)) {
            atomicAdd(primeFactorSum, idx);
        }
    }
}
