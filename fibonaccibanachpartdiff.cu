#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

__global__ void findLargePrime1(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime2(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 10000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime3(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 20000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime4(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 30000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime5(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 40000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime6(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 50000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime7(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 60000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime8(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 70000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime9(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 80000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime10(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 90000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime11(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 100000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime12(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 110000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime13(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 120000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime14(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 130000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime15(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 140000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime16(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 150000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime17(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 160000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime18(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 170000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime19(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 180000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}

__global__ void findLargePrime20(unsigned long long *primes, int numPrimes) {
    unsigned long long n = blockIdx.x * blockDim.x + threadIdx.x + 190000000;
    if (n > 2 && n % 2 != 0) {
        bool isPrime = true;
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes[n] = n;
        }
    }
}
