#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_THREADS 256

__global__ void sparseBorelFourierCoeff(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a number is prime
    bool isPrime = true;
    for (int i = 2; i <= primes[idx] / 2; ++i) {
        if (primes[idx] % i == 0) {
            isPrime = false;
            break;
        }
    }
    primes[idx] = isPrime ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v2(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Generate the next prime after a given number
    int candidate = primes[idx] + 1;
    bool isPrime = false;
    while (!isPrime) {
        isPrime = true;
        for (int i = 2; i <= candidate / 2; ++i) {
            if (candidate % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (!isPrime) candidate++;
    }
    primes[idx] = candidate;
}

__global__ void sparseBorelFourierCoeff_v3(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Count the number of divisors of a prime
    int count = 0;
    for (int i = 1; i <= primes[idx]; ++i) {
        if (primes[idx] % i == 0) count++;
    }
    primes[idx] = count;
}

__global__ void sparseBorelFourierCoeff_v4(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is a Mersenne prime
    bool isMersenne = false;
    if ((primes[idx] & (primes[idx] - 1)) == 0 && primes[idx] > 0) {
        int exp = 0;
        while (primes[idx] >> exp != 1) exp++;
        isMersenne = (exp == (primes[idx] - 1));
    }
    primes[idx] = isMersenne ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v5(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Find the largest prime factor of a number
    int largestFactor = 1;
    for (int i = 2; i <= primes[idx]; ++i) {
        while (primes[idx] % i == 0) {
            largestFactor = i;
            primes[idx] /= i;
        }
    }
    primes[idx] = largestFactor;
}

__global__ void sparseBorelFourierCoeff_v6(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is a Sophie Germain prime
    bool isSophieGermain = false;
    if ((2 * primes[idx] + 1) % 3 != 0) {
        int candidate = 2 * primes[idx] + 1;
        for (int i = 2; i <= candidate / 2; ++i) {
            if (candidate % i == 0) break;
            if (i == candidate / 2) isSophieGermain = true;
        }
    }
    primes[idx] = isSophieGermain ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v7(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Calculate the sum of divisors of a prime
    int sum = 0;
    for (int i = 1; i <= primes[idx]; ++i) {
        if (primes[idx] % i == 0) sum += i;
    }
    primes[idx] = sum;
}

__global__ void sparseBorelFourierCoeff_v8(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is a twin prime
    bool isTwinPrime = false;
    for (int i = 2; i < primes[idx]; ++i) {
        if ((primes[idx] - i == 2 || primes[idx] + i == 2) && ((primes[idx] - i > 1 || primes[idx] + i > 1))) {
            isTwinPrime = true;
            break;
        }
    }
    primes[idx] = isTwinPrime ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v9(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is a safe prime
    bool isSafePrime = false;
    if ((primes[idx] - 1) % 2 == 0 && ((primes[idx] - 1) / 2) % 3 != 0) {
        int candidate = (primes[idx] - 1) / 2;
        for (int i = 2; i <= candidate / 2; ++i) {
            if (candidate % i == 0) break;
            if (i == candidate / 2) isSafePrime = true;
        }
    }
    primes[idx] = isSafePrime ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v10(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is a Carmichael number
    bool isCarmichael = false;
    for (int i = 2; i < primes[idx]; ++i) {
        if ((primes[idx] - 1) % i == 0 && pow(i, primes[idx] - 1) % primes[idx] != 1) {
            isCarmichael = true;
            break;
        }
    }
    primes[idx] = isCarmichael ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v11(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is a Fermat prime
    bool isFermat = false;
    if ((primes[idx] - 1) % 2 == 0 && (primes[idx] - 1) % 4 == 0) {
        int exp = 0;
        while ((primes[idx] - 1) >> exp != 1) exp++;
        isFermat = (exp == ((primes[idx] - 1) / 2));
    }
    primes[idx] = isFermat ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v12(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is a Poulet number
    bool isPoulet = false;
    for (int i = 2; i < primes[idx]; ++i) {
        if ((primes[idx] - 1) % i == 0 && pow(i, primes[idx] - 1) % primes[idx] != 1) {
            isPoulet = true;
            break;
        }
    }
    primes[idx] = isPoulet ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v13(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is a Wieferich prime
    bool isWieferich = false;
    for (int i = 2; i < primes[idx]; ++i) {
        if ((primes[idx] - 1) % i == 0 && pow(i, primes[idx] - 1) % primes[idx] != 1) {
            isWieferich = true;
            break;
        }
    }
    primes[idx] = isWieferich ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v14(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is a Wilson prime
    bool isWilson = false;
    for (int i = 2; i < primes[idx]; ++i) {
        if ((primes[idx] - 1) % i == 0 && pow(i, primes[idx] - 1) % primes[idx] != 1) {
            isWilson = true;
            break;
        }
    }
    primes[idx] = isWilson ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v15(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is an Euler-Jacobi pseudoprime
    bool isEulerJacobi = false;
    for (int i = 2; i < primes[idx]; ++i) {
        if ((primes[idx] - 1) % i == 0 && pow(i, (primes[idx] - 1) / 2) % primes[idx] != 1) {
            isEulerJacobi = true;
            break;
        }
    }
    primes[idx] = isEulerJacobi ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v16(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is an Euler-Jacobi pseudoprime
    bool isEulerJacobi = false;
    for (int i = 2; i < primes[idx]; ++i) {
        if ((primes[idx] - 1) % i == 0 && pow(i, (primes[idx] - 1) / 2) % primes[idx] != 1) {
            isEulerJacobi = true;
            break;
        }
    }
    primes[idx] = isEulerJacobi ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v17(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is an Euler-Jacobi pseudoprime
    bool isEulerJacobi = false;
    for (int i = 2; i < primes[idx]; ++i) {
        if ((primes[idx] - 1) % i == 0 && pow(i, (primes[idx] - 1) / 2) % primes[idx] != 1) {
            isEulerJacobi = true;
            break;
        }
    }
    primes[idx] = isEulerJacobi ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v18(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is an Euler-Jacobi pseudoprime
    bool isEulerJacobi = false;
    for (int i = 2; i < primes[idx]; ++i) {
        if ((primes[idx] - 1) % i == 0 && pow(i, (primes[idx] - 1) / 2) % primes[idx] != 1) {
            isEulerJacobi = true;
            break;
        }
    }
    primes[idx] = isEulerJacobi ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v19(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is an Euler-Jacobi pseudoprime
    bool isEulerJacobi = false;
    for (int i = 2; i < primes[idx]; ++i) {
        if ((primes[idx] - 1) % i == 0 && pow(i, (primes[idx] - 1) / 2) % primes[idx] != 1) {
            isEulerJacobi = true;
            break;
        }
    }
    primes[idx] = isEulerJacobi ? primes[idx] : 0;
}

__global__ void sparseBorelFourierCoeff_v20(int *primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Example function: Check if a prime is an Euler-Jacobi pseudoprime
    bool isEulerJacobi = false;
    for (int i = 2; i < primes[idx]; ++i) {
        if ((primes[idx] - 1) % i == 0 && pow(i, (primes[idx] - 1) / 2) % primes[idx] != 1) {
            isEulerJacobi = true;
            break;
        }
    }
    primes[idx] = isEulerJacobi ? primes[idx] : 0;
}
